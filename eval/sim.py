"""Noise-free NumPy policy inference and C MuJoCo control, without JAX/Brax."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np


class Policy:
    def __init__(self, spec: dict):
        self.spec = spec
        if spec["schema_version"] != 1:
            raise ValueError("Unsupported policy schema")
        network = spec["policy"]
        if network["activation"] not in ("swish", "silu"):
            raise ValueError("Expected the upstream swish policy")
        self.layers = []
        for layer in network["layers"]:
            if layer["encoding"] != "base64-f32-le":
                raise ValueError("Expected little-endian float32 weights")
            self.layers.append(tuple(
                np.frombuffer(base64.b64decode(layer[name], validate=True), dtype="<f4")
                .reshape(layer[f"{name}_shape"])
                for name in ("kernel", "bias")
            ))
        key = network["obs_key"]
        self.mean = np.asarray(spec["normalizer"]["mean"][key], np.float32)
        self.std = np.asarray(spec["normalizer"]["std"][key], np.float32)
        self.action_size = spec["action_size"]

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls(json.loads(Path(path).read_text()))

    def normalize(self, observation):
        if isinstance(observation, Mapping):
            observation = observation[self.spec["policy"]["obs_key"]]
        x = np.asarray(observation, np.float32)
        if x.shape[-1] != self.mean.size:
            raise ValueError(f"Expected {self.mean.size} observations, got {x.shape}")
        if self.spec["normalizer"]["enabled"]:
            x = (x - self.mean) / self.std
            clip = self.spec["normalizer"]["clip"]
            if clip is not None:
                x = np.clip(x, -clip, clip)
        return x

    @staticmethod
    def _swish(x):
        # Stable sigmoid, retaining float32 even for saturated logits.
        e = np.exp(-np.abs(x))
        return x * np.where(x >= 0, 1 / (1 + e), e / (1 + e))

    def _hidden(self, observation):
        x = self.normalize(observation)
        for kernel, bias in self.layers[:-1]:
            x = self._swish(x @ kernel + bias)
        return x

    def logits(self, observation):
        kernel, bias = self.layers[-1]
        return self._hidden(observation) @ kernel + bias

    def action(self, observation):
        # The scale half is irrelevant to deterministic NormalTanh.mode.
        kernel, bias = self.layers[-1]
        return np.tanh(
            self._hidden(observation) @ kernel[:, :self.action_size]
            + bias[:self.action_size]
        )


def apply_model_parameters(model, spec):
    """Apply training-time model edits; also idempotent on exported exact XML."""
    for name, value in spec["model_parameters"]["arrays"].items():
        getattr(model, name)[:] = value
    for name, value in spec["model_parameters"]["options"].items():
        setattr(model.opt, name, value)


def observation_from_data(model, data, spec, last_action, command, phase=None):
    """Actor state in upstream order, at the caller's existing sensor stage.

    In particular, do not call mj_forward here: mj_step returns integrated qpos
    alongside sensors evaluated during its last internal substep, as does MJX.
    """
    sensors = spec["sensors"]

    def sensor(name):
        sensor_id = model.sensor(name).id
        start = model.sensor_adr[sensor_id]
        return np.asarray(
            data.sensordata[start:start + model.sensor_dim[sensor_id]], np.float32
        )

    site_id = model.site(sensors["gravity_site"]).id
    rotation = np.asarray(data.site_xmat[site_id], np.float32).reshape(3, 3)
    values = {
        "gyro": sensor(sensors["gyro"]),
        "gravity": -rotation[2],  # R_site.T @ [0, 0, -1], not framezaxis.
        "joint_position": np.asarray(data.qpos[7:], np.float32)
        - np.asarray(spec["default_pose"], np.float32),
        "joint_velocity": np.asarray(data.qvel[6:], np.float32),
        "last_action": np.asarray(last_action, np.float32),
        "command": np.asarray(command, np.float32),
    }
    if spec["task"] == "walk":
        # A site velocimeter includes omega x IMU offset, unlike root qvel.
        values["local_linear_velocity"] = sensor(sensors["local_linvel"])
    if spec["gait"] is not None:
        phase = np.asarray(phase, np.float32)
        values["phase_cos"] = np.cos(phase)
        values["phase_sin"] = np.sin(phase)
    return np.concatenate([values[f["name"]] for f in spec["observation"]["fields"]])


class Simulator:
    """External-command controller. No training noise, random pushes or autoreset.

    `observation_last_action`/`observation_phase` belong to the returned observation;
    `last_action`/`phase` are the already-advanced upstream info for the next step.
    """

    def __init__(self, policy: Policy, root: Path = Path(".")):
        self.policy = policy
        self.spec = policy.spec
        self.model = mujoco.MjModel.from_xml_path(str(root / self.spec["model_xml"]))
        apply_model_parameters(self.model, self.spec)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = self.spec["n_substeps"]
        self.force_body_id = self.model.body(self.spec["force_body"]).id
        self.default_pose = np.asarray(self.spec["default_pose"], np.float32)
        self.reset()

    def reset(self, qpos=None, qvel=None, command=(0, 0, 0), *,
              gait_frequency=None, phase=None):
        """Set the supplied physical state exactly; no drop or settling is added."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.spec["default_qpos"] if qpos is None else qpos
        self.data.qvel[:] = self.spec["default_qvel"] if qvel is None else qvel
        self.data.ctrl[:] = self.data.qpos[7:]
        self.command = np.asarray(command, np.float32).copy()
        self.last_action = np.zeros(self.model.nu, np.float32)
        self.observation_last_action = self.last_action.copy()
        gait = self.spec["gait"]
        self.gait_frequency = None
        self.phase = None
        self.observation_phase = None
        self.phase_dt = None
        if gait is not None:
            self.gait_frequency = (
                gait["default_frequency"] if gait_frequency is None else gait_frequency
            )
            self.phase = np.asarray(gait["initial_phase"] if phase is None else phase,
                                    np.float32).copy()
            self.observation_phase = self.phase.copy()
            self.phase_dt = np.asarray(
                [2 * np.pi * self.spec["control_dt"] * self.gait_frequency], np.float32
            )
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self, command=None):
        """Read current observation without advancing action or gait history."""
        if command is not None:
            self.command[:] = command
        return observation_from_data(
            self.model, self.data, self.spec, self.observation_last_action,
            self.command, self.observation_phase,
        )

    def physics_step(self):
        """Advance held controls/forces only; subclasses can use an MJX backend."""
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def step(self, command=None, action=None, force=None):
        observation = self.observe(command)
        action = (self.policy.action(observation) if action is None
                  else np.asarray(action, np.float32))
        target = (np.asarray(self.data.qpos[7:], np.float32)
                  if self.spec["control"]["target"] == "current_pose"
                  else self.default_pose)
        self.data.ctrl[:] = target + np.float32(self.spec["control"]["action_scale"]) * action
        self.data.xfrc_applied[:] = 0
        if force is not None:
            self.data.xfrc_applied[self.force_body_id, :3] = force
        self.physics_step()
        # Upstream calls _get_obs BEFORE updating info.last_act and info.phase.
        self.observation_last_action[:] = self.last_action
        if self.phase is not None:
            self.observation_phase[:] = self.phase
        next_observation = self.observe()
        self.last_action[:] = action
        if self.phase is not None:
            pi = np.float32(np.pi)
            self.phase[:] = np.fmod(self.phase + self.phase_dt + pi, 2 * pi) - pi
        return {
            "observation": next_observation,
            "action": action.copy(),
            "qpos": self.data.qpos.copy(),
        }
