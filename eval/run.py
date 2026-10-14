"""Deterministic evaluation trials; protocol and CLI examples are in docs/evaluation.md."""
from __future__ import annotations

import os

# Set before NumPy/JAX imports, including programmatic Modal use.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import uuid

import mujoco
import numpy as np

from .sim import Policy, Simulator


@dataclass(frozen=True)
class SuiteConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    suites: tuple[str, ...] = ("tracking", "push", "getup")
    trial_seed: int = 20261014
    tracking_seconds: float = 10.0
    discard_seconds: float = 2.0
    go1_ranges: tuple = ((-1.5, 1.5), (-0.8, 0.8), (-1.2, 1.2))
    g1_ranges: tuple = ((-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0))
    velocity_points: int = 3
    go1_impulses: tuple[float, ...] = tuple(range(0, 51, 5))
    g1_impulses: tuple[float, ...] = (0, 10, 20, 30, 40, 60, 80, 100)
    push_trials: int = 100
    push_command: tuple[float, ...] = (0.5, 0.0, 0.0)
    push_wait_range: tuple[float, ...] = (2.0, 3.0)
    push_seconds: float = 0.1
    post_push_seconds: float = 3.0
    fall_tilt_degrees: float = 60.0
    getup_trials: int = 200
    drop_height: float = 0.5
    settle_seconds: float = 1.0
    getup_seconds: float = 6.0
    getup_tilt_degrees: float = 15.0
    getup_height: float = 0.2
    getup_linear_speed: float = 0.2
    getup_angular_speed: float = 0.5
    getup_hold_seconds: float = 1.0

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown suite config fields: {sorted(unknown)}")
        return cls(**values)

    def validate(self):
        if not self.seeds or any(s not in (0, 1, 2) for s in self.seeds):
            raise ValueError("Trained seeds must be a nonempty subset of 0,1,2")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate seeds")
        if not self.suites or set(self.suites) - {"tracking", "push", "getup"}:
            raise ValueError("Unknown or empty suite selection")
        if len(set(self.suites)) != len(self.suites):
            raise ValueError("Duplicate suite selection would duplicate raw trial IDs")
        if self.velocity_points < 2 or self.push_trials < 1 or self.getup_trials < 1:
            raise ValueError("Grid/trial counts must be positive (velocity_points >=2)")
        durations = (self.tracking_seconds, self.push_seconds, self.post_push_seconds,
                     self.settle_seconds, self.getup_seconds, self.getup_hold_seconds)
        if not all(math.isfinite(x) and x > 0 for x in durations):
            raise ValueError("Durations must be finite and positive")
        if not 0 <= self.discard_seconds < self.tracking_seconds:
            raise ValueError("Discard must be within the tracking episode")
        if not 0 < self.getup_hold_seconds <= self.getup_seconds:
            raise ValueError("Get-up hold must fit in the episode")
        if len(self.push_wait_range) != 2 or not 0 <= self.push_wait_range[0] <= self.push_wait_range[1]:
            raise ValueError("Invalid push wait range")
        for grid in (self.go1_impulses, self.g1_impulses):
            if not grid or any(not math.isfinite(x) or x < 0 for x in grid):
                raise ValueError("Impulse grid must be finite and nonnegative")
            if list(grid) != sorted(set(grid)):
                raise ValueError("Impulse grid must be strictly increasing")
        for ranges in (self.go1_ranges, self.g1_ranges):
            if len(ranges) != 3 or any(len(r) != 2 or not all(map(math.isfinite, r)) or r[0] >= r[1] for r in ranges):
                raise ValueError("Each robot needs three finite command intervals")
        if len(self.push_command) != 3 or not all(map(math.isfinite, self.push_command)):
            raise ValueError("Push command needs three finite components")
        if not 0 < self.getup_tilt_degrees < self.fall_tilt_degrees < 180:
            raise ValueError("Invalid tilt thresholds")
        if min(self.drop_height, self.getup_height, self.getup_linear_speed, self.getup_angular_speed) <= 0:
            raise ValueError("Get-up height/speed thresholds must be positive")


def _steps(seconds, dt):
    n = round(seconds / dt)
    if not math.isclose(n * dt, seconds, abs_tol=1e-8):
        raise ValueError(f"Duration {seconds} is not a multiple of control_dt={dt}")
    return n


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"Cannot record git provenance: {p.stderr.strip()}")
    return p.stdout.strip()


def _write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    return bool(np.isfinite(np.asarray(value)).all())


class TrialFailure(RuntimeError):
    pass


class TrialSimulator(Simulator):
    """Use the runtime's actor/timing; inspect every physics integration step."""

    def __init__(self, policy, root, config):
        super().__init__(policy, root=root)
        self.config = config
        self.base_id = self.model.body(policy.spec["force_body"]).id
        self.floor_id = self.model.geom(policy.spec.get("floor_geom", "floor")).id
        self.feet_ids = {self.model.geom(name).id for name in policy.spec["feet_geoms"]}
        self.fall_checks = True
        self.failure = None
        self.failure_time = None
        self._velocity = np.empty(6)
        self.recovery_steps = None
        self.recovery_time = None

    def begin(self, *, fall_checks):
        self.fall_checks = fall_checks
        self.failure = self.failure_time = None
        self.data.warning.number[:] = 0
        self.recovery_steps = None
        self.recovery_time = None

    def _fail(self, reason, when=None):
        if self.failure is None:
            self.failure = reason
            value = float(self.data.time if when is None else when)
            self.failure_time = value if math.isfinite(value) else None

    def tilt(self):
        # Read integrated freejoint orientation without recomputing actor sensors.
        q = self.data.qpos[3:7]
        norm2 = float(np.dot(q, q))
        if norm2 == 0 or not math.isfinite(norm2):
            return math.nan
        cosine = 1 - 2 * float(q[1] * q[1] + q[2] * q[2]) / norm2
        return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))

    def velocity(self):
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base_id, self._velocity, 1)
        return self._velocity

    def _inspect(self):
        if not all(_finite(x) for x in (self.data.qpos, self.data.qvel, self.data.qacc,
                                       self.data.sensordata, self.data.ctrl, self.data.xmat)):
            self._fail("nonfinite_state")
        elif not math.isfinite(self.tilt()):
            self._fail("invalid_orientation")
        elif np.any(self.data.warning.number):
            self._fail("mujoco_warning")
        elif self.fall_checks:
            if self.tilt() > self.config.fall_tilt_degrees:
                self._fail("tilt")
            for contact in self.data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                other = b if a == self.floor_id else a if b == self.floor_id else -1
                if contact.dist <= 0 and other >= 0 and self.model.geom_bodyid[other] != 0 and other not in self.feet_ids:
                    self._fail("nonfoot_floor_contact")
                    break
        if not self.failure and self.recovery_steps is not None:
            stable = (self.tilt() < self.config.getup_tilt_degrees and
                      self.data.qpos[2] >= self.config.getup_height and
                      np.linalg.norm(self.data.qvel[:3]) < self.config.getup_linear_speed and
                      np.linalg.norm(self.data.qvel[3:6]) < self.config.getup_angular_speed)
            self.recovery_steps = self.recovery_steps + 1 if stable else 0
            if (self.recovery_time is None and
                    self.recovery_steps * self.model.opt.timestep >= self.config.getup_hold_seconds - 1e-9):
                self.recovery_time = max(0.0, float(self.data.time) - self.config.getup_hold_seconds)

    def physics_step(self):
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            self._inspect()
            if self.failure:
                break

    def checked_step(self, command=None, force=None):
        observation = self.observe(command=command)
        if not _finite(observation):
            self._fail("nonfinite_observation")
        if self.failure:
            raise TrialFailure(self.failure)
        action = self.policy.action(observation)
        if not _finite(action):
            self._fail("nonfinite_action")
            raise TrialFailure(self.failure)
        result = self.step(command=command, action=action, force=force)
        if not _finite(result["observation"]):
            self._fail("nonfinite_observation")
        if self.failure:
            raise TrialFailure(self.failure)
        return result

    def collision_manifest(self):
        enabled = []
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == 0:
                continue
            collides = ((self.model.geom_contype[i] & self.model.geom_conaffinity[self.floor_id]) or
                        (self.model.geom_contype[self.floor_id] & self.model.geom_conaffinity[i]))
            explicit = any(self.floor_id in (int(a), int(b)) and i in (int(a), int(b))
                           for a, b in zip(self.model.pair_geom1, self.model.pair_geom2))
            if collides or explicit:
                enabled.append({"id": i, "name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i),
                                "foot": i in self.feet_ids})
        return {"floor": self.policy.spec.get("floor_geom", "floor"), "enabled_floor_geoms": enabled,
                "feet_only": bool(enabled) and all(g["foot"] for g in enabled)}


class MJXTrialSimulator(TrialSimulator):
    """Training-matched MJX Warp physics on CUDA, retaining actor/state timing."""

    def __init__(self, policy, root, config):
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
        self.jax, self.mjx = jax, mjx
        if policy.spec["training_config"]["impl"] != "warp":
            raise ValueError("The sim-to-sim route requires a policy trained with impl='warp'")
        self.device = jax.devices("cuda")[0]
        self._mjx_data = None
        super().__init__(policy, root, config)
        self._mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        training_config = policy.spec["training_config"]
        self._data_options = {key: training_config[key] for key in
                              ("naconmax", "naccdmax", "njmax", "nvmax") if key in training_config}
        foot = np.zeros(self.model.ngeom, dtype=bool)
        foot[list(self.feet_ids)] = True
        nonfoot_robot = jax.device_put((self.model.geom_bodyid != 0) & ~foot, self.device)
        fall_cos = math.cos(math.radians(config.fall_tilt_degrees))

        def advance(data, fall_checks):
            def substep(d, _):
                d = mjx.step(self._mjx_model, d)
                finite = (jnp.all(jnp.isfinite(d.qpos)) & jnp.all(jnp.isfinite(d.qvel)) &
                          jnp.all(jnp.isfinite(d.qacc)) & jnp.all(jnp.isfinite(d.sensordata)))
                # MJX 3.15.0 DataWarp has flattened global contact buffers.
                # The unused tail can contain stale contacts; inspect nacon only.
                c = d._impl
                a, b = c.contact__geom[:, 0], c.contact__geom[:, 1]
                other = jnp.where(a == self.floor_id, b, jnp.where(b == self.floor_id, a, -1))
                active = jnp.arange(c.contact__dist.shape[0]) < c.nacon[0]
                contact = jnp.any(active & (c.contact__dist <= 0) & (other >= 0) &
                                  nonfoot_robot[jnp.maximum(other, 0)])
                overflow = (c.nacon[0] > c.naconmax) | jnp.any(c.nefc > c.njmax)
                q = d.qpos[3:7]
                norm2 = jnp.dot(q, q)
                finite = finite & (norm2 > 0) & jnp.isfinite(norm2)
                tilt = (1 - 2 * (q[1] ** 2 + q[2] ** 2) / norm2) < fall_cos
                code = jnp.where(~finite, 1, jnp.where(overflow, 4,
                                 jnp.where(fall_checks & tilt, 2,
                                 jnp.where(fall_checks & contact, 3, 0))))
                return d, code
            return jax.lax.scan(substep, data, None, length=self.n_substeps)

        self._advance = jax.jit(advance, device=self.device)
        self._forward = jax.jit(lambda d: mjx.forward(self._mjx_model, d), device=self.device)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._mjx_data = None
        # Simulator.__init__ resets before the Warp model has been constructed.
        if hasattr(self, "_forward"):
            self._mjx_data = self.mjx.put_data(
                self.model, self.data, device=self.device, impl="warp", **self._data_options)
            self._mjx_data = self._forward(self._mjx_data)
            self.mjx.get_data_into(self.data, self.model, self._mjx_data)
        return self.observe()

    def physics_step(self):
        if self._mjx_data is None:
            self._mjx_data = self.mjx.put_data(
                self.model, self.data, device=self.device, impl="warp", **self._data_options)
        start = float(self.data.time)
        self._mjx_data = self._mjx_data.replace(
            ctrl=self.jax.device_put(self.data.ctrl, self.device),
            xfrc_applied=self.jax.device_put(self.data.xfrc_applied, self.device))
        self._mjx_data, codes = self._advance(self._mjx_data, self.fall_checks)
        codes = np.asarray(codes)
        self.mjx.get_data_into(self.data, self.model, self._mjx_data)
        bad = np.flatnonzero(codes)
        if len(bad):
            names = {1: "nonfinite_state", 2: "tilt", 3: "nonfoot_floor_contact", 4: "warp_capacity_overflow"}
            index = int(bad[0])
            self._fail(names[int(codes[index])], start + (index + 1) * self.model.opt.timestep)
        self._inspect()


def _rng(config, seed, suite, index):
    # Excludes policy arm/backend: matched Go1 and backend trials share draws.
    return np.random.default_rng(np.random.SeedSequence([config.trial_seed, seed, suite, index]))


def _tracking(sim, config, command, row):
    sim.reset(command=command)
    sim.begin(fall_checks=False)
    sim._inspect()
    dt = float(sim.policy.spec["control_dt"])
    errors = np.zeros(3)
    count = 0
    try:
        for i in range(_steps(config.tracking_seconds, dt)):
            sim.checked_step(command=command)
            if (i + 1) * dt > config.discard_seconds + 1e-9:
                v = sim.velocity()
                error = np.array([v[3] - command[0], v[4] - command[1], v[2] - command[2]])
                errors += np.abs(error)
                if not _finite(errors):
                    sim._fail("nonfinite_metric")
                    raise TrialFailure(sim.failure)
                count += 1
    finally:
        row["samples"] = count
        row["forward_mae"] = float(errors[0] / count) if count else None
        row["lateral_mae"] = float(errors[1] / count) if count else None
        row["yaw_mae"] = float(errors[2] / count) if count else None


def _push(sim, config, seed, impulse, index, row):
    rng = _rng(config, seed, 1, index)
    theta = float(rng.uniform(0, 2 * math.pi))
    dt = float(sim.policy.spec["control_dt"])
    # Quantize only the onset, never the force duration or impulse.
    wait_steps = round(float(rng.uniform(*config.push_wait_range)) / dt)
    row.update(theta=theta, push_onset=wait_steps * dt)
    sim.reset(command=config.push_command)
    sim.begin(fall_checks=True)
    sim._inspect()
    force = np.array([math.cos(theta), math.sin(theta), 0.0]) * impulse / config.push_seconds
    for _ in range(wait_steps):
        sim.checked_step(command=config.push_command)
    for _ in range(_steps(config.push_seconds, dt)):
        sim.checked_step(command=config.push_command, force=force)
    for _ in range(_steps(config.post_push_seconds, dt)):
        sim.checked_step(command=config.push_command)


def _getup(sim, config, seed, index, row):
    rng = _rng(config, seed, 2, index)
    qpos = np.array(sim.policy.spec["default_qpos"], dtype=float)
    qpos[:3] = (0.0, 0.0, config.drop_height)
    quat = rng.normal(size=4)
    qpos[3:7] = quat / np.linalg.norm(quat)
    row["initial_quaternion"] = qpos[3:7].tolist()
    sim.reset(qpos=qpos, qvel=np.zeros(sim.model.nv))
    sim.begin(fall_checks=False)
    sim.data.ctrl[:] = sim.policy.spec["default_pose"]
    dt = float(sim.policy.spec["control_dt"])
    for _ in range(_steps(config.settle_seconds, dt)):
        sim.physics_step()
        if sim.failure:
            raise TrialFailure(sim.failure)
    row["settled_qpos"] = sim.data.qpos.tolist()
    sim.reset(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy())
    sim.begin(fall_checks=False)
    sim.recovery_steps = 0
    for _ in range(_steps(config.getup_seconds, dt)):
        sim.checked_step()
    row["recovery_time"] = sim.recovery_time
    if sim.recovery_time is None:
        sim._fail("no_recovery")
        raise TrialFailure(sim.failure)


CSV_FIELDS = ("run_id", "trial_id", "backend", "policy", "robot", "task", "seed", "suite", "trial",
              "command", "impulse", "theta", "push_onset", "success", "failure", "error", "failure_time",
              "sim_seconds", "wall_seconds", "samples", "forward_mae", "lateral_mae", "yaw_mae", "recovery_time",
              "initial_quaternion", "settled_qpos")


def _safe_json(value):
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_policy(root, directory, name, backend, config, manifest):
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "manifest.json"
    manifest = dict(manifest, policies=[name], trial_count=0, policy_metadata={})
    started = time.monotonic()
    _write_manifest(path, manifest)
    try:
        policy_path = root / "policies" / f"{name}.json"
        policy = Policy.load(policy_path)
        spec = policy.spec
        simulator_class = MJXTrialSimulator if backend == "mjx" else TrialSimulator
        sim = simulator_class(policy, root, config)
        if f"-s{spec['seed']}" != name[-3:]:
            raise ValueError(f"Policy seed metadata mismatch: {name}")
        manifest["seeds"] = [int(spec["seed"])]
        dt = float(spec["control_dt"])
        for duration in (config.tracking_seconds, config.discard_seconds, config.push_seconds,
                         config.post_push_seconds, config.settle_seconds, config.getup_seconds,
                         config.getup_hold_seconds):
            _steps(duration, dt)
        ranges = config.go1_ranges if spec["robot"] == "go1" else config.g1_ranges
        if spec["task"] != "getup":
            for selected, trained, push in zip(ranges, spec["command_ranges"], config.push_command):
                if selected[0] < trained[0] or selected[1] > trained[1] or not trained[0] <= push <= trained[1]:
                    raise ValueError(f"Command outside trained range for {name}")
        # Hash the checkpoint rather than copying its large weight arrays into manifests.
        recorded_spec = {k: v for k, v in spec.items() if k != "policy"}
        recorded_spec["policy"] = {k: v for k, v in spec["policy"].items() if k != "layers"}
        manifest["policy_metadata"][name] = {
            "policy_sha256": _hash(policy_path), "model_xml": spec["model_xml"],
            "model_sha256": _hash(root / spec["model_xml"]), "spec": recorded_spec,
            "collisions": sim.collision_manifest()}
        _write_manifest(path, manifest)
        with (directory / "trials.jsonl").open("x") as raw, (directory / "trials.csv").open("x", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for suite in config.suites:
                if (spec["task"] == "getup") != (suite == "getup"):
                    continue
                if suite == "tracking":
                    commands = itertools.product(*(np.linspace(lo, hi, config.velocity_points).tolist() for lo, hi in ranges))
                    trials = ((i, list(command), None) for i, command in enumerate(commands))
                elif suite == "push":
                    impulses = config.go1_impulses if spec["robot"] == "go1" else config.g1_impulses
                    trials = ((i, list(config.push_command), float(j)) for j in impulses for i in range(config.push_trials))
                else:
                    trials = ((i, [0.0, 0.0, 0.0], None) for i in range(config.getup_trials))
                for index, command, impulse in trials:
                    identifier = f"{name}/{suite}/{impulse if impulse is not None else '-'}/{index}"
                    row = {"run_id": manifest["run_id"], "trial_id": identifier, "backend": backend,
                           "policy": name, "robot": spec["robot"], "task": spec["task"],
                           "seed": int(spec["seed"]), "suite": suite, "trial": index,
                           "command": command, "impulse": impulse, "success": False,
                           "failure": None, "error": None, "failure_time": None}
                    trial_start = time.monotonic()
                    sim.failure = sim.failure_time = None
                    try:
                        if suite == "tracking":
                            _tracking(sim, config, command, row)
                        elif suite == "push":
                            _push(sim, config, spec["seed"], impulse, index, row)
                        else:
                            _getup(sim, config, spec["seed"], index, row)
                        row["success"] = True
                    except Exception as exc:
                        row["failure"] = sim.failure or "simulator_error"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["failure_time"] = sim.failure_time
                    row["sim_seconds"] = float(sim.data.time)
                    row["wall_seconds"] = time.monotonic() - trial_start
                    row = _safe_json(row)
                    raw.write(json.dumps(row, allow_nan=False) + "\n")
                    raw.flush()
                    writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
                    cf.flush()
                    manifest["trial_count"] += 1
                    if manifest["trial_count"] % 100 == 0:
                        manifest["wall_seconds"] = time.monotonic() - started
                        _write_manifest(path, manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(path, manifest)
    return manifest


def run_suite(root, output, backend="mujoco", seed_filter=None, *, config=None,
              policy_filter=None, modal_app_id=None, provenance=None):
    """Write a new run with separate backend/policy directories; return its Path.

    Modal passes provenance={git_sha, git_dirty, modal_app_id, ...}; without it,
    git provenance is read from root.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    config = config or SuiteConfig()
    if isinstance(config, dict):
        config = SuiteConfig.from_dict(config)
    if seed_filter is not None:
        config = replace(config, seeds=tuple(seed_filter))
    config.validate()
    if backend not in ("mujoco", "mjx"):
        raise ValueError("backend must be mujoco or mjx")
    if backend == "mjx":
        if 0 not in config.seeds or (seed_filter is not None and tuple(config.seeds) != (0,)):
            raise ValueError("The MJX gap protocol uses trained seed 0 only")
        config = replace(config, seeds=(0,), suites=tuple(s for s in config.suites if s != "getup"))
        if not config.suites:
            raise ValueError("MJX gap route evaluates tracking and/or pushes only")
    names = [f"{arm}-s{seed}" for arm in ("go1-walk", "go1-walk-push", "go1-getup", "g1-walk")
             for seed in config.seeds if not (backend == "mjx" and arm == "go1-getup")]
    if policy_filter:
        selected = [policy_filter] if isinstance(policy_filter, (str, Path)) else policy_filter
        requested = {Path(p).stem for p in selected}
        unknown = requested - set(names)
        if unknown:
            raise ValueError(f"Policies outside the selected seed/backend set: {sorted(unknown)}")
        names = [n for n in names if n in requested]
    names = [n for n in names if (("getup" in n and "getup" in config.suites) or
                                 ("getup" not in n and set(config.suites) & {"tracking", "push"}))]
    if not names:
        raise ValueError("No policies selected")
    missing = [n for n in names if not (root / "policies" / f"{n}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing policy exports (select available policies explicitly): {missing}")
    if provenance is None:
        provenance = {"git_sha": _git(root, "rev-parse", "HEAD"),
                      "git_dirty": bool(_git(root, "status", "--porcelain"))}
    if not provenance.get("git_sha") or not isinstance(provenance.get("git_dirty"), bool):
        raise ValueError("Provenance requires git_sha and boolean git_dirty")
    if backend == "mjx" and provenance.get("gpu") in (None, "", "none"):
        raise ValueError("MJX Warp evaluation requires Modal GPU provenance['gpu']")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("numpy", "mujoco", "mujoco-mjx", "mujoco-warp", "warp-lang", "jax", "jaxlib", "brax", "mujoco-playground"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    effective = json.loads(json.dumps(asdict(config)))
    default = json.loads(json.dumps(asdict(SuiteConfig())))
    changes = {k: v for k, v in effective.items() if k not in ("seeds", "suites") and v != default[k]}
    manifest = {"schema_version": 1, "run_id": run_id, "git_sha": provenance["git_sha"],
                "git_dirty": provenance["git_dirty"], "provenance": provenance,
                "config": effective, "standard_protocol": not changes, "protocol_changes": changes,
                "seeds": sorted({int(n.rsplit("-s", 1)[1]) for n in names}), "policies": names,
                "package_versions": versions, "python": platform.python_version(),
                "platform": platform.platform(), "gpu": provenance.get("gpu", "none"), "backend": backend,
                "backend_detail": "mjx-warp" if backend == "mjx" else "mujoco",
                "modal_app_id": modal_app_id or provenance.get("modal_app_id") or os.environ.get("MODAL_APP_ID"),
                "started_at": datetime.now(timezone.utc).isoformat(), "wall_seconds": None,
                "status": "running", "trial_count": 0, "policy_metadata": {}}
    path = run_dir / "manifest.json"
    _write_manifest(path, manifest)
    started = time.monotonic()
    try:
        for name in names:
            completed = _run_policy(root, run_dir / backend / name, name, backend, config, manifest)
            manifest["policy_metadata"].update(completed["policy_metadata"])
            manifest["trial_count"] += completed["trial_count"]
            manifest["wall_seconds"] = time.monotonic() - started
            _write_manifest(path, manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(path, manifest)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("mujoco", "mjx"), default="mujoco")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--policy", action="append")
    parser.add_argument("--suite", choices=("tracking", "push", "getup"), action="append")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--modal-app-id")
    parser.add_argument("--provenance", type=Path, help="JSON git_sha/git_dirty/gpu/Modal metadata")
    args = parser.parse_args()
    config = SuiteConfig.from_dict(json.loads(args.config.read_text())) if args.config else SuiteConfig()
    if args.suite:
        config = replace(config, suites=tuple(args.suite))
    provenance = json.loads(args.provenance.read_text()) if args.provenance else None
    print(run_suite(args.root, args.output, args.backend, args.seeds, config=config,
                    policy_filter=args.policy, modal_app_id=args.modal_app_id, provenance=provenance))


if __name__ == "__main__":
    main()
