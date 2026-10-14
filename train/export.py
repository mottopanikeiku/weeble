"""Export the pinned Playground/Brax PPO policies and reproducible demo traces.

The parameter layout and formulas follow Brax 0.14.2 training/agents/ppo/{train,
networks}.py, training/networks.py and training/acme/running_statistics.py.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform

import numpy as np


TASKS = {
    "Go1JoystickFlatTerrain": ("go1", "walk"),
    "Go1JoystickRoughTerrain": ("go1", "walk"),
    "Go1Getup": ("go1", "getup"),
    "G1JoystickFlatTerrain": ("g1", "walk"),
    "G1Getup": ("g1", "getup"),
}


def _json(value):
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if hasattr(value, "dtype"):
        return _json(np.asarray(value).tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _config_numbers(value):
    if isinstance(value, dict):
        return {key: _config_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_config_numbers(item) for item in value]
    if isinstance(value, str) and value in ("inf", "-inf", "nan"):
        return float(value)
    return value


def _layers(params):
    # Flax MLP.init returns {'params': {'hidden_0': {'kernel', 'bias'}, ...}}.
    dense = params["params"]
    names = [f"hidden_{index}" for index in range(len(dense))]
    if set(dense) != set(names):
        raise ValueError(f"Expected upstream dense MLP layers, got {list(dense)}")
    layers = []
    for name in names:
        kernel, bias = np.asarray(dense[name]["kernel"]), np.asarray(dense[name]["bias"])
        if set(dense[name]) != {"kernel", "bias"} or kernel.shape[1:] != bias.shape:
            raise ValueError(f"Unsupported dense layer {name}")
        layers.append({"name": name, "kernel": kernel, "bias": bias})
    return layers


def _encode_layers(layers):
    encoded = []
    for layer in layers:
        item = {"name": layer["name"], "encoding": "base64-f32-le"}
        for name in ("kernel", "bias"):
            value = np.asarray(layer[name], dtype="<f4")
            item[name] = base64.b64encode(value.tobytes(order="C")).decode("ascii")
            item[f"{name}_shape"] = list(value.shape)
        encoded.append(item)
    return encoded


def _observation_spec(robot, task, nu):
    fields = []

    def add(name, size, source):
        offset = sum(field["size"] for field in fields)
        fields.append({"name": name, "offset": offset, "size": size, "source": source})

    if task == "walk":
        add("local_linear_velocity", 3, "sensordata[local_linvel sensor]; IMU-site local velocity")
    add("gyro", 3, "sensordata[gyro sensor]; IMU-site local angular velocity")
    add("gravity", 3, "site_xmat[gravity_site].T @ [0,0,-1]")
    if robot == "g1" and task == "walk":
        add("command", 3, "[vx,vy,wz] in the robot frame, m/s,m/s,rad/s")
    add("joint_position", nu, "qpos[7:] - default_pose")
    add("joint_velocity", nu, "qvel[6:]")
    add("last_action", nu, "observation_last_action (unsmoothed policy action)")
    if robot == "go1" and task == "walk":
        add("command", 3, "[vx,vy,wz] in the robot frame, m/s,m/s,rad/s")
    if robot == "g1" and task == "walk":
        add("phase_cos", 2, "cos(observation_phase), left then right")
        add("phase_sin", 2, "sin(observation_phase), left then right")
    return {
        "key": "state", "size": sum(field["size"] for field in fields),
        "fields": fields, "noise_level": 0.0, "history_len": 1,
        "history_note": "No frame stack; upstream history_len configuration is unused.",
        "state_update": [
            "reset: last_action=observation_last_action=zeros(nu); phase=observation_phase=initial_phase",
            "choose action from current observation; hold motor targets and force for n_substeps",
            "poststep: observation_last_action=last_action; observation_phase=phase; construct observation",
            "then last_action=action; phase=fmod(phase+phase_dt+pi,2*pi)-pi (G1 only)",
            "observe() reads the observation-time history without advancing it",
        ],
        "action_history": "With a[t] applied at step t, o[t+1] contains a[t-1], not a[t]; reset history is zero.",
        "sensor_timing": "Keep mj_step sensor stage; no extra mj_forward after the final substep.",
    }


def export_policy(env_name, env, params, destination, seed, model_xml,
                  playground_commit, run_id):
    """Write browser JSON + sibling .brax containing the untouched full PPO tuple.

    Use the upstream task's network_factory defaults in train/app.py. The .brax
    sidecar is a trusted, pickle-based Modal artifact, not a browser asset.
    Returns the JSON-compatible spec. No rollout or parity check is run here.
    """
    import jax
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground._src import mjx_env
    from mujoco_playground.config import locomotion_params

    robot, task = TASKS[env_name]
    env = env.unwrapped
    if len(params) != 3:
        raise ValueError("Pass the full (normalizer, policy, value) tuple returned by PPO.train")
    normalizer, policy_params, value_params = params
    layers = _layers(policy_params)
    value_layers = _layers(value_params)
    if env_name == "G1Getup":
        from train.g1_getup import brax_ppo_config
        ppo = brax_ppo_config()
    else:
        ppo = locomotion_params.brax_ppo_config(env_name)
    network = ppo.network_factory.to_dict()
    signature = inspect.signature(ppo_networks.make_ppo_networks)
    activation = network.get("activation", signature.parameters["activation"].default)
    activation = activation if isinstance(activation, str) else activation.__name__
    distribution = network.get("distribution_type", signature.parameters["distribution_type"].default)
    if activation not in ("swish", "silu") or distribution != "tanh_normal":
        raise ValueError("This exporter targets upstream swish + NormalTanh PPO")
    nu = env.action_size
    obs = _observation_spec(robot, task, nu)
    expected = [obs["size"], *network["policy_hidden_layer_sizes"], 2 * nu]
    if [layers[0]["kernel"].shape[0], *[len(x["bias"]) for x in layers]] != expected:
        raise ValueError("Policy dimensions differ from the upstream task defaults")
    mean, std = normalizer.mean, normalizer.std
    value_key = network["value_obs_key"]
    value_expected = [np.asarray(mean[value_key]).size, *network["value_hidden_layer_sizes"], 1]
    if [value_layers[0]["kernel"].shape[0], *[len(x["bias"]) for x in value_layers]] != value_expected:
        raise ValueError("Value dimensions differ from the upstream task defaults")
    config = env._config.to_dict()
    if robot == "g1":
        ranges = ([config[name] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")]
                  if task == "walk" else [[0, 0]] * 3)
        sensors = {"gyro": "gyro_pelvis", "local_linvel": "local_linvel_pelvis",
                   "gravity_site": "imu_in_pelvis", "termination_upvector": "upvector_torso"}
        command_sampling = ("Independent uniform components; entire command zero with probability 0.1; refresh after 501 steps."
                            if task == "walk" else "No command input.")
    else:
        ranges = [[-x, x] for x in config["command_config"]["a"]] if task == "walk" else [[0, 0]] * 3
        sensors = {"gyro": "gyro", "local_linvel": "local_linvel",
                   "gravity_site": "imu", "termination_upvector": "upvector"}
        command_sampling = (
            "Reset uniform(-a,a); then every round(Exp(1)*5/control_dt) steps: "
            "x_next=x-w*(x-y*z), y~uniform(-a,a), w~Bernoulli(0.5), z~Bernoulli(b)."
            if task == "walk" else "No command input."
        )
    model = env.mj_model
    array_fields = ("dof_damping", "actuator_gainprm", "actuator_biasprm",
                    "actuator_ctrlrange", "jnt_range")
    option_fields = ("timestep", "ccd_iterations", "iterations", "ls_iterations",
                     "integrator", "solver", "cone", "disableflags", "enableflags")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".brax")
    brax_model.save_params(str(sidecar), jax.device_get(params))
    spec = {
        "schema_version": 1, "env_name": env_name, "robot": robot, "task": task,
        "seed": int(seed), "run_id": str(run_id), "model_xml": str(model_xml),
        "versions": {
            "python": platform.python_version(), "playground_commit": playground_commit,
            "menagerie_commit": mjx_env.MENAGERIE_COMMIT_SHA,
            **{name: importlib.metadata.version(name)
               for name in ("mujoco", "mujoco-mjx", "brax", "jax", "flax", "numpy")},
        },
        "brax_params_file": sidecar.name,
        "brax_params_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "training_config": config,
        "ppo_network_config": network,
        "default_qpos": env._init_q,
        "default_qvel": np.zeros(model.nv, np.float32),
        "default_pose": env._default_pose,
        "action_size": nu,
        "observation_size": {key: np.asarray(value).size for key, value in mean.items()},
        "force_body": "pelvis" if robot == "g1" else "trunk",
        "feet_geoms": ["left_foot", "right_foot"] if robot == "g1" else ["FR", "FL", "RR", "RL"],
        "floor_geom": "floor",
        "control_dt": env.dt, "simulation_dt": env.sim_dt, "n_substeps": env.n_substeps,
        "command_ranges": ranges, "command_sampling": command_sampling,
        "sensors": sensors, "observation": obs,
        "control": {
            "target": "current_pose" if task == "getup" else "default_pose",
            "action_scale": config["action_scale"],
            "formula": "ctrl=(qpos[7:] if getup else default_pose)+action_scale*action",
            "target_update": "Compute once per control step, hold through all internal substeps.",
            "clip": None,
            "limits": "No Python action/target clipping; MuJoCo actuator ctrl/force limits still apply.",
        },
        "gait": {
            "frequency_range": [1.25, 1.5], "default_frequency": 1.375,
            "initial_phase": np.asarray([0, np.pi], np.float32),
            "phase_dt_formula": "2*pi*control_dt*gait_frequency",
            "phase_update": "fmod(phase+phase_dt+pi,2*pi)-pi; AFTER observation, even at zero command",
        } if robot == "g1" and task == "walk" else None,
        "model_parameters": {
            "arrays": {name: getattr(model, name) for name in array_fields},
            "options": {name: getattr(model.opt, name) for name in option_fields},
        },
        "normalizer": {
            "enabled": bool(ppo.normalize_observations),
            "mean": mean, "std": std,
            "clip": inspect.signature(running_statistics.normalize).parameters["max_abs_value"].default,
            "formula": "(observation-mean)/std; no added epsilon; clip only when non-null",
        },
        "policy": {
            "obs_key": network["policy_obs_key"], "activation": activation,
            "layers": _encode_layers(layers), "layer_sizes": expected,
            "kernel_layout": "input_output", "output_activation": "linear",
            "distribution": "tanh_normal", "action_formula": "tanh(logits[..., :action_size])",
            "scale_formula": "softplus(logits[..., action_size:])+0.001; unused by deterministic inference",
        },
        "value": {"obs_key": value_key, "layer_sizes": value_expected,
                  "weights": "Stored only in brax_params_file; unused by inference."},
        "runtime": {"observation_noise": False, "random_pushes": False,
                    "command_resampling": False, "automatic_reset": False},
    }
    spec = _json(spec)
    destination.write_text(json.dumps(spec, separators=(",", ":"), allow_nan=False) + "\n")
    return spec


def load_reference_env(spec, impl="jax"):
    """The actual pinned task, with observation noise off and chosen MJX backend."""
    from ml_collections import ConfigDict
    from mujoco_playground import registry

    if spec["env_name"] == "G1Getup":
        from train.g1_getup import register
        register()
    config = ConfigDict(_config_numbers(spec["training_config"]))
    config.impl = impl
    config.noise_config.level = 0.0
    return registry.load(spec["env_name"], config=config)


def write_parity_fixture(policy_path, destination, root=Path("."), steps=200,
                         seed=0, command=None, impl="jax"):
    """Write a C MuJoCo rollout from an actual upstream seeded reset.

    This records reference data, not a claim that a second runtime matched it.
    The default demo seed is fixed independently of the training seed.
    """
    import jax
    from eval.sim import Policy, Simulator

    root, policy_path = Path(root), Path(policy_path)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    policy = Policy.load(policy_path)
    env = load_reference_env(policy.spec, impl=impl)
    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    command = (state.info.get("command", np.zeros(3)) if command is None else command)
    gait_frequency = None
    phase = None
    if policy.spec["gait"] is not None:
        phase = np.asarray(state.info["phase"])
        gait_frequency = float(np.asarray(state.info["phase_dt"])[0] / (2 * np.pi * env.dt))
    sim = Simulator(policy, root=root)
    sim.reset(qpos=np.asarray(state.data.qpos), qvel=np.asarray(state.data.qvel),
              command=command, gait_frequency=gait_frequency, phase=phase)
    initial = {
        "qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
        "command": sim.command.copy(), "gait_frequency": sim.gait_frequency,
        "phase": sim.phase, "observation": sim.observe(),
        "last_action": sim.last_action.copy(),
    }
    # Copy phase: the Simulator mutates it during the rollout.
    if initial["phase"] is not None:
        initial["phase"] = initial["phase"].copy()
    frames = []
    for index in range(steps):
        observation = sim.observe()
        result = sim.step()
        frames.append({
            "step": index, "observation": observation, "action": result["action"],
            "motor_targets": sim.data.ctrl.copy(), "qpos": result["qpos"],
            "qvel": sim.data.qvel.copy(), "next_observation": result["observation"],
        })
    fixture = _json({
        "schema_version": 1, "policy": str(Path("policies") / policy_path.name),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "model_xml": policy.spec["model_xml"], "env_name": policy.spec["env_name"],
        "demo_seed": seed, "reset_backend": impl, "physics_backend": "mujoco",
        "versions": policy.spec["versions"], "control_dt": policy.spec["control_dt"],
        "tolerance": {"observation_atol": 1e-5, "action_atol": 1e-5, "rtol": 1e-5},
        "initial": initial, "frames": frames,
    })
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n")
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--command", type=float, nargs=3)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    args = parser.parse_args()
    write_parity_fixture(args.policy, args.destination, root=args.root, steps=args.steps,
                         seed=args.seed, command=args.command, impl=args.impl)


if __name__ == "__main__":
    main()
