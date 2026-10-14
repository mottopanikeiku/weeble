"""Artifact parity; run on Modal with the pinned Playground, models and .brax files.

PUSHABLE_ROOT selects the checkout; PUSHABLE_PARITY_IMPL=jax (CPU) or warp (GPU).
PUSHABLE_PARITY_POLICIES optionally selects comma-separated paths relative to root.
Missing trained artifacts are errors, not skipped tests or initialized substitutes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from eval.sim import Policy, Simulator
from train.export import load_reference_env


ROOT = Path(os.environ.get("PUSHABLE_ROOT", Path(__file__).resolve().parents[1]))
IMPL = os.environ.get("PUSHABLE_PARITY_IMPL", "jax")
ATOL = RTOL = 1e-5
DEFAULT_POLICIES = [
    "policies/go1-walk-s0.json", "policies/go1-getup-s0.json", "policies/g1-walk-s0.json",
]
POLICIES = (
    os.environ["PUSHABLE_PARITY_POLICIES"].split(",")
    if "PUSHABLE_PARITY_POLICIES" in os.environ
    else sorted(set(DEFAULT_POLICIES) | {
        str(path.relative_to(ROOT)) for path in (ROOT / "policies").glob("*-s[012].json")
    })
)


def _load_artifact(path, brax_path=None, root=ROOT, impl=IMPL):
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground.config import locomotion_params

    path, root = Path(path), Path(root)
    assert path.is_file(), f"Missing trained policy artifact: {path}"
    policy = Policy.load(path)
    sidecar = Path(brax_path) if brax_path is not None else path.parent / policy.spec["brax_params_file"]
    assert sidecar.is_file(), f"Missing original PPO parameters: {sidecar}"
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == policy.spec["brax_params_sha256"]
    # Brax uses pickle: load only the sidecar produced by our own training run.
    params = brax_model.load_params(str(sidecar))
    config = locomotion_params.brax_ppo_config(policy.spec["env_name"])
    networks = ppo_networks.make_ppo_networks(
        observation_size={key: value.shape for key, value in params[0].mean.items()},
        action_size=policy.spec["action_size"],
        preprocess_observations_fn=running_statistics.normalize,
        **config.network_factory,
    )
    inference = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
    env = load_reference_env(policy.spec, impl=impl)
    initial = jax.jit(env.reset)(jax.random.PRNGKey(0))
    return policy, Simulator(policy, root=root), env, initial, params, networks, inference


@pytest.fixture(scope="module", params=POLICIES)
def artifact(request):
    return _load_artifact(ROOT / request.param)


def _assert_close(metrics, name, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL,
                               equal_nan=False, err_msg=name)
    error = float(np.max(np.abs(actual - expected)))
    metrics[name] = max(metrics.get(name, 0.0), error)


def _get_observation(env, data, info, is_g1):
    info = dict(info)  # Upstream _get_obs advances its noise PRNG even at level 0.
    if is_g1:
        contact = jp.array([
            data.sensordata[env.mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in env._feet_floor_found_sensor
        ])
        return env._get_obs(data, info, contact)
    return env._get_obs(data, info)


def _check_observation_and_network(artifact, metrics):
    """Independent C/MJX forward sensors, upstream observation, original Brax net."""
    from mujoco_playground._src import mjx_env

    policy, sim, env, initial, params, networks, inference = artifact
    is_g1 = policy.spec["robot"] == "g1"
    forward = jax.jit(lambda data: mjx.forward(env.mjx_model, data))
    get_obs = jax.jit(lambda data, info: _get_observation(env, data, info, is_g1))
    logits = jax.jit(lambda obs: networks.policy_network.apply(params[0], params[1], obs))
    rng = np.random.default_rng(0)
    for index in range(16):
        qpos = np.asarray(initial.data.qpos).copy()
        qpos[:2] += rng.uniform(-0.3, 0.3, 2)
        # Airborne samples isolate the sensor conventions from solver differences.
        qpos[2] += 1.0
        quat = rng.normal(size=4)
        qpos[3:7] = quat / np.linalg.norm(quat)
        qpos[7:] += rng.uniform(-0.15, 0.15, sim.model.nu)
        qvel = rng.uniform(-1.0, 1.0, sim.model.nv).astype(np.float32)
        ranges = np.asarray(policy.spec["command_ranges"])
        command = rng.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)
        sim.reset(qpos, qvel, command)
        previous = rng.uniform(-1, 1, sim.model.nu).astype(np.float32)
        sim.observation_last_action[:] = previous
        info = dict(initial.info)
        info["last_act"] = jp.asarray(previous)
        if policy.spec["task"] == "walk":
            info["command"] = jp.asarray(command)
        if is_g1:
            phase = rng.uniform(-np.pi, np.pi, 2).astype(np.float32)
            sim.observation_phase[:] = phase
            info["phase"] = jp.asarray(phase)
        data = mjx_env.make_data(
            env.mj_model, qpos=jp.asarray(sim.data.qpos), qvel=jp.asarray(sim.data.qvel),
            ctrl=jp.asarray(sim.data.ctrl), impl=env._config.impl,
            naconmax=env._config.naconmax, njmax=env._config.njmax,
        )
        upstream = get_obs(forward(data), info)
        actual = sim.observe()
        _assert_close(metrics, "observation_max_abs", actual, upstream["state"])
        _assert_close(metrics, "logits_max_abs", policy.logits(upstream), logits(upstream))
        expected_action, _ = inference(upstream, jax.random.PRNGKey(index))
        _assert_close(metrics, "action_max_abs", policy.action(upstream), expected_action)
        _assert_close(metrics, "action_from_c_observation_max_abs", policy.action(actual), expected_action)


def _check_step_timing(artifact, metrics):
    """Exercise actual env.step and C physics; compare temporal fields, not trajectories."""
    policy, sim, env, initial, _, _, inference = artifact
    state = initial
    command = np.asarray(state.info.get("command", np.zeros(3)))
    gait_frequency = None
    phase = None
    if policy.spec["gait"] is not None:
        phase = np.asarray(state.info["phase"])
        gait_frequency = float(np.asarray(state.info["phase_dt"])[0] / (2 * np.pi * env.dt))
    sim.reset(np.asarray(state.data.qpos), np.asarray(state.data.qvel), command,
              gait_frequency=gait_frequency, phase=phase)
    step = jax.jit(env.step)
    fields = {field["name"]: slice(field["offset"], field["offset"] + field["size"])
              for field in policy.spec["observation"]["fields"]}
    previous = np.zeros(sim.model.nu, np.float32)
    for index in range(4):
        # Use the original trained Brax network, not an artificial policy.
        action, _ = inference(state.obs, jax.random.PRNGKey(index))
        action = np.asarray(action)
        target_base = (np.asarray(sim.data.qpos[7:], np.float32).copy()
                       if policy.spec["task"] == "getup" else sim.default_pose)
        expected_targets = target_base + np.float32(policy.spec["control"]["action_scale"]) * action
        result = sim.step(action=action)
        state = step(state, jp.asarray(action))
        _assert_close(metrics, "target_max_abs", sim.data.ctrl, expected_targets)
        np.testing.assert_array_equal(result["observation"][fields["last_action"]], previous)
        for name in ("last_action", "phase_cos", "phase_sin"):
            if name in fields:
                _assert_close(metrics, "history_max_abs", result["observation"][fields[name]],
                              state.obs["state"][fields[name]])
        # Reading an observation must not consume a history frame or gait tick.
        np.testing.assert_array_equal(sim.observe(), result["observation"])
        np.testing.assert_array_equal(sim.observe(), result["observation"])
        previous = action.copy()
    sim.reset()
    np.testing.assert_array_equal(sim.observe()[fields["last_action"]], np.zeros(sim.model.nu))


def _check_sensor_stage(artifact, metrics):
    """No extra forward between mj_step and observation; compare identical data fields."""
    policy, sim, env, initial, _, _, _ = artifact
    sim.reset()
    info = dict(initial.info)
    is_g1 = policy.spec["robot"] == "g1"
    get_obs = jax.jit(lambda data, info: _get_observation(env, data, info, is_g1))
    for _ in range(8):
        result = sim.step()
        # put_data carries both integrated qpos/qvel and the existing sensor stage.
        data = mjx.put_data(env.mj_model, sim.data, impl=env._config.impl)
        info["last_act"] = jp.asarray(sim.observation_last_action)
        if policy.spec["task"] == "walk":
            info["command"] = jp.asarray(sim.command)
        if is_g1:
            info["phase"] = jp.asarray(sim.observation_phase)
        upstream = get_obs(data, info)
        _assert_close(metrics, "sensor_stage_max_abs", result["observation"], upstream["state"])
    assert sim.data.time == pytest.approx(8 * policy.spec["control_dt"])
    assert isinstance(sim.model, mujoco.MjModel)
    assert isinstance(sim.data, mujoco.MjData)


def _check_velocity_frame(artifact, metrics, root=ROOT):
    from eval.run import SuiteConfig, TrialSimulator

    sim = TrialSimulator(artifact[0], root, SuiteConfig())
    sim.reset()
    sim.data.qpos[2] = 2.0
    sim.data.qpos[3:7] = [1, 0, 0, 0]
    sim.data.qvel[:] = 0
    sim.data.qvel[:6] = [1.5, 0.8, -0.1, 0.2, -0.3, 1.2]
    mujoco.mj_forward(sim.model, sim.data)
    expected = np.concatenate([sim.data.qvel[3:6], sim.data.qvel[:3]])
    _assert_close(metrics, "velocity_frame_max_abs", sim.velocity(), expected)


def check_policy(policy_path, brax_path, root=Path("."), *, impl="jax"):
    """Run the same real-artifact comparisons as pytest; raise on any mismatch."""
    root = Path(root)
    policy_path = Path(policy_path)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    artifact = _load_artifact(policy_path, brax_path=brax_path, root=root, impl=impl)
    metrics = {
        "policy": policy_path.name, "env_name": artifact[0].spec["env_name"],
        "backend": impl, "atol": ATOL, "rtol": RTOL,
        "observation_samples": 16, "timing_steps": 4, "sensor_stage_steps": 8,
    }
    _check_observation_and_network(artifact, metrics)
    _check_step_timing(artifact, metrics)
    _check_sensor_stage(artifact, metrics)
    _check_velocity_frame(artifact, metrics, root)
    return metrics


def test_observation_and_network_from_identical_states(artifact):
    _check_observation_and_network(artifact, {})


def test_real_step_action_targets_and_delayed_history(artifact):
    _check_step_timing(artifact, {})


def test_model_sensor_stage_matches_upstream_observation(artifact):
    _check_sensor_stage(artifact, {})


def test_tracking_velocity_uses_named_body_frame(artifact):
    _check_velocity_frame(artifact, {})
