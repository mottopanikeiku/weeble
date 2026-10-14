"""G1Getup behavior tests; no trained or artificial policy is required.

Run with the pinned Playground/MuJoCo dependencies and local fallen.xml assets.
PUSHABLE_ROOT and PUSHABLE_PARITY_IMPL follow eval/test_parity.py. Physics tests
exercise real MJX drops and transitions; run them in the main agent's capped
Modal check, not as an unbudgeted local simulation. Trained-network parity still
belongs to eval/test_parity.py with the actual exported .brax checkpoint.

Task formulas and their upstream attribution are documented in train/g1_getup.py.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from eval.sim import observation_from_data
from train.export import _observation_spec
from train.g1_getup import G1Getup, brax_ppo_config, default_config, register
from mujoco_playground._src import mjx_env, wrapper


ROOT = Path(os.environ.get("PUSHABLE_ROOT", Path(__file__).resolve().parents[1]))
IMPL = os.environ.get("PUSHABLE_PARITY_IMPL", "jax")
ATOL = 1e-5


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    cfg.impl = IMPL
    cfg.noise_config.level = 0.0
    return G1Getup(cfg, root=ROOT)


@pytest.fixture(scope="module")
def reset(env):
    return jax.jit(env.reset)


@pytest.fixture(scope="module")
def initial(reset):
    return reset(jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step)


def _info(env, previous=None, older=None):
    return {
        "rng": jax.random.PRNGKey(0),
        "last_act": jp.zeros(env.action_size) if previous is None else jp.asarray(previous),
        "last_last_act": jp.zeros(env.action_size) if older is None else jp.asarray(older),
    }


def _native_data(env, qpos=None, qvel=None):
    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = env._init_q if qpos is None else qpos
    data.qvel[:] = 0 if qvel is None else qvel
    data.ctrl[:] = env._default_pose
    mujoco.mj_forward(env.mj_model, data)
    return data


def _transfer(env, data):
    # Warp put_data refreshes kinematics. Preserve the actual C mj_step sensor
    # stage, as the independent runtime-parity suite does.
    copied = mujoco.MjData(env.mj_model)
    mujoco.mj_copyData(copied, env.mj_model, data)
    matrix = jp.asarray(data.site_xmat.copy()).reshape((-1, 3, 3))
    return mjx.put_data(env.mj_model, copied, impl=env._config.impl).replace(site_xmat=matrix)


def _observation_contract(env):
    # Use the real exporter field layout and the real C runtime observation
    # routine, not a policy with invented weights or a second actor formula.
    return {
        "robot": "g1",
        "task": "getup",
        "gait": None,
        "default_pose": np.asarray(env._default_pose),
        "sensors": {"gyro": "gyro_pelvis", "gravity_site": "imu_in_pelvis"},
        "observation": _observation_spec("g1", "getup", env.action_size),
    }


def test_config_copies_pinned_getup_ppo_and_defines_drop_horizon():
    from mujoco_playground.config import locomotion_params

    cfg = default_config()
    assert cfg.episode_length * cfg.ctrl_dt == pytest.approx(6.0)
    assert cfg.settle_time == 1.0
    assert cfg.drop_height == 1.0
    assert cfg.sim_dt == 0.002
    assert cfg.standing_height >= 0.65
    assert cfg.action_repeat == 1
    expected = locomotion_params.brax_ppo_config("Go1Getup")
    expected.num_resets_per_eval = 4
    assert brax_ppo_config().to_dict() == expected.to_dict()
    cfg.noise_config.level = 0.0
    assert default_config().noise_config.level == 1.0
    ppo = brax_ppo_config()
    ppo.network_factory.policy_hidden_layer_sizes = (16,)
    assert brax_ppo_config().network_factory.policy_hidden_layer_sizes == (512, 256, 128)


def test_forced_training_reset_interval_covers_recovery_episode():
    # Brax 0.14.2 training/agents/ppo/train.py: env_step_per_training_step,
    # num_evals_after_init, and num_training_steps_per_epoch. One GPU, as used
    # by train/app.py. This is schedule arithmetic, not a simulation result.
    cfg, ppo = default_config(), brax_ppo_config()
    env_steps_per_update = ppo.batch_size * ppo.unroll_length * ppo.num_minibatches * ppo.action_repeat
    updates_per_reset = math.ceil(ppo.num_timesteps / (
        max(ppo.num_evals - 1, 1) * env_steps_per_update * max(ppo.num_resets_per_eval, 1)
    ))
    control_steps_per_reset = updates_per_reset * env_steps_per_update / ppo.num_envs
    assert ppo.num_resets_per_eval == 4
    assert control_steps_per_reset >= cfg.episode_length
    assert control_steps_per_reset * cfg.ctrl_dt == pytest.approx(8.0)


def test_registration_exposes_default_config():
    from mujoco_playground import registry
    from mujoco_playground._src import locomotion

    register()
    assert "G1Getup" in locomotion.ALL_ENVS
    assert registry.get_default_config("G1Getup").to_dict() == default_config().to_dict()
    assert registry.get_domain_randomizer("G1Getup") is None


def test_constructor_preserves_local_model_and_export_fields(env):
    source = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1/fallen.xml"))
    assert Path(env.xml_path) == ROOT / "assets/g1/fallen.xml"
    assert env.unwrapped is env
    assert (env.mj_model.nq, env.mj_model.nv, env.action_size) == (36, 35, 29)
    assert env.mjx_model.nu == env.action_size
    assert env.n_substeps == 10
    assert env._settle_steps == 500
    np.testing.assert_allclose(env._init_q, source.keyframe("home").qpos, atol=1e-7)
    np.testing.assert_allclose(env._default_pose, source.keyframe("home").ctrl, atol=1e-7)
    for name in dir(source):
        if name.startswith(("body_", "jnt_", "dof_", "actuator_")):
            value = getattr(source, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(env.mj_model, name), value, err_msg=name)
    for name in ("timestep", "iterations", "ls_iterations", "ccd_iterations", "integrator",
                 "solver", "cone", "disableflags", "enableflags", "gravity"):
        np.testing.assert_array_equal(getattr(env.mj_model.opt, name), getattr(source.opt, name))
    for name in ("pelvis.STL", "torso_link_rev_1_0.STL"):
        assert env.model_assets[name] == (ROOT / "assets/g1/assets" / name).read_bytes()


def test_model_timestep_cannot_be_silently_changed():
    with pytest.raises(ValueError, match="sim_dt must match fallen.xml"):
        G1Getup(config_overrides={"impl": IMPL, "sim_dt": 0.004}, root=ROOT)


def test_drop_distribution_has_nominal_joints_and_uniform_orientation(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    positions = np.asarray(jax.jit(jax.vmap(env._get_random_qpos))(keys))
    np.testing.assert_array_equal(positions[:, :2], 0.0)
    np.testing.assert_array_equal(positions[:, 2], 1.0)
    np.testing.assert_array_equal(
        positions[:, 7:], np.broadcast_to(env._default_pose, (4096, env.action_size)),
    )
    quat = positions[:, 3:7]
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, atol=2e-7)
    # Fixed-seed moment checks distinguish a full SO(3) draw from yaw-only,
    # fixed-face, or uniformly sampled Euler-angle starts.
    np.testing.assert_allclose(quat.mean(axis=0), 0.0, atol=0.04)
    np.testing.assert_allclose(quat.T @ quat / len(quat), np.eye(4) / 4, atol=0.025)
    w, x, y, z = quat.T
    up = np.column_stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x*x + y*y)))
    np.testing.assert_allclose(up.mean(axis=0), 0.0, atol=0.04)
    np.testing.assert_allclose(up.T @ up / len(up), np.eye(3) / 3, atol=0.03)


def test_reset_is_a_full_physical_drop_not_a_pose_assignment(env, monkeypatch):
    # Observe the real integration call. Two independent long Warp contact
    # trajectories need not agree to 1e-5, even for identical random seeds.
    original_step = mjx_env.step
    integrated = {}

    def record_step(model, data, targets, count):
        integrated.update(before=data, targets=targets, count=count)
        result = original_step(model, data, targets, count)
        integrated["after"] = result
        return result

    monkeypatch.setattr(mjx_env, "step", record_step)
    initial = env.reset(jax.random.PRNGKey(0))
    _, drop_key = jax.random.split(jax.random.PRNGKey(0))
    dropped_qpos = env._get_random_qpos(drop_key)
    assert integrated["count"] == round(env._config.settle_time / env.sim_dt)
    np.testing.assert_array_equal(integrated["before"].qpos, dropped_qpos)
    np.testing.assert_array_equal(integrated["before"].qvel, 0.0)
    np.testing.assert_array_equal(integrated["targets"], env._default_pose)
    assert float(integrated["after"].time) == pytest.approx(1.0, abs=2e-5)
    assert float(initial.data.time) == 0.0
    for name in ("qpos", "qvel", "ctrl", "sensordata", "site_xmat"):
        actual = np.asarray(getattr(initial.data, name))
        assert np.isfinite(actual).all(), name
        np.testing.assert_array_equal(actual, getattr(integrated["after"], name))
    assert not np.allclose(initial.data.qpos, dropped_qpos)
    assert float(initial.reward) == float(initial.done) == 0.0
    assert initial.obs["state"].shape == (93,)
    assert initial.obs["privileged_state"].shape == (196,)
    np.testing.assert_array_equal(initial.obs["state"][-env.action_size:], 0.0)
    assert "command" not in initial.info and "phase" not in initial.info


def test_drop_reset_uses_seeded_samples_not_a_constant(env, initial, reset):
    # Seed repeatability applies to the sampled start, not bitwise GPU contact
    # integration. The initial pose sampler remains exactly repeatable.
    key = jax.random.split(jax.random.PRNGKey(0))[1]
    np.testing.assert_array_equal(env._get_random_qpos(key), env._get_random_qpos(key))
    different = reset(jax.random.PRNGKey(1))
    assert not np.allclose(different.data.qpos, initial.data.qpos)


def test_actor_matches_runtime_pelvis_imu_without_commands_or_gait(env):
    qpos = np.asarray(env._init_q).copy()
    qpos[2] = 2.0
    quat = np.array([0.7, 0.3, -0.4, 0.2])
    qpos[3:7] = quat / np.linalg.norm(quat)
    qpos[7:] += np.linspace(-0.1, 0.1, env.action_size)
    qvel = np.linspace(-0.7, 0.8, env.mj_model.nv)
    data = _native_data(env, qpos, qvel)
    previous = np.linspace(-0.5, 0.5, env.action_size, dtype=np.float32)
    obs = env._get_obs(_transfer(env, data), _info(env, previous))
    contract = _observation_contract(env)
    assert [field["name"] for field in contract["observation"]["fields"]] == [
        "gyro", "gravity", "joint_position", "joint_velocity", "last_action",
    ]
    expected = observation_from_data(env.mj_model, data, contract, previous, np.zeros(3))
    np.testing.assert_allclose(obs["state"], expected, atol=ATOL, rtol=ATOL)
    np.testing.assert_array_equal(obs["privileged_state"][:93], obs["state"])
    np.testing.assert_allclose(obs["state"][:3], data.sensor("gyro_pelvis").data, atol=ATOL)
    assert not np.allclose(data.sensor("gyro_pelvis").data, data.sensor("gyro_torso").data)
    pelvis_rotation = data.site("imu_in_pelvis").xmat.reshape(3, 3)
    np.testing.assert_allclose(obs["state"][3:6], -pelvis_rotation[2], atol=ATOL)
    assert not np.allclose(obs["state"][3:6], -data.sensor("upvector_pelvis").data)
    np.testing.assert_allclose(obs["privileged_state"][-4], qpos[2], atol=ATOL)
    np.testing.assert_allclose(obs["privileged_state"][-3:], data.sensor("upvector_torso").data,
                               atol=ATOL)


def test_actor_preserves_final_step_sensor_stage(env):
    qpos = np.asarray(env._init_q).copy()
    qpos[2] = 2.0
    qvel = np.linspace(-0.4, 1.1, env.mj_model.nv)
    data = _native_data(env, qpos, qvel)
    mujoco.mj_step(env.mj_model, data, nstep=env.n_substeps)
    previous = np.full(env.action_size, 0.2, dtype=np.float32)
    expected = observation_from_data(
        env.mj_model, data, _observation_contract(env), previous, np.zeros(3),
    )
    obs = env._get_obs(_transfer(env, data), _info(env, previous))
    np.testing.assert_allclose(obs["state"], expected, atol=ATOL, rtol=ATOL)
    forwarded = mujoco.MjData(env.mj_model)
    mujoco.mj_copyData(forwarded, env.mj_model, data)
    mujoco.mj_forward(env.mj_model, forwarded)
    refreshed = observation_from_data(
        env.mj_model, forwarded, _observation_contract(env), previous, np.zeros(3),
    )
    assert np.max(np.abs(refreshed[:6] - expected[:6])) > 1e-5


def test_noisy_actor_is_seeded_bounded_and_keeps_clean_critic(env):
    cfg = default_config()
    cfg.impl = IMPL
    noisy_env = G1Getup(cfg, root=ROOT)
    data = _transfer(env, _native_data(env))
    previous = np.linspace(-0.3, 0.3, env.action_size, dtype=np.float32)
    clean = env._get_obs(data, _info(env, previous))
    noisy = noisy_env._get_obs(data, _info(env, previous))
    repeat = noisy_env._get_obs(data, _info(env, previous))
    np.testing.assert_array_equal(noisy["state"], repeat["state"])
    scales = cfg.noise_config.scales
    bounds = np.concatenate([
        np.full(3, scales.gyro), np.full(3, scales.gravity),
        np.full(env.action_size, scales.joint_pos), np.full(env.action_size, scales.joint_vel),
        np.zeros(env.action_size),
    ])
    delta = np.abs(np.asarray(noisy["state"]) - np.asarray(clean["state"]))
    assert np.any(delta > 0)
    assert np.all(delta <= bounds + 1e-6)
    np.testing.assert_array_equal(noisy["state"][-env.action_size:], previous)
    np.testing.assert_array_equal(noisy["privileged_state"][93:], clean["privileged_state"][93:])


def test_step_uses_current_pose_and_delayed_unsmoothed_action(env, initial, step):
    state = initial
    older = previous = np.zeros(env.action_size, dtype=np.float32)
    for amplitude in (0.08, -0.04, 0.02):
        action = np.linspace(-amplitude, amplitude, env.action_size, dtype=np.float32)
        expected_target = np.asarray(state.data.qpos[7:]) + env._config.action_scale * action
        before_history = np.asarray(state.info["last_act"]).copy()
        next_state = step(state, jp.asarray(action))
        np.testing.assert_allclose(next_state.data.ctrl, expected_target, atol=ATOL, rtol=ATOL)
        np.testing.assert_array_equal(next_state.obs["state"][-env.action_size:], previous)
        np.testing.assert_array_equal(next_state.info["last_act"], action)
        np.testing.assert_array_equal(next_state.info["last_last_act"], previous)
        np.testing.assert_array_equal(state.info["last_act"], before_history)
        rate = np.sum((action - previous)**2) + np.sum((action - 2 * previous + older)**2)
        assert float(next_state.metrics["reward/action_rate"]) == pytest.approx(
            rate * env._config.reward_config.scales.action_rate, rel=1e-5, abs=1e-9,
        )
        total = sum(float(value) for name, value in next_state.metrics.items()
                    if name.startswith("reward/"))
        assert float(next_state.reward) == pytest.approx(np.clip(total * env.dt, 0, 10000), abs=1e-6)
        assert float(next_state.done) == 0.0
        older, previous, state = previous, action, next_state
    assert float(state.data.time) == pytest.approx(3 * env.dt, abs=1e-6)


def test_dense_rewards_prefer_upright_standing_and_stationary(env):
    action = jp.zeros(env.action_size)
    standing = _transfer(env, _native_data(env))
    up = env._get_reward(standing, action, _info(env))
    qpos = np.asarray(env._init_q).copy()
    qpos[2] = 0.2
    low = env._get_reward(_transfer(env, _native_data(env, qpos)), action, _info(env))
    qpos[3:7] = [0, 1, 0, 0]
    inverted = _transfer(env, _native_data(env, qpos))
    down = env._get_reward(inverted, action, _info(env))
    qpos = np.asarray(env._init_q).copy()
    waist = env.mj_model.joint("waist_pitch_joint").qposadr[0]
    qpos[waist] = 0.4
    bent_torso = env._get_reward(_transfer(env, _native_data(env, qpos)), action, _info(env))
    assert float(up["orientation"]) == pytest.approx(1.0)
    assert float(up["orientation"]) > float(bent_torso["orientation"]) > float(down["orientation"])
    assert float(up["pelvis_height"]) > float(low["pelvis_height"])
    assert float(low["orientation"]) == pytest.approx(1.0)
    assert float(up["posture"]) == float(up["stand_still"]) == 1.0
    assert float(low["posture"]) == float(low["stand_still"]) == 0.0
    assert not bool(env._get_termination(inverted))  # Lying down is the task, not failure.
    moving = standing.replace(qvel=standing.qvel.at[0].set(1.0).at[3].set(0.8))
    moving_reward = env._get_reward(moving, action, _info(env))
    assert float(moving_reward["stand_still"]) < float(up["stand_still"])
    high = standing.replace(qpos=standing.qpos.at[2].set(1.5))
    assert float(env._get_reward(high, action, _info(env))["pelvis_height"]) == pytest.approx(
        float(up["pelvis_height"]),
    )


def test_regularizers_penalize_limits_torque_rate_and_fast_joints(env):
    data = _transfer(env, _native_data(env))
    midpoint = (env._soft_lowers + env._soft_uppers) / 2
    data = data.replace(
        qpos=data.qpos.at[7:].set(midpoint), qvel=jp.zeros(env.mj_model.nv),
        qacc=jp.zeros(env.mj_model.nv), actuator_force=jp.zeros(env.action_size),
    )
    action = jp.zeros(env.action_size)
    baseline = env._get_reward(data, action, _info(env))
    for name in ("dof_pos_limits", "torques", "action_rate", "dof_acc", "dof_vel"):
        assert float(baseline[name]) == 0.0
    torque = jp.linspace(-10.0, 10.0, env.action_size)
    exceeded = data.replace(
        qpos=data.qpos.at[7].set(env._soft_lowers[0] - 0.1).at[8].set(env._soft_uppers[1] + 0.2),
        qvel=data.qvel.at[6].set(2 * jp.pi + 1).at[7].set(-2 * jp.pi - 2),
        qacc=data.qacc.at[6].set(3.0), actuator_force=torque,
    )
    costs = env._get_reward(exceeded, jp.ones(env.action_size), _info(env))
    assert float(costs["dof_pos_limits"]) == pytest.approx(0.3, abs=1e-6)
    assert float(costs["torques"]) == pytest.approx(np.linalg.norm(torque) + np.abs(torque).sum())
    assert float(costs["action_rate"]) == 2 * env.action_size
    assert float(costs["dof_acc"]) == 9.0
    assert float(costs["dof_vel"]) == pytest.approx(5.0, abs=1e-5)
    for name in ("dof_pos_limits", "torques", "action_rate", "dof_acc", "dof_vel"):
        assert env._config.reward_config.scales[name] < 0
    invalid = data.replace(qvel=data.qvel.at[0].set(jp.nan))
    assert bool(env._get_termination(invalid))


def test_brax_horizon_and_cached_autoreset_clear_action_history(env):
    wrapped = wrapper.wrap_for_brax_training(env, episode_length=env._config.episode_length)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 1))
    action = jp.zeros((1, env.action_size))

    @jax.jit
    def before_timeout(state):
        return jax.lax.fori_loop(
            0, env._config.episode_length - 1, lambda _, carry: wrapped.step(carry, action), state,
        )

    state = before_timeout(state)
    np.testing.assert_array_equal(state.done, 0.0)
    np.testing.assert_array_equal(state.info["steps"], 299)
    assert float(state.data.time[0]) == pytest.approx(5.98, abs=3e-4)
    advance = jax.jit(wrapped.step)
    state = advance(state, jp.full_like(action, 0.05))
    np.testing.assert_array_equal(state.done, 1.0)
    np.testing.assert_array_equal(state.info["truncation"], 1.0)
    np.testing.assert_array_equal(state.data.time, 0.0)
    np.testing.assert_array_equal(state.obs["state"][:, -env.action_size:], 0.0)
    state = advance(state, action)
    np.testing.assert_array_equal(state.done, 0.0)
    np.testing.assert_array_equal(state.info["steps"], 1)
    np.testing.assert_array_equal(state.obs["state"][:, -env.action_size:], 0.0)
    np.testing.assert_array_equal(state.info["last_last_act"], 0.0)
