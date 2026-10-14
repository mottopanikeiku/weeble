# Copyright 2026 Google LLC
# Modifications for the local G1 fall-recovery task, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G1 fall recovery using the local, full-body-contact ``assets/g1/fallen.xml``.

Adapted from Google DeepMind MuJoCo Playground at
https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c:
``_src/locomotion/go1/getup.py`` (reset, observation timing, incremental controls,
rewards), ``_src/locomotion/g1/{base,joystick}.py`` (sensor names and 2 ms physics),
and ``config/locomotion_params.py`` (PPO). The Apache-2.0 license is retained at
``assets/g1/LICENSE-Playground``; the G1 assets retain their Menagerie license.

Every reset samples a uniform SO(3) orientation (normalized isotropic Gaussian
quaternion), drops the pelvis from 1 m with home joints and zero velocity, and
holds the home targets for 1 s of real MJX physics. No upright-start mixture or
post-drop pose correction is used. The recovery clock then starts at zero.
As in Playground, the 300-step / 6 s time limit belongs to the Brax EpisodeWrapper;
falling is not terminal. Default cached autoresets reuse the batch's random
drops, as in Go1Getup, rather than paying for another 1 s drop at every step.

The actor is [pelvis gyro, pelvis-frame gravity, joint offsets, joint velocities,
delayed action], with no commands, gait, or frame stack (93 values for 29 joints).
With action a[t] applied at step t, o[t+1] contains a[t-1]. Sensors retain the
final mj_step stage; there is no extra forward after stepping or settling.
Targets are current joint positions + 0.5 * action, held for 10 physics steps.
Joint ranges, gains, damping, inertia, solver settings, and timestep are loaded
unchanged from the XML, not copied from the Go1 model.

Dense orientation uses the *torso* up vector; height uses the pelvis origin,
not the lower IMU site. Home-posture and stillness rewards apply only when
upright and near the nominal 0.785 m standing height. Stillness additionally
penalizes root linear/angular motion, unlike Go1's action-only term. All reward
weights and joint/action/torque/acceleration/velocity costs follow Go1Getup.
The critic adds clean pelvis sensors, joint state, torque, pelvis height, and
torso up vector (196 values). This reward is not the evaluation success rule:
evaluation must independently require tilt < 15 degrees, pelvis height >= 0.65 m,
linear speed < 0.2 m/s and angular speed < 0.5 rad/s continuously for 1 s.

``brax_ppo_config()`` copies the pinned Go1Getup defaults except for
num_resets_per_eval=4 instead of 10: 50M steps, 8192 environments, 20-step unrolls,
32 minibatches, batch size 256, four updates, learning rate 3e-4, discount 0.97,
entropy 0.01, normalized observations, and (512, 256, 128) actor/critic networks
on state/privileged_state. Swish and NormalTanh remain the pinned Brax defaults.
EXT (schedule arithmetic, not measured): four resets allow 8 s of experience
between forced resets at 50M steps; ten allow only 3.2 s, shorter than a recovery
episode. This choice is made before G1 results. The scheduling formula is from
https://github.com/google/brax/blob/v0.14.2/brax/training/agents/ppo/train.py.
These are starting hyperparameters, not measured G1 recovery results or a
budget authorization; the caller must override the step count for capped runs
and reconsider forced-reset frequency if changing that count.
Contact allocation is 64 * 8192 contacts and 512 constraints per environment.

Integration: call ``register()`` before registry.load/get_default_config, or
construct ``G1Getup`` directly (no Menagerie download). Use this module's PPO
config, not upstream locomotion_params.brax_ppo_config("G1Getup"). Wrap with
``wrap_for_brax_training(env, episode_length=env._config.episode_length)``.
Export as robot=g1, task=getup, gait=None, zero command ranges, pelvis IMU
sensors, and current_pose controls. Demo reset seed is always 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env


ROOT = Path(__file__).resolve().parents[1]


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=300,
        drop_height=1.0,
        settle_time=1.0,
        standing_height=0.785,
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        energy_termination_threshold=float("inf"),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                orientation=1.0,
                pelvis_height=1.0,
                posture=1.0,
                stand_still=1.0,
                action_rate=-0.001,
                dof_pos_limits=-0.1,
                torques=-1e-5,
                dof_acc=-2.5e-7,
                dof_vel=-0.1,
            ),
        ),
        impl="warp",
        naconmax=64 * 8192,
        njmax=512,
    )


def brax_ppo_config() -> config_dict.ConfigDict:
    """Go1Getup PPO settings, except four forced resets per evaluation interval."""
    return config_dict.create(
        num_timesteps=50_000_000,
        num_evals=5,
        reward_scaling=1.0,
        episode_length=default_config().episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        num_resets_per_eval=4,
    )


class G1Getup(mjx_env.MjxEnv):
    """Learn recovery from physical drops, without altering the walking dynamics."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        root: str | Path | None = None,
    ):
        config = default_config() if config is None else config.copy_and_resolve_references()
        super().__init__(config, config_overrides)
        self._xml_path = str((ROOT if root is None else Path(root)) / "assets/g1/fallen.xml")
        self._model_assets = {}
        mjx_env.update_assets(self._model_assets, Path(self._xml_path).parent / "assets")
        self._mj_model = mujoco.MjModel.from_xml_path(
            self._xml_path, assets=self._model_assets,
        )
        if not math.isclose(self.sim_dt, self._mj_model.opt.timestep, abs_tol=1e-12):
            raise ValueError("sim_dt must match fallen.xml; G1 dynamics are not overridden")
        if not math.isclose(self.dt, self.n_substeps * self.sim_dt, abs_tol=1e-12):
            raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
        self._settle_steps = int(round(self._config.settle_time / self.sim_dt))
        if not math.isclose(
            self._config.settle_time, self._settle_steps * self.sim_dt, abs_tol=1e-12,
        ):
            raise ValueError("settle_time must be an integer multiple of sim_dt")
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = self._init_q[7:]
        self._lowers, self._uppers = jp.array(self._mj_model.jnt_range[1:]).T
        center = (self._lowers + self._uppers) / 2
        radius = 0.5 * (self._uppers - self._lowers) * self._config.soft_joint_pos_limit_factor
        self._soft_lowers, self._soft_uppers = center - radius, center + radius
        self._imu_site_id = self._mj_model.site("imu_in_pelvis").id

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
        quat = jax.random.normal(rng, (4,))
        quat = quat / jp.linalg.norm(quat)
        return self._init_q.at[:2].set(0.0).at[2].set(
            self._config.drop_height,
        ).at[3:7].set(quat)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, drop_rng = jax.random.split(rng)
        qpos = self._get_random_qpos(drop_rng)
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=jp.zeros(self.mj_model.nv),
            ctrl=self._default_pose,
            impl=self._config.impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        data = mjx_env.step(self.mjx_model, data, self._default_pose, self._settle_steps)
        data = data.replace(time=jp.zeros_like(data.time))
        info = {
            "rng": rng,
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
        }
        metrics = {f"reward/{name}": jp.zeros(()) for name in self._config.reward_config.scales}
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        # Playground's cached autoreset restores data/obs but not info. Its
        # time-zero drop must also start with the reset observation's zero history.
        for name in ("last_act", "last_last_act"):
            info[name] = jp.where(state.data.time == 0, jp.zeros_like(info[name]), info[name])
        targets = state.data.qpos[7:] + self._config.action_scale * action
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)
        obs = self._get_obs(data, info)  # BEFORE advancing action history.
        rewards = self._get_reward(data, action, info)
        rewards = {name: value * self._config.reward_config.scales[name]
                   for name, value in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        done = self._get_termination(data).astype(jp.float32)
        info["last_last_act"], info["last_act"] = info["last_act"], action
        metrics = dict(state.metrics)
        metrics.update({f"reward/{name}": value for name, value in rewards.items()})
        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info, metrics=metrics)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        power = jp.sum(jp.abs(data.actuator_force * data.qvel[6:]))
        invalid = ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        return invalid | (power > self._config.energy_termination_threshold)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "gyro_pelvis")

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        # R_site.T @ [0, 0, -1], NOT the framezaxis sensor in world coordinates.
        return -data.site_xmat[self._imu_site_id, 2, :]

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        def noisy(value, scale):
            info["rng"], key = jax.random.split(info["rng"])
            return value + (2 * jax.random.uniform(key, value.shape) - 1) * (
                self._config.noise_config.level * scale
            )

        scales = self._config.noise_config.scales
        gyro = self.get_gyro(data)
        state = jp.concatenate([
            noisy(gyro, scales.gyro),
            noisy(self.get_gravity(data), scales.gravity),
            noisy(data.qpos[7:], scales.joint_pos) - self._default_pose,
            noisy(data.qvel[6:], scales.joint_vel),
            info["last_act"],
        ])
        privileged = jp.concatenate([
            state,
            gyro,
            mjx_env.get_sensor_data(self.mj_model, data, "accelerometer_pelvis"),
            mjx_env.get_sensor_data(self.mj_model, data, "local_linvel_pelvis"),
            mjx_env.get_sensor_data(self.mj_model, data, "global_angvel_pelvis"),
            data.qpos[7:],
            data.qvel[6:],
            data.actuator_force,
            jp.atleast_1d(data.qpos[2]),
            mjx_env.get_sensor_data(self.mj_model, data, "upvector_torso"),
        ])
        return {"state": state, "privileged_state": privileged}

    def _get_reward(
        self, data: mjx.Data, action: jax.Array, info: dict[str, Any],
    ) -> dict[str, jax.Array]:
        up = mjx_env.get_sensor_data(self.mj_model, data, "upvector_torso")
        orientation_error = jp.sum(jp.square(up - jp.array([0.0, 0.0, 1.0])))
        height = jp.minimum(data.qpos[2], self._config.standing_height)
        standing = (orientation_error < 0.01) & (
            self._config.standing_height - height < 0.005
        )
        action_rate = jp.sum(jp.square(action - info["last_act"])) + jp.sum(
            jp.square(action - 2 * info["last_act"] + info["last_last_act"])
        )
        outside = jp.maximum(self._soft_lowers - data.qpos[7:], 0.0) + jp.maximum(
            data.qpos[7:] - self._soft_uppers, 0.0,
        )
        torque = data.actuator_force
        stillness_cost = jp.sum(jp.square(action)) + jp.sum(jp.square(data.qvel[:6]))
        return {
            "orientation": jp.exp(-2.0 * orientation_error),
            "pelvis_height": jp.expm1(height),
            "posture": standing * jp.exp(-0.5 * jp.sum(jp.square(data.qpos[7:] - self._default_pose))),
            "stand_still": standing * jp.exp(-0.5 * stillness_cost),
            "action_rate": action_rate,
            "dof_pos_limits": jp.sum(outside),
            "torques": jp.linalg.norm(torque) + jp.sum(jp.abs(torque)),
            "dof_acc": jp.sum(jp.square(data.qacc[6:])),
            "dof_vel": jp.sum(jp.square(jp.maximum(jp.abs(data.qvel[6:]) - 2 * jp.pi, 0.0))),
        }


def register() -> None:
    """Make registry.load('G1Getup') available without changing upstream files."""
    from mujoco_playground._src import locomotion

    locomotion.register_environment("G1Getup", G1Getup, default_config)
