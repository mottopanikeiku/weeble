# Policy files

I use [Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c) and [Brax 0.14.2](https://github.com/google/brax/tree/v0.14.2) (Apache-2.0), with Unitree models from [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97) (BSD-3-Clause), Python 3.12, and MuJoCo / mujoco-mjx 3.15.0.

Names are `go1-walk-sN.json`, `go1-getup-sN.json`, `go1-walk-push-sN.json`, and `g1-walk-sN.json`, for training seeds 0,1,2. Only the Go1 push variant enables `pert_config.enable`. Demo seed 0 is fixed before results. File names alone do not establish training completion; exports record run IDs and configurations.

## Export and runtime

[`train/export.py`](../train/export.py):

```python
export_policy(env_name, env, params, destination, seed,
              model_xml, playground_commit, run_id)
```

Pass Brax PPO's complete `(normalizer, policy, value)` tuple. Training must use upstream task network defaults. Export writes JSON and a sibling `.brax` preserving the original tuple, including critic weights. Keep that sidecar on Modal/release, not in the browser payload. Brax uses pickle: only load your own training artifacts.

[`eval/sim.py`](../eval/sim.py) needs only NumPy and MuJoCo:

```python
policy = Policy.load(path)
sim = Simulator(policy, root=Path('.'))
sim.reset(qpos=None, qvel=None, command=(0,0,0))
result = sim.step(command=None, action=None, force=None)
```

`model/data` are C MuJoCo objects. Reset zeroes history/forces and forwards sensors, without randomization or settling. Optional reset keywords `gait_frequency` and `phase` reproduce G1 fixtures. `observe(command=None)` does not advance history. Step returns poststep `observation/qpos` and applied `action`. Force is a world-frame Newton vector at the `force_body` center of mass, held for one control period; absence clears it. Overridable `physics_step()` advances held controls/forces only.

## JSON schema 1

- Metadata: `env_name`, `robot`, `task`, `seed`, `run_id`, `versions`, `training_config`, `ppo_network_config`, original `brax_params_file` and SHA-256.
- Physics: root-relative `model_xml`, `default_qpos/default_qvel/default_pose`, `control_dt/simulation_dt/n_substeps`, `feet_geoms/floor_geom`, and `force_body` (`trunk` for Go1, `pelvis` for G1). `model_parameters.arrays/options` preserve training-time edits, idempotently on exported XML.
- Observation: `sensors` names the gyro, velocimeter and gravity site; `observation.fields` gives ordered names, offsets, sizes and formulas. `observation_size` includes actor and critic widths. `command_ranges` stores three `[min,max]` pairs.
- Weights: each ordered `policy.layers` entry has `name`, `encoding="base64-f32-le"`, base64 `kernel/bias`, and `kernel_shape/bias_shape`. Decode once into exact little-endian IEEE float32. Kernels are row-major `[input,output]`; biases are `[output]`. No quantization. `layer_sizes` includes input/output widths. JSON `value` contains shapes only.
- Normalization: `normalizer.mean/std` are arrays keyed by `state/privileged_state`. Compute `(observation-mean)/std` in float32, **without added epsilon**. Upstream `clip=null` means no clipping. Non-finite configuration constants use strings such as `"inf"`, not nonstandard JSON numbers.

Hidden layers compute `swish(x @ W+b)`, where `swish(z)=z/(1+exp(-z))`. The final layer is linear; action is `tanh(logits[:nu])`. The remaining outputs parameterize `softplus(scale_logits)+0.001`, unused for deterministic inference.

### Observation and control

| Task | Actor fields, in order | Width |
|---|---|---:|
| Go1 walk | local velocity 3, gyro 3, gravity 3, joint offset 12, joint velocity 12, delayed action 12, command 3 | 48 |
| Go1 getup | gyro 3, gravity 3, joint offset 12, joint velocity 12, delayed action 12 | 42 |
| G1 walk | pelvis velocity 3, gyro 3, gravity 3, command 3, joint offset 29, joint velocity 29, delayed action 29, phase cosines 2, sines 2 | 103 |

Offsets are `qpos[7:]-default_pose`; velocities are `qvel[6:]`. Gravity is `R_imu.T@[0,0,-1]`, not the upvector sensor. Use site velocimeters, which include the IMU offset. G1 actor sensors use the pelvis; termination uses torso upvector. Noise is off; no frame stacking occurs.

All tasks observe **before** updating last action: `o[t+1]` contains `a[t-1]`, not `a[t]`. Keep separate observation-time history. G1 also observes phase before advancing `fmod(phase+2*pi*dt*frequency+pi,2*pi)-pi`. Initial phase is `[0,pi]`; frequency trains uniformly over `[1.25,1.5]` Hz, with reference default 1.375 Hz. Phase continues at zero command. Do not add `mj_forward` after `mj_step`.

Walk targets are `default_pose+0.5*action`; getup targets are `current_qpos[7:]+0.5*action`, held through substeps. No additional clipping; actuator limits apply. Go1 uses five 0.004 s substeps; G1 ten 0.002 s. Commands remain external: Go1 limits ±[1.5,0.8,1.2], G1 ±[1,.5,1]; getup ignores commands.

Go1 recovery requires its full-collision XML, not walking's feet-only model. Switch models preserving qpos/qvel and reset policy history at handoff.

## Parity

Inside prepared Modal, run `python -m pytest eval/test_parity.py -q`. Tests require actual models, JSON and `.brax` files: all three seed-0 tasks plus other exported seeds. `PUSHABLE_ROOT`, `PUSHABLE_PARITY_POLICIES` (comma-separated), and `PUSHABLE_PARITY_IMPL=jax|warp` select inputs/backend.

`eval.test_parity.check_policy(policy_path, brax_path, root=Path('.'), *, impl='jax')` returns measured maximum errors or raises. Tests compare actual upstream noise-free observations and original Brax logits/actions at identical states, with `atol=rtol=1e-5`, plus real-step target/history/gait/sensor timing.

`write_parity_fixture(policy_path, destination, root=Path('.'), steps=200, seed=0, command=None, impl='jax')`, or `python -m train.export POLICY DEST`, writes a C MuJoCo trace from an upstream seeded reset. `initial` records qpos/qvel/command/gait_frequency/phase/observation/last_action; frames record pre-action observation, action, motor_targets, poststep qpos/qvel and next_observation. Replay recorded initial state. Fixture generation is not a parity result; tolerances do not imply long C/MJX trajectories stay equal.

Written with AI coding assistance.
