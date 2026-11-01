# Policy files

I use [Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c) and [Brax 0.14.2](https://github.com/google/brax/tree/v0.14.2) (Apache-2.0), with Unitree models from [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97) (BSD-3-Clause), Python 3.12, and MuJoCo / mujoco-mjx 3.15.0.

Files are named `go1-walk-sN.json`, `go1-walk-rough-sN.json`, `go1-getup-sN.json`, `go1-walk-push-sN.json` and `g1-walk-sN.json`, for training seeds 0, 1 and 2. Among Go1 walking policies, only the push variant enables `pert_config.enable`; G1 uses its upstream setting. The demo uses seed 0 for every policy. Each export includes its run ID and training configuration.

Go1 get-up uses a 200M-step training budget. The G1 demo uses the last saved seed-0 checkpoint, at 181,043,200 of 200M requested steps, after training hit its time limit. Its C MuJoCo tracking and Warp parity checks passed. G1 seeds 1 and 2 completed the default 200M-step budget. [Training results](../results/report/training.csv) give completed steps for each seed.

## Export and runtime

[`train/export.py`](../train/export.py):

```python
export_policy(env_name, env, params, destination, seed,
              model_xml, playground_commit, run_id)
```

Pass Brax PPO's complete `(normalizer, policy, value)` tuple and use the upstream task network defaults. Export writes JSON and a sibling `.brax` file with the original tuple, including critic weights. The browser needs only JSON; Python parity tests also need the `.brax` file. Brax uses pickle, so load sidecars only from a trusted source.

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
- Checkpoints: `training_checkpoint` gives the original run ID, checkpoint step, requested steps and selection reason. The export's `run_id` names the directory containing the matching `.brax` sidecar. Check the step counts for interrupted runs.
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
| Experimental G1 getup | pelvis gyro 3, gravity 3, joint offset 29, joint velocity 29, delayed action 29 | 93 |

Offsets are `qpos[7:]-default_pose`; velocities are `qvel[6:]`. Gravity is `R_imu.T@[0,0,-1]`, not the upvector sensor. Use site velocimeters, which include the IMU offset. G1 actor sensors use the pelvis; walking termination uses torso upvector. Noise is off; no frame stacking occurs.

All tasks observe **before** updating last action: `o[t+1]` contains `a[t-1]`, not `a[t]`. Keep separate observation-time history. G1 walking also observes phase before advancing `fmod(phase+2*pi*dt*frequency+pi,2*pi)-pi`. Initial phase is `[0,pi]`; frequency trains uniformly over `[1.25,1.5]` Hz, with reference default 1.375 Hz. Phase continues at zero command. Do not add `mj_forward` after `mj_step`.

Walk targets are `default_pose+0.5*action`; getup targets are `current_qpos[7:]+0.5*action`, held through substeps. No additional clipping; actuator limits apply. Go1 uses five 0.004 s substeps; G1 ten 0.002 s. Commands remain external: Go1 limits ±[1.5,0.8,1.2], G1 ±[1,.5,1]; getup ignores commands.

Go1 recovery requires its full-collision XML, not walking's feet-only model. Switch models preserving qpos/qvel and reset policy history at handoff.

## Rough-ground dog

`go1-walk-rough-sN.json` uses `Go1JoystickRoughTerrain` with the default task and PPO settings, except `njmax=256` to avoid the upstream constraint-buffer overflow at 60. This changes buffer capacity only. [`assets/go1/rough.xml`](../assets/go1/rough.xml) is the training scene: a heightfield with half-widths 10 m, elevation scale 0.05 m and floor friction 1.0. The scene includes its original `hfield.png`, compiled-array comparison and source hashes. Its rock-face texture is [Poly Haven CC0](https://polyhaven.com/a/rock_face) and has no physics effect.

Observation and control match flat Go1 walking. `eval/test_terrain.py` checks that both scenes have identical robot mechanisms and simulation options; their ground and nominal starting height differ. The browser translates a copy of the rough patch beside flat ground. Parity tests replay the original scene.

## Browser fall scene

[`assets/g1/fallen.xml`](../assets/g1/fallen.xml) restores body-floor collisions using the Menagerie mesh/cylinder shapes in the walking model and Playground's limb capsules. It adds floor contact pairs and leaves self-collision pairs unchanged. [`train/fallen.py`](../train/fallen.py) regenerates the model and its source hashes. [`eval/test_fallen.py`](../eval/test_fallen.py) checks unchanged kinematics, masses, inertias, joints, actuators, options and airborne dynamics, plus floor support after drops.

G1 trains and evaluates with `walk.xml`. After a fall, the browser preserves its state, switches to `fallen.xml` and holds its pose until Reset.

## Experimental humanoid recovery

[`train/g1_getup.py`](../train/g1_getup.py) defines a custom task with random physical drops, torso-up and standing-height rewards, and regularizers. Both seed-0 attempts recovered from 0 of 200 fallen starts, so G1 has no demo get-up policy. The exports are [`g1-getup-s0-t50m-v2`](../results/runs/g1-getup-s0-t50m-v2/) and [`g1-getup-s0-t300m-v2`](../results/runs/g1-getup-s0-t300m-v2/). See the [results](../results/report/g1_getup.csv) and [method](../docs/evaluation.md#experimental-humanoid-get-up).

## Parity

Run `python -m pytest eval/test_parity.py -q` with the full install and the `.brax` sidecars below. Tests need the models, JSON exports and matching sidecars. `WEEBLE_ROOT`, `WEEBLE_PARITY_POLICIES` (comma-separated) and `WEEBLE_PARITY_IMPL=jax|warp` select inputs and backend. CI checks all 15 released policies with `impl='jax'` on CPU; `warp` needs a GPU on Modal.

`eval.test_parity.check_policy(policy_path, brax_path, root=Path('.'), *, impl='jax')` returns measured maximum errors or raises. Tests compare actual upstream noise-free observations and original Brax logits/actions at identical states, with `atol=rtol=1e-5`, plus real-step target/history/gait/sensor timing.

`write_parity_fixture(policy_path, destination, root=Path('.'), steps=200, seed=0, command=None, impl='jax')`, or `python -m train.export POLICY DEST`, writes a C MuJoCo trace from an upstream seeded reset. `initial` contains qpos/qvel/command/gait_frequency/phase/observation/last_action. Each frame contains pre-action observation, action, motor_targets, poststep qpos/qvel and next_observation. Replay from the saved initial state. Generating a fixture does not run the parity tests, and long C/MJX trajectories can diverge despite matching single-step calculations.

## Original Brax sidecars

The [`policies-2026-10-14` release](https://github.com/mottopanikeiku/weeble/releases/tag/policies-2026-10-14) contains the original Brax `(normalizer, policy, value)` tuples for all walking and dog get-up policies, including rough-ground seeds. Each policy and training manifest lists package versions; Playground and Menagerie commits are pinned above.

Download and verify the release assets:

```sh
python -m train.fetch_checkpoints
```

The downloader fetches missing files over HTTPS without a GitHub account or CLI. Before installing them, it checks each file against `brax_params_sha256` in the policy JSON. `python -m eval.verify` checks the table below against those fields. To retrieve the files from Modal instead, use `python -m train.launch fetch-checkpoints --gpu none --run-id fetch-checkpoints-release`.

| Policy | SHA-256 |
|---|---|
| g1-walk-s0.brax | 04895d89932c09aad462a92680417d26b6e807e88b849e2bcf995522486e9c8d |
| g1-walk-s1.brax | 84c0ed2a8b85d61d3aceb7816b6d4a665b4b3c667105a68cf2388b63bdaaf17f |
| g1-walk-s2.brax | 835e0a81c714d0c1f2fcb781271035e5761321aec5fe48c7d67846d7dfc2c2a1 |
| go1-getup-s0.brax | 87d51a381561ee24594b8f5b6660a13789cb5c57c4ab8928294bf75351c17d16 |
| go1-getup-s1.brax | aab608dbf80ed8dc7dffbb3addbe42872d503893a91c500362fa6ab891124ca0 |
| go1-getup-s2.brax | a229fa50d803e71c2bbd4742f4c28fdef0af817e3aba1256622de97fd59239f7 |
| go1-walk-push-s0.brax | b1c67073d16290c5492ea011dbd471262920894f0fbadef8c3811dd4790f7ebb |
| go1-walk-push-s1.brax | 58007e001a5d235b43f0803ed28072fef50f650524ca297e656b84daa1634dff |
| go1-walk-push-s2.brax | 849dc5f8295fb30ee4dcb8ccdbbff2185238f3bae3e11f0d6a8fcae38f3279b1 |
| go1-walk-rough-s0.brax | 1f4cf4b649b85fb1785e2924c446040f6fe52f1f75f4447b996fa4fbbb8044c4 |
| go1-walk-rough-s1.brax | 6ec0bdeb664a98b0c39567d8b79e77b145c7eb52c32698174b7596b74ca4fb19 |
| go1-walk-rough-s2.brax | 9ded829ad0bf12106d7a59c43c617010953e095b4464bd8919dfca98d90fde9f |
| go1-walk-s0.brax | 5faa392c8ed5f01d5244c5699758ca462eb52f8fb9d45689490924afdfe3d313 |
| go1-walk-s1.brax | 896a45fbb7b3f703cfb59f6b322b52c42856635cf6ba9e768089eef3c9f6e986 |
| go1-walk-s2.brax | 895775fb8f1b60a528ad9cdb43169b75646517b69d8811affbf933978752081d |
