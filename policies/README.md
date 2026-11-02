# Policy files

Policies are trained with [Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c) and [Brax 0.14.2](https://github.com/google/brax/tree/v0.14.2) (Apache-2.0), on Unitree models from [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97) (BSD-3-Clause), with Python 3.12 and MuJoCo and mujoco-mjx 3.15.0.

Files are named `<policy>-sN.json` for training seeds 0, 1 and 2 of `go1-walk`, `go1-walk-push`, `go1-walk-rough`, `go1-getup` and `g1-walk`, and the demo uses seed 0 of each. Among the Go1 walkers only `go1-walk-push` enables `pert_config.enable`, while G1 keeps its upstream setting. Every export records its run ID and training configuration.

Go1 get-up and G1 walking both train for 200M steps, but the demo's `g1-walk-s0` is the last checkpoint saved before its run hit the time limit, at 181,043,200 steps. G1 seeds 1 and 2 finished, and [`training.csv`](../results/report/training.csv) lists the completed steps per seed.

## Export and runtime

[`train/export.py`](../train/export.py) writes a policy from Brax PPO's `(normalizer, policy, value)` tuple and the upstream task network defaults:

```python
export_policy(env_name, env, params, destination, seed,
              model_xml, playground_commit, run_id)
```

It writes the JSON and a sibling `.brax` file holding the original tuple, critic included. The browser reads only the JSON, while the Python parity tests also need the `.brax` file. Brax uses pickle, so load sidecars only from a source you trust.

[`eval/sim.py`](../eval/sim.py) runs a policy with just NumPy and MuJoCo:

```python
policy = Policy.load(path)
sim = Simulator(policy, root=Path('.'))
sim.reset(qpos=None, qvel=None, command=(0, 0, 0))
result = sim.step(command=None, action=None, force=None)
```

`sim.model` and `sim.data` are C MuJoCo objects. Reset zeroes history and forces and forwards the sensors, with no randomization or settling, and its `gait_frequency` and `phase` keywords reproduce G1 fixtures. `observe(command=None)` reads an observation without advancing history, while `step` returns the post-step `observation`, `qpos` and applied `action`. Its `force` is a world-frame vector in newtons at the `force_body` centre of mass, held for one control period. `physics_step()` advances only the held controls and forces, so a subclass can override it to swap in another physics backend.

## JSON schema 1

- Metadata: `env_name`, `robot`, `task`, `seed`, `run_id`, `versions`, `training_config`, `ppo_network_config`, `brax_params_file` and its SHA-256.
- Checkpoints: `training_checkpoint` gives the original run ID, checkpoint step, requested steps and why it was picked, and `run_id` names the directory with the matching `.brax` file.
- Physics: root-relative `model_xml`, `default_qpos`, `default_qvel`, `default_pose`, `control_dt`, `simulation_dt`, `n_substeps`, `feet_geoms`, `floor_geom` and `force_body` (`trunk` for Go1, `pelvis` for G1), plus `model_parameters` with the training-time model edits.
- Observation: `sensors` names the gyro, velocimeter and gravity site, `observation.fields` lists names, offsets, sizes and formulas in order, and `command_ranges` holds three `[min, max]` pairs.
- Weights: each `policy.layers` entry has `name`, `encoding="base64-f32-le"`, base64 `kernel` and `bias`, and their shapes. They decode once into unquantized little-endian float32 with row-major `[input, output]` kernels, and the JSON `value` entry holds shapes only.
- Normalization: `normalizer.mean` and `std` are keyed by `state` and `privileged_state`, and you compute `(observation - mean) / std` in float32 with no added epsilon. `clip=null` means no clipping, and non-finite constants are strings such as `"inf"`.

Hidden layers compute `swish(x @ W + b)` with `swish(z) = z / (1 + exp(-z))`, and the last layer is linear, with the action `tanh(logits[:nu])`. The other outputs parameterize `softplus(scale_logits) + 0.001`, which is unused at inference.

### Observation and control

| Task | Actor fields, in order | Width |
|---|---|---:|
| Go1 walk | local velocity 3, gyro 3, gravity 3, joint offset 12, joint velocity 12, delayed action 12, command 3 | 48 |
| Go1 getup | gyro 3, gravity 3, joint offset 12, joint velocity 12, delayed action 12 | 42 |
| G1 walk | pelvis velocity 3, gyro 3, gravity 3, command 3, joint offset 29, joint velocity 29, delayed action 29, phase cosines 2, sines 2 | 103 |
| G1 getup (experimental) | pelvis gyro 3, gravity 3, joint offset 29, joint velocity 29, delayed action 29 | 93 |

Joint offsets are `qpos[7:] - default_pose` and velocities are `qvel[6:]`. Gravity is `R_imu.T @ [0, 0, -1]` rather than the upvector sensor, and velocities come from the site velocimeters, which include the IMU offset. G1 actor sensors sit on the pelvis, while walking termination uses the torso upvector, and there is no noise or frame stacking.

Every task observes before updating the last action, so `o[t+1]` holds `a[t-1]`, not `a[t]`, and a port needs its own observation-time history. G1 walking also observes its phase before advancing it with `fmod(phase + 2*pi*dt*frequency + pi, 2*pi) - pi`. The initial phase is in `[0, pi]`, frequency trains uniformly over 1.25–1.5 Hz (reference 1.375), and phase keeps running at zero command. Don't add `mj_forward` after `mj_step`, since `mj_step` already returns sensors evaluated during its last internal substep, as MJX does.

Walking targets are `default_pose + 0.5 * action`, while get-up targets are `current_qpos[7:] + 0.5 * action`, held through the substeps, and there is no extra clipping beyond actuator limits. Go1 uses five 0.004 s substeps and G1 ten 0.002 s. Commands stay external, with limits of ±[1.5, 0.8, 1.2] for Go1 and ±[1, 0.5, 1] for G1, and get-up ignores them.

Go1 recovery needs its full-collision XML rather than the feet-only walking model, and the handoff switches models while keeping `qpos` and `qvel` and resetting policy history.

## Rough-ground dog

`go1-walk-rough-sN.json` uses `Go1JoystickRoughTerrain` with default task and PPO settings, except `njmax=256` because upstream's 60 overflows the constraint buffer. [`assets/go1/rough.xml`](../assets/go1/rough.xml) is the training scene: a heightfield with 10 m half-widths, 0.05 m elevation scale and floor friction 1.0, with its original `hfield.png`. The rock-face texture is [Poly Haven](https://polyhaven.com/a/rock_face) CC0 and has no physics effect.

Observation and control match flat Go1 walking, and `eval/test_terrain.py` checks that both scenes share the same robot and simulation options. The browser shifts a copy of the rough patch beside flat ground, and parity tests replay the original scene.

## Humanoid fall scene

[`assets/g1/fallen.xml`](../assets/g1/fallen.xml) gives G1 body-floor collisions, using Menagerie's mesh and cylinder shapes from the walking model plus Playground's limb capsules. It adds floor contact pairs and leaves self-collisions alone. [`train/fallen.py`](../train/fallen.py) regenerates it with source hashes, and [`eval/test_fallen.py`](../eval/test_fallen.py) checks that kinematics, masses, inertias, joints, actuators and options are unchanged and that the body rests on the floor after drops. G1 trains and evaluates on `walk.xml`, and only after a fall does the browser switch to `fallen.xml` and hold the pose until Reset.

[`train/g1_getup.py`](../train/g1_getup.py) defines an experimental humanoid get-up task, but both seed-0 runs, [`g1-getup-s0-t50m-v2`](../results/runs/g1-getup-s0-t50m-v2/) and [`g1-getup-s0-t300m-v2`](../results/runs/g1-getup-s0-t300m-v2/), recovered 0 of 200 fallen starts, so the demo has none (details in the [method](../docs/evaluation.md#humanoid-get-up-experiment)).

## Parity

With the full install and the `.brax` files below, run `python -m pytest eval/test_parity.py -q`. `WEEBLE_ROOT`, `WEEBLE_PARITY_POLICIES` (comma-separated) and `WEEBLE_PARITY_IMPL=jax|warp` choose inputs and backend, and CI checks all 15 policies with `jax` on CPU, while `warp` needs a GPU.

`eval.test_parity.check_policy(policy_path, brax_path, root=Path('.'), *, impl='jax')` returns the measured maximum errors or raises. It compares upstream's noise-free observations and the original Brax logits and actions at identical states with `atol = rtol = 1e-5`, and checks real-step target, history, gait and sensor timing.

`write_parity_fixture(policy_path, destination, root=Path('.'), steps=200, seed=0, command=None, impl='jax')`, or `python -m train.export POLICY DEST`, writes a C MuJoCo trace from an upstream seeded reset. `initial` holds qpos, qvel, command, gait frequency, phase, observation and last action, while each frame holds the pre-action observation, action, motor targets, post-step qpos and qvel, and the next observation. Replays should start from the saved initial state, and long C and MJX trajectories can diverge even when single steps match.

## Brax sidecars

The [`policies-2026-10-14` release](https://github.com/mottopanikeiku/weeble/releases/tag/policies-2026-10-14) holds the original Brax tuples for all walking and dog get-up policies, and this downloads and verifies them:

```bash
python -m train.fetch_checkpoints
```

The downloader fetches missing files over HTTPS, needing no GitHub account or CLI, and checks each against `brax_params_sha256` in the policy JSON before installing it. `python -m eval.verify` checks the table below against the same fields. To pull the files from Modal instead, run `python -m train.launch fetch-checkpoints --gpu none --run-id fetch-checkpoints-release`.

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
