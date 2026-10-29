# weeble

Walking robots you can shove, live in your browser. I trained the policies with reinforcement learning on one cloud GPU (13 minutes and about $0.24 for the dog) and evaluated them in standard MuJoCo. The page runs the same physics engine compiled to WebAssembly; on recorded rollouts its policy actions match my Python reference to within [5×10⁻⁷](results/parity-checks/wasm.json).

<!-- generated:atlas:start -->
**Robustness atlas (simulation).** I ran the 12 released walking policies, three seeds each of the flat-ground dog, the dog trained with shoves, the rough-ground dog and the humanoid, through 21,600 episodes in standard MuJoCo without retraining. The stresses are held-out rubble, floor friction, payload, weaker motors, sensor delay, sudden full-range command changes, and shoves from every direction at different points in the gait. Each cell has 50 episodes per seed. I paired each trial's random draws across policies and levels. I scaled commands to each robot's range and shoves by its mass and centre-of-mass height. Training never varied friction, mass, motor strength or sensor delay, but it did include sudden command changes. The rough-ground dog trained on 5 cm rubble of the same kind as the 5 cm level, and the shove-trained dog's training kicks reached at most 0.60 in the normalized units below.

The table gives, per seed, the hardest level at which the robot never fell in at least 90% of episodes, and in at least 90% at every easier level.

| Stress (levels tested) | Go1, flat | Go1, shoves | Go1, rough | G1 |
|---|---|---|---|---|
| Held-out rubble height (flat, 5 cm, 7.5 cm, 10 cm, 12.5 cm) | 5 cm / 5 cm / 5 cm | 5 cm / 5 cm / 5 cm | 7.5 cm / 7.5 cm / 5 cm | flat / flat / flat |
| Low-friction stress (μ 0.6, μ 0.4, μ 0.25, μ 0.15) | μ 0.15 / μ 0.15 / μ 0.15 | μ 0.15 / μ 0.15 / μ 0.15 | μ 0.15 / μ 0.15 / μ 0.15 | μ 0.25 / μ 0.15 / μ 0.25 |
| Payload (of body mass) (+0%, +10%, +20%, +30%, +40%) | +40% / +40% / +40% | +40% / +40% / +40% | +40% / +40% / +40% | +10% / +20% / +30% |
| Actuator strength (100%, 85%, 70%, 55%, 40%) | 70% / 70% / 70% | 70% / 70% / 70% | 85% / 70% / 70% | 100% / 85% / 85% |
| Sensor delay (0 ms, 20 ms, 40 ms, 60 ms, 80 ms) | 20 ms / 20 ms / 20 ms | 20 ms / 20 ms / 20 ms | 20 ms / 20 ms / 20 ms | 40 ms / 20 ms / 20 ms |
| Shove, J/(m√(gh)) (0, 0.5, 0.75, 1, 1.25, 1.5) | 0.5 / 0.5 / 0.5 | 1 / 1.25 / 1 | 0.5 / 0.5 / 0.5 | 0 / 0 / 0 |
| Command step: share that never fell in the worst of the 5 full-range changes | 100% / 96% / 100% | 98% / 100% / 100% | 96% / 100% / 94% | 100% / 100% / 100% |

Shoves are normalized by body mass m and nominal centre-of-mass height h, so 1 means J = m√(gh): 20.2 N·s for Go1 (12.7 kg, h = 0.26 m) and 85.3 N·s for G1 (33.3 kg, h = 0.67 m). That removes size, not every difference: both robots get the same 0.1 s pulse while walking at the same 0.5 m/s, which isn't dynamically identical for a 0.26 m and a 0.67 m tall centre of mass. Read the dog-versus-humanoid comparison with that in mind. When a dog fell, I handed it to the get-up policy the page uses (go1-getup-s0) under the same stress. Falls after which it was back on its feet within 6 s, per seed: flat 103/333 / 142/378 / 155/406; shoves 57/290 / 58/279 / 71/320; rough 126/360 / 155/381 / 138/367. Those episodes still count as falls in the table. The humanoid has no get-up policy.

These results describe the recorded CPU and software environment. The Wilson intervals cover variation across sampled trials, not differences between environments. On another runner, seven outcomes changed in a selected 36-episode stress sample. I have not measured how much the full per-cell rates change between environments.

![Never-fell share for every policy seed and stress level](results/atlas/report/atlas.svg)

[Per seed with 95% intervals](results/atlas/report/atlas_seeds.csv) · [margins](results/atlas/report/margins.csv) · [shove direction](results/atlas/report/shove_direction.csv) · [shove gait phase](results/atlas/report/shove_stance.csv) · [raw episodes](results/atlas/) · [clean replay](results/atlas/clean-replay.json) · [protocol](docs/evaluation.md#robustness-atlas)
<!-- generated:atlas:end -->

**[Try it →](https://mottopanikeiku.github.io/weeble/)** Steer with WASD or the touch pad, drag any body, press **Push** to shove or **Throw** to lob a ball. Switch between the dogs trained with and without shoves and on rough ground, or the humanoid; press **I** to see what the policy sees.

![The Go1 dog walks on a test field beside a small obstacle course, is shoved over, gets back up and is hit by a thrown ball; then the G1 humanoid walks and the rough-ground dog crosses its terrain](web/media/hero.gif)

<!-- generated:headline:start -->
**Does training with random shoves make the dog harder to knock over?** Yes. The shove it survives half the time (J50) rises from 24.1 to 32.9 N·s with three seeds per arm pooled (per seed: 22.7–26.8 vs 31.3–35.8 N·s): **+8.8 N·s, 95% interval +4.6 to +13.1**. [Comparison](results/report/ablation.csv)
<!-- generated:headline:end -->

Tasks, rewards and PPO settings are from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c); robot models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97); training uses [Brax](https://github.com/google/brax/tree/v0.14.2) PPO; the terrain texture is CC0 from [Poly Haven](https://polyhaven.com/a/rock_face) ([licenses](assets/)). I trained and evaluated the policies, ran the experiments and wrote the browser runtime.

## What I built

- **Browser runtime** ([physics](web/src/physics.ts), [policy](web/src/policy.ts), [recovery](web/src/recovery.ts), [play scene](web/src/playground.ts)): loads the exact training models and weights, builds observations the way training did, and hands a fallen dog to the get-up policy and back. A [parity test](web/test/parity.test.ts) replays recorded rollouts in the browser engine.
- **Reference controller** ([eval/sim.py](eval/sim.py)): observations, policy and control in NumPy and MuJoCo, matched against Playground's own environment to about 10⁻⁶.
- **Experiments** ([eval/run.py](eval/run.py), [eval/terrain.py](eval/terrain.py), [eval/report.py](eval/report.py)): shove recovery, tracking, get-up from fallen starts and a terrain cross-test, with every raw trial committed.
- **Training** ([train/app.py](train/app.py)): Playground and Brax PPO on MJX, run on [Modal](https://modal.com), with exported checkpoints and a manifest per run.

**The play scene.** The page adds a landscape, a small course of movable crates, cones and balls, thrown balls and a patch of the rough-ground dog's training terrain. The policies never saw props or thrown balls. Robots touch them through extra collision shapes that don't touch the floor, so open-ground walking uses the trained physics; with the policy in the loop, falls and tracking match the training scene ([comparison](web/results/scene-comparison.json)). When the humanoid falls, the page gives it full-body floor contacts so it lies on the ground; it has no get-up policy.

## Results

Standard MuJoCo with deterministic actions, no observation noise and no domain randomization. A shove counts as survived if, from the start of the trial until 3 s after the shove, the body never tilts past 60° and nothing but the feet touches the floor. [Protocol](docs/evaluation.md).

<!-- generated:results:start -->
| Policy | Seeds | Shove survived half the time, J50 (N·s), by seed | Tracking error: forward, lateral (m/s), turning (rad/s) |
|---|---:|---|---|
| Go1, trained without shoves | 3 | 26.8 / 23.4 / 22.7 | 0.068 / 0.058 / 0.057 |
| Go1, trained with shoves | 3 | 31.3 / 31.4 / 35.8 | 0.072 / 0.058 / 0.061 |
| G1 humanoid | 3 | 66.1 / 70.5 / 70.0 | 0.086 / 0.148 / 0.417 |

[Per-seed tracking](results/report/tracking.csv) · [J50 by seed](results/report/j50_seeds.csv) · [survival curves](results/report/push_seeds.svg) · [raw trials](results/eval/)

**Getting up.** From fallen starts, two of the three get-up seeds recover in every trial (181/181 and 179/179); the third almost never does (3/187). Median time back on its feet: 2.1 s. Counting drops where the dog happened to land upright, the totals are 200/200, 199/200 and 16/200. [Per seed](results/report/getup.csv)

**Training simulator vs MuJoCo.** For one dog seed, tracking error differs by at most 0.003 between MJX (Warp), where it trained, and standard MuJoCo; J50 is 30.0 vs 28.3 N·s over 220 matched shove trials. [Comparison](results/report/gap.csv)

<!-- generated:terrain:start -->

**Cross-terrain walking in MuJoCo.** 3 trained seeds per arm; 27 commands per seed and condition. Falls / trials (forward, lateral, yaw MAE):

| Training terrain | Flat test | Rough test |
|---|---|---|
| Flat | 0/81 (0.045 / 0.035 / 0.038) | 1/81 (0.063 / 0.075 / 0.062) |
| Rough | 0/81 (0.037 / 0.026 / 0.037) | 1/81 (0.041 / 0.033 / 0.047) |

Errors are episode means including time after falls: forward/lateral in m/s, yaw in rad/s. [Terrain counts and errors](results/report/terrain.csv) · [Per seed](results/report/terrain_seeds.csv) · [Raw trials](results/terrain/)
<!-- generated:terrain:end -->
<!-- generated:results:end -->

On this terrain, training on rough ground cost nothing on flat ground and tracked commands better on rough ground; neither dog fell more than once in 81 rough-ground runs.

## Reproduce

Evaluation, the report and the artifact checks need only MuJoCo, NumPy and pytest. The JAX, Brax and Playground stack is for training and Python parity.

```bash
uv sync --frozen --only-group eval                     # MuJoCo, NumPy and pytest
uv run --no-sync python -m eval.verify                  # recorded hashes, then a byte-for-byte rebuild of the report and atlas
uv run --no-sync python -m eval.report results/eval --output results/report --update-readme   # every number above, from raw trials
uv run --no-sync python -m eval.atlas                   # the robustness atlas: all 21,600 episodes, its tables and README block
uv run --no-sync python -m eval.atlas --rerun 36        # re-simulate 36 committed atlas episodes; --exact also requires identical rows
cd web && npm ci && npm run dev                         # the demo, locally
uv run python train/launch.py train --env Go1JoystickFlatTerrain --seed 0 --gpu L4 --minutes 25   # retrain on Modal
```

Pull requests that touch code, policies, assets or results run [CI](.github/workflows/ci.yml): the tests, `eval.verify` and an atlas rerun on the minimal install, Python parity for every released checkpoint on CPU, and the browser build and tests. [Details](docs/evaluation.md#checks-on-every-pull-request).

<!-- generated:cost:start -->
NVIDIA L4 training: flat-ground dog 12–14 minutes ($0.22–0.25), dog get-up 15–16 ($0.27–0.28), walking humanoid 45–49 ($0.79–0.86) per policy. Rough-ground dog: 18–19 minutes ($0.31–0.34). The 15 published policies cost $5.65; all cloud jobs, including failures, $12.98. Wall-time/list-price upper bounds, not invoices. [Runs](results/report/training.csv) · [All jobs](results/report/invocations.csv)
<!-- generated:cost:end -->

## Limits

- Simulation only; nothing here ran on a real robot.
- Walking evaluation keeps Playground's feet-only floor contacts, so tilt catches most falls. The page's fall collisions and props are additions outside the evaluation.
- One of three get-up seeds rarely gets up (3 of 187 fallen starts); I report it rather than retrain it.
- The humanoid tracks turning commands poorly, and its demo uses the last checkpoint before a time limit. My own humanoid get-up task failed (0 of 200 fallen starts after 300M steps), so the page shows no humanoid recovery.
- Demo seeds were fixed before evaluation. The cross-terrain test uses the one training heightfield. The atlas's held-out rubble is new grids of the same kind (independent uniform heights at one 7.8 cm cell size), not stairs, slopes or smooth hills. [Configuration history](docs/evaluation.md).
- The atlas changes one thing at a time. Payload sits at the base's centre of mass; strength scales the motors' torque but not the joint damping; delay holds back the sensor readings only. Each cell is 50 episodes per seed, so a never-fell share has a 95% interval about ±8 to ±14 points wide unless it is near 0 or 100%. A fallen dog is handed to get-up seed 0, the page's, whichever walking seed fell.

## Prior work

- Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844) (2025): the tasks, rewards and training settings; their Go1 and G1 policies transfer to real robots.
- Rudin et al., [Learning to walk in minutes](https://arxiv.org/abs/2109.11978): GPU-parallel locomotion training with random pushes.
- Freeman et al., [Brax](https://arxiv.org/abs/2106.13281); Todorov et al., [MuJoCo](https://mujoco.org/), including its official [WebAssembly bindings](https://github.com/google-deepmind/mujoco/tree/main/wasm).

Written with AI coding assistance.
