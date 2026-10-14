# pushable

Walking robots you can shove, live in your browser. I trained the policies with reinforcement learning on one cloud GPU (13 minutes and about $0.24 for the dog) and evaluated them in standard MuJoCo. The page runs the same physics engine compiled to WebAssembly; on recorded rollouts its policy actions match my Python reference to within [5×10⁻⁷](results/parity-checks/wasm.json).

**[Try it →](https://mottopanikeiku.github.io/pushable/)** Steer with WASD or the touch pad, drag any body, press **Push** to shove or **Throw** to lob a ball. Switch between the dogs trained with and without shoves and on rough ground, or the humanoid; press **I** to see what the policy sees.

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

```bash
uv run python -m eval.report results/eval --output results/report --update-readme   # every number above, from raw trials
cd web && npm ci && npm run dev                                                    # the demo, locally
uv run python train/launch.py train --env Go1JoystickFlatTerrain --seed 0 --gpu L4 --minutes 25   # retrain on Modal
```

<!-- generated:cost:start -->
NVIDIA L4 training: flat-ground dog 12–14 minutes ($0.22–0.25), dog get-up 15–16 ($0.27–0.28), walking humanoid 45–49 ($0.79–0.86) per policy. Rough-ground dog: 18–19 minutes ($0.31–0.34). The 15 published policies cost $5.65; all cloud jobs, including failures, $12.98. Wall-time/list-price upper bounds, not invoices. [Runs](results/report/training.csv) · [All jobs](results/report/invocations.csv)
<!-- generated:cost:end -->

## Limits

- Simulation only; nothing here ran on a real robot.
- Walking evaluation keeps Playground's feet-only floor contacts, so tilt catches most falls. The page's fall collisions and props are additions outside the evaluation.
- One of three get-up seeds rarely gets up (3 of 187 fallen starts); I report it rather than retrain it.
- The humanoid tracks turning commands poorly, and its demo uses the last checkpoint before a time limit. My own humanoid get-up task failed (0 of 200 fallen starts after 300M steps), so the page shows no humanoid recovery.
- Demo seeds were fixed before evaluation. The terrain test uses the one training heightfield, not unseen ground. [Configuration history](docs/evaluation.md).

## Prior work

- Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844) (2025): the tasks, rewards and training settings; their Go1 and G1 policies transfer to real robots.
- Rudin et al., [Learning to walk in minutes](https://arxiv.org/abs/2109.11978): GPU-parallel locomotion training with random pushes.
- Freeman et al., [Brax](https://arxiv.org/abs/2106.13281); Todorov et al., [MuJoCo](https://mujoco.org/), including its official [WebAssembly bindings](https://github.com/google-deepmind/mujoco/tree/main/wasm).

Written with AI coding assistance.
