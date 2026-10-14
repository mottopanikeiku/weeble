# pushable

Walking robots you can shove in your browser. I trained the actors on a cloud GPU and measured them in standard MuJoCo. The page uses MuJoCo's WebAssembly build, with [parity checks](results/parity-checks/wasm.json) against my Python controller.

**[Try it →](https://mottopanikeiku.github.io/pushable/)** Steer, drag a body or press **Push**. Compare the dog trained with and without shoves.

![The Go1 dog is shoved over and gets back up; then the G1 humanoid walks](web/media/hero.gif)

<!-- generated:headline:start -->
**Does training with random shoves make the dog harder to knock over?** Yes. The shove it survives half the time (J50) rises from 24.1 to 32.9 N·s with three seeds per arm pooled (per seed: 22.7–26.8 vs 31.3–35.8 N·s): **+8.8 N·s, 95% interval +4.6 to +13.1**. [Comparison](results/report/ablation.csv)
<!-- generated:headline:end -->

Walking tasks and PPO settings: [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c). Robot models: [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97). Training: [Brax PPO](https://github.com/google/brax/tree/v0.14.2). Terrain texture: [Poly Haven CC0](https://polyhaven.com/a/rock_face). [Licenses](assets/). I built the runtime, experiments and custom humanoid recovery task.

## What I built

- **Runtime** ([web/src/physics.ts](web/src/physics.ts), [eval/sim.py](eval/sim.py)): matching observations, control timing and walking/get-up handoffs.
- **Experiments** ([eval/run.py](eval/run.py), [eval/terrain.py](eval/terrain.py)): shove recovery, tracking, fallen starts and terrain cross-tests.
- **Training and reports** ([train/app.py](train/app.py), [eval/report.py](eval/report.py)): exported actors, run manifests and regenerated tables.

**In the browser.** Landscape, movable props, throws and a terrain patch add play. Policies never saw props or throws; the rough-trained dog saw the supplied heightfield. Robot mechanisms and training parameters are unchanged. Labels name the active policy and training scene. Browser-only G1 fall collisions prevent sinking.

## Results

Standard MuJoCo with deterministic actions, no noise or domain randomization. Survival excludes excessive tilt, non-foot floor contact and numerical failure. [Protocol](docs/evaluation.md).

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

## Reproduce

```bash
uv run python -m eval.report results/eval --output results/report --update-readme   # every number above, from raw trials
cd web && npm ci && npm run dev                                                    # the demo, locally
uv run python train/launch.py train --env Go1JoystickFlatTerrain --seed 0 --gpu L4 --minutes 25   # retrain on Modal
```

<!-- generated:cost:start -->
NVIDIA L4 training: flat-ground dog 12–14 minutes ($0.22–0.25), dog get-up 15–16 ($0.27–0.28), walking humanoid 45–49 ($0.79–0.86) per policy. Rough-ground dog: 18–19 minutes ($0.31–0.34). The 15 published policies cost $5.65; all cloud jobs, including failures, $11.37. Wall-time/list-price upper bounds, not invoices. [Runs](results/report/training.csv) · [All jobs](results/report/invocations.csv)
<!-- generated:cost:end -->

## Limits

- Simulation only; nothing here ran on a real robot.
- Walking evaluation retains Playground's feet-only floor contacts; tilt detects most falls. Browser fall collisions and props are separate additions.
- One of three get-up seeds rarely gets up (3 of 187 fallen starts); I report it rather than retrain it.
- The humanoid tracks turning commands poorly. Its demo uses the last checkpoint before a time limit. The initial custom get-up experiment failed; no G1 recovery is shown.
- Demo seeds were chosen before evaluation; the terrain comparison tests one training heightfield, not unseen-ground generalization. [Configuration history](docs/evaluation.md).

## Prior work

- Zakka et al., [MuJoCo Playground](https://arxiv.org/abs/2502.08844) (2025): the tasks, rewards and training settings; their Go1 and G1 policies transfer to real robots.
- Rudin et al., [Learning to walk in minutes](https://arxiv.org/abs/2109.11978): GPU-parallel locomotion training with random pushes.
- Freeman et al., [Brax](https://arxiv.org/abs/2106.13281); Todorov et al., [MuJoCo](https://mujoco.org/), including its official [WebAssembly bindings](https://github.com/google-deepmind/mujoco/tree/main/wasm).

Written with AI coding assistance.
