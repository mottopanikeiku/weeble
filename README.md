# pushable

Environments, rewards and PPO settings come from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c). Robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97), with their licenses kept in [assets](assets/). Training uses [Brax PPO](https://github.com/google/brax/tree/v0.14.2).

I trained walking policies for the Unitree Go1 dog and G1 humanoid, plus a separate Go1 get-up policy. They run live in the official MuJoCo WebAssembly build: steer them and shove them.

**Demo:** publication pending. [Run locally](web/README.md). <!-- RobotWeb: live-demo link -->

<!-- generated:headline:start -->
Training with random pushes changes Go1's estimated 50%-survival impulse (J50) by **+8.83 N·s** compared with default no-push training, with a **95% bootstrap interval [+4.73, +12.85] N·s**. This is **3 seeds per arm**, not a broad algorithm benchmark. [Comparison](results/report/ablation.csv).
<!-- generated:headline:end -->

![Go1 shove and recovery, followed by G1 walking](web/media/hero.gif)
[MP4](web/media/hero.mp4) · [Capture provenance](web/media/hero.json)

The demo runs the preselected seed-0 policies in single-threaded MuJoCo 3.15.0: Go1 walking and get-up, and G1 walking. WASD, the touch pad or a gamepad set the command; dragging a body pulls it, and Push applies the selected impulse in N·s. The clip follows simulation time, not browser frame rate.

## What I built

- [`train/app.py`](train/app.py): guarded GPU training, export and benchmarks.
- [`eval/sim.py`](eval/sim.py): the NumPy/C-MuJoCo reference copied by the browser.
- [`eval/report.py`](eval/report.py): tables, figures and README measurements from raw trials.

## Results

The Go1 comparison changes only the training push setting. Evaluation uses deterministic actions in standard MuJoCo, without observation noise or domain randomization. [Protocol and checks](docs/evaluation.md).

<!-- generated:results:start -->
| Policy | Seeds | J50 by seed (N·s) | Mean tracking MAE: forward / lateral / yaw |
|---|---:|---|---|
| Go1, no training pushes | 3 | 26.82 / 23.42 / 22.71 | 0.068 / 0.058 / 0.057 |
| Go1, training pushes | 3 | 31.35 / 31.36 / 35.79 | 0.072 / 0.058 / 0.061 |
| G1 (seed 0 interrupted) | 3 | 66.12 / 70.50 / 70.00 | 0.086 / 0.148 / 0.417 |

Tracking units are m/s, m/s and rad/s. J50 is the impulse at 50% survival. [Per-seed tracking](results/report/tracking.csv), [thresholds](results/report/j50_seeds.csv), [survival curves](results/report/push_seeds.svg), [raw trials](results/eval/).

Go1 get-up succeeds on **415/600 random drops (69.2%)**. Per seed: **seed 0: 200/200; seed 1: 199/200; seed 2: 16/200**. The median time to the qualifying upright interval among successes is **2.032 s**. [Per-seed results](results/report/getup.csv); already-upright training-reset starts are reported separately in the [diagnosis](results/report/getup_diagnostic.csv).

For the small seed-0 backend comparison, Warp minus C tracking MAE is **+0.003 / +0.001 / +0.002** in those same units. It uses **220 matched push-trial pairs** across the unchanged impulse grid. Matched J50 is **28.33 N·s in C versus 30.00 N·s in Warp**. [Backend comparison](results/report/gap.csv).
<!-- generated:results:end -->

The [get-up diagnosis](docs/evaluation.md#get-up-diagnosis) retains the original failed configuration and explains the longer final training interval.

## Reproduce

The first command regenerates every reported number. The second is an optional training example; use a separate clone because it replaces the export. Cloud runs require Modal credentials and the lab's `rl-modal` spend guard at `$RL_WAVE_HOME/bin/rl-modal` (default `~/Projects/rl-wave/bin/rl-modal`), with a funded `pushable` allocation.

```bash
UV_PROJECT_ENVIRONMENT="$HOME/.cache/rl-wave/venvs/pushable" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 nice -n 19 uv run --locked --python 3.12 python -m eval.report results/eval --output results/report --update-readme
UV_PROJECT_ENVIRONMENT="$HOME/.cache/rl-wave/venvs/pushable" nice -n 19 uv run --locked --python 3.12 python train/launch.py train --env Go1JoystickFlatTerrain --seed 0 --gpu L4 --minutes 25
```

<!-- generated:cost:start -->
The published Go1 walking seed 0 used **L4**, **13.71 min**, and **$0.2407 upper**. All guarded invocations, including failed runs, total **$9.5714 upper**. [Training costs](results/report/training.csv), [all invocations](results/report/invocations.csv), [ledger](results/spend.jsonl). These are guard upper bounds, not invoices.

G1 seed 0 stopped at **181,043,200 of 200,000,000 planned steps** on a time limit; seeds 1 and 2 completed their configured runs. I use seed 0's last persisted checkpoint for the demo, not its best-reward checkpoint. The G1 seeds therefore have unequal completed step counts. [Original curve and manifest](results/runs/g1-walk-s0/).
<!-- generated:cost:end -->

## Limits

- This is simulation. Walking models have foot-only ground collisions; tilt also determines falls.
- There are three Go1 seeds per arm. Get-up varies sharply by seed; its final configuration followed an initial failed run.
- G1 seed 0 ended early and has weak yaw tracking. Its completed step budget differs from the other seeds.
- The backend check uses one Go1 seed and fewer trials, not real hardware.

## Prior work

I use Playground's tasks and settings, not a new learning algorithm. [MuJoCo](https://mujoco.org/) supplies the physics engines.

Written with AI coding assistance.
