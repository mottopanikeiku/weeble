# Evaluation

Everything here runs the exported policies in standard MuJoCo on a CPU. The numbers in the [README](../README.md) come from the [report tables](../results/report/) and the [stress-test atlas](../results/atlas/report/). The demo uses training seed 0 of every policy.

## Policies

The main comparison covers `go1-walk`, `go1-walk-push`, `go1-getup` and `g1-walk`, each with seeds 0, 1 and 2. The two Go1 walking variants differ only in `pert_config.enable`: off for the baseline, on for shove training. Training uses the pinned Playground defaults and the Warp backend. Evaluation uses the actor mean, with observation noise, domain randomization and automatic disturbances off. Reset clears the observation and action history. Physics, gains, action scaling, clipping and observation timing all come from the export. `go1-walk-rough` joins the terrain comparison and the atlas.

## Velocity tracking

Each walking policy runs 27 commands, three evenly spaced values per axis. Go1 ranges are ±1.5 m/s forward, ±0.8 m/s lateral and ±1.2 rad/s yaw. G1 ranges are ±1, ±0.5 and ±1. Episodes last 10 s from the nominal pose. Tracking skips the first 2 s and continues after falls. Errors are mean absolute differences in the body frame (`mjOBJ_XBODY`, the named trunk or pelvis frame), reported as `forward_mae`, `lateral_mae` (m/s) and `yaw_mae` (rad/s). A numerical failure ends an episode early. The principal-inertia frame (`mjOBJ_BODY`) has quite different axes for Go1, so trials measured in it are marked `superseded` and left out of the report.

## Push recovery

Flat-ground Go1 with and without shove training, and G1, get a 0.5 m/s forward command. After a uniformly random 2–3 s of walking, a horizontal world-frame force acts for 0.1 s on the Go1 `trunk` or G1 `pelvis`, in a uniform random direction, with magnitude J/0.1 for impulse J in N·s. The trial runs 3 s past the push. Every seed and impulse gets at least 100 directions and onset times, J = 0 controls included. Go1 trials share directions and onsets across impulses, variants and simulators.

The Go1 grid is 0, 5, …, 50 N·s. The G1 grid is 0, 10, 20, 30, 40, 60, 80 and 100 N·s. A trial fails if base tilt exceeds 60° or a non-foot geometry touches the floor at any step, if any state, observation or action is non-finite, or if MuJoCo raises a numerical warning. There are no retries.

## Get-up

Each Go1 get-up seed gets at least 200 trials. The base starts at z = 0.5 m with a uniform random orientation on SO(3) (four normalized Gaussians), nominal joints and zero velocity. It drops and settles for 1 s under joint position control, with no policy actions. The settled state goes through a fresh policy reset, then the policy runs for 6 s. This isolates orientation recovery, unlike upstream's mix of nominal and random-joint resets.

Success means tilt below 15°, trunk height at least 0.2 m, linear speed under 0.2 m/s and angular speed under 0.5 rad/s, held for 1 s. The report gives the success fraction and the median time to the first qualifying second. A later numerical failure still fails the trial.

The headline counts only starts that settled fallen. A pose is already upright if its tilt is under 15° and its height is at least 0.2 m. [`getup.csv`](../results/report/getup.csv) splits `fallen`, `already_upright` and `all` per seed, and [`getup_trials.csv`](../results/report/getup_trials.csv) classifies every trial.

### Why 200M steps

At the default 50M steps, Go1 get-up seed 0 recovered in 35 of 200 drops, though its training reward was still rising. Upstream's reset distribution is 60% random orientation and joints, 40% nominal pose, with root velocities uniform on ±0.5. On it, C MuJoCo succeeded in 97 of 200 trials (83/84 nominal starts, 14/116 dropped) and Warp in 94 of the 182 it finished before its time limit (79/79 nominal, 15/103 dropped). Brax resets training worlds between optimization epochs as well as at the 6 s episode limit, and 50M steps leaves only 3.2 s between explicit resets. At 200M steps the gap is 12.4 s, long enough for a full recovery. All three released get-up seeds use 200M. [`getup_diagnostic.csv`](../results/report/getup_diagnostic.csv) has the comparison.

## Shove training comparison

Both Go1 variants use all three seeds, with equal trial counts per seed at every impulse. J50, the impulse with 50% survival, comes from a non-increasing weighted isotonic fit to survival, linearly interpolated at its first 50% crossing. A curve still above 50% at the largest impulse is right-censored, and one below 50% at zero is left-censored. Exactly 50% counts as a crossing.

Intervals come from 2,000 hierarchical bootstrap replicates: resample training seeds, then whole trial indices within each seed. One index carries the same direction and onset through every impulse, so its outcomes move together, and both variants use the same sampled indices. Intervals are 95% percentile intervals, omitted if any replicate is censored.

## Cross-terrain walking

`go1-walk` (`Go1JoystickFlatTerrain`) and `go1-walk-rough` (`Go1JoystickRoughTerrain`), seeds 0–2, both without training perturbations. Each policy runs on the exact flat and rough training scenes with its own actor, gains, mechanism and physics options.

Rough training uses `njmax=256` because upstream's 60 overflowed. It changes buffer size only. Rewards, PPO settings, terrain and solver stay at upstream defaults.

Each policy and seed runs 27 commands on both scenes: forward {−0.5, 0, 0.5} m/s × lateral {−0.3, 0, 0.3} m/s × yaw {−0.6, 0, 0.6} rad/s. That keeps a 10 s trajectory inside the 20 m square. Trials start at x = y = 0, base z = 0.35 m (the rough task's nominal height, which clears both floors), identity orientation, nominal joints, zero velocity and an empty history. Tracking uses the last 8 s in the body frame. A fall is tilt over 60° or non-foot floor contact at any step, including the first 2 s. There are no shoves, props, noise or domain randomization. Raw trials are in `results/terrain/`, and the report writes `terrain.csv` and `terrain_seeds.csv`.

## Robustness atlas

`python -m eval.atlas` runs all 12 walking policies (`go1-walk`, `go1-walk-push`, `go1-walk-rough` and `g1-walk`, seeds 0–2) through stresses outside their training. It uses the same deterministic actor, observation timing and fall rule as the push test, on the flat walking scene of each robot (`assets/go1/walk.xml`, `assets/g1/walk.xml`), starting from the nominal pose. The rough-trained Go1 therefore runs on a floor with friction 0.6 instead of its training value of 1.0.

**Environment stresses.** Ten-second episodes. Each trial draws a starting heading and a fixed command, each component uniform within ±50% of the robot's tracking range (Go1: ±0.75 m/s, ±0.4 m/s, ±0.6 rad/s; G1: ±0.5, ±0.25, ±0.5). Levels, nominal first:

- *Held-out rubble:* flat, then random heightfields with a 5, 7.5, 10 or 12.5 cm range. Each trial draws its own 256 × 256 grid of independent uniform heights over 20 × 20 m. The robot starts half the range higher. The training heightfield in `rough.xml` has the same 7.8 cm spacing and a 5 cm range.
- *Floor friction:* μ = 0.6 (the scenes' value), then 0.4, 0.25 and 0.15, plus 1.0. Sets the floor's sliding friction and the G1 foot–floor contact pairs, which override geom friction.
- *Payload:* 0, 10, 20, 30 and 40% of the robot's mass, added to the base body at its centre of mass.
- *Actuator strength:* 100, 85, 70, 55 and 40% of every position actuator's gain and force limit. Joint damping, which these models use as the derivative term, is unchanged.
- *Sensor delay:* 0 to 4 control steps (0 to 80 ms) on gyro, gravity, joint position, joint velocity and local velocity. The first reading is held until the buffer fills. Command, previous action and gait clock stay current.

The nominal levels of the five stresses are the same episodes, and the report requires their rows to be identical.

**Command steps.** Ten-second episodes starting at heading zero. The command switches once, at a uniform 2–3 s, between full-range values: stand to forward, forward to stand, forward to backward, left to right, and left turn to right turn.

**Shoves.** The push protocol again: 0.5 m/s forward, a uniform 2–3 s wait, a 0.1 s horizontal force on the base in a uniform body-frame direction, then 3 s. Impulses are normalized as J/(m√(gh)) for total mass m and nominal centre-of-mass height h, which is the change in centre-of-mass speed over the inverted-pendulum speed √(gh), so dog and humanoid share one scale. One normalized unit is 20.2 N·s for Go1 and 85.3 N·s for G1. Levels are 0 (control), 0.5, 0.75, 1, 1.25 and 1.5. Playground's training kicks have an impulse of m·v/π for speed v; the manifest gives the largest, at v = 3 m/s. Directions are binned into eight 45° sectors, and each row records which feet touched the floor at onset.

**Fall and recovery.** A fallen Go1 switches to the demo's get-up policy (`go1-getup-s0`) in the full-contact `getup.xml`, keeping its state, stress and terrain. It must meet the get-up rule within 6 s. Recovered episodes still count as falls. G1 has no get-up policy.

**Pairing and statistics.** Draws use `SeedSequence([20261029, group, trial])`, independent of policy and level, so every policy sees the same commands, terrain grids and shoves, scaled to its command range, mass and height. There are 50 trials per policy and level, 36 levels × 12 policies = 21,600 episodes. For each seed and level the report gives the never-fell share, fall count and recovered share of falls, with 95% Wilson intervals. [`margins.csv`](../results/atlas/report/margins.csv) gives each seed's hardest level where at least 90% of episodes never fell, at that level and every easier one.

`python -m eval.atlas --report-only` rebuilds tables and figure from the raw rows. `python -m eval.atlas --rerun N` re-simulates a fixed sample of committed episodes, and fails on numerical failures, missing rows or exceptions (`--strict` also fails on any outcome change). Individual outcomes can differ on other CPUs: on one GitHub runner 29 of 36 sampled outcomes matched, so CI reports changes without failing on them.

## Humanoid get-up experiment

`G1Getup` is a custom task: the full-contact model drops from 1.0 m with nominal joints and a uniform random orientation, holds its targets for 1 s, then runs a 6 s recovery episode. The actor has no commands or gait clock. Rewards favour torso-up alignment and standing pelvis height, with joint-limit, action-rate and torque penalties. PPO settings follow Go1Getup except `num_resets_per_eval=4`.

Evaluation uses 200 drops with a 1 s settle and a fresh controller. Success needs tilt under 15°, pelvis height at least 0.65 m, linear speed under 0.2 m/s and angular speed under 0.5 rad/s for 1 s. The demo bar was 80% of fallen starts.

| Training run | Requested steps | Reset interval | Recovered fallen starts |
|---|---:|---:|---:|
| [`g1-getup-s0-t50m-v2`](../results/runs/g1-getup-s0-t50m-v2/) | 50M | 8 s | [0/200](../results/getup-diagnostic/20261014T184451Z-87018d1a/) |
| [`g1-getup-s0-t300m-v2`](../results/runs/g1-getup-s0-t300m-v2/) | 300M | 46 s | [0/200](../results/getup-diagnostic/20261014T202800Z-cff38d9a/) |

Training reward rose in both, but neither policy recovered, so G1 has no get-up policy in the demo. Both results are in [`g1_getup.csv`](../results/report/g1_getup.csv).

## Sim-to-sim check

The `mjx` runner uses MJX Warp, the training physics, on a Modal GPU. It needs `training_config.impl='warp'` and matches the C MuJoCo runner in model, deterministic actor, observation timing, commands, push directions, onset times and trial IDs. The comparison uses Go1 walking seed 0, the 27-command grid and 20 trials per impulse:

```bash
python -m train.launch gap --policy go1-walk-s0 --backend mjx --gpu L4 --minutes 40
```

Warp trials are paired with the first 20 C MuJoCo trials at each impulse. J50 came out at 30.0 N·s in MJX and 28.3 N·s in standard MuJoCo ([`gap.csv`](../results/report/gap.csv)).

## Running it

```bash
python -m eval.run --root . --output results/c-mujoco
python -m eval.report results/eval --output results/report
python -m eval.atlas
python -m eval.verify
```

`python train/launch.py eval --policy go1-walk-s0 --gpu none --minutes 20` runs the evaluator on Modal CPU (a configured Modal account is required). The launcher saves launch settings and wall time and estimates cost from [Modal list prices](https://modal.com/pricing).

Each evaluation run has a `manifest.json`, and each `<backend>/<policy>/` directory holds a policy manifest, an append-only `trials.jsonl` and a matching `trials.csv`. Manifests record the configuration, the policy and model SHA-256, package versions, hardware, timing and completion status. The report reads raw JSONL or `.jsonl.gz`, rejects duplicate trial IDs, skips `superseded` runs, and writes the tables, SVG figures and `summary.json` with input hashes and bootstrap settings. It also reads `results/terrain/` and `results/getup-diagnostic/`.

`web/src/data/survival.json` feeds the browser's shove percentages: pooled counts across seeds at each evaluated impulse, without interpolation. There is none for the rough-trained dog.

Training costs come from `results/spend.jsonl` and each launch manifest, matched by run ID. `invocations.csv` includes failed jobs.

## Checks

[`ci.yml`](../.github/workflows/ci.yml) runs three jobs:

- **Evaluation:** `uv sync --frozen --only-group eval` (MuJoCo, NumPy, pytest), the Python tests without the GPU-heavy ones, `python -m eval.verify` and `python -m eval.atlas --rerun 36`.
- **Parity:** the 15 released `.brax` files are downloaded, hash-checked and compared with the JSON exports, one process per policy (`eval/test_parity.py`, JAX on CPU). Warp parity and `eval/test_g1_getup.py` need a GPU and run on Modal.
- **Browser:** `npm ci`, `npm run build` and `npm test`, which replays C MuJoCo fixtures in WASM.

`python -m eval.verify` checks asset, export and fixture hashes, the sidecar table in `policies/README.md`, and every evaluation manifest. It then rebuilds the report and the atlas from the raw trials and compares them byte for byte with the committed files. [`eval/test_verify.py`](../eval/test_verify.py) corrupts each of those in turn and expects a failure.
