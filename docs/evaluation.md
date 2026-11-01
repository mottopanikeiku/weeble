# Evaluation

I measure velocity tracking, shove survival and get-up recovery in MuJoCo. The [result tables](../results/report/) and [robustness atlas](../results/atlas/report/) use the methods below. Every demo policy uses training seed 0.

## Policies and inference

The main evaluation covers `go1-walk`, `go1-walk-push`, `go1-getup` and `g1-walk`, each trained with seeds 0, 1 and 2. The Go1 walking variants differ only in `pert_config.enable`: false for the baseline, true for push training. Training uses the pinned Playground defaults and Warp backend. Evaluation uses the actor mean with observation noise, domain randomization and automatic disturbances disabled. Reset clears observation/action history. Physics, gains, action scaling, clipping and observation timing come from the export. The terrain comparison and atlas also include `go1-walk-rough`.

## Velocity tracking

Each walking policy runs 27 commands: the Cartesian product of three evenly spaced values per axis. Go1 ranges are forward ±1.5 m/s, lateral ±0.8 m/s and yaw ±1.2 rad/s; G1 ranges are ±1 m/s, ±0.5 m/s and ±1 rad/s. These are checked against each export's trained ranges. Episodes last 10 s from the nominal pose; tracking excludes the first 2 s and continues after falls. Mean absolute errors are `forward_mae` and `lateral_mae` in m/s, and `yaw_mae` in rad/s, all in the body frame. Numerical failures end an episode early; their partial-sample errors are labelled conditional and they count against the completed-episode fraction.

Velocities use the named trunk/pelvis body frame (`mjOBJ_XBODY`). The principal-inertia frame (`mjOBJ_BODY`) has substantially different axes for Go1. Earlier trials measured in that frame are marked `superseded` and excluded from the report. A freejoint-velocity test checks this convention.

## Cross-terrain walking

The comparison uses default `Go1JoystickFlatTerrain` (`go1-walk`) and `Go1JoystickRoughTerrain` (`go1-walk-rough`), each with training seeds 0, 1 and 2. Both disable training perturbations. Each policy runs on the exact flat and rough training scenes with unchanged actor, gains, robot mechanism and physics options.

Rough training uses `njmax=256` after the upstream value of 60 overflowed in [`rough-smoke-v2`](../results/runs/rough-smoke-v2/). That smoke run is excluded. The change increases constraint-buffer capacity; rewards, PPO settings, terrain and solver settings remain at their upstream defaults.

Each policy/seed runs both scenes on 27 commands: forward {-0.5, 0, 0.5} m/s × lateral {-0.3, 0, 0.3} m/s × yaw {-0.6, 0, 0.6} rad/s. These speeds keep ten-second trajectories inside the 20 m square. Trials start at x=y=0, base z=0.35 m, identity orientation, nominal joints and zero velocities, with empty controller history. The starting height is the rough task's nominal pose and clears both terrains. Tracking measures the final 8 s in the body frame.

A fall is tilt over 60° or an enabled non-foot geometry touching the floor, checked at every integration step, including the first 2 s. Tracking continues after falls. Numerical failures stop the episode and keep their partial-sample errors. There are no shoves, browser props, observation noise or domain randomization. The report gives forward/lateral/yaw mean absolute error and fall frequency per seed and averaged equally across seeds. Raw trials are in `results/terrain/`; the report writes `terrain.csv` and `terrain_seeds.csv`.

## Experimental humanoid get-up

The custom `G1Getup` task drops the full-contact model from 1.0 m with nominal joints and uniform SO(3) orientations, holds the joint targets for 1 s, then runs a six-second recovery episode. Its actor has no commands or gait clock. Rewards favour torso-up alignment and standing pelvis height, with joint-limit, action-rate and torque penalties.

PPO settings follow Go1Getup except `num_resets_per_eval=4`. With 50M requested steps, explicit resets are 8 s apart, long enough for the six-second recovery episode. Physics options are unchanged; full-body contacts use `naconmax=64*8192` and `njmax=512` allocation.

Evaluation uses 200 drops with zero velocities, nominal joints, one-second settling and a reset controller. Success requires tilt below 15°, pelvis height at least 0.65 m, linear speed below 0.2 m/s and angular speed below 0.5 rad/s continuously for 1 s, checked at every physics step. Fallen and already-upright starts are counted separately. The demo threshold is 80% success among fallen starts.

| Training run | Requested steps | Explicit reset interval | Recovered fallen starts |
|---|---:|---:|---:|
| [`g1-getup-s0-t50m-v2`](../results/runs/g1-getup-s0-t50m-v2/) | 50M | 8 s | [0/200](../results/getup-diagnostic/20261014T184451Z-87018d1a/) |
| [`g1-getup-s0-t300m-v2`](../results/runs/g1-getup-s0-t300m-v2/) | 300M | 46 s | [0/200](../results/getup-diagnostic/20261014T202800Z-cff38d9a/) |

The 300M-step run started from scratch with seed 0. Only the requested timestep count changed; it also lengthened the reset interval. Training reward increased, but neither policy recovered in the drop test, so neither is in the demo. Exports and curves remain in the run directories; [`g1_getup.csv`](../results/report/g1_getup.csv) contains both results.

## Push recovery

Flat-ground Go1 with and without push training, plus G1, receive a forward command of 0.5 m/s. Trials start at the nominal pose and walk for a uniformly sampled 2–3 s before the push. A horizontal world-frame force acts on Go1 `trunk` or G1 `pelvis` for 0.1 s, with direction uniform on [0, 2π) and magnitude J / 0.1, where J is impulse in N·s. The trial continues for 3 s after the force ends. Each trained seed and impulse has at least 100 directions/onset times, including J=0 controls. Trial seeds are fixed in the configuration; Go1 trial indices share directions and onset times across impulses, training variants and C MuJoCo/MJX.

The Go1 grid is 0, 5, …, 50 N·s. The G1 grid is 0, 10, 20, 30, 40, 60, 80, 100 N·s. Failure is base tilt strictly greater than 60° or a non-foot robot geometry contacting the floor at any integration step, including before the push. Non-finite states, observations or actions, simulator errors and MuJoCo numerical warnings also count as failures. Trials are not retried. Survival requires completing the whole trial without a failure. Walking models have feet-only floor collisions, so the tilt criterion detects most falls. Each manifest lists the enabled floor-contact geometry.

## Get-up

For each Go1 get-up seed I sample at least 200 trials. The starting base orientation is uniform on SO(3), generated by normalizing four independent standard normal samples. The base starts at z=0.5 m with zero velocity and nominal joint angles. It drops and settles for 1 s under joint position control holding those initial angles, without policy actions. The resulting physical state is passed through a fresh policy reset, then the policy runs for 6 s. This isolates orientation recovery rather than the upstream mixture of nominal/random poses and random joints.

Success requires tilt strictly below 15°, trunk height at least 0.2 m, linear speed below 0.2 m/s and angular speed below 0.5 rad/s continuously for 1 s, checked at every physics step. Tilt and height use the integrated freejoint pose without recomputing the actor's sensor stage. The report gives success fraction and median time from policy start to the first qualifying one-second interval. The policy runs the full 6 s; a later numerical failure still makes the trial fail. Floor contact is allowed. Raw rows include initial orientation, settled pose and recovery time.

The headline counts starts whose saved `settled_qpos` is fallen. A pose is already upright if its integrated freejoint tilt is below 15° and its height is at least 0.2 m; speed does not affect this classification. `getup.csv` separates `fallen`, `already_upright` and `all` starts per seed. Missing or invalid settled poses are `unclassified`. Recovery-time medians include successful trials in the named group.

In `getup.csv`, `equal_seed_success` is the mean of per-seed success fractions; `count_success` is total successes divided by total trials. They can differ when settled-pose groups have different numbers of trials per seed. `ci_low` and `ci_high` describe the equal-seed estimate. [getup_trials.csv](../results/report/getup_trials.csv) gives every trial's classification.

### Get-up diagnosis

The original 50M-step Go1 seed 0 recovered in 35 of 200 C MuJoCo drops. Its last two [training evaluations](../results/runs/go1-getup-s0/curve.csv) rose from 6.265 to 7.036 reward. I compared this drop test with the upstream training reset distribution in both backends.

Upstream reset chooses a random orientation and random joints at 0.5 m with probability 0.6, otherwise the nominal home pose. Root linear and angular velocities are uniform on [-0.5, 0.5]. Joint-position control holds the sampled angles while settling for 0.5 s. The diagnostic keeps the cached sensor stage after settling, as upstream reset does. Both distributions use a 300-control-step (6 s) policy window and the same one-second standing criterion.

On the upstream reset distribution, C MuJoCo succeeded in 97/200 trials: 83/84 nominal starts and 14/116 dropped starts. Warp completed 182 of 200 trials before its time limit: 79/79 nominal starts and 15/103 dropped starts succeeded. [Diagnostic results](../results/report/getup_diagnostic.csv) separate backend, initialization, training run, completion status and recovery time, with trial-bootstrap intervals for seed 0.

Brax resets training worlds between optimization epochs as well as at the six-second episode limit. The 50M-step defaults allow only 160 control steps (3.2 s) between explicit resets. Increasing the budget to 200M steps, with all other settings unchanged, extends that interval to 12.4 s and allows the full recovery episode. [`training.csv`](../results/report/training.csv) calculates the interval from each run's configuration. All three final get-up seeds use this 200M-step configuration.

Use `getup-diagnostic --initialization drop` for the original test and `--initialization upstream` for training resets. `--remote-run <training-run-id>` selects a stored policy on the Modal Volume, including older runs. `fetch-getup-diagnostic --remote-run <diagnostic-launch-id>` retrieves saved trials. These are stages of `python -m train.launch`.

For interrupted G1 walking training, `export-checkpoint --remote-run g1-walk-s0 --checkpoint-steps 181043200` exports the last saved checkpoint. Its manifest links the original configuration and curve. `check-policy --policy g1-walk-s0 --gpu none` runs the 27-command C MuJoCo tracking grid; its results are in `results/policy-checks/`.

G1 seeds 1 and 2 completed the default 200M-step configuration with a 60-minute L4 limit. Seed 0 uses the interrupted checkpoint above. `training.csv` lists each seed's completed steps.

## Go1 ablation and uncertainty

The Go1 comparison uses all three training seeds per variant, with equal trial counts per seed at each impulse. I fit a non-increasing survival curve with weighted isotonic regression, then linearly interpolate its first 50% crossing to obtain J50. Curves still above 50% at the largest impulse are right-censored; curves below 50% at zero are left-censored. The report gives the grid bound in those cases. Exactly 50% is an observed crossing. Each seed also has its own survival curve and J50.

The report uses 2,000 hierarchical bootstrap replicates, resampling training seeds, then whole trial indices within each sampled seed. One index carries the same direction and onset through every impulse, so its outcomes are resampled together. The Go1 comparison uses the same sampled indices in both variants.

Percentile intervals are 95%. If any replicate is censored, the J50 interval is omitted and the censored fraction and grid bounds are shown. Incomplete trial-index grids have no joint bootstrap interval; fewer than three complete seeds produces a partial result. The [older report](../results/superseded/independent-impulse-report.zip) sampled impulses independently.

## Robustness atlas

`python -m eval.atlas` evaluates all 12 walking policies (`go1-walk`, `go1-walk-push`, `go1-walk-rough` and `g1-walk`, seeds 0, 1 and 2) in C MuJoCo on CPU. It uses the same deterministic actor, observation timing and fall rule as the push test, without retraining.

**Episodes.** Every policy runs on its robot's flat walking scene (`assets/go1/walk.xml` or `assets/g1/walk.xml`) from the nominal pose. This includes rough-trained Go1, whose training floor friction was 1.0 rather than 0.6. A fall is tilt over 60° or a non-foot floor contact at any physics step. Numerical failures are a separate outcome.

**Environment stresses.** These use 10-second episodes. Each trial draws a starting heading and a fixed command. Each command component is uniform within ±50% of the robot's standard tracking range: Go1 ±0.75 m/s forward, ±0.4 m/s lateral and ±0.6 rad/s yaw; G1 ±0.5, ±0.25 and ±0.5. Levels, nominal first:

- *Held-out rubble:* flat, then heightfields with a 5, 7.5, 10 or 12.5 cm range. Each trial gets a 256 × 256 grid of independent uniform heights over 20 × 20 m, centred on floor height. The robot starts half the height range higher to clear the ground. The training heightfield in `rough.xml` has the same 7.8 cm spacing and a 5 cm range. The manifest includes the spread and neighbour correlation of both the training and generated grids.
- *Floor friction:* μ = 0.6 (the scenes' value), then 0.4, 0.25 and 0.15, plus 1.0. I set the floor's sliding friction and the G1 foot–floor contact pairs, which override geom friction.
- *Payload:* +0, 10, 20, 30 and 40% of the robot's mass, added to the base body (trunk or pelvis) at its centre of mass.
- *Actuator strength:* 100, 85, 70, 55 and 40% of the position actuators' gains and force limits, so every motor torque scales by the same factor. Joint damping, which these models use as the derivative term, is unchanged.
- *Sensor delay:* 0 to 4 control steps (0 to 80 ms). The policy sees gyro, gravity, joint position, joint velocity and local linear velocity from that many steps earlier. It holds the first reading until the buffer fills. The command, previous action and gait clock stay current.

The nominal levels of these five stresses are the same episodes. The report requires their rows to be identical.

**Command steps.** These also use 10-second episodes, starting with heading zero. The command switches once, at a time uniform in 2–3 s, between full values of the standard tracking ranges. The five changes are: standing to full forward, full forward to standing, full forward to full backward, full left to full right, and full left turn to full right turn.

**Shoves.** These follow the push protocol: a 0.5 m/s forward command, a uniform 2–3 s wait, a horizontal force on the base for 0.1 s, then 3 s more. The direction is uniform in the body frame at onset. I normalize the impulse as J/(m√(gh)), with total mass m and nominal centre-of-mass height h. That is the change in centre-of-mass speed divided by the inverted-pendulum speed √(gh), so the dog and the humanoid share one scale. Rows also keep the impulse in N·s. The levels are 0 (a control), 0.5, 0.75, 1, 1.25 and 1.5. Playground's training kicks push with 0.5 sin(πt/T)·m·v/T for T seconds, an impulse of m·v/π. The manifest gives the largest one, at v = 3 m/s, in the same units. The report bins directions into eight 45° sectors. As the gait phase it records which feet touched the floor at onset.

**Fall and recovery.** A fallen Go1 switches to the demo's get-up policy (`go1-getup-s0`) in the full-contact `getup.xml`, keeping its state, stress and terrain. Recovery must meet the get-up rule within 6 s: tilt below 15°, height at least 0.2 m and low speed, held for 1 s. A limb can overlap the floor when full contacts are enabled; each row gives that overlap. Recovered episodes still count as falls. The report separates the never-fell share from the recovered share of falls. G1 has no get-up policy.

**Paired trials.** Random draws use `SeedSequence([20261029, group, trial])`, independent of policy and level. Every policy sees the same commands, terrain grids and shoves, scaled to its command range, mass and centre-of-mass height. Each row includes a SHA-256 of its level and shared settings. Raw rows omit wall time and use a fixed gzip header. The original manifest has `git_dirty: true`; a replay from `1d7b688` matched all 21,600 episodes and generated files ([replay record](../results/atlas/clean-replay.json)).

`python -m eval.atlas --rerun N` re-simulates a fixed sample spanning every stress, observed outcome and policy variant, excluding nominal and zero-shove levels. It prints changed fields and outcomes, and fails on numerical failures during walking or recovery, missing rows or exceptions. `--strict` also fails on an outcome change; `--exact` fails on any changed field.

**Statistics.** There are 50 trials per policy and level: 36 levels × 12 policies = 21,600 episodes. For each seed and level, the report gives the never-fell share, fall count and recovered share of falls, with 95% Wilson intervals. Seeds have separate intervals. The README summarizes the three seeds as ranges: each seed's hardest level with at least 90% survival at that level and every easier one, or "below nominal" if the nominal level misses 90%. For command steps, it gives the range of survival rates on each seed's worst change. `python -m eval.atlas --report-only` rebuilds the tables, figure and README block from raw rows; `python -m eval.verify` checks them against the saved outputs.

## Sim-to-sim check

The `mjx` runner uses MJX Warp, the training physics backend, on a Modal CUDA GPU. It requires `training_config.impl='warp'`. The model, deterministic actor, observation timing, commands, push directions, onset times and trial IDs match C MuJoCo. Model parameters, integration settings and contact/constraint capacities come from the export. Reset uses Warp forward dynamics; a compiled scan checks state, tilt, active contacts and capacity overflow at every physics step.

The backend comparison uses Go1 walking seed 0, the 27-command tracking grid and 20 trials per impulse. Run it with `python -m train.launch gap --policy go1-walk-s0 --backend mjx --gpu L4 --minutes 40`. Directions, onset times, force duration, episode windows and failure rules match the standard test. The report pairs Warp trials with the first 20 C MuJoCo trial IDs at each impulse; both J50 estimates use those paired trials.

Warp model preparation uses Playground's `mjx_env.put_model` and its default warning options. Only arrays needed for observations and tracking cross back to NumPy; contacts and capacity checks stay on the GPU. GPU parity tests compare this transfer with MuJoCo's full `get_data_into`, including observations and body velocities. The first invocation timed out while copying full contact buffers and using extra warnings; its partial results are excluded.

Single-world Warp evaluation caps its contact allocation at `mjMAXCONPAIR` times every mask-enabled or explicit geom pair, including excluded and same-body pairs in the bound. It uses the smaller of this bound and the 8,192-world training allocation. Contact filtering, solver settings and per-step overflow checks are unchanged. GPU parity also compares a control step with the original full allocation.

Each parity invocation has its own Volume directory. The sensor-stage test copies native data before Warp conversion, which calls kinematics in place, and restores the cached site matrices before comparing fields.

## Source records

Get-up training seeds 1 and 2 used commit `94fa026ee21b822a6673d23a8dece21def21702a` with uncommitted changes. Their configurations, package versions, models, weights and curves are available, but the working-tree patch was not saved. Tracked changes through `c06840c` affect only `eval/report.py`; the exact source snapshot for those two training runs cannot be reconstructed.

Get-up seed 2 and push-trained walking seed 2 evaluations were repeated from clean commit `1a7f7d5`. [`reruns.csv`](../results/report/reruns.csv) compares the raw rows. Outcomes, recovery times and tracking metrics agree. The push run differs only in run ID and wall time. In 22 get-up trials, initial quaternions differ by at most 2.23×10⁻¹⁶ and settled poses by at most 5.45×10⁻¹³. The report uses the repeated evaluations and excludes the superseded originals.

## Running and reporting

The evaluator runs locally or on Modal. For a CPU run on Modal, use `python train/launch.py eval --policy go1-walk-s0 --gpu none --minutes 20`; `--minutes` sets the timeout. The launcher saves source revision, launch settings and wall time, and estimates cost from [Modal list prices](https://modal.com/pricing), including CPU and memory with a 10% margin. Remote runs require a configured Modal account. The direct entrypoints are:

```sh
python -m eval.run --root . --output results/c-mujoco
python -m eval.run --root . --output results/mjx --backend mjx --seeds 0 --provenance evaluation-provenance.json
python -m eval.report results/c-mujoco results/mjx --output results/report
```

`run_suite(root, output, backend='mujoco', seed_filter=None, *, config=None, policy_filter=None, modal_app_id=None, provenance=None, getup_initialization='drop')` returns the new run directory. `provenance={'git_sha': ..., 'git_dirty': ..., 'modal_app_id': ..., 'gpu': ...}` supplies source and machine details when the container has no git checkout. Use the selected GPU type for MJX Warp and `none` for CPU; the GPU route requires this field. The CLI reads the dictionary with `--provenance path.json`. `policy_filter` accepts one policy stem or an iterable; `--policy` is repeatable. `SuiteConfig` sets grids, trial counts, thresholds, durations, trial seed and suite selection. Use `getup-diagnostic` for training-reset comparisons.

Each run has a summary `manifest.json`. Each `<backend>/<policy>/` directory has a policy manifest, append-only `trials.jsonl` and matching `trials.csv`. Rows flush immediately; policy manifests update every 100 rows and at completion. Manifests include source revision, effective configuration, selected policies/seeds, policy/model SHA-256, package versions, hardware, backend, timing, Modal app ID, completion status and enabled contacts. The backend key is `mjx`, with `backend_detail='mjx-warp'`. Actor weights are referenced by hash. Runs use separate directories. `modal_app_id` comes from the argument, `provenance` or the environment, otherwise null. Trial failures keep error text, and non-finite diagnostic values become null. Caught interruptions set status `interrupted`; a hard termination can leave `running`. Completed rows remain usable.

The Modal wrapper compresses JSONL and CSV before download. To rebuild the main report, run:

```sh
python -m eval.report results/eval --output results/report --update-readme
```

The report reads raw JSONL or `.jsonl.gz` plus manifests, rejects duplicate trial IDs and excludes `superseded` runs. It writes tracking, push, get-up, J50, ablation and backend-comparison tables, SVG figures and `summary.json`, including input hashes and bootstrap settings. It also reads `results/terrain/` and `results/getup-diagnostic/`.

`web/src/data/survival.json` supplies the browser's shove percentages: evaluated impulses, survived/total counts, seed IDs and source paths. Values are pooled across seeds without interpolation. There are no shove results for rough-trained Go1 in this file.

Training costs come from `results/spend.jsonl` and per-launch manifests, matched by run ID to avoid double counting. `invocations.csv` includes failed jobs and distinguishes historical charges from later list-price estimates. Benchmark tables separate steady-state physics cost per million control steps from total invocation cost per measured step, which includes compilation, warmup and download. These benchmarks use zero actions.

If download fails after a Volume commit, `python train/launch.py fetch-eval --gpu none --minutes 2 --remote-run <evaluation-directory-or-launch-id> --run-id <download-id>` retrieves the saved trials and manifest. It does not repeat simulation or change the original job's failed status.

`fetch-training --remote-run <training-run-id>` retrieves the last saved training manifest, curve and checkpoint inventory without overwriting the billing manifest. Original PPO tuples are on the Volume and the [checkpoint release](https://github.com/mottopanikeiku/weeble/releases/tag/policies-2026-10-14); `python -m train.fetch_checkpoints` downloads and verifies them for parity tests.

Raw `TrialFailure` errors such as `no_recovery` describe trial outcomes. `failure=simulator_error` identifies an unexpected exception.

`--update-readme` also updates the marked README sections and get-up count from the tables. Incomplete ablation, get-up, G1 or backend-comparison inputs raise an error. Omit this option when generating partial analyses or alternate reports.

## Checks on every pull request

[`ci.yml`](../.github/workflows/ci.yml) runs on pull requests and pushes to `main` that touch evaluation or training code, policies, assets, results, the web runtime, root README, dependency files or the workflow itself.

- **Evaluation:** `uv sync --frozen --only-group eval` installs MuJoCo 3.15.0, NumPy and pytest. CI runs Python tests excluding `eval/test_parity.py` and `eval/test_g1_getup.py`, then `python -m eval.verify` and `python -m eval.atlas --rerun 36 --workers 2`.
- **Python parity:** `uv sync --frozen --no-extra cuda` installs the full CPU dependencies. CI downloads and verifies all 15 `.brax` sidecars, tests the downloader, then runs `eval/test_parity.py` with `impl='jax'` in a separate process per policy. Warp parity and `eval/test_g1_getup.py` run separately on Modal because they need a GPU or more memory.
- **Browser:** `npm ci`, `npm run build` and `npm test`, including C MuJoCo fixture replay in WASM.

`python -m eval.verify` checks asset hashes, the sidecar table in `policies/README.md`, parity fixture/export hashes and evaluation model/policy hashes. Main evaluation and terrain results require a matching training export; diagnostics may refer to older runs whose exports are unavailable.

It rebuilds the report with the saved bootstrap settings and compares `results/report/`, `web/src/data/survival.json` and generated README sections with the saved versions. Atlas checks cover raw-file hashes, settings, one row per level/trial, model/export hashes and identical nominal episodes across stresses, then compare every regenerated atlas file and its README block. Any difference exits with status 1 and is listed.

[`eval/test_verify.py`](../eval/test_verify.py) checks these failures by changing exports, models, raw outcomes, hashes, report tables, browser counts and README numbers, and by deleting or corrupting files.

## Limitations

- The cross-terrain comparison uses one training heightfield. Atlas rubble uses new random heights at the same 7.8 cm spacing; it does not cover stairs, slopes or smooth hills.
- The atlas changes one factor at a time. Its intervals describe trial sampling within each seed.
- Atlas outcomes can differ across machines. A 36-episode sample matched exactly on the original machine; on one GitHub runner, 4 rows matched exactly and 29/36 outcomes agreed. Seven stressed episodes fell or recovered differently. Full per-level rates have not been compared across machines.

Upstream: [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c), MuJoCo and mujoco-mjx 3.15.0; robot meshes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97).
