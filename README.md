# pushable

Environments, rewards and PPO settings come from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c). Robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97), with their licenses kept beside the models. Training uses [Brax PPO](https://github.com/google/brax/tree/v0.14.2).

I train locomotion policies for the Unitree Go1 dog and G1 humanoid, then run the same policies in standard MuJoCo and in a browser. A visitor can steer them and push them over. The dog also has a get-up policy.

<!-- RobotWeb: live-demo link and hero image -->

The default demo policies are seed 0, chosen before training. The Go1 walking comparison tests the default training setup against the same setup with random pushes enabled. Results are not reported until the runs and evaluation have finished.

## What I built

- [`train/app.py`](train/app.py): pinned training, model export and benchmark runs on Modal.
- [`eval/sim.py`](eval/sim.py): NumPy inference and standard-MuJoCo control, shared with the browser port.
- [`eval/run.py`](eval/run.py): velocity tracking, push recovery and get-up trials.

## Reproduce

Use Python 3.12 and a Modal account. All Modal runs go through the project's spend guard; `train/launch.py` enforces the authorized project cap. The package versions are in [`uv.lock`](uv.lock), and each run records its configuration, wall time and upper-bound charge under `results/runs/`.

## Limits

- This is simulation, not a hardware deployment.
- The comparison uses three seeds per arm, not a broad algorithm benchmark.
- Walking models have foot-only ground collisions; fall detection also uses trunk tilt.

## Prior work

I use Playground's tasks and PPO settings rather than claiming a new learning algorithm. [MuJoCo](https://mujoco.org/) supplies the physics engines; Playground and Menagerie supply the environments and robot models.

Written with AI coding assistance.
