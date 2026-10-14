# Browser runtime

The browser uses the official single-threaded [`@mujoco/mujoco` 3.15.0](https://github.com/google-deepmind/mujoco/tree/3.15.0/wasm), [three.js](https://threejs.org/), and the exact Playground/Menagerie models in [`assets/`](../assets/). Model provenance and upstream licenses travel with those assets. The npm lockfile records the renderer and build-tool versions.

```sh
npm ci
npm run dev
npm run build && npm test
```

Vite serves `/pushable/`. The manual-only Pages workflow builds and tests the site; it does not enable Pages or change repository visibility. The demo always selects `go1-walk-s0`, `go1-getup-s0`, and `g1-walk-s0`. It does not substitute the push-trained Go1 comparator.

## Simulation

`src/policy.ts` ports [`eval/sim.py`](../eval/sim.py). It uses exported observation block order, sensor names, normalization, float32 network parameters, swish hidden layers, `tanh(mean)` actions, and position targets. G1 gait phase and both action-history buffers reset with each policy. The next observation contains the previous history before the latest action and phase are written. There is no extra `mj_forward` after a control step: sensors retain MuJoCo's final-substep timing.

Layer weights are lossless base64-encoded, little-endian IEEE float32 arrays, decoded once when a policy loads. They are not quantized.

`src/physics.ts` steps at `simulation_dt`, with exactly `n_substeps` per control step. The fixed accumulator runs at wall-clock speed, or one-quarter speed when selected. Catch-up is capped at four control steps per frame; excess elapsed time is discarded, not accumulated. A slow device therefore runs slower instead of blocking the page. Switching tabs clears held steering input.

The renderer constructs mesh geometry from compiled model vertices/faces and primitives from geom sizes, and interpolates between control frames. Robot resources load on demand. Soft shadows use a 1024-pixel map and device pixel ratio is capped at 1.5. Camera orbit is independent of steering.

`npm run simplify` regenerates the visual-only STL copies with [meshoptimizer](https://github.com/zeux/meshoptimizer) 1.3.0: target 10% of input triangles, maximum relative simplification error 0.002, `Prune` enabled. [`visual/manifest.json`](visual/manifest.json) records every input/output hash, size, triangle count and returned error. The upstream [Go1](../assets/go1/LICENSE-Menagerie) and [G1](../assets/g1/LICENSE-Menagerie) licenses also cover these derived meshes. Original XML and assets remain unchanged. Only zero-collision-mask meshes outside explicit contact pairs may be reduced. Tests compare inertias, joints, actuators, sensors, collision primitives, and a short zero-action trajectory before and after replacement; parity uses the reduced assets actually served by the browser.

## Input and recovery

W/S or up/down commands forward/backward motion. A/D or left/right commands lateral motion; Q/E turns. The touch joystick provides planar motion, with separate turn buttons. A standard-mapped gamepad uses its left stick for motion and right stick for yaw; buttons 0 and 1 push and reset. The exported command ranges set all limits. Releasing or cancelling a touch clears only that pointer's command, leaving other held touches active. Robot selection returns keyboard focus to the scene.

Dragging a body applies a mass-scaled damped spring at the picked point, including the resulting torque about its centre of mass. Push applies a uniformly random horizontal world-frame impulse to the exported force body over exactly 0.1 simulated seconds. The slider is in N·s; the force is the selected impulse divided by 0.1 s, as in the [evaluation protocol](../docs/evaluation.md). A partial final integration interval is weighted so it does not add extra impulse.

Go1 enters get-up when base height is below 0.16 m or the body/world up-vector dot product is below 0.45, continuously for 0.12 simulated seconds. It returns to walking after 0.5 continuous seconds above 0.24 m, up-vector dot product above 0.92, horizontal speed below 0.6 m/s, and angular speed below 1 rad/s. These are demo switching rules, not the evaluation's success criterion.

The walking model has feet-only floor collisions; the get-up model has different torso/limb/self-collision geometry. Handoff switches the entire compiled model, preserving positions and velocities, and resets action/gait history. It does not splice collision parameters into the walking model. Reset restores the walking policy's exported initial pose.

## Parity

`npm test` replays every `results/parity/*.json` using the same WASM engine as the browser. It checks the policy file hash, observation vectors, deterministic network actions, motor targets, and resulting positions/velocities. It prints maximum absolute errors per policy. The test applies the recorded action **after** comparing inference, isolating float32 matrix-multiplication differences from contact divergence. It uses fixture-provided absolute/relative tolerances for observations and actions. Empty or shortened fixtures are errors, not skipped tests.

Performance and learned recovery are only claims after measurement on the actual policies. A successful browser test alone does not establish 60 fps on every laptop or phone.

Written with AI coding assistance.
