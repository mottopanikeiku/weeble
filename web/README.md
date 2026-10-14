# Browser runtime

The browser uses the official single-threaded [`@mujoco/mujoco` 3.15.0](https://github.com/google-deepmind/mujoco/tree/3.15.0/wasm), [three.js](https://threejs.org/), and the exact Playground/Menagerie models in [`assets/`](../assets/). Model provenance and upstream licenses travel with those assets. The npm lockfile records the renderer and build-tool versions.

```sh
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 UV_THREADPOOL_SIZE=2
nice -n 19 npm ci
nice -n 19 npm run dev
nice -n 19 npm run build && nice -n 19 npm test
```

Vite serves `/pushable/`. The manual-only Pages workflow builds and tests the site; it does not enable Pages or change repository visibility. The initial selection is Go1 trained with shoves (`go1-walk-push-s0`). The selector also offers Go1 trained without shoves (`go1-walk-s0`) and G1 (`g1-walk-s0`). Both Go1 entries use `go1-getup-s0` after a fall. Every entry uses training seed 0.

The initial download includes the selected walking policy and Go1 get-up. The other Go1 walking policy and G1 load on demand. A production build includes all four policy files.

## Simulation

`src/policy.ts` ports [`eval/sim.py`](../eval/sim.py). It uses exported observation block order, sensor names, normalization, float32 network parameters, swish hidden layers, `tanh(mean)` actions, and position targets. G1 gait phase and both action-history buffers reset with each policy. The next observation contains the previous history before the latest action and phase are written. There is no extra `mj_forward` after a control step: sensors retain MuJoCo's final-substep timing.

Layer weights are lossless base64-encoded, little-endian IEEE float32 arrays, decoded once when a policy loads. They are not quantized.

`src/physics.ts` steps at `simulation_dt`, with exactly `n_substeps` per control step. The fixed accumulator runs at wall-clock speed, or one-quarter speed when selected. Catch-up is capped at four control steps per frame; excess elapsed time is discarded, not accumulated. A slow device therefore runs slower instead of blocking the page. Switching tabs clears held steering input.

The renderer constructs mesh geometry from compiled model vertices/faces and primitives from geom sizes, and interpolates between control frames. MuJoCo array views are shared across each transform loop. Robot resources load on demand. Soft shadows use a 512-pixel variance map with four blur samples, and device pixel ratio is capped at 1.5. The diffuse floor has a world-fixed, half-metre grid generated as a small mipmapped texture, so the reference remains visible in light and shadow without a separate transparent line layer. Mipmaps and the scene fog (12–45 m) fade the pattern with distance. The camera follows the base in three dimensions, with faster vertical tracking to keep airborne recoveries in view. Orbit is independent of steering; portrait framing keeps the robot in view.

Like MuJoCo's default viewer, rendering and body picking include geom groups 0–2 only. Groups 3 and above remain in the physics model but create no three.js geometry.

`npm run simplify` regenerates the visual-only STL copies with [meshoptimizer](https://github.com/zeux/meshoptimizer) 1.3.0: target 10% of input triangles, maximum relative simplification error 0.002, `Prune` enabled. [`visual/manifest.json`](visual/manifest.json) records every input/output hash, size, triangle count and returned error. The upstream [Go1](../assets/go1/LICENSE-Menagerie) and [G1](../assets/g1/LICENSE-Menagerie) licenses also cover these derived meshes. Original XML and assets remain unchanged. Only zero-collision-mask meshes outside explicit contact pairs may be reduced. Tests compare inertias, joints, actuators, sensors, collision primitives, and a short zero-action trajectory before and after replacement; parity uses the reduced assets actually served by the browser.

## Input and recovery

W/S or up/down commands forward/backward motion. A/D or left/right commands lateral motion; Q/E turns. The touch joystick provides planar motion, with separate turn buttons. A standard-mapped gamepad uses its left stick for motion and right stick for yaw; buttons 0 and 1 push and reset. The exported command ranges set all limits. Releasing or cancelling a touch clears only that pointer's command, leaving other held touches active. Robot selection stays disabled during loading and returns keyboard focus to the scene afterward.

Dragging a body applies a mass-scaled damped spring at the picked point, including the resulting torque about its centre of mass. Push applies a uniformly random horizontal world-frame impulse to the exported force body over exactly 0.1 simulated seconds. The slider is in N·s; the force is the selected impulse divided by 0.1 s, as in the [evaluation protocol](../docs/evaluation.md). A partial final integration interval is weighted so it does not add extra impulse.

Go1 enters get-up when base height is below 0.20 m or the body/world up-vector dot product is below 0.55, continuously for 0.04 simulated seconds. This short entry guard avoids waiting for deep floor penetration in the feet-only walking model before enabling get-up's body contacts. It returns to walking after 0.5 continuous seconds above 0.24 m, up-vector dot product above 0.92, horizontal speed below 0.6 m/s, and angular speed below 1 rad/s. These are demo switching rules, not the evaluation's success criterion.

The walking model has feet-only floor collisions; the get-up model has different torso/limb/self-collision geometry. Handoff switches the entire compiled model, preserving positions and velocities, and resets action/gait history. It does not splice collision parameters into the walking model. Both Go1 entries share one get-up controller and return to their selected walking policy after recovery. Reset restores that walking policy's exported initial pose.

Recovery is not guaranteed. After eight simulated seconds without returning to walking, the page says “Still down. Reset to start again.” The policy keeps running; it never silently resets or hides a fall.

## Recording

`?record=1` disables the wall-clock animation loop. `window.pushableRecording.step()` advances exactly one complete policy control step and renders once; it uses the same controller, force application and recovery switch as the live demo. `selectPolicy()` selects a walking-policy ID, while the other recording methods set bounded commands and a push direction. Rendering may be slow in headless software graphics; encoded playback still follows simulation time.

With an owned headless Chromium listening for debugging on port 9228 and the built site served with `npm run preview -- --port 5176`, run `nice -n 19 npm run record`. The script captures the actual page, not a separate renderer. Its fixed sequence explicitly selects the without-shoves Go1 (`go1-walk-s0`) at 0.5 m/s, a 20 N·s world +Y push at 1.5 s, then G1 at 7 s, for a 10 s loop. It refuses to publish media unless Go1 actually enters get-up, recovers, and is still walking at the robot switch. It writes MP4, a compact GIF, and provenance under `media/`. Neither encoding nor a successful clip measures real-time browser frame rate.

Recording requires FFmpeg with the `libopenh264` encoder. Full-range screenshot JPEGs are explicitly converted to limited-range YUV420 for H.264 Main, with QP bounds fixed at 26, rate control off and no frame skipping. The 560-pixel-wide GIF uses 10 fps and a 96-colour palette. The two files together must be below 5 MB before the script publishes them.

The committed [MP4](media/hero.mp4) and [GIF](media/hero.gif) show the without-shoves Go1 and G1, not the new initial selection. [Capture provenance](media/hero.json) records the fixed sequence, policy hashes and observed recovery transitions. [Viewport captures](media/screenshots.json) record the earlier build at 1280 × 800 and 390 × 844.

## Parity

`npm test` replays every `results/parity/*.json` using the same WASM engine as the browser. It checks the policy file hash, observation vectors, deterministic network actions, motor targets, and resulting positions/velocities. It prints maximum absolute errors per policy. The test applies the recorded action **after** comparing inference, isolating float32 matrix-multiplication differences from contact divergence. It uses fixture-provided absolute/relative tolerances for observations and actions. Empty or shortened fixtures are errors, not skipped tests.

Performance and learned recovery are only claims after measurement on the actual policies. A successful browser test alone does not establish 60 fps on every laptop or phone.

Written with AI coding assistance.
