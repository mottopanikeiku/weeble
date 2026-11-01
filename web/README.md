# Browser runtime

The demo runs exported policies in single-threaded [`@mujoco/mujoco` 3.15.0](https://github.com/google-deepmind/mujoco/tree/3.15.0/wasm), rendered with [three.js](https://threejs.org/). Robot licenses and source models are in [assets](../assets/). The scenery is procedural.

From `web/`:

```sh
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 UV_THREADPOOL_SIZE=2
nice -n 19 npm ci && nice -n 19 npm run dev
nice -n 19 npm run build && nice -n 19 npm test
```

Vite serves `/weeble/`; GitHub Pages deploys changes to the demo. Every policy uses seed 0. Push-trained Go1 loads first; the others load on demand. The loading screen shows download bytes and model-file progress.

## Physics and play

[`policy.ts`](src/policy.ts) ports [`eval/sim.py`](../eval/sim.py), including observation order, normalization, float32 inference and sensor timing.

[`playground.ts`](src/playground.ts) adds physical props and zero-mass robot bumpers. Separate collision channels keep bumpers away from the training floor and each other. Bumpers touch props only, and Reset restores the whole scene. The policies were trained without these obstacles.

The rough patch copies [`rough.xml`](../assets/go1/rough.xml)'s heightfield, dimensions and friction beside the flat start. Rough-trained Go1 starts near its edge, facing across it. This changes only the browser starting pose.

[Tests](test/playground.test.ts) compare robot masses, inertias, joints, actuators and original collision geometry, and check throws, resets, G1 floor support and rendered heights. [Parity tests](test/parity.test.ts) replay every full fixture on the original training scene and start.

Even distant props can change MuJoCo's [global warmstart choice](https://github.com/google-deepmind/mujoco/blob/3.5.0/src/engine/engine_forward.c). The demo uses native `mj_step` with unchanged solver settings, so its trajectories can differ from the training scene.[^open-loop]

`npm run compare:scenes` compares the two scenes for three Go1 walking policies on seven commands: zero and each policy's signed axis limits. Each trial lasts ten seconds; tracking excludes the first two. Props are distant, and the original training ground and starting pose are retained. [Per-command results](results/scene-comparison.json) include errors after falls.

All 21 paired fall outcomes match, with no prop contacts or numerical failures. Maximum per-command MAE differences among trials without falls were 0.0093 m/s forward, 0.0110 m/s lateral and 0.0097 rad/s yaw.

| Policy | Scene | Falls | Forward / lateral / yaw MAE |
|---|---|---:|---|
| Flat | Training | 0/7 | 0.0443 / 0.0365 / 0.0388 |
| Flat | Composed | 0/7 | 0.0419 / 0.0370 / 0.0369 |
| Push-trained | Training | 0/7 | 0.0467 / 0.0352 / 0.0362 |
| Push-trained | Composed | 0/7 | 0.0477 / 0.0338 / 0.0357 |
| Rough | Training | 2/7 | 0.9627 / 0.7182 / 0.4565 |
| Rough | Composed | 2/7 | 0.9740 / 0.7517 / 0.4021 |

Units: m/s, m/s, rad/s. Rough-trained Go1 fell on the ±1.5 m/s forward commands, which dominate its aggregate errors. The [cross-terrain evaluation](../docs/evaluation.md#cross-terrain-walking) uses lower speeds.

## Controls and falls

WASD/arrows move; Q/E turns. Touch and standard gamepads also steer. Drag a robot or prop to pull it. Push delivers the slider's impulse over 0.1 simulated seconds; Throw/T launches a ball and shows its launch momentum. I shows observations, actions and contacts; ¼ speed slows simulation without changing the physics step.

Shove survival percentages come from [`survival.json`](src/data/survival.json), pooled across training seeds at each evaluated impulse, without interpolation. These measurements use flat ground without props. Rough-trained Go1 has no shove result in this dataset.

Go1 switches to its learned get-up policy after a fall. G1 switches to [full collisions](../assets/g1/fallen.xml), preserving state and forces, then holds its pose until Reset. The browser's [fall detection](src/recovery.ts) uses different thresholds from the evaluation. Simplified [visual meshes](visual/manifest.json) leave collision meshes unchanged.

## Rendering and media

The scene has a clear start pad, a side course and layered hills. Rough ground uses height-derived normals, height shading and a feathered border. Light and dark themes follow the OS; pixel ratio and shadows are bounded.

`npm run record` captures `?record=1` through Chromium/CDP and FFmpeg. The 15-second sequence uses seed 0 and a light theme: Go1 trained without shoves walks at 0.5 m/s, takes a 20 N·s +Y shove at 1.5 s, and must recover before Throw at 7 s. G1 starts at 9 s, then rough-trained Go1 at 12 s. [Media files](media/) play back in simulation time.

An earlier version measured 60.1 fps both idle and walking on a Radeon 840M with Chromium ANGLE GL-EGL, with no errors. The [measurement](results/rendering.json) used six-second windows and a 1280×657 captured viewport from a requested 1280×800 window. It predates the terrain refinement; phone performance has not been measured.

[^open-loop]: At `f9a54bd`, a three-second open-loop replay produced maximum root-coordinate differences of 0.788 m (flat), 0.689 m (push), 0.952 m (rough), 0.0189 m (get-up) and 7.4×10⁻¹³ m (G1). Open-loop replay applies recorded actions; the table above lets each policy respond to its own scene.
