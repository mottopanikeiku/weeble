# Browser runtime

I run exported policies in single-threaded [`@mujoco/mujoco` 3.15.0](https://github.com/google-deepmind/mujoco/tree/3.15.0/wasm), with [three.js](https://threejs.org/) rendering. Robot licenses and provenance remain in [assets](../assets/). Scenery is procedural.

```sh
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 UV_THREADPOOL_SIZE=2
nice -n 19 npm ci && nice -n 19 npm run dev
nice -n 19 npm run build && nice -n 19 npm test
```

Vite serves `/pushable/`; Pages deploys relevant changes. Every demo uses seed 0, chosen before results. Push-trained Go1 starts first; other policies load on demand. Loading reports download bytes and model-file progress.

## Physics and play

[`policy.ts`](src/policy.ts) ports [`eval/sim.py`](../eval/sim.py), including observation order, normalization, float32 inference and sensor timing.

[`playground.ts`](src/playground.ts) adds browser-only physical props and zero-mass robot bumpers. Collision channels separate training contacts, bumpers, props and their floor. Bumpers touch props, never the training floor or each other. Reset restores everything. No policy learned these obstacles.

The patch copies [`rough.xml`](../assets/go1/rough.xml)'s exact heightfield, dimensions and friction, translated beyond the flat start. Rough Go1 starts near its edge facing across it; this changes only the browser starting pose. Flat-trained policies may stumble there.

[Tests](test/playground.test.ts) compare robot masses, inertias, joints, actuators and original collision geometry; check physical throws, resets, G1 support and rendered heights. [Parity](test/parity.test.ts) checks every full fixture on the original training scene and start.

Disconnected props can change MuJoCo's global warmstart choice ([implementation](https://github.com/google-deepmind/mujoco/blob/3.5.0/src/engine/engine_forward.c#L605-L678)). I use native `mj_step`, not solver modifications. Model invariance does not imply trajectory identity.[^open-loop]

`npm run compare:scenes` measures three Go1 walking policies on seven commands: zero and each policy's signed axis limits. Each trial lasts ten seconds; the first two are discarded. Props are distant; training ground and starting pose are retained. [Per-command tables/provenance](results/scene-comparison.json) include postfall errors.

**MEAS:** all 21 paired fall outcomes match, with no prop contacts or numerical failures. Maximum nonfallen per-command MAE differences: 0.0093 m/s forward, 0.0110 m/s lateral, 0.0097 rad/s yaw. This is agreement on this subset, not general equivalence.

| Policy | Scene | Falls | Forward / lateral / yaw MAE |
|---|---|---:|---|
| Flat | Training | 0/7 | 0.0443 / 0.0365 / 0.0388 |
| Flat | Composed | 0/7 | 0.0419 / 0.0370 / 0.0369 |
| Push-trained | Training | 0/7 | 0.0467 / 0.0352 / 0.0362 |
| Push-trained | Composed | 0/7 | 0.0477 / 0.0338 / 0.0357 |
| Rough | Training | 2/7 | 0.9627 / 0.7182 / 0.4565 |
| Rough | Composed | 2/7 | 0.9740 / 0.7517 / 0.4021 |

Units: m/s, m/s, rad/s. Rough's fallen ±1.5 m/s trials dominate aggregate errors. The research cross-terrain grid uses lower speeds.

## Controls and falls

WASD/arrows move; Q/E turns. Touch and standard gamepads also steer. Drag pulls physically. Push delivers its impulse over 0.1 simulated seconds; Throw/T launches a ballistic ball. Its label is launch momentum, not delivered impulse. I shows observations, actions and contacts; ¼ speed slows simulation.

Survival percentages read generated [`survival.json`](src/data/survival.json): exact evaluated impulses, counts and seeds, without interpolation. These are flat-ground results, not predictions among props.

Go1 switches to learned get-up after falls. G1 switches to [full collisions](../assets/g1/fallen.xml), preserving state and forces, holds its pose, and asks for Reset: **no G1 get-up policy**. [Fall heuristics](src/recovery.ts) are not evaluation criteria. Simplified [visual meshes](visual/manifest.json) never replace fallen collision meshes.

## Fixed bugs

- G1 sank: feet-only contacts persisted. Transfer to full collisions.
- Focused sliders swallowed steering: letters now steer; arrows retain native behavior.
- Landscape controls clipped: remove minimum height and compact the layout.
- Keyboard turn buttons failed: add held Enter/Space and release/blur handling.
- Failed loading left stale selection/actions: Reset restores the loaded policy.
- Browser shortcuts steered: preserve Ctrl/Cmd/Alt combinations.
- Repeated Push truncated pulses: lock until delivery finishes.
- Compressed loading reached 100% early: show decoded bytes without a false denominator.
- Dark active controls lacked contrast: use dark ink on pale green.
- Prop coordinates entered actuators: copy only robot joints.
- Unevaluated policies kept stale tooltips: clear provenance when no row exists.
- Missing WebGL hung at loading: catch graphics startup; show requirements and README instead.

## Rendering and media

The clear start pad and side course sit before layered hills. Matte rough ground uses height-derived normals, subtle height shading, low sunlight and a feathered border, never exaggerated geometry. Light/dark follows the OS; pixel ratio and shadows are bounded.

`npm run record` captures `?record=1` through Chromium/CDP and FFmpeg. **Predeclared:** without-shoves Go1 at 0.5 m/s; 20 N·s +Y shove at 1.5 s; require recovery before Throw at 7 s; G1 at 9 s; rough Go1 at 12 s; end at 15 s. Light theme, fixed seed. [Media/provenance](media/) records simulation-time playback, not real-time performance.

**MEAS, lead:** Radeon 840M, Chromium ANGLE GL-EGL: **60.1 fps idle and walking**, no errors. [Record](results/rendering.json): six-second windows; requested window 1280×800, captured viewport 1280×657. This predates terrain refinement. Physical-phone FPS is unmeasured.

[^open-loop]: MEAS, initial layout `5123569`: three-second open-loop replay produced maximum root-coordinate differences of 0.788 m (flat), 0.689 m (push), 0.952 m (rough), 0.0189 m (get-up), 7.4×10⁻¹³ m (G1). This unstable replay is not closed-loop walking performance.

Written with AI coding assistance.
