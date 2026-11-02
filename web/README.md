# Browser demo

Exported policies run in single-threaded [`@mujoco/mujoco`](https://github.com/google-deepmind/mujoco/tree/3.15.0/wasm) 3.15.0 and are drawn with [three.js](https://threejs.org/) in a procedural scene, while the robot models and their licenses are in [`assets/`](../assets/).

```bash
npm ci
npm run dev                      # http://127.0.0.1:5173/weeble/
npm run build && npm test
```

`npm test` replays the C MuJoCo fixtures in `results/parity/` through the WASM build and checks the demo scene, and GitHub Pages builds, tests and deploys on every push to `main` that touches the demo. Every policy in the demo is training seed 0, and the push-trained Go1 loads first while the rest load on demand.

## How it works

[`policy.ts`](src/policy.ts) ports [`eval/sim.py`](../eval/sim.py): observation order, normalization, float32 inference and sensor timing.

[`playground.ts`](src/playground.ts) adds the props and zero-mass robot bumpers, which the policies never trained with. Separate collision channels keep the bumpers off the floor and off each other, so they touch props only.

The rough patch copies the heightfield, size and friction of [`rough.xml`](../assets/go1/rough.xml) next to the flat start, and the rough-trained dog starts near its edge, facing across it.

Go1 switches to its get-up policy after a fall ([`recovery.ts`](src/recovery.ts)) and back to walking once it stands steadily. G1 instead switches to the [full-collision model](../assets/g1/fallen.xml), keeps its state and holds the pose until Reset. The simplified [visual meshes](visual/manifest.json) leave the collision meshes untouched.

Shove percentages come from [`survival.json`](src/data/survival.json), which pools flat-ground trials without props over training seeds and has no entry for the rough-trained dog.

## Props and the physics

`npm run compare:scenes` runs three Go1 policies for 10 s on 7 commands (zero and each axis limit), once on the training scene and once in the demo scene. All 21 pairs agree on whether the dog fell, with no prop contacts or numerical failures, and tracking error on runs without falls differs by at most 0.011 m/s or rad/s.

| Policy | Scene | Falls | Forward / lateral / yaw error |
|---|---|---:|---|
| Flat | Training | 0/7 | 0.044 / 0.037 / 0.039 |
| Flat | Demo | 0/7 | 0.042 / 0.037 / 0.037 |
| Push-trained | Training | 0/7 | 0.047 / 0.035 / 0.036 |
| Push-trained | Demo | 0/7 | 0.048 / 0.034 / 0.036 |
| Rough | Training | 2/7 | 0.96 / 0.72 / 0.46 |
| Rough | Demo | 2/7 | 0.97 / 0.75 / 0.40 |

Errors, with raw numbers in [`scene-comparison.json`](results/scene-comparison.json), are in m/s, m/s and rad/s and include time after falls. The rough-trained dog fell on the ±1.5 m/s forward commands, which dominate its numbers, whereas the [terrain comparison](../docs/evaluation.md#cross-terrain-walking) uses gentler speeds.

## Controls

WASD or the arrow keys move the robot and Q and E turn it, while touch and gamepads can steer too. Dragging a robot or prop pulls it, Push delivers the slider's impulse over 0.1 simulated seconds, and Throw (T) fires a ball and shows its launch momentum. I shows observations, actions and contacts, and ¼ speed slows time without changing the physics step.

## Recording the GIF

`npm run record` drives Chromium over CDP (frames from `?record=1`, one per five control steps) and encodes `docs/shove.gif` with ffmpeg. The comments at the top of [`scripts/record-demo.mjs`](scripts/record-demo.mjs) have the exact commands.
