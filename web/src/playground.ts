import type { MainModule, MjModel } from '@mujoco/mujoco';
import { mountFile, type Physics, type Vec3 } from './physics';
import { addTerrain } from './terrain';

// Separate collision channels: training floor/feet 1, robot bumpers 2,
// play objects 4, and a props-only floor 8. Nothing changes the training XML.
const types = ['', '', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box'];
export const BALL_MASS = 0.6;
export const BALL_SPEED = 8;
export const BALL_MOMENTUM = BALL_MASS * BALL_SPEED;
const playGeom = 'contype="4" conaffinity="14" condim="3" friction="0.65 0.01 0.001"';
const ballNames = ['play_ball_0', 'play_ball_1'];

function props(): string {
  const crate = (name: string, pos: string) => `<body name="${name}" pos="${pos}"><freejoint/><geom ${playGeom} type="box" size="0.17 0.17 0.17" mass="1.2" rgba="0.48 0.37 0.23 1"/></body>`;
  const cone = (name: string, pos: string) => `<body name="${name}" pos="${pos}"><freejoint/><geom ${playGeom} type="cylinder" size=".105 .005" mass=".12" rgba=".16 .18 .16 1"/><geom ${playGeom} type="mesh" mesh="play_cone_shape" pos="0 0 .005" mass=".18" rgba=".84 .34 .10 1"/><geom type="mesh" mesh="play_cone_stripe" pos="0 0 .005" mass="0" contype="0" conaffinity="0" group="2" rgba=".94 .94 .84 1"/></body>`;
  const ring = (radius: number, z: number) => Array.from({ length: 16 }, (_, i) =>
    `${radius * Math.cos(i * Math.PI / 8)} ${radius * Math.sin(i * Math.PI / 8)} ${z}`).join(' ');
  return `<asset><mesh name="play_cone_shape" vertex="${ring(.085, 0)} 0 0 .24"/>
    <mesh name="play_cone_stripe" vertex="${ring(.085 * (1 - .10 / .24) + .0005, .10)} ${ring(.085 * (1 - .15 / .24) + .0005, .15)}"/></asset><worldbody>
    <geom name="play_floor" type="plane" size="0 0 0.01" contype="8" conaffinity="4" group="3" friction="0.6" condim="3"/>
    ${crate('play_crate_0', '1.3 1.6 0.17')}${crate('play_crate_1', '1.3 1.6 0.51')}
    ${cone('play_cone_0', '-0.9 1.3 .005')}${cone('play_cone_1', '-0.2 1.65 .005')}${cone('play_cone_2', '0.5 1.3 .005')}
    ${ballNames.map((name, i) => `<body name="${name}" pos="${-1.45 + i * .4} ${1.2 + i * .45} .12"><freejoint name="${name}_joint"/><geom ${playGeom} type="sphere" size=".12" mass="${BALL_MASS}" rgba=".30 .47 .49 1"/></body>`).join('\n')}
    <body name="play_step" pos="2.15 1.4 .045"><geom ${playGeom} type="box" size=".45 .3 .045" rgba=".50 .53 .46 1"/></body>
  </worldbody>`;
}

/** Clone existing primitive collision outlines, never the visual triangle meshes. */
export function composeScene(mj: MainModule, xml: string, model: MjModel): string {
  if (/<include\b/.test(xml)) throw new Error('Play scenes require self-contained exported XML');
  const bumpers = new Map<string, string>();
  for (let id = 0; id < model.ngeom; id++) {
    const body = model.geom_bodyid[id], kind = model.geom_type[id];
    if (body === 0 || model.geom_group[id] < 2 || !types[kind]) continue;
    const name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body);
    const size = Array.from(model.geom_size.subarray(id * 3, id * 3 + 3)).join(' ');
    const pos = Array.from(model.geom_pos.subarray(id * 3, id * 3 + 3)).join(' ');
    const quat = Array.from(model.geom_quat.subarray(id * 4, id * 4 + 4)).join(' ');
    const geom = `<geom name="play_bumper_${id}" type="${types[kind]}" size="${size}" pos="${pos}" quat="${quat}" mass="0" contype="2" conaffinity="4" group="3" condim="3"/>`;
    bumpers.set(name, (bumpers.get(name) ?? '') + geom);
  }
  // G1's torso/pelvis outlines are meshes. Use simple play-only boxes there,
  // rather than paying for triangle-mesh contacts or changing their floor masks.
  for (const [name, size, pos] of [
    ['pelvis', '.10 .095 .075', '0 0 -.055'], ['torso_link', '.12 .10 .20', '0 0 .16'],
  ]) {
    if (mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, name) < 0) continue;
    bumpers.set(name, (bumpers.get(name) ?? '') + `<geom name="play_bumper_${name}" type="box" size="${size}" pos="${pos}" mass="0" contype="2" conaffinity="4" group="3" condim="3"/>`);
  }
  const composed = xml.replace(/<body\b[^>]*\bname="([^"]+)"[^>]*>/g,
    (tag, name: string) => tag + (bumpers.get(name) ?? ''));
  // Exported initial poses are used by Controller.reset. Training keyframes have
  // no coordinates for the additional free bodies and do not belong to this scene.
  return composed.replace(/<keyframe\b[^>]*>[\s\S]*?<\/keyframe>/g, '')
    .replace('</mujoco>', `${props()}</mujoco>`);
}

export function mountPlayScene(mj: MainModule, path: string): string {
  const model = mj.MjModel.from_xml_path(`/${path}`);
  try {
    const xml = new TextDecoder().decode(mj.FS.readFile(`/${path}`));
    const destination = path.replace(/\.xml$/, '.play.xml');
    const rough = new TextDecoder().decode(mj.FS.readFile('/assets/go1/rough.xml'));
    const scene = addTerrain(composeScene(mj, xml, model), rough, model.nhfield > 0);
    mountFile(mj, destination, new TextEncoder().encode(scene));
    return destination;
  } finally { model.delete(); }
}

export class Thrower {
  private next = 0;
  reset(): void { this.next = 0; }

  launch(physics: Physics, origin: Vec3, target: Vec3): number {
    const { mj, model, data } = physics;
    const joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, `${ballNames[this.next]}_joint`);
    this.next = (this.next + 1) % ballNames.length;
    if (joint < 0) throw new Error('Missing play ball');
    const q = model.jnt_qposadr[joint], v = model.jnt_dofadr[joint];
    const dx = target[0] - origin[0], dy = target[1] - origin[1], dz = target[2] - origin[2];
    const distance = Math.hypot(dx, dy, dz);
    const options = model.opt, gravity = -options.gravity[2];
    options.delete();
    const horizontal = Math.hypot(dx, dy), speed2 = BALL_SPEED ** 2;
    const discriminant = speed2 ** 2 - gravity * (gravity * horizontal ** 2 + 2 * dz * speed2);
    // Low ballistic arc at the labelled speed. Beyond its physical range, aim
    // directly; the ball is still simulated, not guided toward a moving robot.
    const angle = horizontal > 1e-6 && gravity > 0 && discriminant >= 0
      ? Math.atan((speed2 - Math.sqrt(discriminant)) / (gravity * horizontal)) : Math.atan2(dz, horizontal);
    const planarSpeed = BALL_SPEED * Math.cos(angle);
    for (let i = 0; i < 3; i++) {
      data.qpos[q + i] = origin[i];
      data.qvel[v + i] = i === 2 ? BALL_SPEED * Math.sin(angle)
        : (target[i] - origin[i]) * planarSpeed / (horizontal || distance);
      data.qvel[v + 3 + i] = 0;
    }
    data.qpos.set([1, 0, 0, 0], q + 3);
    // Do not forward here: policy sensors retain their trained substep timing.
    return BALL_MOMENTUM;
  }
}
