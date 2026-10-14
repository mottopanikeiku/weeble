import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, readdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco, { type MjModel } from '@mujoco/mujoco';
import { Physics, mountFile } from '../src/physics.ts';
import { Controller, Policy, type PolicySpec } from '../src/policy.ts';
import { mountPlayScene, Thrower, BALL_MOMENTUM } from '../src/playground.ts';
import { fallenAssets } from '../src/fallen.ts';
import { g1Fallen } from '../src/recovery.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const mj = await loadMujoco();
for (const robot of ['go1', 'g1']) {
  for (const entry of await readdir(resolve(root, `assets/${robot}`), { recursive: true, withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const path = resolve(entry.parentPath, entry.name).slice(root.length + 1);
    mountFile(mj, path, await readFile(resolve(root, /\.stl$/i.test(path) ? 'web/visual' : '.', path)));
  }
}
const fallen = fallenAssets(await readFile(resolve(root, 'assets/g1/fallen.xml'), 'utf8'));
for (const path of fallen.files) mountFile(mj, path, await readFile(resolve(root, path.replace('/assets/full/', '/assets/'))));
mountFile(mj, 'assets/g1/fallen.browser.xml', new TextEncoder().encode(fallen.xml));
const specFor = async (id: string): Promise<PolicySpec> => JSON.parse(await readFile(resolve(root, `policies/${id}.json`), 'utf8'));
function physics(spec: PolicySpec, play: boolean, xml = spec.model_xml): Physics {
  return new Physics(mj, play ? mountPlayScene(mj, xml) : xml, spec.simulation_dt, spec.control_dt, spec.force_body);
}
function assertMechanism(a: MjModel, b: MjModel): void {
  assert.equal(a.nu, b.nu);
  for (const field of [
    'body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat', 'body_parentid',
    'jnt_type', 'jnt_bodyid', 'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_stiffness', 'jnt_qposadr', 'jnt_dofadr',
    'dof_damping', 'dof_armature', 'dof_frictionloss', 'dof_M0',
    'actuator_gainprm', 'actuator_biasprm', 'actuator_dynprm', 'actuator_ctrlrange', 'actuator_forcerange',
    'actuator_trnid', 'actuator_gear', 'actuator_gaintype', 'actuator_biastype', 'actuator_dyntype',
  ]) {
    const left = (a as unknown as Record<string, Float64Array>)[field];
    const right = (b as unknown as Record<string, Float64Array>)[field];
    assert.deepEqual(Array.from(right.subarray(0, left.length)), Array.from(left), field);
  }
}
for (const id of ['go1-walk-s0', 'go1-walk-push-s0', 'go1-getup-s0', 'g1-walk-s0']) {
  test(`${id}: play composition preserves robot masses, inertias, joints, actuators and open-ground motion`, async () => {
    const spec = await specFor(id), a = physics(spec, false), b = physics(spec, true);
    try {
      const ca = new Controller(new Policy(spec), a), cb = new Controller(new Policy(spec), b);
      assertMechanism(a.model, b.model);
      const floor = mj.mj_name2id(b.model, mj.mjtObj.mjOBJ_GEOM.value, spec.floor_geom);
      let bumpers = 0;
      for (let geom = 0; geom < b.model.ngeom; geom++) {
        const name = mj.mj_id2name(b.model, mj.mjtObj.mjOBJ_GEOM.value, geom) ?? '';
        if (!name.startsWith('play_bumper_')) continue;
        bumpers++;
        assert.equal(b.model.geom_contype[geom], 2);
        assert.equal(b.model.geom_conaffinity[geom], 4);
        assert.equal((b.model.geom_contype[geom] & b.model.geom_conaffinity[floor])
          | (b.model.geom_contype[floor] & b.model.geom_conaffinity[geom]), 0, 'bumper/floor pair');
      }
      assert.ok(bumpers > 0);
      for (let step = 0; step < 25; step++) {
        ca.step([0, 0, 0]); cb.step([0, 0, 0]);
        for (let i = 0; i < a.model.nq; i++) assert.ok(Math.abs(a.data.qpos[i] - b.data.qpos[i]) < 1e-10, `${id} qpos ${i}`);
        for (let i = 0; i < a.model.nv; i++) assert.ok(Math.abs(a.data.qvel[i] - b.data.qvel[i]) < 1e-10, `${id} qvel ${i}`);
      }
    } finally { a.dispose(); b.dispose(); }
  });
}

test('throw has the labelled momentum, hits a robot bumper, and reset restores every prop', async () => {
  const spec = await specFor('go1-walk-s0'), p = physics(spec, true);
  try {
    const controller = new Controller(new Policy(spec), p), thrower = new Thrower();
    const initial = p.data.qpos.slice();
    const j = mj.mj_name2id(p.model, mj.mjtObj.mjOBJ_JOINT.value, 'play_ball_0_joint');
    const body = p.model.jnt_bodyid[j], v = p.model.jnt_dofadr[j];
    assert.equal(thrower.launch(p, [0, -1, .35], [0, 0, .35]), BALL_MOMENTUM);
    assert.ok(Math.abs(p.model.body_mass[body] * Math.hypot(...p.data.qvel.subarray(v, v + 3)) - BALL_MOMENTUM) < 1e-12);
    let hit = false;
    for (let step = 0; step < 150; step++) {
      mj.mj_step(p.model, p.data);
      const contacts = p.data.contact;
      for (let k = 0; k < p.data.ncon; k++) {
        const c = contacts.get(k)!;
        const names = [c.geom1, c.geom2].map(g => mj.mj_id2name(p.model, mj.mjtObj.mjOBJ_GEOM.value, g) ?? '');
        const bodies = [p.model.geom_bodyid[c.geom1], p.model.geom_bodyid[c.geom2]];
        if (bodies.includes(body) && names.some(name => name.startsWith('play_bumper_'))) hit = true;
        c.delete();
      }
      contacts.delete();
    }
    assert.ok(hit, 'ball physically contacts robot bumper');
    controller.reset();
    assert.deepEqual(Array.from(p.data.qpos), Array.from(initial));
    assert.ok(p.data.qvel.every((value: number) => value === 0));
  } finally { p.dispose(); }
});

test('G1 fall handoff preserves robot and prop state and supports its body on the floor', async () => {
  const spec = await specFor('g1-walk-s0'), walk = physics(spec, true), down = physics(spec, true, 'assets/g1/fallen.browser.xml');
  try {
    const controller = new Controller(new Policy(spec), walk), passive = new Controller(controller.policy, down);
    const original = physics(spec, false);
    try { assertMechanism(original.model, down.model); } finally { original.dispose(); }
    for (let i = 0; i < 100; i++) controller.step([.5, 0, 0]);
    walk.push(100, Math.PI / 2);
    let switched = false;
    for (let i = 0; i < 200; i++) {
      controller.step([.5, 0, 0]);
      if (!g1Fallen(walk.data.qpos)) continue;
      const q = walk.data.qpos.slice(), v = walk.data.qvel.slice();
      passive.reset(q, v);
      down.inheritForces(walk);
      assert.deepEqual(Array.from(down.data.qpos), Array.from(q));
      assert.deepEqual(Array.from(down.data.qvel), Array.from(v));
      switched = true;
      break;
    }
    assert.ok(switched);
    for (let i = 0; i < 150; i++) down.step();
    assert.ok(down.data.qpos[2] > .03, `pelvis height ${down.data.qpos[2]}`);
    const torso = mj.mj_name2id(down.model, mj.mjtObj.mjOBJ_BODY.value, 'torso_link');
    assert.ok(down.data.xpos[torso * 3 + 2] > .03);
    assert.ok(down.data.qpos.every(Number.isFinite));
  } finally { walk.dispose(); down.dispose(); }
});
