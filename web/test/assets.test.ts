import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, readdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import loadMujoco, { type MjModel } from '@mujoco/mujoco';
import { mountFile } from '../src/physics.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const mj = await loadMujoco();
const report = JSON.parse(await readFile(resolve(root, 'web/visual/manifest.json'), 'utf8')) as {
  models: Record<string, string>;
  meshes: Record<string, { original_sha256: string; reduced_sha256: string }>;
};

for (const robot of ['go1', 'g1']) {
  test(`${robot} visual reduction leaves dynamics and collision geometry unchanged`, async () => {
    const entries = await readdir(resolve(root, `assets/${robot}`), { recursive: true, withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const path = resolve(entry.parentPath, entry.name).slice(root.length + 1);
      const bytes = await readFile(resolve(root, path));
      if (/\.stl$/i.test(path)) assert.equal(createHash('sha256').update(bytes).digest('hex'), report.meshes[path].original_sha256);
      // The manifest covers visual-only training models. Fallen collision meshes
      // use originals in a separate namespace, exercised in playground.test.ts.
      if (report.models[path]) assert.equal(createHash('sha256').update(bytes).digest('hex'), report.models[path]);
      mountFile(mj, path, bytes);
    }
    const tasks = robot === 'go1' ? ['walk', 'getup'] : ['walk'];
    const originals = tasks.map(task => mj.MjModel.from_xml_path(`/assets/${robot}/${task}.xml`));
    const reducedModels: MjModel[] = [];
    try {
      for (const [path, hashes] of Object.entries(report.meshes)) {
        if (!path.startsWith(`assets/${robot}/`)) continue;
        const bytes = await readFile(resolve(root, 'web/visual', path));
        assert.equal(createHash('sha256').update(bytes).digest('hex'), hashes.reduced_sha256);
        mountFile(mj, path, bytes);
      }
      for (const [index, task] of tasks.entries()) {
        const original = originals[index];
        const reduced = mj.MjModel.from_xml_path(`/assets/${robot}/${task}.xml`);
        reducedModels.push(reduced);
        for (const field of ['nq', 'nv', 'nu', 'nbody', 'njnt', 'ngeom', 'npair', 'nsensor'] as const) {
          assert.equal(reduced[field], original[field], `${task}: ${field}`);
        }
        for (const field of [
          'body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat', 'body_parentid',
          'body_invweight0', 'jnt_type', 'jnt_bodyid', 'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_stiffness',
          'dof_damping', 'dof_armature', 'dof_frictionloss', 'dof_M0', 'qpos0',
          'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange', 'actuator_trnid',
          'pair_geom1', 'pair_geom2', 'exclude_signature', 'sensor_type', 'sensor_objid', 'sensor_adr',
          'site_pos', 'site_quat', 'geom_contype', 'geom_conaffinity',
        ]) {
          const a = (original as unknown as Record<string, ArrayLike<number>>)[field];
          const b = (reduced as unknown as Record<string, ArrayLike<number>>)[field];
          assert.deepEqual(Array.from(b), Array.from(a), `${task}: ${field}`);
        }
        for (let geom = 0; geom < original.ngeom; geom++) {
          if (original.geom_type[geom] === 7) {
            assert.equal(original.geom_contype[geom], 0);
            assert.equal(original.geom_conaffinity[geom], 0);
            for (let p = 0; p < original.npair; p++) {
              assert.notEqual(original.pair_geom1[p], geom);
              assert.notEqual(original.pair_geom2[p], geom);
            }
          } else {
            for (const [field, width] of [['geom_size', 3], ['geom_pos', 3], ['geom_quat', 4], ['geom_friction', 3], ['geom_solref', 2], ['geom_solimp', 5]] as const) {
              assert.deepEqual(Array.from(reduced[field].subarray(geom * width, (geom + 1) * width)),
                Array.from(original[field].subarray(geom * width, (geom + 1) * width)), `${task}: ${field}[${geom}]`);
            }
          }
        }
        const a = new mj.MjData(original), b = new mj.MjData(reduced);
        try {
          mj.mj_resetDataKeyframe(original, a, 0);
          mj.mj_resetDataKeyframe(reduced, b, 0);
          for (let step = 0; step < 100; step++) {
            mj.mj_step(original, a);
            mj.mj_step(reduced, b);
          }
          assert.deepEqual(Array.from(b.qpos), Array.from(a.qpos), `${task}: zero-action trajectory`);
          assert.deepEqual(Array.from(b.qvel), Array.from(a.qvel), `${task}: zero-action velocity`);
        } finally { a.delete(); b.delete(); }
      }
    } finally {
      for (const model of [...originals, ...reducedModels]) model.delete();
    }
  });
}
