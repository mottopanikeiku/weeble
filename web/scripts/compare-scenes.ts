import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import { Physics, mountFile } from '../src/physics.ts';
import { Controller, Policy, type PolicySpec, type Command } from '../src/policy.ts';
import { mountPlayScene } from '../src/playground.ts';
import { TERRAIN_Y } from '../src/terrain.ts';

// Predeclared subset of docs/evaluation.md's 27-command grid: zero and each
// signed axis endpoint. All three Go1 walking seed-0 policies, no selection.
const policies = ['go1-walk-s0', 'go1-walk-push-s0', 'go1-walk-rough-s0'];
const web = resolve(dirname(fileURLToPath(import.meta.url)), '..'), root = resolve(web, '..');
const mj = await loadMujoco();
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
for (const entry of await readdir(resolve(root, 'assets/go1'), { recursive: true, withFileTypes: true })) {
  if (!entry.isFile()) continue;
  const path = resolve(entry.parentPath, entry.name).slice(root.length + 1);
  mountFile(mj, path, await readFile(resolve(root, /\.stl$/i.test(path) ? 'web/visual' : '.', path)));
}
const rows: object[] = [], summaries: object[] = [], hashes: Record<string, string> = {};
for (const path of ['src/physics.ts', 'src/policy.ts', 'src/playground.ts', 'src/terrain.ts', 'scripts/compare-scenes.ts']) {
  hashes[`web/${path}`] = hash(await readFile(resolve(web, path)));
}
for (const id of policies) {
  const bytes = await readFile(resolve(root, `policies/${id}.json`));
  hashes[`policies/${id}.json`] = hash(bytes);
  const spec: PolicySpec = JSON.parse(bytes.toString()), rough = spec.model_xml.endsWith('/rough.xml');
  hashes[spec.model_xml] = hash(await readFile(resolve(root, spec.model_xml)));
  let scene = new TextDecoder().decode(mj.FS.readFile('/' + mountPlayScene(mj, spec.model_xml)));
  scene = scene.replace(/<body\b[^>]*\bname="play_(?:crate|cone|ball|step)[^"]*"[^>]*>/g, tag =>
    tag.replace(/pos="([^"]+)"/, (_, position: string) => {
      const xyz = position.split(/\s+/).map(Number); xyz[0] += 100;
      return `pos="${xyz.join(' ')}"`;
    }));
  // Compare identical training ground, not the browser's extra terrain/borders.
  if (rough) scene = scene.replace(/<geom\b[^>]*\bname="play_outer_[^"]*"[^>]*\/>/g, '');
  else scene = scene.replace(/<geom\b[^>]*\bname="play_(?:prop_)?terrain"[^>]*\/>/g,
    tag => tag.replace(/pos="[^"]*"/, 'pos="1000 12 0"'));
  const composed = spec.model_xml.replace('.xml', '.comparison.xml');
  mountFile(mj, composed, new TextEncoder().encode(scene));
  hashes[composed] = hash(new TextEncoder().encode(scene));
  const commands: Command[] = [[0, 0, 0]];
  for (let axis = 0; axis < 3; axis++) for (const endpoint of spec.command_ranges[axis]) {
    const command: Command = [0, 0, 0]; command[axis] = endpoint; commands.push(command);
  }
  for (const mode of ['training', 'composed'] as const) {
    const p = new Physics(mj, mode === 'training' ? spec.model_xml : composed,
      spec.simulation_dt, spec.control_dt, spec.force_body, mode === 'composed' && rough ? TERRAIN_Y : 0);
    const controller = new Controller(new Policy(spec), p);
    const floor = mj.mj_name2id(p.model, mj.mjtObj.mjOBJ_GEOM.value, spec.floor_geom);
    const feet = new Set(spec.feet_geoms.map(name => mj.mj_name2id(p.model, mj.mjtObj.mjOBJ_GEOM.value, name)));
    const warningVector = p.data.warning;
    const warnings = Array.from({ length: warningVector.size() }, (_, i) => warningVector.get(i)!);
    const buffer = new mj.DoubleBuffer(6), velocity: Float64Array = buffer.GetView();
    let fallen = false, propContacts = 0;
    // No applied forces in this experiment. Native integration is unchanged;
    // inspecting every substep does not forward or mutate the simulation.
    p.step = () => {
      for (let substep = 0; substep < p.substeps; substep++) {
        mj.mj_step(p.model, p.data);
        const q = p.data.qpos, norm = q[3] ** 2 + q[4] ** 2 + q[5] ** 2 + q[6] ** 2;
        if (1 - 2 * (q[4] ** 2 + q[5] ** 2) / norm < .5) fallen = true;
        const contacts = p.data.contact;
        try {
          for (let j = 0; j < p.data.ncon; j++) {
            const c = contacts.get(j)!;
            try {
              if (c.dist > 0) continue;
              const [a, b] = c.geom;
              const other = a === floor ? b : b === floor ? a : -1;
              if (other >= 0 && p.model.geom_bodyid[other] !== 0 && !feet.has(other)) fallen = true;
              if ((p.model.geom_contype[a] === 4 && p.model.geom_contype[b] === 2)
                || (p.model.geom_contype[b] === 4 && p.model.geom_contype[a] === 2)) propContacts++;
            } finally { c.delete(); }
          }
        } finally { contacts.delete(); }
      }
    };
    const totals = [0, 0, 0]; let falls = 0, failures = 0;
    try {
      for (const command of commands) {
        controller.reset(undefined, undefined, command); fallen = false; propContacts = 0;
        const sums = [0, 0, 0]; let samples = 0, error: string | null = null;
        for (let step = 0; step < Math.round(10 / spec.control_dt); step++) {
          controller.step(command);
          if ([p.data.qpos, p.data.qvel, p.data.qacc, p.data.sensordata, p.data.ctrl].some(a => !a.every(Number.isFinite))) {
            error = 'nonfinite_state'; break;
          }
          if (warnings.some(warning => warning.number > 0)) { error = 'mujoco_warning'; break; }
          if ((step + 1) * spec.control_dt <= 2) continue;
          mj.mj_objectVelocity(p.model, p.data, mj.mjtObj.mjOBJ_XBODY.value, p.body, velocity, 1);
          sums[0] += Math.abs(velocity[3] - command[0]);
          sums[1] += Math.abs(velocity[4] - command[1]);
          sums[2] += Math.abs(velocity[2] - command[2]);
          samples++;
        }
        assert.equal(propContacts, 0, 'comparison must remain physically disconnected');
        const mae = sums.map(value => samples ? value / samples : null);
        rows.push({ policy: id, scene: mode, training_model: spec.model_xml, command_ranges: spec.command_ranges,
          command, fallen, error, samples, robot_prop_contacts: propContacts,
          forward_mae: mae[0], lateral_mae: mae[1], yaw_mae: mae[2] });
        if (fallen) falls++; if (error) failures++;
        for (let i = 0; i < 3; i++) totals[i] += mae[i] ?? NaN;
      }
      const summary = { policy: id, scene: mode, training_model: spec.model_xml, command_ranges: spec.command_ranges,
        trials: commands.length, falls, numerical_failures: failures, forward_mae: totals[0] / commands.length,
        lateral_mae: totals[1] / commands.length, yaw_mae: totals[2] / commands.length };
      summaries.push(summary); console.log(JSON.stringify(summary));
    } finally { buffer.delete(); p.dispose(); }
  }
}
const report = { measurement: 'MEAS', runtime: `Node ${process.version}, MuJoCo ${mj.mj_versionString()} WASM`,
  source_commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(), hashes,
  protocol: { commands: 'zero and signed axis endpoints: 7 of the 27 tracking commands, fixed before running',
    episode_seconds: 10, discard_seconds: 2, feedback: 'each policy infers from its own scene at every control step',
    fall: 'tilt >60 degrees or nonfoot floor contact at any physics substep; continue tracking after falls',
    velocity_frame: 'mjOBJ_XBODY, local=1', props: 'translated +100m in x; original training ground retained',
    solver: 'native mj_step; no warmstart, option, robot or policy changes', seed: 0 }, summaries, rows };
await mkdir(resolve(web, 'results'), { recursive: true });
await writeFile(resolve(web, 'results/scene-comparison.json'), JSON.stringify(report, null, 2) + '\n');
console.log('Wrote web/results/scene-comparison.json');
