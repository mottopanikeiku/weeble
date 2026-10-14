import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { resolve, dirname, basename, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import loadMujoco from '@mujoco/mujoco';
import { Physics, mountFile } from '../src/physics.ts';
import { Policy, Controller, type PolicySpec } from '../src/policy.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const started = performance.now();
const mj = await loadMujoco();
const mounted = new Set<string>();

interface Fixture {
  policy: string;
  policy_sha256: string;
  initial: { qpos: number[]; qvel: number[]; command: number[]; gait_frequency: number | null; phase: number[] | null };
  tolerance: { observation_atol: number; action_atol: number; rtol: number };
  frames: { observation: number[]; action: number[]; motor_targets: number[]; qpos: number[]; qvel: number[]; next_observation: number[] }[];
}

const directory = resolve(root, 'results/parity');
const fixtures = (await readdir(directory)).filter(name => name.endsWith('.json')).sort();
for (const id of ['go1-walk-push-s0', 'go1-walk-s0', 'go1-walk-rough-s0', 'go1-getup-s0', 'g1-walk-s0']) {
  assert.ok(fixtures.includes(`${id}.json`), `Missing demo parity fixture: ${id}`);
}

const reportPath = process.env.PUSHABLE_PARITY_REPORT;
const checks: Record<string, {
  policy: string; policy_sha256: string; fixture: string; backend: 'wasm'; control_steps: number;
  observation_max_abs: number; action_max_abs: number;
}> = {};

after(async () => {
  // Reports are opt-in and only replaced after every fixture passes.
  if (!reportPath || Object.keys(checks).length !== fixtures.length) return;
  const packageInfo = JSON.parse(await readFile(
    new URL('./package.json', import.meta.resolve('@mujoco/mujoco')), 'utf8'));
  const report = {
    checked_at: new Date().toISOString(),
    git_sha: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    git_dirty: !!execFileSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' }).trim(),
    command: process.env.npm_lifecycle_event === 'test' ? 'npm test'
      : ['node', ...process.execArgv, relative(process.cwd(), process.argv[1])].join(' '),
    test_script: process.env.npm_lifecycle_script,
    working_directory: relative(root, process.cwd()) || '.',
    versions: {
      node: process.version,
      npm: execFileSync('npm', ['--version'], { encoding: 'utf8' }).trim(),
      mujoco: mj.mj_versionString(),
      '@mujoco/mujoco': packageInfo.version,
    },
    scope: 'C MuJoCo fixtures replayed in single-threaded WASM; inference is compared before applying recorded actions.',
    checks,
    wall_seconds: (performance.now() - started) / 1000,
  };
  const destination = resolve(reportPath);
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, JSON.stringify(report, null, 2) + '\n');
});

for (const name of fixtures) {
  test(`C MuJoCo → WASM: ${name}`, async () => {
    const fixture = JSON.parse(await readFile(resolve(directory, name), 'utf8')) as Fixture;
    const id = basename(fixture.policy, '.json');
    const policyBytes = await readFile(resolve(root, fixture.policy));
    const policyHash = createHash('sha256').update(policyBytes).digest('hex');
    assert.equal(policyHash, fixture.policy_sha256);
    const spec = JSON.parse(policyBytes.toString()) as PolicySpec;
    const assetDirectory = `assets/${spec.robot}`;
    if (!mounted.has(assetDirectory)) {
      const paths = await readdir(resolve(root, assetDirectory), { recursive: true, withFileTypes: true });
      for (const entry of paths) {
        if (entry.isFile()) {
          const path = resolve(entry.parentPath, entry.name).slice(root.length + 1);
          const source = resolve(root, /\.stl$/i.test(path) ? 'web/visual' : '.', path);
          mountFile(mj, path, new Uint8Array(await readFile(source)));
        }
      }
      mounted.add(assetDirectory);
    }
    const physics = new Physics(mj, spec.model_xml, spec.simulation_dt, spec.control_dt, spec.force_body);
    const max = { max_obs_err: 0, max_action_err: 0, max_ctrl_err: 0, max_qpos_err: 0, max_qvel_err: 0 };
    const mismatches: string[] = [];
    const compare = (actual: ArrayLike<number>, expected: number[], key: keyof typeof max, atol: number, step: number) => {
      assert.equal(actual.length, expected.length);
      for (let i = 0; i < expected.length; i++) {
        const error = Math.abs(actual[i] - expected[i]);
        max[key] = Math.max(max[key], error);
        if ((!Number.isFinite(error) || error > atol + fixture.tolerance.rtol * Math.abs(expected[i])) && mismatches.length < 8) {
          mismatches.push(`${key} step=${step} index=${i}: ${actual[i]} vs ${expected[i]}`);
        }
      }
    };
    try {
      const controller = new Controller(new Policy(spec), physics);
      const initial = fixture.initial;
      controller.reset(initial.qpos, initial.qvel, initial.command, initial.gait_frequency ?? undefined, initial.phase ?? undefined);
      assert.ok(fixture.frames.length >= 190, 'Expected approximately 200 control steps, not a shortened trace');
      for (const [step, frame] of fixture.frames.entries()) {
        const obs = controller.observe();
        compare(obs, frame.observation, 'max_obs_err', fixture.tolerance.observation_atol, step);
        compare(controller.policy.infer(obs), frame.action, 'max_action_err', fixture.tolerance.action_atol, step);
        // Replay the recorded action after checking inference. This separates tiny
        // float32 GEMM differences from exponentially growing contact divergence.
        controller.step(undefined, frame.action);
        compare(physics.data.ctrl, frame.motor_targets, 'max_ctrl_err', 1e-6, step);
        compare(physics.data.qpos, frame.qpos, 'max_qpos_err', 1e-5, step);
        compare(physics.data.qvel, frame.qvel, 'max_qvel_err', 1e-5, step);
        compare(controller.observe(), frame.next_observation, 'max_obs_err', fixture.tolerance.observation_atol, step);
      }
      const qpos = physics.data.qpos.slice(), qvel = physics.data.qvel.slice();
      controller.reset(qpos, qvel);
      assert.deepEqual(Array.from(physics.data.qpos), Array.from(qpos));
      assert.deepEqual(Array.from(physics.data.qvel), Array.from(qvel));
      assert.ok(controller.lastAction.every(value => value === 0));
      assert.ok(controller.observationLastAction.every(value => value === 0));
      if (spec.gait) {
        assert.deepEqual(controller.phase, new Float32Array(spec.gait.initial_phase));
        assert.deepEqual(controller.observationPhase, new Float32Array(spec.gait.initial_phase));
      }
      controller.step(undefined, new Float32Array(spec.action_size).fill(0.25));
      assert.ok(controller.observationLastAction.every(value => value === 0));
      controller.step(undefined, new Float32Array(spec.action_size).fill(-0.5));
      assert.ok(controller.observationLastAction.every(value => value === 0.25));
      assert.ok(controller.lastAction.every(value => value === -0.5));
    } finally {
      console.log(JSON.stringify({ policy: id, ...max }));
      physics.dispose();
    }
    assert.deepEqual(mismatches, [], mismatches.join('\n'));
    if (reportPath) checks[id] = {
      policy: fixture.policy, policy_sha256: policyHash, fixture: `results/parity/${name}`,
      backend: 'wasm', control_steps: fixture.frames.length,
      observation_max_abs: max.max_obs_err, action_max_abs: max.max_action_err,
    };
  });
}
