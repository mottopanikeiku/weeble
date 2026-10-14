import assert from 'node:assert/strict';
import test from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { FixedClock, Physics, mountFile } from '../src/physics.ts';
import { RecoverySwitch } from '../src/recovery.ts';

const mj = await loadMujoco();
mountFile(mj, 'test/impulse.xml', new TextEncoder().encode(
  '<mujoco><option gravity="0 0 0"/><worldbody><body name="root" pos="0 0 2"><freejoint/><geom type="sphere" size="0.1" mass="2"/></body></worldbody></mujoco>'));

test('single-threaded MuJoCo 3.15 applies the requested horizontal impulse over 0.1 s', () => {
  assert.equal(mj.mj_versionString(), '3.15.0');
  const physics = new Physics(mj, 'test/impulse.xml', 0.003, 0.03, 'root');
  try {
    physics.push(2, 0.7);
    for (let i = 0; i < 4; i++) physics.step();
    assert.ok(Math.abs(physics.data.qvel[0] - Math.cos(0.7)) < 1e-12);
    assert.ok(Math.abs(physics.data.qvel[1] - Math.sin(0.7)) < 1e-12);
    assert.equal(physics.data.qvel[2], 0);
    physics.step();
    assert.ok(Math.abs(Math.hypot(physics.data.qvel[0], physics.data.qvel[1]) - 1) < 1e-12);
    physics.reset();
    assert.equal(physics.data.qvel[0], 0);
    assert.equal(physics.data.time, 0);
  } finally {
    physics.dispose();
  }
});

test('model handoff keeps the remaining impulse and remaps the picked body by name', () => {
  mountFile(mj, 'test/handoff.xml', new TextEncoder().encode(
    '<mujoco><option gravity="0 0 0"/><worldbody><body name="marker" pos="10 0 0"><geom type="sphere" size="0.01"/></body><body name="root" pos="0 0 2"><freejoint/><geom type="sphere" size="0.1" mass="2"/></body></worldbody></mujoco>'));
  const from = new Physics(mj, 'test/impulse.xml', 0.003, 0.03, 'root');
  const to = new Physics(mj, 'test/handoff.xml', 0.002, 0.02, 'root');
  try {
    from.push(2, 0);
    from.step();
    from.step();
    to.reset(from.data.qpos, from.data.qvel);
    to.inheritForces(from);
    to.step();
    to.step();
    assert.ok(Math.abs(to.data.qvel[0] - 1) < 1e-12);
    from.drag = { body: from.body, local: [0, 0, 0], target: [1, 0, 2], previous: [0, 0, 2] };
    to.inheritForces(from);
    assert.notEqual(from.body, to.body);
    assert.equal(to.drag?.body, to.body);
    assert.deepEqual(to.drag?.target, [1, 0, 2]);
    assert.equal(to.drag?.previous, null);
    assert.equal(from.drag, null);
  } finally { from.dispose(); to.dispose(); }
});

test('fixed clock caps catch-up, preserves remainder, and slows simulation not physics', () => {
  const clock = new FixedClock();
  let steps = 0;
  clock.advance(1, 0.02, false, () => steps++);
  assert.equal(steps, 4);
  assert.ok(Math.abs(clock.droppedSeconds - 0.92) < 1e-12);
  clock.reset();
  steps = 0;
  const alpha = clock.advance(0.12, 0.02, true, () => steps++);
  assert.equal(steps, 1);
  assert.ok(Math.abs(alpha - 0.5) < 1e-12);
});

test('Go1 switches only after a fall and returns after a continuous half-second of steady stance', () => {
  const recovery = new RecoverySwitch();
  assert.equal(recovery.update(0.02, 0.19, 1, 0, 0), 'walk');
  assert.equal(recovery.update(0.02, 0.3, 1, 0, 0), 'walk');
  assert.equal(recovery.update(0.02, 0.3, 0.5, 0, 0), 'walk');
  assert.equal(recovery.update(0.02, 0.3, 0.5, 0, 0), 'getup');
  for (let i = 0; i < 24; i++) assert.equal(recovery.update(0.02, 0.3, 1, 0, 0), 'getup');
  assert.equal(recovery.update(0.02, 0.3, 1, 0, 2), 'getup');
  for (let i = 0; i < 24; i++) assert.equal(recovery.update(0.02, 0.3, 1, 0, 0), 'getup');
  assert.equal(recovery.update(0.02, 0.3, 1, 0, 0), 'walk');
  assert.equal(recovery.update(0.02, 0.19, 1, 0, 0), 'walk');
  assert.equal(recovery.update(0.02, 0.19, 1, 0, 0), 'getup');
  recovery.reset();
  assert.equal(recovery.active, 'walk');
});
