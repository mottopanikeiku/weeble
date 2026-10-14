import assert from 'node:assert/strict';
import test from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { FixedClock, Physics, mountFile } from '../src/physics.ts';
import { RecoverySwitch } from '../src/recovery.ts';

const mj = await loadMujoco();

test('single-threaded MuJoCo 3.15 applies the requested horizontal impulse over 0.1 s', () => {
  assert.equal(mj.mj_versionString(), '3.15.0');
  mountFile(mj, 'test/impulse.xml', new TextEncoder().encode(
    '<mujoco><option gravity="0 0 0"/><worldbody><body name="root" pos="0 0 2"><freejoint/><geom type="sphere" size="0.1" mass="2"/></body></worldbody></mujoco>'));
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
  for (let i = 0; i < 5; i++) assert.equal(recovery.update(0.02, 0.1, 0, 0, 0), 'walk');
  assert.equal(recovery.update(0.02, 0.1, 0, 0, 0), 'getup');
  for (let i = 0; i < 24; i++) assert.equal(recovery.update(0.02, 0.3, 1, 0, 0), 'getup');
  assert.equal(recovery.update(0.02, 0.3, 1, 0, 2), 'getup');
  for (let i = 0; i < 24; i++) assert.equal(recovery.update(0.02, 0.3, 1, 0, 0), 'getup');
  assert.equal(recovery.update(0.02, 0.3, 1, 0, 0), 'walk');
  recovery.update(0.12, 0.1, 0, 0, 0);
  recovery.reset();
  assert.equal(recovery.active, 'walk');
});
