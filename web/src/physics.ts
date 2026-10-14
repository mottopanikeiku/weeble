import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

export type Vec3 = [number, number, number];

/** Files retain their repository-relative paths, including XML includes. */
export function mountFile(mj: MainModule, path: string, bytes: Uint8Array): void {
  const fullPath = `/${path}`;
  mj.FS.mkdirTree(fullPath.slice(0, fullPath.lastIndexOf('/')), 0o755);
  mj.FS.writeFile(fullPath, bytes);
}

export class FixedClock {
  private pending = 0;
  droppedSeconds = 0;

  reset(): void { this.pending = 0; }

  advance(elapsed: number, dt: number, slow: boolean, step: () => void): number {
    this.pending += Math.max(0, elapsed) * (slow ? 0.25 : 1);
    // Never replay a suspended tab or let a slow device build an unbounded queue.
    const budget = 4 * dt;
    if (this.pending > budget) {
      this.droppedSeconds += this.pending - budget;
      this.pending = budget;
    }
    while (this.pending + 1e-12 >= dt) {
      step();
      this.pending -= dt;
    }
    return Math.max(0, this.pending / dt);
  }
}

export class Physics {
  readonly model: MjModel;
  readonly data: MjData;
  readonly body: number;
  readonly substeps: number;
  private pushRemaining = 0;
  private readonly pushForce: Vec3 = [0, 0, 0];
  drag: { body: number; local: Vec3; target: Vec3; previous: Vec3 | null } | null = null;

  constructor(readonly mj: MainModule, xml: string, readonly simDt: number,
              readonly ctrlDt: number, bodyName: string, readonly spawnY = 0) {
    this.model = mj.MjModel.from_xml_path(`/${xml}`);
    this.data = new mj.MjData(this.model);
    this.body = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_BODY.value, bodyName);
    this.substeps = Math.round(ctrlDt / simDt);
    if (this.body < 1 || Math.abs(this.substeps * simDt - ctrlDt) > 1e-9) {
      this.dispose();
      throw new Error('Invalid root body or nonintegral control interval');
    }
    const options = this.model.opt;
    options.timestep = simDt;
    options.delete();
    this.reset();
  }

  reset(qpos?: ArrayLike<number>, qvel?: ArrayLike<number>): void {
    this.mj.mj_resetData(this.model, this.data);
    if (qpos) this.data.qpos.set(qpos);
    if (qvel) this.data.qvel.set(qvel);
    this.data.xfrc_applied.fill(0);
    this.pushRemaining = 0;
    this.drag = null;
    this.mj.mj_forward(this.model, this.data);
  }

  get pushing(): boolean { return this.pushRemaining > 1e-12; }

  push(impulse: number, angle = Math.random() * 2 * Math.PI): void {
    if (this.pushing) return;
    this.pushRemaining = 0.1;
    this.pushForce[0] = Math.cos(angle) * impulse / 0.1;
    this.pushForce[1] = Math.sin(angle) * impulse / 0.1;
  }

  inheritForces(source: Physics): void {
    this.pushRemaining = source.pushRemaining;
    for (let i = 0; i < 3; i++) this.pushForce[i] = source.pushForce[i];
    if (source.drag) {
      const name = this.mj.mj_id2name(source.model, this.mj.mjtObj.mjOBJ_BODY.value, source.drag.body);
      const body = this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_BODY.value, name);
      this.drag = body > 0 ? { ...source.drag, body, previous: null } : null;
      source.drag = null;
    }
  }

  point(body: number, local: Vec3, out: Vec3): Vec3 {
    const x = this.data.xpos, r = this.data.xmat;
    for (let i = 0; i < 3; i++) {
      out[i] = x[3 * body + i] + r[9 * body + 3 * i] * local[0]
        + r[9 * body + 3 * i + 1] * local[1] + r[9 * body + 3 * i + 2] * local[2];
    }
    return out;
  }

  private applyForces(): void {
    const forces = this.data.xfrc_applied;
    forces.fill(0);
    if (this.pushRemaining > 0) {
      const fraction = Math.min(this.pushRemaining, this.simDt) / this.simDt;
      for (let i = 0; i < 3; i++) forces[6 * this.body + i] = this.pushForce[i] * fraction;
      this.pushRemaining = Math.max(0, this.pushRemaining - this.simDt);
    }
    if (!this.drag) return;
    const { body, local, target, previous } = this.drag;
    const point = this.point(body, local, [0, 0, 0]);
    const mass = Math.max(0.2, this.model.body_subtreemass[body]);
    const stiffness = mass * 120;
    const damping = 0.6 * Math.sqrt(stiffness * mass);
    const force: Vec3 = [0, 0, 0];
    const lever: Vec3 = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      const velocity = previous ? (point[i] - previous[i]) / this.simDt : 0;
      force[i] = stiffness * (target[i] - point[i]) - damping * velocity;
      lever[i] = point[i] - this.data.xipos[3 * body + i];
    }
    const scale = Math.min(1, mass * 60 / (Math.hypot(...force) || 1));
    for (let i = 0; i < 3; i++) {
      force[i] *= scale;
      forces[6 * body + i] += force[i];
    }
    forces[6 * body + 3] += lever[1] * force[2] - lever[2] * force[1];
    forces[6 * body + 4] += lever[2] * force[0] - lever[0] * force[2];
    forces[6 * body + 5] += lever[0] * force[1] - lever[1] * force[0];
    this.drag.previous = point;
  }

  step(): void {
    for (let i = 0; i < this.substeps; i++) {
      this.applyForces();
      this.mj.mj_step(this.model, this.data);
    }
  }

  dispose(): void {
    this.data.delete();
    this.model.delete();
  }
}
