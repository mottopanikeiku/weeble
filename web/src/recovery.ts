export type Go1Task = 'walk' | 'getup';

/** Hysteresis is in simulated seconds, including during quarter-speed playback. */
export class RecoverySwitch {
  active: Go1Task = 'walk';
  private fallenFor = 0;
  private steadyFor = 0;

  reset(): void {
    this.active = 'walk';
    this.fallenFor = this.steadyFor = 0;
  }

  update(dt: number, height: number, up: number, speed: number, angularSpeed: number): Go1Task {
    if (this.active === 'walk') {
      this.fallenFor = height < 0.2 || up < 0.55 ? this.fallenFor + dt : 0;
      if (this.fallenFor + 1e-9 >= 0.04) {
        this.active = 'getup';
        this.steadyFor = 0;
      }
    } else {
      const steady = height > 0.24 && up > 0.92 && speed < 0.6 && angularSpeed < 1;
      this.steadyFor = steady ? this.steadyFor + dt : 0;
      if (this.steadyFor + 1e-9 >= 0.5) {
        this.active = 'walk';
        this.fallenFor = 0;
      }
    }
    return this.active;
  }
}

/** Switch before the feet-only G1 can put its torso through the floor. */
export function g1Fallen(qpos: ArrayLike<number>): boolean {
  const up = 1 - 2 * (qpos[4] * qpos[4] + qpos[5] * qpos[5]);
  return qpos[2] < 0.55 || up < 0.65;
}
