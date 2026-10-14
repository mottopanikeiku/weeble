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
      this.fallenFor = height < 0.16 || up < 0.45 ? this.fallenFor + dt : 0;
      if (this.fallenFor + 1e-9 >= 0.12) {
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
