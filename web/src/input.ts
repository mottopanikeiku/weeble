export type Command = [number, number, number];
export type CommandLimits = [readonly [number, number], readonly [number, number], readonly [number, number]];
const bindings: Record<string, [number, number]> = {
  KeyW: [0, 1], ArrowUp: [0, 1], KeyS: [0, -1], ArrowDown: [0, -1],
  KeyA: [1, 1], ArrowLeft: [1, 1], KeyD: [1, -1], ArrowRight: [1, -1],
  KeyQ: [2, 1], KeyE: [2, -1],
};
const deadzone = (x: number) => Math.abs(x) < 0.15 ? 0 : Math.sign(x) * (Math.abs(x) - 0.15) / 0.85;

export class Steering {
  private readonly keys = new Set<string>();
  private readonly touch: Command = [0, 0, 0];
  private stickPointer: number | null = null;
  private gamepadPush = false;
  private gamepadReset = false;
  private readonly turnPointers = new Map<number, number>();
  readonly command: Command = [0, 0, 0];

  constructor(private readonly onInput: () => void, private readonly push: () => void,
              private readonly reset: () => void) {
    window.addEventListener('keydown', event => {
      if ((event.target as HTMLElement).matches('input, select')) return;
      if (bindings[event.code]) {
        this.keys.add(event.code);
        event.preventDefault();
        this.onInput();
      }
    });
    window.addEventListener('keyup', event => this.keys.delete(event.code));
    window.addEventListener('blur', () => this.clear());
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.clear(); });
    const joystick = document.querySelector<HTMLElement>('#joystick')!;
    const stick = document.querySelector<HTMLElement>('#stick')!;
    const move = (event: PointerEvent) => {
      if (event.pointerId !== this.stickPointer) return;
      const bounds = joystick.getBoundingClientRect();
      let x = (event.clientX - bounds.left - bounds.width / 2) / 36;
      let y = (event.clientY - bounds.top - bounds.height / 2) / 36;
      const scale = Math.max(1, Math.hypot(x, y));
      x /= scale; y /= scale;
      this.touch[0] = -y; this.touch[1] = -x;
      stick.style.transform = `translate(${x * 31}px, ${y * 31}px)`;
    };
    joystick.addEventListener('pointerdown', event => {
      if (this.stickPointer !== null) return;
      this.stickPointer = event.pointerId;
      joystick.setPointerCapture(event.pointerId);
      this.onInput();
      move(event);
    });
    joystick.addEventListener('pointermove', move);
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      joystick.addEventListener(name, () => {
        this.stickPointer = null;
        this.touch[0] = this.touch[1] = 0;
        stick.style.transform = '';
      });
    }
    for (const [id, direction] of [['turn-left', 1], ['turn-right', -1]] as const) {
      const button = document.getElementById(id)!;
      button.addEventListener('pointerdown', event => {
        button.setPointerCapture(event.pointerId);
        this.turnPointers.set(event.pointerId, direction);
        this.onInput();
      });
      for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        button.addEventListener(name, event => this.turnPointers.delete((event as PointerEvent).pointerId));
      }
    }
  }

  clear(): void {
    this.keys.clear();
    this.touch.fill(0);
    this.turnPointers.clear();
    this.command.fill(0);
    this.stickPointer = null;
    document.querySelector<HTMLElement>('#stick')!.style.transform = '';
  }

  sample(limits: CommandLimits): Command {
    this.command.fill(0);
    for (const key of this.keys) {
      const [axis, direction] = bindings[key];
      this.command[axis] += direction;
    }
    for (let i = 0; i < 3; i++) this.command[i] += this.touch[i];
    for (const direction of this.turnPointers.values()) this.command[2] += direction;
    const pad = Array.from(navigator.getGamepads?.() ?? []).find(p => p?.connected && p.mapping === 'standard');
    if (pad && document.hasFocus()) {
      this.command[0] -= deadzone(pad.axes[1] ?? 0);
      this.command[1] -= deadzone(pad.axes[0] ?? 0);
      this.command[2] -= deadzone(pad.axes[2] ?? 0);
      if (pad.buttons[0]?.pressed && !this.gamepadPush) this.push();
      if (pad.buttons[1]?.pressed && !this.gamepadReset) this.reset();
      this.gamepadPush = !!pad.buttons[0]?.pressed;
      this.gamepadReset = !!pad.buttons[1]?.pressed;
      if (this.command.some(x => x !== 0)) this.onInput();
    } else {
      this.gamepadPush = this.gamepadReset = false;
    }
    for (let i = 0; i < 3; i++) {
      const normalized = Math.min(1, Math.max(-1, this.command[i]));
      this.command[i] = normalized >= 0 ? normalized * limits[i][1] : -normalized * limits[i][0];
    }
    return this.command;
  }
}
