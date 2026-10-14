import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import './style.css';
import { FixedClock, Physics, mountFile } from './physics';
import { Policy, Controller, type PolicySpec } from './policy';
import { RobotView } from './render';
import { Steering } from './input';
import { RecoverySwitch } from './recovery';

type Robot = 'go1' | 'g1';
interface Manifest { robots: Record<Robot, { files: string[]; policies: string[] }> }

const canvas = document.querySelector<HTMLCanvasElement>('#scene')!;
const selector = document.querySelector<HTMLSelectElement>('#robot')!;
const message = document.querySelector<HTMLParagraphElement>('#message')!;
const label = document.querySelector<HTMLElement>('#active')!;
const hint = document.querySelector<HTMLElement>('#hint')!;
if (matchMedia('(pointer: coarse), (max-width: 600px)').matches) {
  hint.textContent = 'Move with the pad · use the turn buttons · drag the robot to pull';
}
const pushButton = document.querySelector<HTMLButtonElement>('#push')!;
const resetButton = document.querySelector<HTMLButtonElement>('#reset')!;
const slowButton = document.querySelector<HTMLButtonElement>('#slow')!;
const impulse = document.querySelector<HTMLInputElement>('#impulse')!;
const impulseValue = document.querySelector<HTMLOutputElement>('#impulse-value')!;
const commandSpeed = document.querySelector<HTMLOutputElement>('#command-speed')!;
const measuredSpeed = document.querySelector<HTMLOutputElement>('#measured-speed')!;
const clock = new FixedClock();
const recovery = new RecoverySwitch();
const cache = new Map<Robot, Controller[]>();
let active: Controller | null = null;
let loading = true;
let stopped = false;
let slow = false;
let robot: Robot = 'go1';

function acknowledgeInput(): void { hint.hidden = true; }

function push(): void {
  if (!loading && active) active.physics.push(Number(impulse.value));
  acknowledgeInput();
}

function reset(): void {
  if (loading) return;
  view.endDrag();
  steering.clear();
  recovery.reset();
  active = cache.get(robot)!.find(controller => controller.policy.spec.task === 'walk')!;
  active.reset();
  view.setRobot(active.physics);
  clock.reset();
  stopped = false;
  message.hidden = true;
  message.classList.remove('error');
  label.textContent = 'walk · seed 0';
  acknowledgeInput();
}

const steering = new Steering(acknowledgeInput, push, reset);
const view = new RobotView(canvas, acknowledgeInput);
pushButton.addEventListener('click', push);
resetButton.addEventListener('click', reset);
slowButton.addEventListener('click', () => {
  slow = !slow;
  slowButton.setAttribute('aria-pressed', String(slow));
  acknowledgeInput();
});
impulse.addEventListener('input', () => {
  impulseValue.value = impulse.value;
  acknowledgeInput();
});

function showError(error: unknown): void {
  stopped = true;
  message.textContent = error instanceof Error ? error.message : String(error);
  message.classList.add('error');
  message.hidden = false;
  label.textContent = 'stopped';
}

async function fetchData(path: string): Promise<Response> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${path}`);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
  return response;
}

async function boot(): Promise<void> {
  const [mj, manifest] = await Promise.all([
    loadMujoco({ locateFile: () => wasmUrl }),
    fetchData('manifest.json').then(response => response.json() as Promise<Manifest>),
  ]);
  if (mj.mj_versionString() !== '3.15.0') throw new Error('MuJoCo version differs from policy contract');

  async function loadRobot(next: Robot): Promise<void> {
    loading = true;
    selector.disabled = pushButton.disabled = resetButton.disabled = true;
    view.endDrag();
    steering.clear();
    message.hidden = false;
    message.classList.remove('error');
    message.textContent = `Loading ${next === 'go1' ? 'Go1' : 'G1'}`;
    label.textContent = 'loading';
    try {
      if (!cache.has(next)) {
        const entry = manifest.robots[next];
        // Two simultaneous downloads keep loading modest on mobile connections.
        const files = [...entry.files];
        await Promise.all([0, 1].map(async () => {
          for (let path = files.pop(); path; path = files.pop()) {
            const bytes = new Uint8Array(await (await fetchData(path)).arrayBuffer());
            mountFile(mj, path, bytes);
          }
        }));
        const controllers: Controller[] = [];
        try {
          for (const path of entry.policies) {
            const spec = await (await fetchData(path)).json() as PolicySpec;
            const physics = new Physics(mj, spec.model_xml, spec.simulation_dt, spec.control_dt, spec.force_body);
            try { controllers.push(new Controller(new Policy(spec), physics)); }
            catch (error) { physics.dispose(); throw error; }
          }
        } catch (error) {
          for (const controller of controllers) controller.physics.dispose();
          throw error;
        }
        cache.set(next, controllers);
      }
      robot = next;
      recovery.reset();
      active = cache.get(next)!.find(controller => controller.policy.spec.task === 'walk')!;
      active.reset();
      view.setRobot(active.physics);
      clock.reset();
      stopped = false;
      label.textContent = 'walk · seed 0';
      message.hidden = true;
      pushButton.disabled = resetButton.disabled = false;
    } finally {
      loading = false;
      selector.disabled = false;
    }
  }

  selector.addEventListener('change', () => {
    acknowledgeInput();
    loadRobot(selector.value as Robot).then(() => canvas.focus({ preventScroll: true })).catch(showError);
  });
  await loadRobot('go1');
}

let previousTime = performance.now();
let previousHud = 0;
function frame(now: number): void {
  const elapsed = (now - previousTime) / 1000;
  previousTime = now;
  let alpha = 1;
  if (active && !loading && !stopped && !document.hidden) {
    try {
      const walk = cache.get(robot)!.find(controller => controller.policy.spec.task === 'walk')!;
      const command = steering.sample(walk.policy.spec.command_ranges);
      alpha = clock.advance(elapsed, active.policy.spec.control_dt, slow, () => {
        view.capture();
        active!.step(command);
        if (robot === 'go1') {
          const { data, body } = active!.physics;
          const task = recovery.update(active!.policy.spec.control_dt, data.qpos[2], data.xmat[body * 9 + 8],
            Math.hypot(data.qvel[0], data.qvel[1]), Math.hypot(data.qvel[3], data.qvel[4], data.qvel[5]));
          if (task !== active!.policy.spec.task) {
            const next = cache.get(robot)!.find(controller => controller.policy.spec.task === task)!;
            const time = data.time;
            next.reset(data.qpos, data.qvel, command);
            next.physics.data.time = time;
            next.physics.inheritForces(active!.physics);
            active = next;
            view.setRobot(active.physics, false);
            label.textContent = `${task === 'getup' ? 'get-up' : 'walk'} · seed 0`;
          }
        }
      });
      if (!Number.isFinite(active.physics.data.qpos[2])) throw new Error('Non-finite simulation state. Reset to restart.');
      if (now - previousHud >= 100) {
        commandSpeed.value = Math.hypot(command[0], command[1]).toFixed(2);
        measuredSpeed.value = Math.hypot(active.physics.data.qvel[0], active.physics.data.qvel[1]).toFixed(2);
        previousHud = now;
      }
    } catch (error) { showError(error); }
  } else clock.reset();
  view.render(alpha);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
boot().catch(showError);
