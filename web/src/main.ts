import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import './style.css';
import { FixedClock, Physics, mountFile } from './physics';
import { Policy, Controller, type PolicySpec } from './policy';
import { RobotView } from './render';
import { Steering, type Command } from './input';
import { RecoverySwitch, g1Fallen } from './recovery';
import { mountPlayScene, Thrower, BALL_MOMENTUM } from './playground';
import { fallenAssets } from './fallen';
import { Inspector } from './inspector';
import survival from './data/survival.json';
import { TERRAIN_Y } from './terrain';

type Robot = 'go1' | 'g1';
type WalkingPolicy = 'go1-walk-push-s0' | 'go1-walk-s0' | 'g1-walk-s0' | 'go1-walk-rough-s0';
interface Manifest { robots: Record<Robot, { files: string[]; policies: string[] }> }
interface SurvivalArm { source: string[]; trials: { impulse: number; survived: number; total: number; seeds: number[] }[] }

declare global {
  interface Window {
    pushableRecording?: {
      ready: Promise<void>;
      selectPolicy(policy: WalkingPolicy): Promise<void>;
      setCommand(command: Command): void;
      push(angle: number): void;
      throwBall(): void;
      reset(): void;
      step(): { time: number; robot: Robot; task: string; control_dt: number; height: number; up: number };
    };
  }
}

const recording = new URLSearchParams(location.search).has('record');
const recordedCommand: Command = [0, 0, 0];
const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = element<HTMLCanvasElement>('scene');
const selector = element<HTMLSelectElement>('robot');
const message = element<HTMLParagraphElement>('message');
const label = element<HTMLElement>('active');
const hint = element<HTMLElement>('hint');
const training = element<HTMLElement>('training');
const loadingPanel = element<HTMLElement>('loading');
const progress = element<HTMLProgressElement>('load-progress');
const pushButton = element<HTMLButtonElement>('push');
const throwButton = element<HTMLButtonElement>('throw');
const resetButton = element<HTMLButtonElement>('reset');
const slowButton = element<HTMLButtonElement>('slow');
const infoButton = element<HTMLButtonElement>('info');
const inspectorPanel = element<HTMLElement>('inspector');
const inspector = new Inspector(inspectorPanel);
const impulse = element<HTMLInputElement>('impulse');
const clock = new FixedClock();
const recovery = new RecoverySwitch();
const thrower = new Thrower();
const cache = new Map<Robot, Map<string, Controller>>();
let active: Controller | null = null;
let walking: Controller | null = null;
let getup: Controller | null = null;
let fallenController: Controller | null = null;
let loading = true, stopped = false, slow = false, fallen = false;
let robot: Robot = 'go1';
let selected: WalkingPolicy = 'go1-walk-push-s0';
let recoveryBegan = 0;
let selectPolicy: ((policy: WalkingPolicy) => Promise<void>) | null = null;

if (matchMedia('(pointer: coarse), (max-width: 600px)').matches) {
  hint.textContent = 'Move with the pad · use the turn buttons · drag to pull';
}
element<HTMLOutputElement>('momentum').value = `${BALL_MOMENTUM.toFixed(1)} N·s at launch`;
function acknowledgeInput(): void { hint.hidden = true; }

function updateSurvival(): void {
  const arms = survival.policies as Record<string, SurvivalArm>;
  const arm = arms[selected] ?? arms[selected.replace(/-s0$/, '')];
  const row = arm?.trials.find(row => row.impulse === Number(impulse.value));
  const output = element<HTMLElement>('survival');
  if (!row) {
    output.removeAttribute('title');
    output.textContent = `Not evaluated at ${impulse.value} N·s${selected.includes('rough') ? ' for this terrain policy' : ''}.`;
    return;
  }
  output.textContent = `In my evaluation, this policy survived a shove this size ${(100 * row.survived / row.total).toFixed(0)}% of the time (${row.total} trials${row.seeds?.length > 1 ? `, ${row.seeds.length} seeds` : ''}).`;
  output.title = `${arm.source} · flat ground, 0.5 m/s command, random horizontal shove · not a prediction among props`;
}

function updatePolicyLabel(): void {
  const spec = active!.policy.spec;
  label.textContent = fallen ? 'fallen · no get-up policy' : `${spec.task === 'getup' ? 'get-up' : 'walk'} · seed ${spec.seed}`;
  training.textContent = fallen ? 'G1 · full-body contacts · Reset to walk again' : spec.task === 'getup'
    ? 'Go1 get-up · trained from fallen poses'
    : selected.includes('rough') ? 'Go1 · trained on rough ground · browser play scene'
    : `${robot === 'g1' ? 'G1' : 'Go1'} · trained on flat ground${selected.includes('push') ? ' with random shoves' : robot === 'go1' ? ' without shoves' : ''}`;
  updateSurvival();
}

function push(angle?: number): void {
  if (!loading && !stopped && active) {
    active.physics.push(Number(impulse.value), angle);
    pushButton.disabled = active.physics.pushing;
  }
  acknowledgeInput();
}
function throwBall(): void {
  if (!loading && !stopped && active) {
    const { origin, target } = view.throwRay();
    thrower.launch(active.physics, origin, target);
  }
  acknowledgeInput();
}
function reset(): void {
  if (loading || !walking) return;
  view.endDrag();
  steering.clear();
  recovery.reset();
  thrower.reset();
  recordedCommand.fill(0);
  recoveryBegan = 0;
  fallen = false;
  active = walking;
  active.reset();
  view.setRobot(active.physics);
  clock.reset();
  stopped = false;
  selector.value = selected;
  pushButton.disabled = throwButton.disabled = resetButton.disabled = false;
  message.hidden = true;
  message.classList.remove('error', 'recovery');
  updatePolicyLabel();
  updateHud(recordedCommand);
  acknowledgeInput();
}
const steering = new Steering(acknowledgeInput, push, reset);
const view = new RobotView(canvas, acknowledgeInput);
pushButton.addEventListener('click', () => push());
throwButton.addEventListener('click', throwBall);
resetButton.addEventListener('click', reset);
slowButton.addEventListener('click', () => {
  slow = !slow;
  slowButton.setAttribute('aria-pressed', String(slow));
  acknowledgeInput();
});
infoButton.addEventListener('click', () => {
  inspectorPanel.hidden = !inspectorPanel.hidden;
  infoButton.setAttribute('aria-pressed', String(!inspectorPanel.hidden));
  if (active) inspector.update(active, fallen);
});
window.addEventListener('keydown', event => {
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey
      || (event.target as HTMLElement).matches('textarea, [contenteditable="true"], input:not([type="range"])')) return;
  if (event.code === 'KeyT') { event.preventDefault(); throwBall(); }
  if (event.code === 'KeyI') { event.preventDefault(); infoButton.click(); }
});
impulse.addEventListener('input', () => {
  element<HTMLOutputElement>('impulse-value').value = impulse.value;
  updateSurvival();
  acknowledgeInput();
});
function showError(error: unknown): void {
  stopped = true;
  loadingPanel.hidden = true;
  message.textContent = error instanceof Error ? error.message : String(error);
  message.classList.remove('recovery');
  message.classList.add('error');
  message.hidden = false;
  label.textContent = 'stopped';
  pushButton.disabled = throwButton.disabled = true;
  resetButton.disabled = !walking;
}
async function fetchData(path: string): Promise<Response> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${path}`);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
  return response;
}
async function boot(): Promise<void> {
  const response = await fetch(wasmUrl);
  if (!response.ok) throw new Error(`Could not load physics (${response.status})`);
  // Fetch streams decoded bytes; compressed Content-Length is not their total.
  const encoding = response.headers.get('content-encoding');
  const length = !encoding || encoding === 'identity' ? Number(response.headers.get('content-length')) : 0;
  const reader = response.body!.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (length) progress.value = Math.min(100, received * 100 / length);
    element<HTMLElement>('load-detail').textContent = `${(received / 1048576).toFixed(1)} MiB · physics`;
  }
  progress.value = 100;
  const binary = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { binary.set(chunk, offset); offset += chunk.length; }
  element<HTMLElement>('loading-label').textContent = 'Starting physics';
  const [mj, manifest] = await Promise.all([
    loadMujoco({ wasmBinary: binary, locateFile: () => wasmUrl }),
    fetchData('manifest.json').then(response => response.json() as Promise<Manifest>),
  ]);
  if (mj.mj_versionString() !== '3.15.0') throw new Error('MuJoCo version differs from policy contract');

  async function loadPolicy(next: WalkingPolicy): Promise<void> {
    const nextRobot: Robot = next === 'g1-walk-s0' ? 'g1' : 'go1';
    const selectedPath = `policies/${next}.json`;
    loading = true;
    selector.disabled = pushButton.disabled = throwButton.disabled = resetButton.disabled = true;
    view.endDrag();
    steering.clear();
    recordedCommand.fill(0);
    loadingPanel.hidden = false;
    progress.value = 0;
    element<HTMLElement>('loading-label').textContent = `Loading ${nextRobot === 'go1' ? 'Go1' : 'G1'}`;
    message.hidden = true;
    label.textContent = 'loading';
    try {
      if (!cache.has(nextRobot)) {
        const files = [...manifest.robots[nextRobot].files];
        const total = files.length;
        let completed = 0;
        await Promise.all([0, 1].map(async () => {
          for (let path = files.pop(); path; path = files.pop()) {
            mountFile(mj, path, new Uint8Array(await (await fetchData(path)).arrayBuffer()));
            completed++;
            progress.value = completed * 100 / total;
            element<HTMLElement>('load-detail').textContent = `${completed} / ${total} model files`;
          }
        }));
        if (nextRobot === 'g1') {
          const original = new TextDecoder().decode(mj.FS.readFile('/assets/g1/fallen.xml'));
          mountFile(mj, 'assets/g1/fallen.browser.xml', new TextEncoder().encode(fallenAssets(original).xml));
        }
        cache.set(nextRobot, new Map());
      }
      const controllers = cache.get(nextRobot)!;
      for (const path of manifest.robots[nextRobot].policies) {
        if (path !== selectedPath && path !== 'policies/go1-getup-s0.json') continue;
        if (controllers.has(path)) continue;
        element<HTMLElement>('load-detail').textContent = 'Loading policy · composing play scene';
        const spec = await (await fetchData(path)).json() as PolicySpec;
        const physics = new Physics(mj, mountPlayScene(mj, spec.model_xml), spec.simulation_dt, spec.control_dt, spec.force_body,
          spec.model_xml.endsWith('/rough.xml') ? TERRAIN_Y : 0);
        try { controllers.set(path, new Controller(new Policy(spec), physics)); }
        catch (error) { physics.dispose(); throw error; }
      }
      const nextWalking = controllers.get(selectedPath);
      if (!nextWalking) throw new Error(`Policy not in the published manifest: ${next}`);
      if (nextRobot === 'g1' && !fallenController) {
        const spec = nextWalking.policy.spec;
        const physics = new Physics(mj, mountPlayScene(mj, 'assets/g1/fallen.browser.xml'), spec.simulation_dt, spec.control_dt, spec.force_body);
        try { fallenController = new Controller(nextWalking.policy, physics); }
        catch (error) { physics.dispose(); throw error; }
      }
      robot = nextRobot;
      selected = next;
      selector.value = next;
      walking = nextWalking;
      getup = controllers.get('policies/go1-getup-s0.json') ?? null;
      loading = false;
      reset();
      loadingPanel.hidden = true;
      pushButton.disabled = throwButton.disabled = resetButton.disabled = false;
    } finally {
      loading = false;
      selector.disabled = false;
    }
  }
  selectPolicy = loadPolicy;
  selector.addEventListener('change', () => {
    acknowledgeInput();
    loadPolicy(selector.value as WalkingPolicy).then(() => canvas.focus({ preventScroll: true })).catch(showError);
  });
  await loadPolicy(selected);
}
function handoff(next: Controller, command: Command): void {
  const previous = active!;
  const { data } = previous.physics;
  const time = data.time;
  next.reset(data.qpos, data.qvel, command);
  next.physics.data.time = time;
  next.physics.inheritForces(previous.physics);
  active = next;
  view.setRobot(active.physics, false);
  recoveryBegan = time;
  updatePolicyLabel();
}
function controlStep(command: Command): void {
  view.capture();
  if (fallen) active!.physics.step();
  else active!.step(command);
  if (robot === 'go1') {
    const { data } = active!.physics;
    const up = 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2);
    const task = recovery.update(active!.policy.spec.control_dt, data.qpos[2], up,
      Math.hypot(data.qvel[0], data.qvel[1]), Math.hypot(data.qvel[3], data.qvel[4], data.qvel[5]));
    if (task !== active!.policy.spec.task) {
      handoff(task === 'walk' ? walking! : getup!, command);
      message.hidden = true;
      message.classList.remove('recovery');
    }
    if (task === 'getup' && active!.physics.data.time - recoveryBegan >= 8 && message.hidden) {
      message.textContent = 'Still down. Reset to start again.';
      message.classList.add('recovery');
      message.hidden = false;
    }
  } else if (!fallen && g1Fallen(active!.physics.data.qpos)) {
    fallen = true;
    handoff(fallenController!, command);
    message.textContent = 'G1 has no get-up policy. Reset to walk again.';
    message.classList.add('recovery');
    message.hidden = false;
  }
  pushButton.disabled = active!.physics.pushing;
  if (!active!.physics.data.qpos.every(Number.isFinite)) throw new Error('Non-finite simulation state. Reset to restart.');
}
function updateHud(command: Command): void {
  element<HTMLOutputElement>('command-speed').value = (fallen ? 0 : Math.hypot(command[0], command[1])).toFixed(2);
  element<HTMLOutputElement>('measured-speed').value = Math.hypot(active!.physics.data.qvel[0], active!.physics.data.qvel[1]).toFixed(2);
  inspector.update(active!, fallen);
}
let previousTime = performance.now(), previousHud = 0;
function frame(now: number): void {
  const elapsed = (now - previousTime) / 1000;
  previousTime = now;
  let alpha = 1;
  if (active && !loading && !stopped && !document.hidden) {
    try {
      const command = steering.sample(walking!.policy.spec.command_ranges);
      alpha = clock.advance(elapsed, active.policy.spec.control_dt, slow, () => controlStep(command));
      if (now - previousHud >= 100) { updateHud(command); previousHud = now; }
    } catch (error) { showError(error); }
  } else clock.reset();
  view.render(alpha);
  requestAnimationFrame(frame);
}
if (!recording) requestAnimationFrame(frame);
const ready = boot();
ready.catch(showError);
if (recording) {
  window.pushableRecording = {
    ready,
    async selectPolicy(next) { await ready; await selectPolicy!(next); },
    setCommand(command) {
      for (let i = 0; i < 3; i++) recordedCommand[i] = command[i];
      acknowledgeInput();
    },
    push, throwBall, reset,
    step() {
      if (!active || loading || stopped) throw new Error('Recording requires a loaded, running policy');
      const limits = walking!.policy.spec.command_ranges;
      for (let i = 0; i < 3; i++) recordedCommand[i] = Math.max(limits[i][0], Math.min(limits[i][1], recordedCommand[i]));
      controlStep(recordedCommand);
      updateHud(recordedCommand);
      view.render(1);
      const { data } = active.physics;
      return { time: data.time, robot, task: fallen ? 'fallen' : active.policy.spec.task,
        control_dt: active.policy.spec.control_dt, height: data.qpos[2], up: 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2) };
    },
  };
}
