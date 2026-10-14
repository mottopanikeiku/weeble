// Capture actual browser frames at one control step each, then encode simulation time.
import { mkdir, writeFile, rename, rm, stat } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: {
  cdp: { type: 'string', default: 'http://127.0.0.1:9228' },
  url: { type: 'string', default: 'http://127.0.0.1:5176/pushable/' },
} });
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const temporary = resolve(root, 'web/.browser/recording');
const destination = resolve(root, 'web/media');
await mkdir(temporary, { recursive: true });
await mkdir(destination, { recursive: true });
const url = new URL(values.url);
url.searchParams.set('record', '1');
const target = await (await fetch(`${values.cdp}/json/new?about:blank`, { method: 'PUT' })).json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await once(socket, 'open');
let sequence = 0;
const pending = new Map();
const events = new Map();
const errors = [];
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    clearTimeout(request.timer);
    if (message.error) request.reject(new Error(JSON.stringify(message.error)));
    else request.resolve(message.result);
  } else {
    events.get(message.method)?.(message.params);
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
    if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') errors.push(message.params.args);
  }
});
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 30000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
const started = Date.now();
const mp4 = resolve(temporary, 'hero.mp4'), gif = resolve(temporary, 'hero.gif');
let encoder;
let encoderFinished;
try {
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 960, height: 600, deviceScaleFactor: 1, mobile: false });
  await send('Page.bringToFront');
  const loaded = new Promise(resolve => events.set('Page.loadEventFired', resolve));
  await send('Page.navigate', { url: url.href });
  await loaded;
  await evaluate('window.pushableRecording.ready');
  // Warm both models before recording; no loading screen is cut from the footage.
  await evaluate("window.pushableRecording.selectRobot('g1')");
  await evaluate("window.pushableRecording.selectRobot('go1')");
  await evaluate('window.pushableRecording.setCommand([0.5,0,0])');
  let state = await evaluate('window.pushableRecording.step()');
  const fps = Math.round(1 / state.control_dt);
  if (Math.abs(fps * state.control_dt - 1) > 1e-8) throw new Error('Control interval is not an integer recording frame rate');
  const choices = { seconds: 10, fps, width: 960, height: 600, forward_command: 0.5,
    push_time: 1.5, push_impulse_ns: 20, push_angle_rad: Math.PI / 2, g1_time: 7 };
  const encoderArgs = ['-n', '19', 'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
    '-threads', '2', '-f', 'image2pipe', '-framerate', String(fps), '-vcodec', 'mjpeg', '-i', 'pipe:0',
    '-an', '-c:v', 'libopenh264', '-rc_mode', 'off', '-qmin', '26', '-qmax', '26',
    '-profile:v', 'main', '-threads', '2', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', mp4];
  encoder = spawn('nice', encoderArgs, { stdio: ['pipe', 'inherit', 'inherit'] });
  encoderFinished = once(encoder, 'exit');
  let sawGetup = false, recovered = false, previousTask = '';
  const transitions = [];
  for (let frame = 0; frame < choices.seconds * fps; frame++) {
    if (frame === choices.push_time * fps) {
      await evaluate('window.pushableRecording.setCommand([0,0,0])');
      await evaluate(`document.querySelector('#impulse').value='20'; document.querySelector('#impulse').dispatchEvent(new Event('input',{bubbles:true})); window.pushableRecording.push(${choices.push_angle_rad})`);
    }
    if (frame === choices.g1_time * fps) {
      if (!sawGetup || !recovered || state.task !== 'walk') throw new Error('Fixed demonstration push did not produce a sustained learned recovery; no hero is published');
      await evaluate("window.pushableRecording.selectRobot('g1')");
      await evaluate('window.pushableRecording.setCommand([0.5,0,0])');
    }
    if (frame > 0) state = await evaluate('window.pushableRecording.step()');
    if (Math.abs(state.control_dt * fps - 1) > 1e-8) throw new Error('Tasks use different recording rates');
    if (state.robot === 'go1') {
      if (state.task === 'getup') sawGetup = true;
      if (sawGetup && state.task === 'walk') recovered = true;
    }
    const task = `${state.robot}-${state.task}`;
    if (task !== previousTask) { transitions.push({ frame, ...state }); previousTask = task; }
    const screenshot = await send('Page.captureScreenshot', { format: 'jpeg', quality: 90, fromSurface: true });
    if (!encoder.stdin.write(Buffer.from(screenshot.data, 'base64'))) await once(encoder.stdin, 'drain');
  }
  encoder.stdin.end();
  const [code] = await encoderFinished;
  if (code !== 0) throw new Error(`MP4 encoding exited ${code}`);
  if (errors.length) throw new Error(`Browser errors: ${JSON.stringify(errors)}`);
  const gifArgs = ['-n', '19', 'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-threads', '2',
    '-i', mp4, '-filter_complex_threads', '2', '-filter_complex',
    'fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
    '-loop', '0', gif];
  const gifEncoder = spawn('nice', gifArgs, { stdio: 'inherit' });
  const [gifCode] = await once(gifEncoder, 'exit');
  if (gifCode !== 0) throw new Error(`GIF encoding exited ${gifCode}`);
  const sizes = { mp4: (await stat(mp4)).size, gif: (await stat(gif)).size };
  if (sizes.mp4 >= 5_000_000 || sizes.gif >= 5_000_000) throw new Error(`Media exceeds 5 MB: ${JSON.stringify(sizes)}`);
  const policies = {};
  for (const id of ['go1-walk-s0', 'go1-getup-s0', 'g1-walk-s0']) {
    const response = await fetch(new URL(`data/policies/${id}.json`, url));
    if (!response.ok) throw new Error(`Could not read recorded policy ${id}: ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    policies[id] = { sha256: createHash('sha256').update(bytes).digest('hex'), versions: JSON.parse(bytes).versions };
  }
  const manifest = { recorded_at: new Date().toISOString(),
    git_sha: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    dirty: !!execFileSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' }).trim(),
    cadence: 'One control step per frame, encoded in simulation time; capture wall time is not playback time.',
    ffmpeg_version: execFileSync('ffmpeg', ['-version'], { encoding: 'utf8' }).split('\n')[0],
    browser_version: await send('Browser.getVersion'),
    choices, transitions, policies, bytes: sizes, wall_seconds: (Date.now() - started) / 1000,
    ffmpeg_mp4: ['nice', ...encoderArgs.map(arg => arg === mp4 ? 'hero.mp4' : arg)],
    ffmpeg_gif: ['nice', ...gifArgs.map(arg => arg === mp4 ? 'hero.mp4' : arg === gif ? 'hero.gif' : arg)] };
  await rename(mp4, resolve(destination, 'hero.mp4'));
  await rename(gif, resolve(destination, 'hero.gif'));
  await writeFile(resolve(destination, 'hero.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log(JSON.stringify({ media: ['web/media/hero.mp4', 'web/media/hero.gif'], ...sizes, transitions }));
} finally {
  if (encoder && encoder.exitCode === null) { encoder.kill('SIGTERM'); await encoderFinished; }
  socket.close();
  await fetch(`${values.cdp}/json/close/${target.id}`);
  await rm(mp4, { force: true });
  await rm(gif, { force: true });
}
