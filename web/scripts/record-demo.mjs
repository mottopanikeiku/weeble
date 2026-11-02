// Records docs/shove.gif: the flat-ground Go1 takes a 30 N·s shove, falls, gets up and walks on.
// Frames come from a real browser (one every 5 control steps) and play back in simulation time.
//
//   chrome --headless=new --remote-debugging-port=9228 --use-angle=swiftshader --enable-unsafe-swiftshader
//   npm run dev &
//   npm run record -- --cdp http://127.0.0.1:9228 --url http://127.0.0.1:5173/weeble/
//
// Needs ffmpeg on PATH.
import { mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: {
  cdp: { type: 'string', default: 'http://127.0.0.1:9228' },
  url: { type: 'string', default: 'https://mottopanikeiku.github.io/weeble/' },
} });
const destination = resolve(dirname(fileURLToPath(import.meta.url)), '../../docs');
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

const skip = 15;      // control steps before the first frame
const stride = 5;     // control steps per frame
const frames = 40;
const gif = resolve(destination, 'shove.gif');
const filter = 'scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];'
  + '[b][p]paletteuse=dither=none:diff_mode=rectangle';
let encoder, finished;
try {
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 960, height: 600, deviceScaleFactor: 1, mobile: false });
  await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] });
  await send('Page.bringToFront');
  const loaded = new Promise(resolve => events.set('Page.loadEventFired', resolve));
  await send('Page.navigate', { url: url.href });
  await loaded;
  await evaluate('window.weebleRecording.ready');
  // Load every model first so the footage has no loading screen.
  await evaluate("window.weebleRecording.selectPolicy('go1-walk-s0')");
  await evaluate('window.weebleRecording.reset()');
  await evaluate('window.weebleRecording.setCommand([0.5,0,0])');
  let state = await evaluate('window.weebleRecording.step()');
  const fps = Math.round(1 / state.control_dt);
  encoder = spawn('nice', ['-n', '19', 'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-threads', '2',
    '-f', 'image2pipe', '-framerate', String(fps / stride), '-vcodec', 'mjpeg', '-i', 'pipe:0',
    '-filter_complex', filter, '-loop', '0', gif], { stdio: ['pipe', 'inherit', 'inherit'] });
  finished = once(encoder, 'exit');
  let sawGetup = false, walkingAgain = -1, captured = 0;
  for (let frame = 0; captured < frames; frame++) {
    if (frame === fps) {
      await evaluate('window.weebleRecording.setCommand([0,0,0])');
      await evaluate("(() => { const slider = document.querySelector('#impulse'); slider.value = '30';"
        + " slider.dispatchEvent(new Event('input', { bubbles: true })); window.weebleRecording.push(Math.PI / 2); })()");
    }
    if (frame > 0) state = await evaluate('window.weebleRecording.step()');
    if (state.task === 'getup') sawGetup = true;
    if (sawGetup && walkingAgain < 0 && state.task === 'walk') walkingAgain = frame;
    if (walkingAgain >= 0 && frame === walkingAgain + 40) await evaluate('window.weebleRecording.setCommand([0.5,0,0])');
    if (frame >= skip && (frame - skip) % stride === 0) {
      const shot = await send('Page.captureScreenshot', { format: 'jpeg', quality: 92, fromSurface: true });
      if (!encoder.stdin.write(Buffer.from(shot.data, 'base64'))) await once(encoder.stdin, 'drain');
      captured++;
    }
    if (frame > 20 * fps) throw new Error('The dog never recovered from the shove');
  }
  if (!sawGetup) throw new Error('The shove did not knock the dog over');
  encoder.stdin.end();
  const [code] = await finished;
  if (code !== 0) throw new Error(`ffmpeg exited ${code}`);
  if (errors.length) throw new Error(`Browser errors: ${JSON.stringify(errors)}`);
  console.log(`Wrote ${gif}`);
} finally {
  if (encoder && encoder.exitCode === null) { encoder.kill('SIGTERM'); await finished; }
  socket.close();
  await fetch(`${values.cdp}/json/close/${target.id}`);
}
