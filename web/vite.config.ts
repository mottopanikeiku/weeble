import { defineConfig, type Plugin } from 'vite';
import { readdir, readFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { dirname, resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const base = '/pushable/';

async function filesIn(directory: string): Promise<string[]> {
  const entries = await readdir(resolve(root, directory), { withFileTypes: true });
  const paths = await Promise.all(entries.map(entry => entry.isDirectory()
    ? filesIn(`${directory}/${entry.name}`) : [`${directory}/${entry.name}`]));
  return paths.flat().sort();
}

async function manifest() {
  const robots = {} as Record<string, { files: string[]; policies: string[] }>;
  for (const robot of ['go1', 'g1']) {
    const selected = (robot === 'go1' ? ['go1-walk-push-s0', 'go1-walk-s0', 'go1-getup-s0'] : ['g1-walk-s0'])
      .map(id => `policies/${id}.json`);
    robots[robot] = {
      files: (await filesIn(`assets/${robot}`)).filter(path => !path.endsWith('.md')),
      policies: selected,
    };
  }
  return { robots, visual_meshes: 'visual-meshes.json' };
}

function simulationData(): Plugin {
  return {
    name: 'simulation-data',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
        if (!pathname.startsWith(`${base}data/`)) return next();
        const path = decodeURIComponent(pathname.slice(`${base}data/`.length));
        try {
          if (path === 'manifest.json') {
            response.setHeader('Content-Type', 'application/json');
            response.end(JSON.stringify(await manifest()));
          } else if (path === 'visual-meshes.json') {
            response.setHeader('Content-Type', 'application/json');
            response.end(await readFile(resolve(root, 'web/visual/manifest.json')));
          } else {
            const fullPath = resolve(root, path);
            const safe = relative(root, fullPath).split(sep);
            if (!['assets', 'policies'].includes(safe[0]) || safe.includes('..')) {
              response.writeHead(404).end();
              return;
            }
            response.setHeader('Content-Type', path.endsWith('.json') ? 'application/json' : 'application/octet-stream');
            const stream = createReadStream(/\.stl$/i.test(path) ? resolve(root, 'web/visual', path) : fullPath);
            stream.on('error', () => { if (!response.headersSent) response.writeHead(404); response.end(); });
            stream.pipe(response);
          }
        } catch (error) {
          response.writeHead(503).end(String(error));
        }
      });
    },
    async generateBundle() {
      const data = await manifest();
      this.emitFile({ type: 'asset', fileName: 'data/manifest.json', source: JSON.stringify(data) });
      this.emitFile({ type: 'asset', fileName: 'data/visual-meshes.json',
        source: await readFile(resolve(root, 'web/visual/manifest.json')) });
      for (const { files, policies } of Object.values(data.robots)) {
        for (const path of [...files, ...policies]) {
          const source = await readFile(resolve(root, /\.stl$/i.test(path) ? 'web/visual' : '.', path));
          this.emitFile({ type: 'asset', fileName: `data/${path}`, source });
        }
      }
    },
  };
}

export default defineConfig({
  base,
  plugins: [simulationData()],
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 800 },
});
