// Offline visual-only reduction. Original MJCF and collision primitives are untouched.
import { readFile, writeFile, readdir, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import loadMujoco from '@mujoco/mujoco';
import { MeshoptSimplifier } from 'meshoptimizer';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const output = resolve(root, 'web/visual');
const settings = { tool: 'meshoptimizer', version: '1.3.0', target_fraction: 0.1, target_error: 0.002, flags: ['Prune'] };
const mj = await loadMujoco();
await MeshoptSimplifier.ready;
const report = { settings, meshes: {} };
for (const robot of ['go1', 'g1']) {
  const directory = `assets/${robot}`;
  const entries = await readdir(resolve(root, directory), { recursive: true, withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const source = resolve(entry.parentPath, entry.name);
    const path = `/${source.slice(root.length + 1)}`;
    mj.FS.mkdirTree(dirname(path), 0o755);
    mj.FS.writeFile(path, new Uint8Array(await readFile(source)));
  }
  for (const task of robot === 'go1' ? ['walk', 'getup'] : ['walk']) {
    const model = mj.MjModel.from_xml_path(`/${directory}/${task}.xml`);
    try {
      for (let i = 0; i < model.ngeom; i++) {
        if (model.geom_type[i] !== mj.mjtGeom.mjGEOM_MESH.value) continue;
        if (model.geom_contype[i] || model.geom_conaffinity[i]) throw new Error(`Mesh participates in collisions: ${robot}/${task} geom ${i}`);
        for (let pair = 0; pair < model.npair; pair++) {
          if (model.pair_geom1[pair] === i || model.pair_geom2[pair] === i) throw new Error('Mesh participates in an explicit contact pair');
        }
      }
    } finally { model.delete(); }
  }
  for (const name of (await readdir(resolve(root, directory, 'assets'))).sort()) {
    if (!/\.stl$/i.test(name)) continue;
    const path = `${directory}/assets/${name}`;
    const source = await readFile(resolve(root, path));
    const triangles = source.readUInt32LE(80);
    if (source.length !== 84 + triangles * 50) throw new Error(`Expected binary STL: ${path}`);
    const vertices = [], index = new Uint32Array(triangles * 3), unique = new Map();
    for (let t = 0; t < triangles; t++) {
      for (let v = 0; v < 3; v++) {
        const start = 84 + t * 50 + 12 + v * 12;
        const x = source.readFloatLE(start), y = source.readFloatLE(start + 4), z = source.readFloatLE(start + 8);
        const key = `${x},${y},${z}`;
        let id = unique.get(key);
        if (id === undefined) { id = vertices.length / 3; unique.set(key, id); vertices.push(x, y, z); }
        index[t * 3 + v] = id;
      }
    }
    const positions = new Float32Array(vertices);
    const target = Math.max(12, Math.floor(index.length * settings.target_fraction / 3) * 3);
    const [reduced, error] = MeshoptSimplifier.simplify(index, positions, 3, target, settings.target_error, settings.flags);
    const bytes = Buffer.alloc(84 + reduced.length / 3 * 50);
    bytes.write('Visual mesh reduced with meshoptimizer 1.3.0');
    bytes.writeUInt32LE(reduced.length / 3, 80);
    for (let t = 0; t < reduced.length / 3; t++) {
      const a = reduced[t * 3] * 3, b = reduced[t * 3 + 1] * 3, c = reduced[t * 3 + 2] * 3;
      const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
      const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
      const normal = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
      const length = Math.hypot(...normal) || 1;
      for (let k = 0; k < 3; k++) bytes.writeFloatLE(normal[k] / length, 84 + t * 50 + k * 4);
      for (let v = 0; v < 3; v++) for (let k = 0; k < 3; k++) {
        bytes.writeFloatLE(positions[reduced[t * 3 + v] * 3 + k], 84 + t * 50 + 12 + v * 12 + k * 4);
      }
    }
    await mkdir(dirname(resolve(output, path)), { recursive: true });
    await writeFile(resolve(output, path), bytes);
    report.meshes[path] = {
      original_sha256: createHash('sha256').update(source).digest('hex'),
      reduced_sha256: createHash('sha256').update(bytes).digest('hex'),
      original_triangles: triangles, reduced_triangles: reduced.length / 3,
      original_bytes: source.length, reduced_bytes: bytes.length, relative_error: error,
    };
    console.log(`${path}: ${triangles} -> ${reduced.length / 3} triangles; error=${error}`);
  }
}
await mkdir(output, { recursive: true });
await writeFile(resolve(output, 'manifest.json'), JSON.stringify(report, null, 2) + '\n');
