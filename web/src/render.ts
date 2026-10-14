import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { MjModel } from '@mujoco/mujoco';
import type { Physics, Vec3 } from './physics';
import { Outdoor, groundTexture } from './environment';

function geometry(model: MjModel, id: number): THREE.BufferGeometry {
  const kind = model.geom_type[id];
  const s = model.geom_size.subarray(3 * id, 3 * id + 3);
  switch (kind) {
    case 0: {
      const width = s[0] ? 2 * s[0] : 200, height = s[1] ? 2 * s[1] : 200;
      const plane = new THREE.PlaneGeometry(width, height);
      const uv = plane.getAttribute('uv');
      for (let i = 0; i < uv.count; i++) {
        uv.setXY(i, (uv.getX(i) - 0.5) * width * 2, (uv.getY(i) - 0.5) * height * 2);
      }
      return plane;
    }
    case 1: {
      const field = model.geom_dataid[id], rows = model.hfield_nrow[field], cols = model.hfield_ncol[field];
      const size = model.hfield_size.subarray(4 * field, 4 * field + 4), start = model.hfield_adr[field];
      const surface = new THREE.PlaneGeometry(2 * size[0], 2 * size[1], cols - 1, rows - 1);
      const positions = surface.getAttribute('position'), uv = surface.getAttribute('uv');
      for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
        const index = row * cols + col;
        positions.setZ(index, model.hfield_data[start + (rows - 1 - row) * cols + col] * size[2]);
        uv.setXY(index, col * 2 * size[0] / (cols - 1) * 2, row * 2 * size[1] / (rows - 1) * 2);
      }
      surface.computeVertexNormals();
      return surface;
    }
    case 2: return new THREE.SphereGeometry(s[0], 20, 12);
    case 3: return new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 16).rotateX(Math.PI / 2);
    case 4: return new THREE.SphereGeometry(1, 20, 12).scale(s[0], s[1], s[2]);
    case 5: return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 20).rotateX(Math.PI / 2);
    case 6: return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case 7: {
      const mesh = model.geom_dataid[id];
      const v = model.mesh_vertadr[mesh], f = model.mesh_faceadr[mesh];
      const result = new THREE.BufferGeometry();
      // Compiled vertices already contain MuJoCo's centering, rotation and scale.
      result.setAttribute('position', new THREE.BufferAttribute(new Float32Array(
        model.mesh_vert.subarray(3 * v, 3 * (v + model.mesh_vertnum[mesh]))), 3));
      result.setIndex(new THREE.BufferAttribute(new Uint32Array(
        model.mesh_face.subarray(3 * f, 3 * (f + model.mesh_facenum[mesh]))), 1));
      result.computeVertexNormals();
      return result;
    }
    default: throw new Error(`Unsupported geom type ${kind} at ${id}`);
  }
}


export class RobotView {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(42, 1, 0.02, 250);
  readonly renderer: THREE.WebGLRenderer;
  readonly orbit: OrbitControls;
  readonly meshes: THREE.Mesh[] = [];
  private physics: Physics | null = null;
  private readonly previous: { position: THREE.Vector3; rotation: THREE.Quaternion }[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly rotation = new THREE.Quaternion();
  private readonly position = new THREE.Vector3();
  private readonly follow = new THREE.Vector3();
  private followHeightOffset = 0;
  private readonly ray = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly dragPlane = new THREE.Plane();
  private readonly hitPoint = new THREE.Vector3();
  private dragPointer: number | null = null;
  private readonly light = new THREE.DirectionalLight(0xfff9ee, 2.2);
  private readonly groundMap = groundTexture();
  private readonly crateMap: THREE.CanvasTexture;
  private readonly outdoor: Outdoor;
  private readonly resizeObserver: ResizeObserver;

  constructor(readonly canvas: HTMLCanvasElement, onInput: () => void) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.VSMShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.outdoor = new Outdoor(this.scene, this.light, () => {
      for (const mesh of this.meshes) {
        if (mesh.userData.ground) (mesh.material as THREE.MeshLambertMaterial).color
          .set(this.outdoor.groundColor).multiplyScalar(mesh.userData.terrain ? .86 : 1);
      }
    });
    const stencil = document.createElement('canvas');
    stencil.width = 512; stencil.height = 64;
    const ink = stencil.getContext('2d')!;
    ink.font = '500 34px monospace';
    ink.fillStyle = '#f0f2e3';
    ink.textAlign = 'center';
    ink.fillText('ROUGH GROUND', 256, 43);
    const marker = new THREE.Mesh(new THREE.PlaneGeometry(1.8, .225),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(stencil), transparent: true, depthWrite: false, opacity: .8 }));
    marker.position.set(0, 1.8, .003);
    this.scene.add(marker);
    const boards = document.createElement('canvas');
    boards.width = boards.height = 128;
    const wood = boards.getContext('2d')!;
    wood.fillStyle = '#f0d9ad'; wood.fillRect(0, 0, 128, 128);
    wood.fillStyle = '#b59d76';
    for (let y = 0; y < 128; y += 32) wood.fillRect(0, y, 128, 2);
    wood.fillStyle = '#dcc394';
    wood.fillRect(8, 0, 14, 128); wood.fillRect(106, 0, 14, 128);
    wood.fillStyle = '#796b53';
    for (const x of [14, 112]) for (const y of [8, 40, 72, 104]) wood.fillRect(x, y, 2, 2);
    this.crateMap = new THREE.CanvasTexture(boards);
    this.crateMap.colorSpace = THREE.SRGBColorSpace;
    this.light.position.set(-2, -3, 6);
    this.light.castShadow = true;
    this.light.shadow.mapSize.set(512, 512);
    this.light.shadow.blurSamples = 4;
    this.light.shadow.radius = 3;
    this.light.shadow.camera.left = this.light.shadow.camera.bottom = -3;
    this.light.shadow.camera.right = this.light.shadow.camera.top = 3;
    this.light.shadow.camera.near = 0.2;
    this.light.shadow.camera.far = 15;
    this.light.shadow.normalBias = 0.008;
    this.light.shadow.bias = -0.0002;
    this.scene.add(this.light, this.light.target);
    this.orbit = new OrbitControls(this.camera, canvas);
    this.orbit.enableDamping = true;
    this.orbit.enablePan = false;
    this.orbit.minDistance = 0.8;
    this.orbit.maxDistance = 10;
    this.orbit.maxPolarAngle = Math.PI * 0.49;
    this.orbit.addEventListener('start', onInput);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    canvas.addEventListener('pointerdown', event => this.startDrag(event, onInput), true);
    canvas.addEventListener('pointermove', event => this.moveDrag(event));
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      canvas.addEventListener(name, event => {
        if ((event as PointerEvent).pointerId === this.dragPointer) this.endDrag();
      });
    }
    window.addEventListener('blur', () => this.endDrag());
  }

  private resize(): void {
    const { width, height } = this.canvas.getBoundingClientRect();
    this.renderer.setSize(width, height, false);
    const framing = Math.max(1, 0.7 * height / width) / Math.max(1, 0.7 / this.camera.aspect);
    this.camera.position.sub(this.orbit.target).multiplyScalar(framing).add(this.orbit.target);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  setRobot(physics: Physics, resetCamera = true): void {
    if (resetCamera || !physics.drag) this.endDrag();
    for (const mesh of this.meshes) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.meshes.length = this.previous.length = 0;
    this.physics = physics;
    const m = physics.model;
    for (let i = 0; i < m.ngeom; i++) {
      // Match MuJoCo's default viewer without changing collision geometry.
      if (m.geom_group[i] >= 3) continue;
      const materialId = m.geom_matid[i];
      const rgba = materialId < 0 ? m.geom_rgba.subarray(i * 4, i * 4 + 4)
        : m.mat_rgba.subarray(materialId * 4, materialId * 4 + 4);
      const ground = m.geom_type[i] <= 1;
      const bodyName = physics.mj.mj_id2name(m, physics.mj.mjtObj.mjOBJ_BODY.value, m.geom_bodyid[i]) ?? '';
      const material = ground
        ? new THREE.MeshLambertMaterial({ color: this.outdoor.groundColor, map: this.groundMap })
        : new THREE.MeshStandardMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          map: bodyName.startsWith('play_crate_') ? this.crateMap : null,
          opacity: rgba[3], transparent: rgba[3] < 1, roughness: 0.72, metalness: 0.08,
        });
      if (m.geom_type[i] === 1) material.color.multiplyScalar(.86);
      const mesh = new THREE.Mesh(geometry(m, i), material);
      mesh.castShadow = m.geom_bodyid[i] > 0;
      mesh.receiveShadow = true;
      mesh.visible = rgba[3] > 0;
      mesh.userData.body = m.geom_bodyid[i];
      mesh.userData.geom = i;
      mesh.userData.ground = ground;
      mesh.userData.terrain = m.geom_type[i] === 1;
      this.meshes.push(mesh);
      this.previous.push({ position: new THREE.Vector3(), rotation: new THREE.Quaternion() });
      this.scene.add(mesh);
    }
    if (resetCamera) this.home();
    this.capture();
    this.render(1);
  }

  home(): void {
    if (!this.physics) return;
    const p = this.physics.data.qpos;
    const height = Math.max(0.25, p[2]);
    this.followHeightOffset = height * 0.2;
    this.follow.set(p[0], p[1], height * 0.8);
    this.orbit.target.copy(this.follow);
    const distance = (height > 0.5 ? 2.2 : 1.65) * Math.max(1, 0.7 / this.camera.aspect);
    this.camera.position.copy(this.follow).add(new THREE.Vector3(0.95, -1.35, 0.38).multiplyScalar(distance));
    this.orbit.update();
  }

  throwRay(): { origin: Vec3; target: Vec3 } {
    const data = this.physics!.data;
    const target: Vec3 = [data.qpos[0], data.qpos[1], data.qpos[2]];
    this.position.set(...target).sub(this.camera.position).normalize();
    // Start clear of the near plane and aim at the robot's current base.
    const origin = this.camera.position.clone().addScaledVector(this.position, 0.35).toArray() as Vec3;
    return { origin, target };
  }

  private transform(id: number, r: Float64Array, p: Float64Array): void {
    const k = id * 9;
    this.position.set(p[id * 3], p[id * 3 + 1], p[id * 3 + 2]);
    this.matrix.set(r[k], r[k + 1], r[k + 2], 0,
      r[k + 3], r[k + 4], r[k + 5], 0, r[k + 6], r[k + 7], r[k + 8], 0, 0, 0, 0, 1);
    this.rotation.setFromRotationMatrix(this.matrix);
  }

  capture(): void {
    const { geom_xmat: rotations, geom_xpos: positions } = this.physics!.data;
    for (let i = 0; i < this.meshes.length; i++) {
      this.transform(this.meshes[i].userData.geom, rotations, positions);
      this.previous[i].position.copy(this.position);
      this.previous[i].rotation.copy(this.rotation);
    }
  }

  render(alpha: number): void {
    if (this.physics) {
      const { geom_xmat: rotations, geom_xpos: positions } = this.physics.data;
      for (let i = 0; i < this.meshes.length; i++) {
        this.transform(this.meshes[i].userData.geom, rotations, positions);
        this.meshes[i].position.lerpVectors(this.previous[i].position, this.position, alpha);
        this.meshes[i].quaternion.slerpQuaternions(this.previous[i].rotation, this.rotation, alpha);
        if (this.meshes[i].userData.ground && !this.meshes[i].userData.terrain) {
          // Keep the finite visual plane beyond the haze as the robot travels.
          // Whole grid periods preserve its world-fixed motion reference.
          this.meshes[i].position.x += Math.round(this.follow.x / 50) * 50;
          this.meshes[i].position.y += Math.round(this.follow.y / 50) * 50;
        }
      }
      const q = this.physics.data.qpos;
      this.position.set(q[0], q[1], Math.max(0.25, q[2]) - this.followHeightOffset);
      this.position.sub(this.follow);
      this.position.set(this.position.x * 0.12, this.position.y * 0.12, this.position.z * 0.35);
      this.follow.add(this.position);
      this.camera.position.add(this.position);
      this.orbit.target.add(this.position);
      this.light.position.set(this.follow.x - 2, this.follow.y - 3, 6);
      this.light.target.position.set(this.follow.x, this.follow.y, 0);
    }
    this.orbit.update();
    this.outdoor.follow(this.camera.position);
    this.renderer.render(this.scene, this.camera);
  }

  private setRay(event: PointerEvent): void {
    const bounds = this.canvas.getBoundingClientRect();
    this.pointer.set(2 * (event.clientX - bounds.left) / bounds.width - 1,
      1 - 2 * (event.clientY - bounds.top) / bounds.height);
    this.ray.setFromCamera(this.pointer, this.camera);
  }

  private startDrag(event: PointerEvent, onInput: () => void): void {
    if (!this.physics || event.button !== 0 || this.dragPointer !== null) return;
    this.setRay(event);
    const hit = this.ray.intersectObjects(this.meshes, false).find(h => h.object.userData.body > 0 && h.object.visible);
    if (!hit) return;
    const body = hit.object.userData.body as number;
    const r = this.physics.data.xmat, p = this.physics.data.xpos;
    const delta = hit.point.clone().sub(new THREE.Vector3(p[body * 3], p[body * 3 + 1], p[body * 3 + 2]));
    const local: Vec3 = [0, 0, 0];
    for (let i = 0; i < 3; i++) local[i] = r[body * 9 + i] * delta.x + r[body * 9 + 3 + i] * delta.y + r[body * 9 + 6 + i] * delta.z;
    this.camera.getWorldDirection(this.position);
    this.dragPlane.setFromNormalAndCoplanarPoint(this.position, hit.point);
    this.physics.drag = { body, local, target: hit.point.toArray() as Vec3, previous: null };
    this.dragPointer = event.pointerId;
    this.orbit.enabled = false;
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.classList.add('dragging');
    event.stopImmediatePropagation();
    onInput();
  }

  private moveDrag(event: PointerEvent): void {
    if (!this.physics?.drag || event.pointerId !== this.dragPointer) return;
    this.setRay(event);
    if (this.ray.ray.intersectPlane(this.dragPlane, this.hitPoint)) {
      this.hitPoint.toArray(this.physics.drag.target);
    }
  }

  endDrag(): void {
    if (this.physics) this.physics.drag = null;
    this.dragPointer = null;
    this.orbit.enabled = true;
    this.canvas.classList.remove('dragging');
  }
}
