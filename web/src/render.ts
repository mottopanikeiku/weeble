import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { MjModel } from '@mujoco/mujoco';
import type { Physics, Vec3 } from './physics';

function geometry(model: MjModel, id: number): THREE.BufferGeometry {
  const kind = model.geom_type[id];
  const s = model.geom_size.subarray(3 * id, 3 * id + 3);
  switch (kind) {
    case 0: return new THREE.PlaneGeometry(s[0] ? 2 * s[0] : 200, s[1] ? 2 * s[1] : 200);
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
  readonly camera = new THREE.PerspectiveCamera(36, 1, 0.02, 250);
  readonly renderer: THREE.WebGLRenderer;
  readonly orbit: OrbitControls;
  readonly meshes: THREE.Mesh[] = [];
  private physics: Physics | null = null;
  private readonly previous: { position: THREE.Vector3; rotation: THREE.Quaternion }[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly rotation = new THREE.Quaternion();
  private readonly position = new THREE.Vector3();
  private readonly follow = new THREE.Vector3();
  private readonly ray = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly dragPlane = new THREE.Plane();
  private readonly hitPoint = new THREE.Vector3();
  private dragPointer: number | null = null;
  private readonly light = new THREE.DirectionalLight(0xfff9ee, 2.2);
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
    this.renderer.setClearColor('#eeede8');
    this.scene.fog = new THREE.Fog('#eeede8', 12, 45);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x827b70, 1.6));
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
    const grid = new THREE.GridHelper(200, 200, '#cbc9c0', '#dedcd4');
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.002;
    const gridMaterial = grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.32;
    this.scene.add(grid);
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
    const framing = Math.max(1, 0.9 * height / width) / Math.max(1, 0.9 / this.camera.aspect);
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
      const materialId = m.geom_matid[i];
      const rgba = materialId < 0 ? m.geom_rgba.subarray(i * 4, i * 4 + 4)
        : m.mat_rgba.subarray(materialId * 4, materialId * 4 + 4);
      const material = m.geom_type[i] === 0
        ? new THREE.MeshLambertMaterial({ color: '#cecdc5' })
        : new THREE.MeshStandardMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          opacity: rgba[3], transparent: rgba[3] < 1, roughness: 0.72, metalness: 0.08,
        });
      const mesh = new THREE.Mesh(geometry(m, i), material);
      mesh.castShadow = m.geom_bodyid[i] > 0;
      mesh.receiveShadow = true;
      mesh.visible = rgba[3] > 0;
      mesh.userData.body = m.geom_bodyid[i];
      mesh.userData.geom = i;
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
    this.follow.set(p[0], p[1], height * 0.8);
    this.orbit.target.copy(this.follow);
    const distance = (height > 0.5 ? 1.7 : 1) * Math.max(1, 0.9 / this.camera.aspect);
    this.camera.position.copy(this.follow).add(new THREE.Vector3(0.95, -1.35, 0.7).multiplyScalar(distance));
    this.orbit.update();
  }

  private transform(id: number): void {
    const d = this.physics!.data, r = d.geom_xmat, p = d.geom_xpos, k = id * 9;
    this.position.set(p[id * 3], p[id * 3 + 1], p[id * 3 + 2]);
    this.matrix.set(r[k], r[k + 1], r[k + 2], 0,
      r[k + 3], r[k + 4], r[k + 5], 0, r[k + 6], r[k + 7], r[k + 8], 0, 0, 0, 0, 1);
    this.rotation.setFromRotationMatrix(this.matrix);
  }

  capture(): void {
    for (let i = 0; i < this.meshes.length; i++) {
      this.transform(i);
      this.previous[i].position.copy(this.position);
      this.previous[i].rotation.copy(this.rotation);
    }
  }

  render(alpha: number): void {
    if (this.physics) {
      for (let i = 0; i < this.meshes.length; i++) {
        this.transform(i);
        this.meshes[i].position.lerpVectors(this.previous[i].position, this.position, alpha);
        this.meshes[i].quaternion.slerpQuaternions(this.previous[i].rotation, this.rotation, alpha);
      }
      const q = this.physics.data.qpos;
      this.position.set(q[0], q[1], this.follow.z);
      this.position.sub(this.follow).multiplyScalar(0.12);
      this.follow.add(this.position);
      this.camera.position.add(this.position);
      this.orbit.target.add(this.position);
      this.light.position.set(this.follow.x - 2, this.follow.y - 3, 6);
      this.light.target.position.set(this.follow.x, this.follow.y, 0);
    }
    this.orbit.update();
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
