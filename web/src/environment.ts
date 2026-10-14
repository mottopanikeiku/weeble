import * as THREE from 'three';

export function groundTexture(): THREE.DataTexture {
  const size = 128, pixels = new Uint8Array(size * size * 4);
  let seed = 41;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    seed = (1664525 * seed + 1013904223) >>> 0;
    const shade = 240 + (seed >>> 29) - (x < 1 || y < 1 ? 19 : 0);
    const i = 4 * (y * size + x);
    pixels[i] = pixels[i + 1] = pixels[i + 2] = shade;
    pixels[i + 3] = 255;
  }
  const texture = new THREE.DataTexture(pixels, size, size);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = 4;
  texture.needsUpdate = true;
  return texture;
}

export class Outdoor {
  readonly dark = matchMedia('(prefers-color-scheme: dark)');
  private readonly sky: THREE.Mesh;
  private readonly skyMaterial: THREE.ShaderMaterial;
  private readonly hillsMaterials = Array.from({ length: 3 }, () => new THREE.MeshBasicMaterial());
  private readonly padMaterial = new THREE.MeshLambertMaterial();
  readonly ambient = new THREE.HemisphereLight(0xffffff, 0x7d806b, 1.5);
  constructor(private readonly scene: THREE.Scene, private readonly sun: THREE.DirectionalLight,
              private readonly recolor: () => void) {
    this.skyMaterial = new THREE.ShaderMaterial({
      uniforms: { zenith: { value: new THREE.Color() }, horizon: { value: new THREE.Color() } },
      vertexShader: 'varying vec3 direction; void main(){direction=position; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader: 'uniform vec3 zenith; uniform vec3 horizon; varying vec3 direction; void main(){float h=normalize(direction).z; gl_FragColor=vec4(mix(horizon,zenith,smoothstep(-.02,.65,h)),1.);\n#include <colorspace_fragment>\n}',
      side: THREE.BackSide, depthWrite: false, toneMapped: false,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(110, 24, 12), this.skyMaterial);
    this.sky.renderOrder = -1;
    scene.add(this.sky, this.ambient);
    const matrix = new THREE.Matrix4(), rotation = new THREE.Quaternion();
    const hillGeometry = new THREE.SphereGeometry(1, 20, 10);
    for (let layer = 2; layer >= 0; layer--) {
      const hills = new THREE.InstancedMesh(hillGeometry, this.hillsMaterials[layer], 12);
      for (let i = 0; i < 12; i++) {
        const angle = i * Math.PI / 6 + layer * .19, radius = 34 + layer * 13 + (i % 3) * 2;
        matrix.compose(new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, -1), rotation,
          new THREE.Vector3(10 + i % 4, 8 + i % 3, 3.5 + layer * 1.6 + (i % 3) * .6));
        hills.setMatrixAt(i, matrix);
      }
      scene.add(hills);
    }
    const padTexture = document.createElement('canvas');
    padTexture.width = 768; padTexture.height = 512;
    const marks = padTexture.getContext('2d')!;
    marks.fillStyle = '#fff'; marks.fillRect(0, 0, 768, 512);
    marks.strokeStyle = 'rgba(65,75,56,.32)'; marks.lineWidth = 2;
    marks.strokeRect(2, 2, 764, 508);
    marks.beginPath();
    for (const x of [128, 384, 640]) {
      marks.moveTo(x, 2); marks.lineTo(x, 18);
      marks.moveTo(x, 510); marks.lineTo(x, 494);
    }
    marks.moveTo(2, 256); marks.lineTo(18, 256);
    marks.moveTo(766, 256); marks.lineTo(750, 256); marks.stroke();
    this.padMaterial.map = new THREE.CanvasTexture(padTexture);
    this.padMaterial.map.colorSpace = THREE.SRGBColorSpace;
    const pad = new THREE.Mesh(new THREE.PlaneGeometry(3, 2), this.padMaterial);
    pad.position.z = .002; pad.receiveShadow = true; scene.add(pad);
    this.theme();
    this.dark.addEventListener('change', () => { this.theme(); recolor(); });
  }
  get groundColor(): string { return this.dark.matches ? '#69765f' : '#b8bba1'; }
  get terrainColor(): string { return this.dark.matches ? '#66735c' : '#adb29b'; }
  private theme(): void {
    const dark = this.dark.matches, haze = dark ? '#394747' : '#dce5df';
    this.skyMaterial.uniforms.zenith.value.set(dark ? '#162b3b' : '#8bb2cb');
    this.skyMaterial.uniforms.horizon.value.set(haze);
    this.scene.fog = new THREE.Fog(haze, 16, 74);
    const hills = dark ? ['#233c34', '#3b514a', '#52625c'] : ['#82957f', '#a2b2a2', '#c1cdc3'];
    this.hillsMaterials.forEach((material, i) => material.color.set(hills[i]));
    this.padMaterial.color.set(dark ? '#879779' : '#d4d5b5');
    this.ambient.color.set(dark ? '#a7c2d5' : '#e9f2f5');
    this.ambient.intensity = dark ? 1.4 : 1.1;
    this.sun.color.set(dark ? '#f0dfc3' : '#ffe8bd');
    this.sun.intensity = dark ? 2.5 : 2.7;
  }
  follow(position: THREE.Vector3): void { this.sky.position.copy(position); }
}
