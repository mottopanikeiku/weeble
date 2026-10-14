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
  private readonly hillsMaterial = new THREE.MeshLambertMaterial();
  readonly ambient = new THREE.HemisphereLight(0xffffff, 0x7d806b, 2.0);
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
    const hills = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 6), this.hillsMaterial, 22);
    const matrix = new THREE.Matrix4(), rotation = new THREE.Quaternion();
    for (let i = 0; i < 22; i++) {
      const angle = i * 2.39996, radius = 43 + (i % 4) * 5;
      matrix.compose(new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, -1), rotation,
        new THREE.Vector3(9 + i % 5, 7 + i % 3, 2.5 + i % 4));
      hills.setMatrixAt(i, matrix);
    }
    scene.add(hills);
    this.theme();
    this.dark.addEventListener('change', () => { this.theme(); recolor(); });
  }
  get groundColor(): string { return this.dark.matches ? '#535d50' : '#b8bba1'; }
  private theme(): void {
    const dark = this.dark.matches, haze = dark ? '#323f43' : '#dce5df';
    this.skyMaterial.uniforms.zenith.value.set(dark ? '#162b3b' : '#8bb2cb');
    this.skyMaterial.uniforms.horizon.value.set(haze);
    this.scene.fog = new THREE.Fog(haze, 14, 72);
    this.hillsMaterial.color.set(dark ? '#283a32' : '#7c937b');
    this.ambient.color.set(dark ? '#a7c2d5' : '#e9f2f5');
    this.ambient.intensity = 2;
    this.sun.color.set(dark ? '#cad7eb' : '#fff1d5');
    this.sun.intensity = dark ? 2.2 : 2.4;
  }
  follow(position: THREE.Vector3): void { this.sky.position.copy(position); }
}
