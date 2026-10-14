import { Physics } from './physics';
import type { Command, CommandLimits } from './input';

export interface PolicySpec {
  schema_version: number;
  robot: 'go1' | 'g1';
  task: 'walk' | 'getup';
  seed: number;
  model_xml: string;
  force_body: string;
  action_size: number;
  default_qpos: number[];
  default_qvel: number[];
  default_pose: number[];
  simulation_dt: number;
  control_dt: number;
  n_substeps: number;
  command_ranges: CommandLimits;
  sensors: { gyro: string; local_linvel: string; gravity_site: string };
  observation: { fields: { name: string; offset: number; size: number }[] };
  control: { target: 'default_pose' | 'current_pose'; action_scale: number };
  model_parameters: { arrays: Record<string, number[] | number[][]>; options: Record<string, number | number[]> };
  normalizer: { enabled: boolean; mean: Record<string, number[]>; std: Record<string, number[]>; clip: number | null };
  policy: {
    obs_key: string; activation: 'swish' | 'silu';
    layers: { encoding: 'base64-f32-le'; kernel_shape: [number, number]; bias_shape: [number]; kernel: string; bias: string }[];
  };
  gait: { default_frequency: number; initial_phase: number[] } | null;
}

const f32 = Math.fround;

/** A direct float32 port of eval/sim.py; no stochastic action sampling. */
export class Policy {
  private readonly layers: { kernel: Float32Array; bias: Float32Array; input: number; output: Float32Array }[];
  private readonly mean: Float32Array;
  private readonly std: Float32Array;
  private readonly normalized: Float32Array;
  readonly action: Float32Array;

  constructor(readonly spec: PolicySpec) {
    if (spec.schema_version !== 1 || !['swish', 'silu'].includes(spec.policy.activation)) {
      throw new Error('Unsupported policy export');
    }
    const decode = (encoded: string, size: number): Float32Array => {
      const raw = atob(encoded);
      if (raw.length !== size * 4) throw new Error('Encoded weight shape differs');
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      const view = new DataView(bytes.buffer);
      const values = new Float32Array(size);
      for (let i = 0; i < size; i++) values[i] = view.getFloat32(i * 4, true);
      return values;
    };
    this.layers = spec.policy.layers.map(layer => {
      if (layer.encoding !== 'base64-f32-le') throw new Error('Unsupported weight encoding');
      const [input, output] = layer.kernel_shape;
      return {
        kernel: decode(layer.kernel, input * output), bias: decode(layer.bias, output),
        input, output: new Float32Array(output),
      };
    });
    this.mean = new Float32Array(spec.normalizer.mean[spec.policy.obs_key]);
    this.std = new Float32Array(spec.normalizer.std[spec.policy.obs_key]);
    this.normalized = new Float32Array(this.mean.length);
    this.action = new Float32Array(spec.action_size);
  }

  infer(observation: Float32Array): Float32Array {
    if (observation.length !== this.mean.length) throw new Error('Observation size differs from exported network');
    const normalizer = this.spec.normalizer;
    for (let i = 0; i < observation.length; i++) {
      let x = normalizer.enabled ? f32(f32(observation[i] - this.mean[i]) / this.std[i]) : observation[i];
      if (normalizer.enabled && normalizer.clip !== null) x = Math.min(normalizer.clip, Math.max(-normalizer.clip, x));
      this.normalized[i] = x;
    }
    let input = this.normalized;
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l], width = layer.bias.length;
      const final = l === this.layers.length - 1;
      // The scale half of the final Gaussian output is unused by tanh(mean).
      const outputs = final ? this.action.length : width;
      for (let j = 0; j < outputs; j++) {
        let sum = 0;
        for (let i = 0; i < layer.input; i++) sum += input[i] * layer.kernel[i * width + j];
        const x = f32(f32(sum) + layer.bias[j]);
        if (final) this.action[j] = Math.tanh(x);
        else {
          const e = f32(Math.exp(-Math.abs(x)));
          const sigmoid = f32(x >= 0 ? 1 / f32(1 + e) : e / f32(1 + e));
          layer.output[j] = f32(x * sigmoid);
        }
      }
      input = layer.output;
    }
    return this.action;
  }
}

export class Controller {
  readonly physics: Physics;
  readonly observation: Float32Array;
  readonly command = new Float32Array(3);
  readonly lastAction: Float32Array;
  readonly observationLastAction: Float32Array;
  readonly phase: Float32Array | null;
  readonly observationPhase: Float32Array | null;
  private phaseDt = 0;
  private readonly gyro: number;
  private readonly velocity: number;
  private readonly gravitySite: number;
  private readonly defaultPose: Float32Array;

  constructor(readonly policy: Policy, physics: Physics) {
    this.physics = physics;
    const { spec } = policy;
    if (physics.substeps !== spec.n_substeps) throw new Error('Exported substep count differs');
    const model = physics.model as unknown as Record<string, Float64Array>;
    for (const [key, value] of Object.entries(spec.model_parameters.arrays)) model[key].set(value.flat());
    const options = physics.model.opt;
    const settings = options as unknown as Record<string, number | Float64Array>;
    for (const [key, value] of Object.entries(spec.model_parameters.options)) {
      if (Array.isArray(value)) (settings[key] as Float64Array).set(value);
      else settings[key] = value;
    }
    options.delete();
    const sensorAddress = (name: string) => {
      const id = physics.mj.mj_name2id(physics.model, physics.mj.mjtObj.mjOBJ_SENSOR.value, name);
      if (id < 0) throw new Error(`Missing policy sensor ${name}`);
      return physics.model.sensor_adr[id] as number;
    };
    this.gyro = sensorAddress(spec.sensors.gyro);
    this.velocity = spec.task === 'walk' ? sensorAddress(spec.sensors.local_linvel) : -1;
    this.gravitySite = physics.mj.mj_name2id(physics.model, physics.mj.mjtObj.mjOBJ_SITE.value, spec.sensors.gravity_site);
    if (this.gravitySite < 0) throw new Error('Missing gravity reference site');
    this.defaultPose = new Float32Array(spec.default_pose);
    this.observation = new Float32Array(spec.observation.fields.reduce((sum, field) => sum + field.size, 0));
    this.lastAction = new Float32Array(spec.action_size);
    this.observationLastAction = new Float32Array(spec.action_size);
    this.phase = spec.gait ? new Float32Array(spec.gait.initial_phase) : null;
    this.observationPhase = this.phase?.slice() ?? null;
    this.reset();
  }

  reset(qpos?: ArrayLike<number>, qvel?: ArrayLike<number>, command: ArrayLike<number> = [0, 0, 0],
        gaitFrequency?: number, phase?: ArrayLike<number>): void {
    const spec = this.policy.spec;
    this.physics.reset(qpos ?? spec.default_qpos, qvel ?? spec.default_qvel);
    this.physics.data.ctrl.set(this.physics.data.qpos.subarray(7));
    this.physics.mj.mj_forward(this.physics.model, this.physics.data);
    this.command.set(command);
    this.lastAction.fill(0);
    this.observationLastAction.fill(0);
    if (spec.gait && this.phase && this.observationPhase) {
      this.phase.set(phase ?? spec.gait.initial_phase);
      this.observationPhase.set(this.phase);
      this.phaseDt = f32(2 * Math.PI * spec.control_dt * (gaitFrequency ?? spec.gait.default_frequency));
    }
  }

  observe(command?: ArrayLike<number>): Float32Array {
    if (command) this.command.set(command);
    const { data } = this.physics;
    for (const field of this.policy.spec.observation.fields) {
      for (let i = 0; i < field.size; i++) {
        let value: number;
        switch (field.name) {
          case 'gyro': value = data.sensordata[this.gyro + i]; break;
          case 'gravity': value = -f32(data.site_xmat[this.gravitySite * 9 + 6 + i]); break;
          case 'joint_position': value = f32(data.qpos[7 + i]) - this.defaultPose[i]; break;
          case 'joint_velocity': value = data.qvel[6 + i]; break;
          case 'last_action': value = this.observationLastAction[i]; break;
          case 'command': value = this.command[i]; break;
          case 'local_linear_velocity': value = data.sensordata[this.velocity + i]; break;
          case 'phase_cos': value = Math.cos(this.observationPhase![i]); break;
          case 'phase_sin': value = Math.sin(this.observationPhase![i]); break;
          default: throw new Error(`Unknown observation block ${field.name}`);
        }
        this.observation[field.offset + i] = value;
      }
    }
    return this.observation;
  }

  step(command?: Command, action?: ArrayLike<number>): Float32Array {
    const chosen = action ?? this.policy.infer(this.observe(command));
    if (action && command) this.command.set(command);
    const { data } = this.physics;
    const control = this.policy.spec.control;
    for (let i = 0; i < chosen.length; i++) {
      const pose = control.target === 'current_pose' ? f32(data.qpos[7 + i]) : this.defaultPose[i];
      data.ctrl[i] = f32(pose + f32(f32(control.action_scale) * f32(chosen[i])));
    }
    this.physics.step();
    // Playground constructs the next observation before advancing info history.
    this.observationLastAction.set(this.lastAction);
    if (this.phase && this.observationPhase) this.observationPhase.set(this.phase);
    this.lastAction.set(chosen);
    if (this.phase) {
      const pi = f32(Math.PI);
      for (let i = 0; i < this.phase.length; i++) {
        this.phase[i] = f32(f32(f32(this.phase[i] + this.phaseDt) + pi) % f32(2 * pi) - pi);
      }
    }
    return this.observe();
  }
}
