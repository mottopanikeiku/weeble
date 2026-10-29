"""Robustness atlas: the released walking checkpoints under stresses outside their training.

Native C MuJoCo on the CPU and no retraining. The protocol is in docs/evaluation.md.

python -m eval.atlas                  # simulate every episode, then write the report and README block
python -m eval.atlas --report-only    # rebuild the report from the committed raw outcomes
python -m eval.atlas --rerun 36       # re-simulate committed episodes: same outcomes (--exact: identical rows)
"""
from __future__ import annotations

import os

# Set before NumPy imports: one BLAS thread per process keeps reruns bit-identical.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import gzip
import hashlib
import html
from importlib import metadata
import json
import math
from multiprocessing import get_context
from pathlib import Path
import platform
import subprocess

import mujoco
import numpy as np

from eval.run import SuiteConfig, TrialFailure, TrialSimulator, _finite, _steps
from eval.sim import Policy


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("go1-walk", "go1-walk-push", "go1-walk-rough", "g1-walk")
ARM_LABELS = {"go1-walk": "Go1, flat", "go1-walk-push": "Go1, shoves", "go1-walk-rough": "Go1, rough",
              "g1-walk": "G1"}
SCENES = {"go1": "assets/go1/walk.xml", "g1": "assets/g1/walk.xml"}
ENVIRONMENT_AXES = ("terrain", "friction", "payload", "strength", "delay")
AXES = ENVIRONMENT_AXES + ("transition", "shove")
# Same nominal episodes on every environment axis: the report requires identical rows.
NOMINAL = {"terrain": 0.0, "friction": 0.6, "payload": 0.0, "strength": 1.0, "delay": 0}
SENSOR_FIELDS = ("local_linear_velocity", "gyro", "gravity", "joint_position", "joint_velocity")
FALLS = ("tilt", "nonfoot_floor_contact")
OUTCOMES = ("never_fell", "fell_recovered", "fell", "numerical_failure")
# Fractions of each robot's standard tracking range (forward, lateral, yaw).
TRANSITIONS = {
    "start": ((0, 0, 0), (1, 0, 0)), "stop": ((1, 0, 0), (0, 0, 0)),
    "reverse": ((1, 0, 0), (-1, 0, 0)), "lateral_flip": ((0, 1, 0), (0, -1, 0)),
    "turn_flip": ((0, 0, 1), (0, 0, -1)),
}
SECTORS = ("forward", "forward-left", "left", "back-left", "back", "back-right", "right", "forward-right")
MARGIN = 0.9


@dataclass(frozen=True)
class AtlasConfig:
    trial_seed: int = 20261029
    trials: int = 50
    episode_seconds: float = 10.0
    discard_seconds: float = 2.0
    command_fraction: float = 0.5
    terrain: tuple[float, ...] = (0.0, 0.05, 0.075, 0.1, 0.125)
    terrain_cells: int = 256
    terrain_half_width: float = 10.0
    friction: tuple[float, ...] = (1.0, 0.6, 0.4, 0.25, 0.15)
    payload: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
    strength: tuple[float, ...] = (1.0, 0.85, 0.7, 0.55, 0.4)
    delay: tuple[int, ...] = (0, 1, 2, 3, 4)
    transition: tuple[str, ...] = tuple(TRANSITIONS)
    transition_window: tuple[float, float] = (2.0, 3.0)
    shove: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)
    shove_command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    shove_wait: tuple[float, float] = (2.0, 3.0)
    shove_seconds: float = 0.1
    post_shove_seconds: float = 3.0
    recovery_policy: str = "go1-getup-s0"
    recovery_seconds: float = 6.0

    def validate(self):
        if self.trials < 1 or self.terrain_cells < 2 or self.terrain_half_width <= 0:
            raise ValueError("Trials and terrain size must be positive")
        if not 0 <= self.discard_seconds < self.episode_seconds:
            raise ValueError("Discard must fall inside the episode")
        for axis, nominal in NOMINAL.items():
            if nominal not in getattr(self, axis):
                raise ValueError(f"The {axis} levels must include the nominal level {nominal}")
        if set(self.transition) - set(TRANSITIONS) or not self.transition:
            raise ValueError("Unknown transition")
        if any(not 0 <= x for x in self.terrain + self.payload + self.shove + self.delay):
            raise ValueError("Terrain, payload, shove and delay levels must be nonnegative")
        if any(x <= 0 for x in self.friction + self.strength):
            raise ValueError("Friction and strength levels must be positive")
        for low, high in (self.transition_window, self.shove_wait):
            if not 0 < low <= high:
                raise ValueError("Invalid timing window")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cells(config):
    """Every (axis, level) with its full parameter set and the SHA-256 of that set."""
    shared = {key: value for key, value in asdict(config).items() if key not in AXES}
    result = []
    for axis in AXES:
        for level in getattr(config, axis):
            params = {"axis": axis, "level": level, **shared}
            result.append({"axis": axis, "level": level, "sha256": _digest(params)})
    return result


def policy_names():
    return [f"{arm}-s{seed}" for arm in ARMS for seed in (0, 1, 2)]


def _rng(config, group, trial):
    # Excludes policy and level: every policy and level sees the same draws for a trial.
    return np.random.default_rng(np.random.SeedSequence([config.trial_seed, group, trial]))


def environment_draws(config, trial):
    rng = _rng(config, 0, trial)
    heading = float(rng.uniform(-math.pi, math.pi))
    command = rng.uniform(-1.0, 1.0, 3) * config.command_fraction
    field = rng.random((config.terrain_cells, config.terrain_cells))
    return heading, command, field


def transition_draws(config, trial):
    return float(_rng(config, 1, trial).uniform(*config.transition_window))


def shove_draws(config, trial):
    rng = _rng(config, 2, trial)
    return float(rng.uniform(*config.shove_wait)), float(rng.uniform(0.0, 2 * math.pi))


def _ranges(robot):
    standard = SuiteConfig()
    return np.array([high for _, high in (standard.go1_ranges if robot == "go1" else standard.g1_ranges)])


def terrain_model(root, scene, amplitude, config):
    """The scene with its floor replaced by a heightfield spanning [-amplitude/2, amplitude/2]."""
    spec = mujoco.MjSpec.from_file(str(root / scene))
    field = spec.add_hfield()
    field.name = "atlas_terrain"
    field.size = [config.terrain_half_width, config.terrain_half_width, amplitude, 0.1]
    field.nrow = field.ncol = config.terrain_cells
    field.userdata = [0.0] * config.terrain_cells ** 2  # Each trial writes its own grid.
    floor = spec.geom("floor")
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = "atlas_terrain"
    floor.pos = [0.0, 0.0, -amplitude / 2]
    return spec.compile()


def _field_statistics(heights):
    """Spread and neighbour correlation of heights normalized to [0, 1]."""
    centred = heights - heights.mean()
    variance = float(centred.var())
    return {"cells": list(heights.shape), "min": float(heights.min()), "max": float(heights.max()),
            "std": float(heights.std()),
            "neighbour_correlation_rows": float((centred[1:] * centred[:-1]).mean() / variance),
            "neighbour_correlation_columns": float((centred[:, 1:] * centred[:, :-1]).mean() / variance)}


def terrain_statistics(root, config):
    """The training heightfield next to the first generated one, both normalized to [0, 1]."""
    model = mujoco.MjModel.from_xml_path(str(root / "assets/go1/rough.xml"))
    training = model.hfield_data.reshape(model.hfield_nrow[0], model.hfield_ncol[0])
    return {
        "training": {"scene": "assets/go1/rough.xml", "elevation_m": float(model.hfield_size[0][2]),
                     "cell_m": float(2 * model.hfield_size[0][0] / (model.hfield_ncol[0] - 1)),
                     **_field_statistics(training)},
        "generated_trial_0": {"cell_m": 2 * config.terrain_half_width / (config.terrain_cells - 1),
                              **_field_statistics(environment_draws(config, 0)[2])},
    }


def apply_stress(model, spec, axis, level):
    """Change one physical property after the export's own model parameters are applied."""
    if axis == "friction":
        floor = model.geom("floor").id
        model.geom_friction[floor, 0] = level
        for index in range(model.npair):
            if floor in (model.pair_geom1[index], model.pair_geom2[index]):
                model.pair_friction[index, :2] = level
    elif axis == "payload":
        base = model.body(spec["force_body"]).id
        model.body_mass[base] += level * model.body_subtreemass[base]
    elif axis == "strength":
        if not (np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)):
            raise ValueError("Strength scaling expects position actuators")
        model.actuator_gainprm[:, 0] *= level
        model.actuator_biasprm[:, 1:3] *= level
        model.actuator_forcerange *= level
        model.jnt_actfrcrange *= level


class AtlasSimulator(TrialSimulator):
    """Trial simulator with an optionally stressed model and delayed sensor readings."""

    def __init__(self, policy, root, model, axis=None, level=None):
        self.delay_steps = int(level) if axis == "delay" else 0
        self._history = deque(maxlen=self.delay_steps + 1)
        super().__init__(policy, root, SuiteConfig(), model=model)
        apply_stress(self.model, policy.spec, axis, level)
        # Payload changes masses; every atlas model gets the same constant recomputation.
        mujoco.mj_setConst(self.model, self.data)
        self._sensor = np.zeros(sum(f["size"] for f in policy.spec["observation"]["fields"]), bool)
        for field in policy.spec["observation"]["fields"]:
            if field["name"] in SENSOR_FIELDS:
                self._sensor[field["offset"]:field["offset"] + field["size"]] = True
        self.reset()

    def reset(self, *args, **kwargs):
        if hasattr(self, "_history"):
            self._history.clear()
        return super().reset(*args, **kwargs)

    def checked_step(self, command=None, force=None):
        # Sensor fields come from delay_steps control steps ago, holding the first reading
        # until the buffer fills; commands, the previous action and the gait clock are current.
        current = self.observe(command=command)
        self._history.append(current)
        observation = np.where(self._sensor, self._history[0], current)
        if not _finite(observation):
            self._fail("nonfinite_observation")
        if self.failure:
            raise TrialFailure(self.failure)
        action = self.policy.action(observation)
        if not _finite(action):
            self._fail("nonfinite_action")
            raise TrialFailure(self.failure)
        self.step(command=command, action=action, force=force)
        if self.failure:
            raise TrialFailure(self.failure)

    def foot_contacts(self):
        touching = set()
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a == self.floor_id else a if b == self.floor_id else -1
            if contact.dist <= 0 and other in self.feet_ids:
                touching.add(self.model.geom(other).name)
        return sorted(touching)

    def heading(self):
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def stance(robot, feet):
    """Gait phase at shove onset, read from which feet touch the floor."""
    feet = set(feet)
    if robot == "g1":
        return {0: "flight", 2: "double support"}.get(len(feet)) or (
            "left stance" if "left_foot" in feet else "right stance")
    if len(feet) >= 3:
        return "three or four feet"
    if feet == {"FL", "RR"}:
        return "diagonal FL+RR"
    if feet == {"FR", "RL"}:
        return "diagonal FR+RL"
    return "other"


def _start(policy, heading, lift=0.0):
    qpos = np.array(policy.spec["default_qpos"], dtype=float)
    qpos[2] += lift
    qpos[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
    return qpos


class Episodes:
    """Simulators for one policy and one cell; the Go1 recovery simulator shares the stress."""

    def __init__(self, root, config, name, cell):
        self.config, self.cell = config, cell
        self.policy = Policy.load(root / "policies" / f"{name}.json")
        self.robot = self.policy.spec["robot"]
        axis, level = cell["axis"], cell["level"]
        stress = (axis, level) if axis in ENVIRONMENT_AXES else (None, None)
        terrain = axis == "terrain" and level > 0

        def model(scene):
            return terrain_model(root, scene, level, config) if terrain else None

        self.walk = AtlasSimulator(self._scene_policy(SCENES[self.robot]), root,
                                   model(SCENES[self.robot]), *stress)
        self.recovery = None
        if self.robot == "go1":
            recovery = Policy.load(root / "policies" / f"{config.recovery_policy}.json")
            self.recovery = AtlasSimulator(recovery, root, model(recovery.spec["model_xml"]), *stress)
        self.terrain = terrain
        self.reference_impulse = reference_impulse(self.walk, self.policy.spec)

    def _scene_policy(self, scene):
        # Every walking arm, including the rough-ground one, runs on the same flat scene.
        return Policy(dict(self.policy.spec, model_xml=scene))

    def run(self, trial):
        axis, level, config = self.cell["axis"], self.cell["level"], self.config
        spec = self.policy.spec
        dt = float(spec["control_dt"])
        ranges = _ranges(self.robot)
        row = {"axis": axis, "level": level, "trial": trial, "cell_sha256": self.cell["sha256"]}
        heading, schedule, force_at = 0.0, None, None
        lift = 0.0
        if axis in ENVIRONMENT_AXES:
            heading, fraction, field = environment_draws(config, trial)
            command = (fraction * ranges).tolist()
            schedule = [(0, command)]
            steps = _steps(config.episode_seconds, dt)
            row.update(heading=heading, command=command)
            if self.terrain:
                for sim in (self.walk, self.recovery):
                    if sim is not None:
                        sim.model.hfield_data[:] = field.ravel()
                lift = level / 2
        elif axis == "transition":
            switch = round(transition_draws(config, trial) / dt)
            first, second = (np.asarray(c, float) * ranges for c in TRANSITIONS[level])
            schedule = [(0, first.tolist()), (switch, second.tolist())]
            steps = _steps(config.episode_seconds, dt)
            row.update(heading=heading, commands=[first.tolist(), second.tolist()], switch_time=switch * dt)
        else:
            wait, direction = shove_draws(config, trial)
            onset = round(wait / dt)
            push_steps = _steps(config.shove_seconds, dt)
            schedule = [(0, list(config.shove_command))]
            steps = onset + push_steps + _steps(config.post_shove_seconds, dt)
            force_at = (onset, onset + push_steps)
            impulse = level * self.reference_impulse
            row.update(heading=heading, command=list(config.shove_command), impulse_normalized=level,
                       impulse_ns=impulse, direction_body=direction, onset=onset * dt,
                       sector=SECTORS[int(((direction + math.pi / 8) % (2 * math.pi)) // (math.pi / 4))])
        sim = self.walk
        sim.reset(qpos=_start(self.policy, heading, lift), command=schedule[0][1])
        sim.begin(fall_checks=True)
        sim._inspect()
        errors, samples, command = np.zeros(3), 0, schedule[0][1]
        force = None
        row.update(outcome="never_fell", fall_time=None, fall_reason=None, recovery_time=None,
                   handoff_penetration=None, numerical_failure=None)
        try:
            for index in range(steps):
                for start, value in schedule:
                    if index == start:
                        command = value
                if force_at is not None and index == force_at[0]:
                    feet = sim.foot_contacts()
                    angle = sim.heading() + row["direction_body"]
                    force = np.array([math.cos(angle), math.sin(angle), 0.0]) * row["impulse_ns"] / config.shove_seconds
                    row.update(stance=stance(self.robot, feet), feet=feet)
                if force_at is not None and index == force_at[1]:
                    force = None
                sim.checked_step(command=command, force=force)
                if axis != "shove" and (index + 1) * dt > config.discard_seconds + 1e-9:
                    v = sim.velocity()
                    errors += np.abs([v[3] - command[0], v[4] - command[1], v[2] - command[2]])
                    samples += 1
        except TrialFailure as failure:
            if sim.failure not in FALLS:
                row.update(outcome="numerical_failure", numerical_failure=str(failure))
            else:
                row.update(outcome="fell", fall_time=sim.failure_time, fall_reason=sim.failure)
                if self.recovery is not None:
                    self._recover(row)
        except Exception as exc:  # A software defect is recorded, never counted as survival.
            row.update(outcome="numerical_failure", numerical_failure=f"{type(exc).__name__}: {exc}")
        if axis != "shove":
            row.update(samples=samples, **{f"{name}_mae": (float(errors[i] / samples) if samples else None)
                                           for i, name in enumerate(("forward", "lateral", "yaw"))})
        return row

    def _recover(self, row):
        """Hand the fallen state to the get-up policy, as the browser does, under the same stress."""
        sim, recovery = self.walk, self.recovery
        recovery.reset(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy())
        depths = [-float(c.dist) for c in recovery.data.contact if c.dist < 0]
        row["handoff_penetration"] = max(depths, default=0.0)
        recovery.begin(fall_checks=False)
        recovery.recovery_steps = 0
        try:
            for _ in range(_steps(self.config.recovery_seconds, float(recovery.policy.spec["control_dt"]))):
                recovery.checked_step()
        except TrialFailure as failure:
            row["recovery_failure"] = str(failure)
            return
        if recovery.recovery_time is not None:
            row.update(outcome="fell_recovered", recovery_time=recovery.recovery_time)


def reference_impulse(sim, spec):
    """m sqrt(g h): total mass times the inverted-pendulum speed at the nominal COM height."""
    model = sim.model
    data = mujoco.MjData(model)
    data.qpos[:] = spec["default_qpos"]
    mujoco.mj_forward(model, data)
    base = model.body(spec["force_body"]).id
    gravity = float(np.linalg.norm(model.opt.gravity))
    return float(model.body_subtreemass[base] * math.sqrt(gravity * data.subtree_com[base][2]))


def _robot_metadata(root, name):
    policy = Policy.load(root / "policies" / f"{name}.json")
    sim = AtlasSimulator(Policy(dict(policy.spec, model_xml=SCENES[policy.spec["robot"]])), root, None)
    base = sim.model.body(policy.spec["force_body"]).id
    data = mujoco.MjData(sim.model)
    data.qpos[:] = policy.spec["default_qpos"]
    mujoco.mj_forward(sim.model, data)
    reference = reference_impulse(sim, policy.spec)
    # Playground's Go1 kicks push with 0.5 sin(pi t/T) m v/T for T seconds: impulse m v/pi.
    kicks = policy.spec["training_config"].get("pert_config") or {}
    kick = (float(sim.model.body_subtreemass[base]) * kicks["velocity_kick"][1] / math.pi
            if kicks.get("enable") else None)
    return {"mass_kg": float(sim.model.body_subtreemass[base]), "com_height_m": float(data.subtree_com[base][2]),
            "reference_impulse_ns": reference, "training_kick_max_ns": kick,
            "training_kick_max_normalized": None if kick is None else kick / reference,
            "command_ranges": _ranges(policy.spec["robot"]).tolist()}


def simulate_policy(task):
    """All cells and trials for one policy, in a fixed order; rows carry no wall time."""
    root, config_values, name, selection = task
    config = AtlasConfig(**config_values)
    rows = []
    spec = json.loads((root / "policies" / f"{name}.json").read_text())
    for cell in cells(config):
        trials = [t for (c, t) in selection if c == cell["sha256"]] if selection is not None else range(config.trials)
        if not trials:
            continue
        episodes = Episodes(root, config, name, cell)
        for trial in trials:
            row = episodes.run(trial)
            row.update(policy=name, arm=name.rsplit("-s", 1)[0], seed=int(spec["seed"]), robot=spec["robot"])
            rows.append(_ordered(row))
    return name, rows


ROW_ORDER = ("policy", "arm", "robot", "seed", "axis", "level", "trial", "cell_sha256", "outcome",
             "fall_time", "fall_reason", "recovery_time", "handoff_penetration", "recovery_failure",
             "numerical_failure", "heading", "command", "commands", "switch_time", "impulse_normalized",
             "impulse_ns", "direction_body", "sector", "onset", "stance", "feet", "samples",
             "forward_mae", "lateral_mae", "yaw_mae")


def _ordered(row):
    unknown = set(row) - set(ROW_ORDER)
    if unknown:
        raise ValueError(f"Unexpected atlas fields {sorted(unknown)}")
    return {key: row[key] for key in ROW_ORDER if key in row}


def _write_rows(path, rows):
    # Fixed gzip header (no name, mtime 0): identical rows give identical bytes.
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, allow_nan=False) + "\n").encode())


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def _provenance(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "platform": platform.platform(),
            "package_versions": {name: metadata.version(name) for name in ("mujoco", "numpy")}}


def simulate(root, output, config, workers):
    config.validate()
    names = policy_names()
    # Humanoid episodes are the slowest; start them first.
    order = sorted(names, key=lambda n: not n.startswith("g1"))
    tasks = [(root, asdict(config), name, None) for name in order]
    with get_context("spawn").Pool(workers) as pool:
        results = dict(pool.imap_unordered(simulate_policy, tasks))
    output.mkdir(parents=True, exist_ok=True)
    raw = {}
    for name in names:
        path = output / f"{name}.jsonl.gz"
        _write_rows(path, results[name])
        raw[name] = {"path": path.name, "sha256": _sha(path), "rows": len(results[name])}
    recovery = root / "policies" / f"{config.recovery_policy}.json"
    scenes = sorted(set(SCENES.values()) | {json.loads(recovery.read_text())["model_xml"]})
    manifest = {
        "schema_version": 1, "command": "python -m eval.atlas", "config": asdict(config),
        "config_sha256": _digest(asdict(config)), "cells": cells(config),
        "policies": {name: {"policy_sha256": _sha(root / "policies" / f"{name}.json"),
                            **_robot_metadata(root, name)} for name in names},
        "recovery_policy": {"name": config.recovery_policy, "policy_sha256": _sha(recovery)},
        "scenes": {scene: _sha(root / scene) for scene in scenes},
        "terrain_statistics": terrain_statistics(root, config),
        "raw": raw, "provenance": _provenance(root),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load(root, output):
    """Read the manifest and raw rows, checking hashes, configuration and completeness."""
    manifest = json.loads((output / "manifest.json").read_text())
    config = AtlasConfig(**{key: tuple(value) if isinstance(value, list) else value
                            for key, value in manifest["config"].items()})
    if _digest(asdict(config)) != manifest["config_sha256"] or cells(config) != manifest["cells"]:
        raise ValueError("Atlas configuration does not match its recorded hashes")
    expected = {(c["sha256"], trial) for c in manifest["cells"] for trial in range(config.trials)}
    rows = []
    for name in policy_names():
        record = manifest["raw"][name]
        path = output / record["path"]
        if _sha(path) != record["sha256"]:
            raise ValueError(f"{record['path']} does not match its recorded SHA-256")
        found = read_rows(path)
        keys = [(r["cell_sha256"], r["trial"]) for r in found]
        if len(keys) != len(set(keys)) or set(keys) != expected or any(r["policy"] != name for r in found):
            raise ValueError(f"{record['path']} is not exactly one row per cell and trial")
        if _sha(root / "policies" / f"{name}.json") != manifest["policies"][name]["policy_sha256"]:
            raise ValueError(f"policies/{name}.json changed after the atlas ran")
        rows += found
    recovery = manifest["recovery_policy"]
    if _sha(root / "policies" / f"{recovery['name']}.json") != recovery["policy_sha256"]:
        raise ValueError(f"policies/{recovery['name']}.json changed after the atlas ran")
    for scene, digest in manifest["scenes"].items():
        if _sha(root / scene) != digest:
            raise ValueError(f"{scene} changed after the atlas ran")
    return manifest, config, rows


def wilson(successes, trials, z=1.959963984540054):
    """95% Wilson score interval; the bounds are exactly 0 or 1 at the extremes."""
    if trials == 0:
        return None, None
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return (0.0 if successes == 0 else centre - half), (1.0 if successes == trials else centre + half)


def _rate(rows, outcomes):
    count = sum(r["outcome"] in outcomes for r in rows)
    low, high = wilson(count, len(rows))
    return count, count / len(rows) if rows else None, low, high


def _summaries(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in groups.items():
        never, never_rate, never_low, never_high = _rate(group, ("never_fell",))
        fell = [r for r in group if r["outcome"] in ("fell", "fell_recovered")]
        recovered, _, rec_low, rec_high = _rate(fell, ("fell_recovered",)) if fell else (0, None, None, None)
        result.append({**dict(zip(keys, key)), "trials": len(group), "never_fell": never,
                       "never_fell_rate": never_rate, "never_fell_ci_low": never_low,
                       "never_fell_ci_high": never_high, "fell": len(fell), "fell_recovered": recovered,
                       "recovered_given_fall": recovered / len(fell) if fell else None,
                       "recovered_ci_low": rec_low, "recovered_ci_high": rec_high,
                       "numerical_failures": sum(r["outcome"] == "numerical_failure" for r in group)})
    return result


def check_nominal(rows, config):
    """Nominal levels of the environment axes are the same episodes and must agree exactly."""
    ignore = {"axis", "level", "cell_sha256"}
    nominal = defaultdict(dict)
    for row in rows:
        if row["axis"] in ENVIRONMENT_AXES and row["level"] == NOMINAL[row["axis"]]:
            nominal[(row["policy"], row["trial"])][row["axis"]] = {k: v for k, v in row.items() if k not in ignore}
    for key, by_axis in nominal.items():
        values = list(by_axis.values())
        if len(values) != len(ENVIRONMENT_AXES) or any(v != values[0] for v in values[1:]):
            raise ValueError(f"Nominal episodes differ between axes for {key}")
    return len(nominal)


def margin(levels, rates):
    """Hardest level reached with never-fell rate >= MARGIN at it and every easier level."""
    reached = None
    for level, rate in zip(levels, rates):
        if rate < MARGIN:
            break
        reached = level
    return reached


def _ordered_levels(config, axis):
    """Levels from nominal toward the stress the margin measures."""
    levels = list(getattr(config, axis))
    if axis == "friction":
        # The margin follows the low-friction sweep. μ=1.0 is a separate higher-friction level,
        # not an easier one: some seeds fall more there than at 0.6. Every table keeps it.
        return [level for level in levels if level <= NOMINAL[axis]]
    return levels


def tables(config, rows):
    seeds = _summaries(rows, ("arm", "robot", "seed", "policy", "axis", "level"))
    order = {(axis, level): i for i, (axis, level) in enumerate(
        (c["axis"], c["level"]) for c in cells(config))}
    seeds.sort(key=lambda r: (ARMS.index(r["arm"]), r["seed"], order[(r["axis"], r["level"])]))
    arms = []
    for (arm, axis, level), group in _group(seeds, ("arm", "axis", "level")).items():
        group.sort(key=lambda r: r["seed"])
        rates = [r["never_fell_rate"] for r in group]
        recovered = [r["recovered_given_fall"] for r in group]
        arms.append({"arm": arm, "axis": axis, "level": level, "seeds": [r["seed"] for r in group],
                     "trials_per_seed": group[0]["trials"], "never_fell_by_seed": rates,
                     "never_fell_equal_seed": float(np.mean(rates)), "never_fell_min": min(rates),
                     "never_fell_max": max(rates), "recovered_given_fall_by_seed": recovered,
                     "fell_by_seed": [r["fell"] for r in group],
                     "numerical_failures": sum(r["numerical_failures"] for r in group)})
    arms.sort(key=lambda r: (ARMS.index(r["arm"]), order[(r["axis"], r["level"])]))
    shoves = [r for r in rows if r["axis"] == "shove" and r["level"] > 0]
    direction = _summaries(shoves, ("arm", "seed", "level", "sector"))
    direction.sort(key=lambda r: (ARMS.index(r["arm"]), r["seed"], r["level"], SECTORS.index(r["sector"])))
    phase = _summaries(shoves, ("arm", "seed", "level", "stance"))
    phase.sort(key=lambda r: (ARMS.index(r["arm"]), r["seed"], r["level"], r["stance"]))
    margins = []
    for arm in ARMS:
        for seed in (0, 1, 2):
            entry = {"arm": arm, "seed": seed}
            by_cell = {(r["axis"], r["level"]): r for r in seeds if r["arm"] == arm and r["seed"] == seed}
            for axis in ENVIRONMENT_AXES + ("shove",):
                levels = _ordered_levels(config, axis)
                entry[axis] = margin(levels, [by_cell[(axis, level)]["never_fell_rate"] for level in levels])
            entry["transition_worst"] = min(by_cell[("transition", kind)]["never_fell_rate"]
                                            for kind in config.transition)
            margins.append(entry)
    return {"seeds": seeds, "arms": arms, "shove_direction": direction, "shove_stance": phase,
            "margins": margins}


def _group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def _csv(path, rows, columns):
    import csv
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                         for row in rows)


LEVEL_LABELS = {
    "terrain": lambda v: "flat" if v == 0 else f"{v * 100:g} cm",
    "friction": lambda v: f"μ {v:g}", "payload": lambda v: f"+{v * 100:g}%",
    "strength": lambda v: f"{v * 100:g}%", "delay": lambda v: f"{v * 20} ms",
    "transition": lambda v: v.replace("_", " "), "shove": lambda v: f"{v:g}",
}
AXIS_TITLES = {
    "terrain": "Held-out rubble height", "friction": "Floor friction", "payload": "Payload (of body mass)",
    "strength": "Actuator strength", "delay": "Sensor delay", "transition": "Command step",
    "shove": "Shove, J/(m√(gh))",
}


def heatmap(path, config, seeds):
    """Never-fell share for every policy seed (columns) and stress level (rows)."""
    columns = [(arm, seed) for arm in ARMS for seed in (0, 1, 2)]
    by = {(r["arm"], r["seed"], r["axis"], r["level"]): r for r in seeds}
    rows = [(c["axis"], c["level"]) for c in cells(config)]
    cell_w, cell_h, left, top = 46, 17, 190, 64
    width, height = left + cell_w * len(columns) + 20, top + cell_h * (len(rows) + len(AXES)) + 40
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="11">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="16" font-size="13">Share of episodes in which the robot never fell '
             f'(simulation, {config.trials} per cell)</text>']
    for index, (arm, seed) in enumerate(columns):
        x = left + index * cell_w + cell_w / 2
        if seed == 0:
            parts.append(f'<text x="{x + cell_w}" y="36" text-anchor="middle">{html.escape(ARM_LABELS[arm])}</text>')
        parts.append(f'<text x="{x}" y="52" text-anchor="middle" fill="#555">seed {seed}</text>')
    y = top
    for axis in AXES:
        parts.append(f'<text x="8" y="{y + 12}" font-weight="600">{html.escape(AXIS_TITLES[axis])}</text>')
        y += cell_h
        for level in getattr(config, axis):
            parts.append(f'<text x="{left - 8}" y="{y + 12}" text-anchor="end">'
                         f'{html.escape(LEVEL_LABELS[axis](level))}</text>')
            for index, (arm, seed) in enumerate(columns):
                rate = by[(arm, seed, axis, level)]["never_fell_rate"]
                # Red at 0%, green at 100%.
                fill = f"rgb({round(222 - 102 * rate)},{round(90 + 110 * rate)},{round(90 + 50 * rate)})"
                x = left + index * cell_w
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{fill}"/>')
                parts.append(f'<text x="{x + cell_w / 2 - 1}" y="{y + 12}" text-anchor="middle">{rate * 100:.0f}</text>')
            y += cell_h
    parts.append(f'<text x="8" y="{height - 12}" fill="#555">Percent of episodes; per-seed 95% Wilson '
                 'intervals and fall-and-recovery counts are in atlas_seeds.csv.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


SEED_COLUMNS = ("arm", "robot", "seed", "policy", "axis", "level", "trials", "never_fell", "never_fell_rate",
                "never_fell_ci_low", "never_fell_ci_high", "fell", "fell_recovered", "recovered_given_fall",
                "recovered_ci_low", "recovered_ci_high", "numerical_failures")


def report(root, output, destination=None):
    """Write the tables and figure; return the summary and the README block."""
    manifest, config, rows = load(root, output)
    episodes = check_nominal(rows, config)
    result = tables(config, rows)
    directory = Path(destination or output / "report")
    directory.mkdir(parents=True, exist_ok=True)
    _csv(directory / "atlas_seeds.csv", result["seeds"], SEED_COLUMNS)
    _csv(directory / "atlas_arms.csv", result["arms"],
         ("arm", "axis", "level", "seeds", "trials_per_seed", "never_fell_by_seed", "never_fell_equal_seed",
          "never_fell_min", "never_fell_max", "fell_by_seed", "recovered_given_fall_by_seed", "numerical_failures"))
    for name in ("shove_direction", "shove_stance"):
        key = "sector" if name == "shove_direction" else "stance"
        _csv(directory / f"{name}.csv", result[name],
             ("arm", "seed", "level", key, "trials", "never_fell", "never_fell_rate", "never_fell_ci_low",
              "never_fell_ci_high", "fell", "fell_recovered", "numerical_failures"))
    _csv(directory / "margins.csv", result["margins"], ("arm", "seed", *ENVIRONMENT_AXES, "shove", "transition_worst"))
    heatmap(directory / "atlas.svg", config, result["seeds"])
    summary = {
        "schema_version": 1, "config_sha256": manifest["config_sha256"], "episodes": len(rows),
        "nominal_episodes_checked": episodes, "raw": manifest["raw"],
        "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES},
        "policies": manifest["policies"], "margin_threshold": MARGIN,
        "notes": ["Simulation only: standard MuJoCo, deterministic actions, no observation noise.",
                  "Intervals are per-seed 95% Wilson score intervals over trials.",
                  "Fall-and-recovery applies to Go1 only, using the get-up policy the page uses."],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary, readme_block(config, manifest, rows, result)


def _percent(value):
    return f"{value * 100:.0f}%"


def _margin_text(axis, value):
    return "below nominal" if value is None else LEVEL_LABELS[axis](value)


def readme_block(config, manifest, rows, result):
    robots = manifest["policies"]
    go1, g1 = robots["go1-walk-s0"], robots["g1-walk-s0"]
    kick = robots["go1-walk-push-s0"]["training_kick_max_normalized"]
    lines = [
        f"**Robustness atlas (simulation).** I ran the {len(policy_names())} released walking policies, three "
        f"seeds each of the flat-ground dog, the dog trained with shoves, the rough-ground dog and the humanoid, "
        f"through {len(rows):,} episodes in standard MuJoCo without retraining. The stresses are held-out rubble, "
        f"floor friction, payload, weaker motors, sensor delay, sudden full-range command changes, and shoves from "
        f"every direction at different points in the gait. Each cell has {config.trials} episodes per seed. "
        f"I paired each trial's random draws across policies and levels. I scaled commands to each robot's range "
        f"and shoves by its mass and centre-of-mass height. Training never varied friction, mass, motor strength or sensor "
        f"delay, but it did include sudden command changes. The rough-ground dog trained on 5 cm rubble of the "
        f"same kind as the 5 cm level, and the shove-trained dog's training kicks reached at most {kick:.2f} in "
        f"the normalized units below.",
        "",
        f"The table gives, per seed, the hardest level at which the robot never fell in at least "
        f"{_percent(MARGIN)} of episodes, and in at least {_percent(MARGIN)} at every easier level.",
        "",
        "| Stress (levels tested) | " + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    by = {(m["arm"], m["seed"]): m for m in result["margins"]}
    for axis in ENVIRONMENT_AXES + ("shove",):
        levels = ", ".join(LEVEL_LABELS[axis](v) for v in _ordered_levels(config, axis))
        values = [" / ".join(_margin_text(axis, by[(arm, seed)][axis]) for seed in (0, 1, 2)) for arm in ARMS]
        label = "Low-friction stress" if axis == "friction" else AXIS_TITLES[axis]
        lines.append(f"| {label} ({levels}) | " + " | ".join(values) + " |")
    worst = [" / ".join(_percent(by[(arm, seed)]["transition_worst"]) for seed in (0, 1, 2)) for arm in ARMS]
    lines.append(f"| Command step: share that never fell in the worst of the {len(config.transition)} "
                 "full-range changes | " + " | ".join(worst) + " |")
    recovery = []
    for arm in ARMS[:3]:
        counts = []
        for seed in (0, 1, 2):
            fell = [r for r in rows if r["arm"] == arm and r["seed"] == seed
                    and r["outcome"] in ("fell", "fell_recovered")]
            counts.append(f"{sum(r['outcome'] == 'fell_recovered' for r in fell)}/{len(fell)}")
        recovery.append(f"{ARM_LABELS[arm].split(', ')[1]} {' / '.join(counts)}")
    lines += [
        "",
        f"Shoves are normalized by body mass m and nominal centre-of-mass height h, so 1 means J = m√(gh): "
        f"{go1['reference_impulse_ns']:.1f} N·s for Go1 ({go1['mass_kg']:.1f} kg, h = {go1['com_height_m']:.2f} m) and "
        f"{g1['reference_impulse_ns']:.1f} N·s for G1 ({g1['mass_kg']:.1f} kg, h = {g1['com_height_m']:.2f} m). "
        f"That removes size, not every difference: both robots get the same 0.1 s pulse while walking at the same "
        f"0.5 m/s, which isn't dynamically identical for a 0.26 m and a 0.67 m tall centre of mass. Read the "
        f"dog-versus-humanoid comparison with that in mind. "
        f"When a dog fell, I handed it to the get-up policy the page uses ({config.recovery_policy}) under the same "
        f"stress. Falls after which it was back on its feet within {config.recovery_seconds:g} s, per seed: "
        f"{'; '.join(recovery)}. Those episodes still count as falls in the table. The humanoid has no get-up policy.",
        "",
        "![Never-fell share for every policy seed and stress level](results/atlas/report/atlas.svg)",
        "",
        "[Per seed with 95% intervals](results/atlas/report/atlas_seeds.csv) · "
        "[margins](results/atlas/report/margins.csv) · "
        "[shove direction](results/atlas/report/shove_direction.csv) · "
        "[shove gait phase](results/atlas/report/shove_stance.csv) · [raw episodes](results/atlas/) · "
        "[clean replay](results/atlas/clean-replay.json) · [protocol](docs/evaluation.md#robustness-atlas)",
    ]
    return "\n".join(lines)


def render_readme(text, block):
    start, end = "<!-- generated:atlas:start -->", "<!-- generated:atlas:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("README must contain one atlas block")
    before, _, remainder = text.partition(start)
    _, _, after = remainder.partition(end)
    return before + start + "\n" + block + "\n" + end + after


def rerun_sample(rows, count):
    """A fixed sample spread over every axis, outcome, policy and trial, never a nominal or control level."""
    baseline = dict(NOMINAL, shove=0.0)
    groups = defaultdict(list)
    for row in rows:
        if row["level"] != baseline.get(row["axis"]):
            groups[(AXES.index(row["axis"]), OUTCOMES.index(row["outcome"]))].append(row)
    keys = sorted(groups)
    per_group = -(-count // len(keys))
    picks = {}
    for key in keys:
        group, taken = groups[key], min(per_group, len(groups[key]))
        picks[key] = [group[(2 * k + 1) * len(group) // (2 * taken)] for k in range(taken)]
    sample = [picks[key][k] for k in range(per_group) for key in keys if k < len(picks[key])]
    return sample[:count]


def rerun(root, output, count, workers):
    """Re-simulate rerun_sample(); return the sample and the rows that changed."""
    manifest, config, rows = load(root, output)
    sample = rerun_sample(rows, count)
    selections = defaultdict(list)
    for row in sample:
        selections[row["policy"]].append((row["cell_sha256"], row["trial"]))
    tasks = [(root, asdict(config), name, selection) for name, selection in selections.items()]
    with get_context("spawn").Pool(workers) as pool:
        repeated = {}
        for _, found in pool.imap_unordered(simulate_policy, tasks):
            repeated.update({(r["policy"], r["cell_sha256"], r["trial"]): r for r in found})
    pairs = [(row, repeated.get((row["policy"], row["cell_sha256"], row["trial"]))) for row in sample]
    return sample, [(row, again) for row, again in pairs if again != row]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "results/atlas")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--trials", type=int, help="Smaller smoke runs; publish only the default")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--rerun", type=int, metavar="N",
                        help="Re-simulate N committed episodes; fail if any outcome changes")
    parser.add_argument("--exact", action="store_true", help="With --rerun, also fail on any changed field")
    parser.add_argument("--no-readme", action="store_true", help="Leave README.md unchanged")
    args = parser.parse_args()
    if args.trials and not args.no_readme:
        parser.error("--trials is for smoke runs; add --no-readme")
    output = args.output.resolve()
    if args.rerun:
        sample, differing = rerun(ROOT, output, args.rerun, args.workers)
        for row in sample:
            print(f"RERUN {row['policy']} {row['axis']}={row['level']} trial {row['trial']} ({row['outcome']})")
        checked, changed = len(sample), 0
        for row, again in differing:
            fields = sorted(k for k in row.keys() | (again or {}).keys() if again is None or row.get(k) != again.get(k))
            changed += again is None or row["outcome"] != again["outcome"]
            print(f"DIFFERS {row['policy']} {row['axis']}={row['level']} trial {row['trial']}: {', '.join(fields)}")
        print(f"Re-simulated {checked} committed episodes: {checked - len(differing)} identical rows, "
              f"{checked - changed} with the same outcome.")
        raise SystemExit(1 if changed or (args.exact and differing) else 0)
    if not args.report_only:
        config = AtlasConfig(**({"trials": args.trials} if args.trials else {}))
        simulate(ROOT, output, config, args.workers)
    summary, block = report(ROOT, output)
    if not args.no_readme:
        readme = ROOT / "README.md"
        readme.write_text(render_readme(readme.read_text(), block))
    print(f"{summary['episodes']} episodes; outcomes {summary['outcomes']}; report in {output / 'report'}")


if __name__ == "__main__":
    main()
