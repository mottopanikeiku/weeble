"""Paired flat/rough walking cross-test defined in docs/evaluation.md."""
from dataclasses import asdict
from importlib import metadata
import itertools
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from eval.run import SuiteConfig, TrialSimulator, _hash, _safe_json, _tracking, _write_manifest
from eval.sim import Policy

CONFIG = SuiteConfig(suites=("tracking",), go1_ranges=((-0.5, 0.5), (-0.3, 0.3), (-0.6, 0.6)))
SCENES = {"flat": "assets/go1/walk.xml", "rough": "assets/go1/rough.xml"}
ARMS = ("go1-walk", "go1-walk-rough")


class TerrainSimulator(TrialSimulator):
    """Record physical falls without truncating the tracking measurement."""

    def begin(self, *, fall_checks):
        super().begin(fall_checks=True)
        self.fallen = False
        self.fall_time = None

    def _fail(self, reason, when=None):
        if reason in ("tilt", "nonfoot_floor_contact"):
            self.fallen = True
            if self.fall_time is None:
                self.fall_time = float(self.data.time if when is None else when)
        else:
            super()._fail(reason, when)


def assert_same_robot(flat, rough):
    """A scene swap may not silently change the robot mechanism."""
    for name in dir(flat):
        if name.startswith(("body_", "jnt_", "dof_", "actuator_")):
            value = getattr(flat, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(rough, name), err_msg=name)
    for name in ("timestep", "iterations", "ls_iterations", "ccd_iterations", "integrator",
                 "solver", "cone", "disableflags", "enableflags", "gravity"):
        np.testing.assert_array_equal(getattr(flat.opt, name), getattr(rough.opt, name), err_msg=name)


def run_terrain(root, output, provenance, policy_filter=""):
    root, output = Path(root), Path(output)
    CONFIG.validate()
    models = [mujoco.MjModel.from_xml_path(str(root / path)) for path in SCENES.values()]
    assert_same_robot(*models)
    names = [f"{arm}-s{seed}" for arm in ARMS for seed in CONFIG.seeds]
    if policy_filter:
        if policy_filter not in names:
            raise ValueError(f"Unknown terrain policy {policy_filter}")
        names = [policy_filter]
    run_id = provenance["run_id"]
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        **provenance, "run_id": run_id, "backend": "mujoco", "gpu": "none",
        "config": {**asdict(CONFIG), "initial_height": 0.35, "scenes": SCENES},
        "policies": names, "seeds": list(CONFIG.seeds), "status": "running", "trial_count": 0,
        "package_versions": {name: metadata.version(name) for name in ("numpy", "mujoco")},
        "python": platform.python_version(), "policy_metadata": {},
    }
    started = time.monotonic()
    try:
        for name in names:
            policy_path = root / "policies" / f"{name}.json"
            spec = json.loads(policy_path.read_text())
            arm, seed = name.rsplit("-s", 1)
            child = directory / name
            child.mkdir()
            current = {**manifest, "policies": [name], "seeds": [int(seed)], "trial_count": 0,
                       "policy_metadata": {name: {"policy_sha256": _hash(policy_path),
                           "models": {terrain: {"path": path, "sha256": _hash(root / path)}
                                      for terrain, path in SCENES.items()}, "training_run": spec["run_id"]}}}
            _write_manifest(child / "manifest.json", current)
            with (child / "trials.jsonl").open("x") as raw:
                for terrain, scene in SCENES.items():
                    effective = dict(spec, model_xml=scene, default_qpos=list(spec["default_qpos"]))
                    effective["default_qpos"][:3] = [0.0, 0.0, 0.35]
                    sim = TerrainSimulator(Policy(effective), root, CONFIG)
                    commands = itertools.product(*(np.linspace(lo, hi, CONFIG.velocity_points).tolist()
                                                   for lo, hi in CONFIG.go1_ranges))
                    for index, command in enumerate(commands):
                        row = {"run_id": run_id, "trial_id": f"{name}/{terrain}/{index}",
                               "suite": "terrain", "backend": "mujoco", "policy": name,
                               "seed": int(seed), "arm": arm, "terrain": terrain, "trial": index,
                               "command": list(command), "success": False, "completed": False,
                               "failure": None, "error": None}
                        trial_started = time.monotonic()
                        try:
                            _tracking(sim, CONFIG, command, row)
                            row.update(completed=True, success=not sim.fallen,
                                       failure="fall" if sim.fallen else None,
                                       failure_time=sim.fall_time)
                        except Exception as exc:
                            row.update(failure=sim.failure or "simulator_error",
                                       error=f"{type(exc).__name__}: {exc}", failure_time=sim.failure_time)
                        row["wall_seconds"] = time.monotonic() - trial_started
                        row["sim_seconds"] = float(sim.data.time)
                        row["fell"] = sim.fallen
                        row["final_xy"] = sim.data.qpos[:2].tolist()
                        raw.write(json.dumps(_safe_json(row), allow_nan=False) + "\n")
                        raw.flush()
                        current["trial_count"] += 1
                        manifest["trial_count"] += 1
            current.update(status="complete", wall_seconds=time.monotonic() - started)
            _write_manifest(child / "manifest.json", current)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        _write_manifest(directory / "manifest.json", manifest)
    return directory
