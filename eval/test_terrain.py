"""Cross-terrain trials preserve the mechanism and retain post-fall samples."""
from pathlib import Path

import mujoco
import numpy as np

from eval.sim import Policy
from eval.terrain import CONFIG, SCENES, TerrainSimulator, assert_same_robot

ROOT = Path(__file__).resolve().parents[1]


def test_terrain_scene_changes_only_ground():
    models = [mujoco.MjModel.from_xml_path(str(ROOT / path)) for path in SCENES.values()]
    assert_same_robot(*models)
    assert models[0].geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE
    assert models[1].geom("floor").type == mujoco.mjtGeom.mjGEOM_HFIELD
    np.testing.assert_array_equal(models[1].hfield_size[0], [10, 10, .05, 1])


def test_tracking_records_fall_without_stopping_physics():
    policy = Policy.load(ROOT / "policies/go1-walk-s0.json")
    sim = TerrainSimulator(policy, ROOT, CONFIG)
    pose = np.asarray(policy.spec["default_qpos"], dtype=float)
    pose[2] = 1.0
    pose[3:7] = [0, 1, 0, 0]
    sim.reset(qpos=pose)
    sim.begin(fall_checks=False)
    sim._inspect()
    assert sim.fallen and sim.failure is None
    sim.checked_step(command=[0, 0, 0])
    assert sim.data.time == policy.spec["control_dt"]
    assert sim.failure is None
    sim._fail("nonfinite_state")
    assert sim.failure == "nonfinite_state"
    sim.begin(fall_checks=False)
    assert not sim.fallen and sim.failure is None
