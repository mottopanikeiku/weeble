"""The browser fall scene changes contacts, not the trained G1 mechanism."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def models():
    return tuple(mujoco.MjModel.from_xml_path(str(ROOT / f"assets/g1/{name}.xml"))
                 for name in ("walk", "fallen"))


def test_fallen_preserves_robot_dynamics(models):
    walk, fallen = models
    assert (walk.nq, walk.nv, walk.nu, walk.nbody, walk.ngeom) == (
        fallen.nq, fallen.nv, fallen.nu, fallen.nbody, fallen.ngeom)
    fields = [name for name in dir(walk) if name.startswith(("body_", "jnt_", "dof_", "actuator_"))
              and isinstance(getattr(walk, name), np.ndarray)]
    # Name-table offsets change because formerly unnamed collision copies now
    # have names; no body/joint/actuator array contains geom-name offsets.
    for name in fields + ["qpos0", "qpos_spring", "key_qpos", "key_qvel", "key_ctrl"]:
        np.testing.assert_array_equal(getattr(walk, name), getattr(fallen, name), err_msg=name)
    for name in ("timestep", "iterations", "ls_iterations", "ccd_iterations", "integrator",
                 "solver", "cone", "disableflags", "enableflags", "gravity"):
        np.testing.assert_array_equal(getattr(walk.opt, name), getattr(fallen.opt, name), err_msg=name)
    for name in ("geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size",
                 "geom_contype", "geom_conaffinity", "geom_dataid"):
        np.testing.assert_array_equal(getattr(walk, name), getattr(fallen, name), err_msg=name)


def test_fallen_airborne_trajectory_unchanged(models):
    states = []
    for model in models:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[2] += 2
        data.ctrl[:] = data.qpos[7:]
        mujoco.mj_forward(model, data)
        for _ in range(25):
            mujoco.mj_step(model, data)
        states.append(data)
    np.testing.assert_array_equal(states[0].qpos, states[1].qpos)
    np.testing.assert_array_equal(states[0].qvel, states[1].qvel)


@pytest.mark.parametrize("quaternion", [[0, 1, 0, 0], [2**-.5, 0, 2**-.5, 0]])
def test_fallen_body_is_supported_by_floor(models, quaternion):
    model = models[1]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 1.3
    data.qpos[3:7] = quaternion
    data.ctrl[:] = data.qpos[7:]
    mujoco.mj_forward(model, data)
    floor = model.geom("floor").id
    feet = {model.geom(name).id for name in ("left_foot", "right_foot")}
    body_contact = False
    for _ in range(1000):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            body_contact |= floor in pair and not (pair & feet)
    assert body_contact
    assert data.qpos[2] > 0.03
    assert data.xpos[model.body("torso_link").id, 2] > 0.03
