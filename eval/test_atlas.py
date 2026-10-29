"""Atlas stresses change only what they name; bookkeeping is hashed, paired and exact."""
from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from eval.atlas import (
    AXES, ROOT, AtlasConfig, AtlasSimulator, Episodes, NOMINAL, SCENES, _write_rows, cells, check_nominal,
    environment_draws, load, margin, read_rows, rerun_sample, shove_draws, stance, terrain_model, wilson,
)
from eval.sim import Policy
from eval.terrain import assert_same_robot


def _policy(name):
    policy = Policy.load(ROOT / "policies" / f"{name}.json")
    return Policy(dict(policy.spec, model_xml=SCENES[policy.spec["robot"]]))


def test_each_cell_hash_covers_its_level_and_the_shared_protocol():
    base = {(c["axis"], c["level"]): c["sha256"] for c in cells(AtlasConfig())}
    changed = {(c["axis"], c["level"]): c["sha256"]
               for c in cells(AtlasConfig(friction=(1.0, 0.6, 0.4, 0.25, 0.1)))}
    differing = {key for key in base.keys() | changed.keys() if base.get(key) != changed.get(key)}
    assert differing == {("friction", 0.15), ("friction", 0.1)}
    longer = {c["sha256"] for c in cells(AtlasConfig(episode_seconds=12.0))}
    assert not longer & set(base.values())


def test_trial_draws_are_shared_by_every_policy_and_level():
    config = AtlasConfig()
    first, second = environment_draws(config, 3), environment_draws(config, 3)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert np.all(np.abs(first[1]) <= config.command_fraction)
    assert environment_draws(config, 4)[0] != first[0]
    assert shove_draws(config, 3) == shove_draws(config, 3)
    assert shove_draws(replace(config, trial_seed=1), 3) != shove_draws(config, 3)


def test_config_requires_nominal_levels():
    for axis in NOMINAL:
        values = tuple(v for v in getattr(AtlasConfig(), axis) if v != NOMINAL[axis])
        with pytest.raises(ValueError, match=axis):
            replace(AtlasConfig(), **{axis: values}).validate()


@pytest.mark.parametrize("name", ["go1-walk-s0", "g1-walk-s0"])
def test_friction_sets_every_floor_contact(name):
    policy = _policy(name)
    sim = AtlasSimulator(policy, ROOT, None, "friction", 0.25)
    sim.reset()
    for _ in range(10):
        sim.step(command=(0.3, 0, 0))
    floor = sim.model.geom("floor").id
    frictions = [c.friction[0] for c in sim.data.contact if floor in (c.geom1, c.geom2)]
    assert frictions and all(f == pytest.approx(0.25) for f in frictions)


@pytest.mark.parametrize("name", ["go1-walk-s0", "g1-walk-s0"])
def test_payload_adds_a_fraction_of_body_mass_at_the_base(name):
    policy = _policy(name)
    nominal = AtlasSimulator(policy, ROOT, None)
    loaded = AtlasSimulator(policy, ROOT, None, "payload", 0.3)
    base = nominal.model.body(policy.spec["force_body"]).id
    total = nominal.model.body_subtreemass[base]
    assert loaded.model.body_subtreemass[base] == pytest.approx(1.3 * total)
    assert loaded.model.body_mass[base] - nominal.model.body_mass[base] == pytest.approx(0.3 * total)
    np.testing.assert_array_equal(np.delete(loaded.model.body_mass, base), np.delete(nominal.model.body_mass, base))


@pytest.mark.parametrize("name", ["go1-walk-s0", "g1-walk-s0"])
def test_strength_scales_actuator_torque_and_its_limit(name):
    policy = _policy(name)
    torques = {}
    for scale in (1.0, 0.55):
        sim = AtlasSimulator(policy, ROOT, None, "strength", scale)
        sim.reset()
        sim.data.ctrl[:] = sim.data.qpos[7:] + 0.01
        mujoco.mj_forward(sim.model, sim.data)
        torques[scale] = sim.data.qfrc_actuator[6:].copy()
        limits = np.concatenate([sim.model.actuator_forcerange.ravel(), sim.model.jnt_actfrcrange.ravel()])
        torques[f"limit {scale}"] = limits
    np.testing.assert_allclose(torques[0.55], 0.55 * torques[1.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(torques["limit 0.55"], 0.55 * torques["limit 1.0"], rtol=1e-12)
    assert np.any(torques[1.0] != 0)


def test_nominal_stresses_leave_the_model_unchanged():
    policy = _policy("go1-walk-s0")
    plain = AtlasSimulator(policy, ROOT, None)
    for axis, level in NOMINAL.items():
        if axis == "terrain":
            continue
        stressed = AtlasSimulator(policy, ROOT, None, axis, level)
        for field in ("body_mass", "body_invweight0", "dof_invweight0", "geom_friction", "pair_friction",
                      "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
            np.testing.assert_array_equal(getattr(stressed.model, field), getattr(plain.model, field), err_msg=field)


def test_delay_feeds_old_sensor_readings_but_current_action_and_command():
    policy = _policy("go1-walk-s0")
    fields = {f["name"]: slice(f["offset"], f["offset"] + f["size"]) for f in policy.spec["observation"]["fields"]}
    sim = AtlasSimulator(policy, ROOT, None, "delay", 2)
    sim.reset()
    seen, fresh = [], []
    act = sim.policy.action

    def record(observation):
        seen.append(np.array(observation))
        return act(observation)

    sim.policy.action = record
    for _ in range(5):
        fresh.append(sim.observe(command=(0.4, 0.1, 0.2)))
        sim.checked_step(command=(0.4, 0.1, 0.2))
    for step, observation in enumerate(seen):
        source = fresh[max(0, step - 2)]
        for name in ("gyro", "gravity", "joint_position", "joint_velocity", "local_linear_velocity"):
            np.testing.assert_array_equal(observation[fields[name]], source[fields[name]])
        for name in ("last_action", "command"):
            np.testing.assert_array_equal(observation[fields[name]], fresh[step][fields[name]])
    assert not np.array_equal(fresh[3][fields["joint_velocity"]], fresh[1][fields["joint_velocity"]])
    sim.reset()
    assert len(sim._history) == 0


@pytest.mark.parametrize("scene", sorted(set(SCENES.values()) | {"assets/go1/getup.xml"}))
def test_terrain_replaces_only_the_floor(scene):
    config = AtlasConfig()
    model = terrain_model(ROOT, scene, 0.1, config)
    flat = mujoco.MjModel.from_xml_path(str(ROOT / scene))
    assert_same_robot(flat, model)
    floor = model.geom("floor").id
    assert model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_HFIELD
    np.testing.assert_array_equal(model.geom_friction[floor], flat.geom_friction[flat.geom("floor").id])
    assert model.geom_pos[floor][2] == pytest.approx(-0.05)
    assert model.hfield_size[0][2] == pytest.approx(0.1)
    assert model.hfield_nrow[0] == model.hfield_ncol[0] == config.terrain_cells


def test_terrain_episode_writes_the_trial_field_to_both_controllers():
    config = AtlasConfig()
    cell = next(c for c in cells(config) if c["axis"] == "terrain" and c["level"] == 0.1)
    episodes = Episodes(ROOT, replace(config), "go1-walk-s0", cell)
    episodes.run(2)
    field = environment_draws(config, 2)[2].ravel()
    np.testing.assert_allclose(episodes.walk.model.hfield_data, field, rtol=1e-6)
    np.testing.assert_array_equal(episodes.recovery.model.hfield_data, episodes.walk.model.hfield_data)


def test_check_nominal_requires_identical_episodes():
    rows = [{"policy": "p", "trial": 0, "axis": axis, "level": level, "cell_sha256": axis, "outcome": "never_fell"}
            for axis, level in NOMINAL.items()]
    assert check_nominal(rows, AtlasConfig()) == 1
    rows[2]["outcome"] = "fell"
    with pytest.raises(ValueError, match="Nominal episodes differ"):
        check_nominal(rows, AtlasConfig())


def test_wilson_interval_and_margin():
    low, high = wilson(45, 50)
    assert low == pytest.approx(0.7864, abs=1e-4) and high == pytest.approx(0.9565, abs=1e-4)
    assert wilson(0, 50)[0] == 0 and wilson(50, 50)[1] == 1
    assert margin([0, 1, 2, 3], [1.0, 0.92, 0.95, 0.5]) == 2
    assert margin([0, 1, 2], [1.0, 0.8, 1.0]) == 0
    assert margin([0, 1], [0.5, 1.0]) is None


def test_stance_classes():
    assert stance("go1", ["FL", "RR"]) == "diagonal FL+RR"
    assert stance("go1", ["FR", "RL"]) == "diagonal FR+RL"
    assert stance("go1", ["FL", "FR", "RR"]) == "three or four feet"
    assert stance("go1", ["FL"]) == "other"
    assert stance("g1", []) == "flight"
    assert stance("g1", ["left_foot", "right_foot"]) == "double support"
    assert stance("g1", ["right_foot"]) == "right stance"


def test_raw_rows_have_reproducible_bytes(tmp_path):
    rows = [{"policy": "p", "trial": index, "heading": math.pi / (index + 1)} for index in range(3)]
    _write_rows(tmp_path / "a.jsonl.gz", rows)
    _write_rows(tmp_path / "b.jsonl.gz", rows)
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()
    assert read_rows(tmp_path / "a.jsonl.gz") == rows


def test_rerun_sample_covers_axes_outcomes_policies_and_trials():
    _, _, rows = load(ROOT, ROOT / "results/atlas")
    sample = rerun_sample(rows, 36)
    assert sample == rerun_sample(rows, 36)
    assert len({(r["policy"], r["cell_sha256"], r["trial"]) for r in sample}) == 36
    assert {r["axis"] for r in sample} == set(AXES)
    assert {r["outcome"] for r in sample} == {r["outcome"] for r in rows} >= {"never_fell", "fell", "fell_recovered"}
    baseline = dict(NOMINAL, shove=0.0)
    assert not [r for r in sample if r["level"] == baseline.get(r["axis"])]
    for axis in AXES:
        assert len({r["level"] for r in sample if r["axis"] == axis}) >= 2, axis
    assert len({r["trial"] for r in sample}) >= 12
    assert {r["arm"] for r in sample} == {"go1-walk", "go1-walk-push", "go1-walk-rough", "g1-walk"}
