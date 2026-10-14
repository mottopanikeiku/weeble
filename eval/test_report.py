"""Regression tests for trial blocks, settled-pose groups and run accounting."""
import json

import numpy as np
import pytest

from eval.report import ablation_table, compare_rerun, getup_start_pose, getup_table, push_tables
from train.launch import write_manifest


def test_ablation_resamples_whole_direction_blocks_across_impulses():
    # Every trial's second arm is exactly a one-unit horizontal shift. A joint
    # block bootstrap must preserve that shift, not invent interval width by
    # drawing the same trial independently at neighboring impulses.
    rows = []
    config = {"go1_impulses": list(range(6)), "push_trials": 10}
    for arm, shift in (("go1-walk", 0), ("go1-walk-push", 1)):
        for seed in range(3):
            for impulse in range(6):
                for trial in range(10):
                    threshold = (1.5 if trial < 5 else 2.5) + shift
                    rows.append({
                        "suite": "push", "backend": "mujoco", "arm": arm,
                        "robot": "go1", "seed": seed, "trial": trial,
                        "impulse": impulse, "theta": trial / 10,
                        "push_onset": 2 + trial / 50,
                        "success": impulse < threshold, "_config": config,
                        "_standard": False,
                    })
    rng = np.random.default_rng(7319)
    _, _, fits = push_tables(rows, rng, 300)
    result = ablation_table(fits, rng, 300)[0]
    assert result["paired_trials"] == 180
    assert result["unpaired_trials"] == 0
    assert result["bootstrap_censored_fraction"] == 0
    np.testing.assert_allclose(
        [result["j50_delta"], result["ci_low"], result["ci_high"]], 1, atol=1e-12,
    )


@pytest.mark.parametrize("pose, expected", [
    ([0, 0, 0.3, 1, 0, 0, 0], "already_upright"),
    ([0, 0, 0.1, 1, 0, 0, 0], "fallen"),
    ([0, 0, 0.3, 0, 2, 0, 0], "fallen"),
    ([0, 0, 0.3, 0, 0, 0, 0], "unclassified"),
    (None, "unclassified"),
])
def test_getup_classifies_settled_pose_not_drop_orientation(pose, expected):
    row = {"settled_qpos": pose, "initial_quaternion": [0, 1, 0, 0],
           "_config": {"getup_tilt_degrees": 15, "getup_height": 0.2}}
    assert getup_start_pose(row) == expected


def test_getup_keeps_fallen_and_all_start_denominators_separate():
    rows = []
    config = {"getup_tilt_degrees": 15, "getup_height": 0.2, "getup_trials": 2}
    for seed in range(3):
        for upright in (False, True):
            success = upright or seed != 2
            rows.append({
                "suite": "getup", "backend": "mujoco", "arm": "go1-getup",
                "seed": seed, "success": success,
                "settled_qpos": [0, 0, 0.3, *([1, 0, 0, 0] if upright else [0, 1, 0, 0])],
                "recovery_time": (0.1 if upright else 2.0) if success else None,
                "_config": config, "_standard": False,
            })
    table = getup_table(rows, np.random.default_rng(7319), 100)
    by_group = {r["start_pose"]: r for r in table if r["seed"] == "all"}
    assert (by_group["fallen"]["successes"], by_group["fallen"]["trials"]) == (2, 3)
    assert (by_group["all"]["successes"], by_group["all"]["trials"]) == (5, 6)
    assert by_group["fallen"]["median_recovery_seconds"] == 2
    assert by_group["all"]["median_recovery_seconds"] == 0.1
    assert all(r["complete"] for r in table)
    # Unequal fallen-group sizes make a pooled count rate differ from the
    # equal-seed estimate; both must be named rather than sharing 'success'.
    rows[3]["settled_qpos"] = [0, 0, 0.3, 0, 1, 0, 0]
    uneven = getup_table(rows, np.random.default_rng(7319), 100)
    fallen = next(r for r in uneven if r["seed"] == "all" and r["start_pose"] == "fallen")
    assert fallen["equal_seed_success"] == pytest.approx(2 / 3)
    assert fallen["count_success"] == 3 / 4


@pytest.mark.parametrize("exit_code", [0, 1])
def test_standalone_manifest_retains_remote_config_and_prices_actual_wall_time(tmp_path, exit_code):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"config": {"num_timesteps": 200000000},
                                "versions": {"mujoco": "3.15.0"}}))
    config = {"stage": "train", "env": "Go1JoystickFlatTerrain", "seed": 0,
              "gpu": "L4", "cores": 2, "mem_gib": 8, "run_id": "example", "minutes": 45}
    source = {"git_sha": "a" * 40, "git_dirty": False}
    write_manifest(path, source, config, exit_code, 1800)
    manifest = json.loads(path.read_text())
    assert manifest["config"] == {"num_timesteps": 200000000}
    assert manifest["versions"] == {"mujoco": "3.15.0"}
    assert manifest["launch_config"] == config
    assert manifest["minutes"] == 30
    assert manifest["usd_upper"] == 0.5266
    assert manifest["git_sha"] == source["git_sha"] and not manifest["git_dirty"]
    assert manifest["status"] == ("complete" if exit_code == 0 else "failed")


def test_rerun_comparison_distinguishes_state_roundoff_from_changed_outcomes(tmp_path):
    original, repeat = tmp_path / "old" / "policy", tmp_path / "new" / "policy"
    for directory in (original, repeat):
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({
            "run_id": directory.parent.name, "git_sha": "a" * 40, "git_dirty": False,
            "config": {}, "policy_metadata": {},
        }))
    old = {"trial_id": "trial-0", "run_id": "old", "wall_seconds": 1,
           "success": True, "recovery_time": 2, "initial_qpos": [0.3]}
    new = {**old, "run_id": "new", "wall_seconds": 9, "initial_qpos": [0.3 + 1e-15]}
    before, after = original / "trials.jsonl", repeat / "trials.jsonl"
    before.write_text(json.dumps(old) + "\n")
    after.write_text(json.dumps(new) + "\n")
    result = compare_rerun(before, after)
    assert result["outcomes_equal"]
    assert set(result["differing_fields"]) == {"initial_qpos"}
    assert 0 < result["differing_fields"]["initial_qpos"]["max_abs"] < 1e-14
    new["success"] = False
    after.write_text(json.dumps(new) + "\n")
    result = compare_rerun(before, after)
    assert not result["outcomes_equal"]
    assert result["successes_original"] == 1 and result["successes_repeat"] == 0
