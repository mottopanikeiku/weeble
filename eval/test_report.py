"""Regression tests for trial blocks, settled-pose groups and run accounting."""
import csv
import gzip
import hashlib
from itertools import product
import json

import numpy as np
import pytest

from eval.report import (
    ablation_table, compare_rerun, generate_report, getup_start_pose, getup_table,
    load_terrain_trials, load_trials, push_tables, survival_data, terrain_readme_lines,
    terrain_tables,
)
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


def _raw_trials(root, suite, arm, seed, rows, *, run="example", status="complete", compressed=False):
    policy = f"{arm}-s{seed}"
    directory = root / "results" / ("terrain" if suite == "terrain" else "eval") / run
    directory = directory / policy if suite == "terrain" else directory / "mujoco" / policy
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": suite, "run_id": run, "backend": "mujoco", "seeds": [seed],
        "status": status, "git_sha": "a" * 40, "git_dirty": False, "wall_seconds": 1,
        "modal_app_id": None, "config": {"seeds": [seed]},
        "policy_metadata": {policy: {"policy_sha256": arm + str(seed), "model_sha256": "b" * 64}},
    }
    if suite == "push":
        manifest["config"].update(go1_impulses=[0, 2], g1_impulses=[0, 2], push_trials=2)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    path = directory / ("trials.jsonl.gz" if compressed else "trials.jsonl")
    opener = gzip.open if compressed else open
    with opener(path, "wt") as stream:
        for index, row in enumerate(rows):
            record = {
                "run_id": run, "suite": suite, "backend": "mujoco", "policy": policy,
                "arm": arm, "seed": seed, "trial_id": f"{policy}/{suite}/{index}",
                "wall_seconds": 0.1, **row,
            }
            stream.write(json.dumps(record) + "\n")
    return path


def _push_rows(seed):
    return [
        {"robot": "go1", "impulse": impulse, "trial": trial, "theta": trial / 2,
         "push_onset": 2 + trial / 10, "command": [0, 0, 0],
         "success": (impulse == 2) != (seed == 0 and trial == 0)}
        for impulse in (0, 2) for trial in range(2)
    ]


def _terrain_row(terrain="flat", command=(0, 0, 0), **changes):
    return {
        "terrain": terrain, "command": list(command), "success": True, "failure": None,
        "completed": True, "forward_mae": 1.0, "lateral_mae": 2.0, "yaw_mae": 3.0,
        **changes,
    }


def test_report_emits_raw_survival_counts_seeds_and_accepted_sources(tmp_path):
    sources = {}
    for arm in ("go1-walk", "go1-walk-push", "g1-walk"):
        sources[arm] = []
        for seed in range(3):
            path = _raw_trials(tmp_path, "push", arm, seed, _push_rows(seed), compressed=seed == 2)
            sources[arm].append(path.relative_to(tmp_path).as_posix())
    _raw_trials(tmp_path, "push", "go1-walk", 0, _push_rows(0), run="old", status="superseded")
    terrain_path = _raw_trials(tmp_path, "terrain", "go1-walk", 0, [_terrain_row()])
    output = tmp_path / "results/report"
    summary = generate_report([tmp_path / "results"], output, root=tmp_path, bootstrap_replicates=3)
    data = json.loads((tmp_path / "web/src/data/survival.json").read_text())
    assert data["schema_version"] == 1
    assert set(data["policies"]) == {"go1-walk-s0", "go1-walk-push-s0", "g1-walk-s0"}
    for arm, paths in sources.items():
        policy = data["policies"][f"{arm}-s0"]
        assert policy["source"] == sorted(paths)
        assert policy["impulse_unit"] == "N·s"
        assert policy["trials"] == [
            {"impulse": 0, "survived": 1, "total": 6, "seeds": [0, 1, 2], "survival_fraction": 1 / 6},
            {"impulse": 2, "survived": 5, "total": 6, "seeds": [0, 1, 2], "survival_fraction": 5 / 6},
        ]
    assert summary["trial_count"] == 36
    assert summary["terrain_trial_count"] == 1
    assert summary["tables"]["terrain"][0]["trials"] == 1
    assert len(summary["tables"]["push"]) == 6
    terrain_source = next(r for r in summary["inputs"] if r.get("suite") == "terrain")
    assert terrain_source["raw"] == terrain_path.relative_to(tmp_path).as_posix()
    assert terrain_source["raw_sha256"] == hashlib.sha256(terrain_path.read_bytes()).hexdigest()
    manifest_path = terrain_path.parent / "manifest.json"
    assert terrain_source["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with (output / "terrain.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["trials"] == "1"
    assert (output / "terrain_seeds.csv").is_file()
    assert json.loads((output / "summary.json").read_text())["terrain_trial_count"] == 1


def test_survival_uses_only_present_impulses_and_mujoco_push_rows():
    rows = [
        {"suite": "push", "backend": "mujoco", "arm": "go1-walk", "seed": seed,
         "impulse": impulse, "success": success, "_source": source}
        for seed, impulse, success, source in (
            (0, 0, True, "results/a/trials.jsonl"), (2, 0, False, "results/b/trials.jsonl"),
            (2, 8, True, "results/b/trials.jsonl"),
        )
    ]
    rows += [{**rows[0], "backend": "mjx"}, {**rows[0], "suite": "tracking"}]
    data = survival_data(rows)["policies"]["go1-walk-s0"]
    assert [r["impulse"] for r in data["trials"]] == [0, 8]
    assert [r["total"] for r in data["trials"]] == [2, 1]
    assert [r["seeds"] for r in data["trials"]] == [[0, 2], [2]]
    assert [r["survival_fraction"] for r in data["trials"]] == [0.5, 1]
    assert data["source"] == ["results/a/trials.jsonl", "results/b/trials.jsonl"]


def test_terrain_keeps_falls_and_numerical_failures_in_denominators(tmp_path):
    paths = [
        _raw_trials(tmp_path, "terrain", "go1-walk", 0, [
            _terrain_row(),
            _terrain_row(success=False, failure="fall", forward_mae=5.0),
            _terrain_row(success=False, failure="nonfinite_state", completed=False,
                         forward_mae=None, lateral_mae=None, yaw_mae=None),
        ], compressed=True),
        _raw_trials(tmp_path, "terrain", "go1-walk", 1, [
            _terrain_row(forward_mae=9.0),
            _terrain_row(success=False, failure="mujoco_warning", completed=False,
                         forward_mae=13.0),
        ]),
    ]
    rows, _ = load_terrain_trials(paths, root=tmp_path)
    pooled, seeds = terrain_tables(rows)
    row = pooled[0]
    assert row["seeds"] == [0, 1]
    assert (row["trials"], row["completed"], row["successes"], row["falls"]) == (5, 3, 2, 1)
    assert row["fall_rate"] == 1 / 5
    assert row["numerical_failures"] == 2
    assert row["failures"] == {"fall": 1, "mujoco_warning": 1, "nonfinite_state": 1}
    assert row["forward_mae_trials"] == 4
    assert row["forward_mae"] == 7
    assert row["lateral_mae_trials"] == 4
    assert row["lateral_mae"] == 2
    assert not row["complete_grid"] and not row["three_seed_protocol"]
    assert [r["trials"] for r in seeds] == [3, 2]
    assert [r["forward_mae"] for r in seeds] == [3, 11]
    assert terrain_readme_lines(pooled) == []


def test_terrain_three_seed_table_requires_all_four_complete_cells(tmp_path):
    for arm in ("go1-walk", "go1-walk-rough"):
        for seed in range(3):
            rows = [
                _terrain_row(terrain, command, success=not (terrain == "rough" and command == (0, 0, 0)),
                             failure="fall" if terrain == "rough" and command == (0, 0, 0) else None)
                for terrain in ("flat", "rough") for command in product((-1, 0, 1), repeat=3)
            ]
            _raw_trials(tmp_path, "terrain", arm, seed, rows, compressed=seed == 2)
    rows, _ = load_terrain_trials([tmp_path / "results/terrain"], root=tmp_path)
    pooled, seeds = terrain_tables(rows)
    assert len(pooled) == 4 and len(seeds) == 12
    assert all(r["three_seed_protocol"] and r["complete_grid"] for r in pooled)
    assert all(r["trials"] == 81 and r["seeds"] == [0, 1, 2] for r in pooled)
    assert [r["falls"] for r in pooled] == [0, 3, 0, 3]
    text = "\n".join(terrain_readme_lines(pooled))
    assert "| Flat | 0/81 (1.000 / 2.000 / 3.000) | 3/81 (1.000 / 2.000 / 3.000) |" in text
    assert "| Rough | 0/81 (1.000 / 2.000 / 3.000) | 3/81 (1.000 / 2.000 / 3.000) |" in text
    partial, _ = terrain_tables(rows[:-1])
    assert terrain_readme_lines(partial) == []
    assert terrain_readme_lines(pooled[:-1]) == []


@pytest.mark.parametrize("suite", ["push", "terrain"])
def test_duplicate_trial_ids_rejected(tmp_path, suite):
    rows = _push_rows(0) if suite == "push" else [_terrain_row()]
    path = _raw_trials(tmp_path, suite, "go1-walk", 0, rows)
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    loader = load_trials if suite == "push" else load_terrain_trials
    with pytest.raises(ValueError, match="Duplicate"):
        loader([path], root=tmp_path)


@pytest.mark.parametrize("superseded_at", ["policy", "invocation"])
def test_terrain_excludes_superseded_inputs_only(tmp_path, superseded_at):
    path = _raw_trials(tmp_path, "terrain", "go1-walk", 0, [_terrain_row()])
    if superseded_at == "policy":
        target = path.parent / "manifest.json"
        manifest = json.loads(target.read_text())
    else:
        target = path.parent.parent / "manifest.json"
        manifest = {"run_id": "example"}
    target.write_text(json.dumps({**manifest, "status": "superseded"}))
    rows, sources = load_terrain_trials([path], root=tmp_path)
    assert rows == [] and sources == []


def test_terrain_rejects_completed_episode_without_tracking_measurements(tmp_path):
    path = _raw_trials(tmp_path, "terrain", "go1-walk", 0, [_terrain_row(forward_mae=None)])
    with pytest.raises(ValueError, match="Completed terrain trial lacks forward_mae"):
        load_terrain_trials([path], root=tmp_path)
