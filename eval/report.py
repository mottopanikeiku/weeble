"""Regenerate all evaluation tables and SVG figures from committed raw trials."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 7319


def monotone_survival(values, weights):
    """Weighted non-increasing least-squares fit (pool adjacent violators)."""
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([i, i + 1, float(value) * weight, float(weight)])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] < blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = np.empty(len(values))
    for start, end, total, weight in blocks:
        result[start:end] = total / weight
    return result


def j50(grid, survival):
    """Return (estimate, censoring, tested bound), without extrapolation."""
    if survival[0] < 0.5:
        return None, "left", float(grid[0])
    if survival[-1] > 0.5:
        return None, "right", float(grid[-1])
    for i, probability in enumerate(survival):
        if probability <= 0.5:
            if probability == 0.5 or i == 0:
                return float(grid[i]), "none", None
            previous = survival[i - 1]
            fraction = (previous - 0.5) / (previous - probability)
            return float(grid[i - 1] + fraction * (grid[i] - grid[i - 1])), "none", None
    raise ValueError("Survival curve has no crossing or censoring classification")


def _interval(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return np.percentile(values, [2.5, 97.5]).tolist() if values else [None, None]


def _arm(row):
    return row["policy"].rsplit("-s", 1)[0]


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(inputs):
    found = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            found.update(path.rglob("trials.jsonl"))
            found.update(path.rglob("trials.jsonl.gz"))
        elif path.name.endswith((".jsonl", ".jsonl.gz")):
            found.add(path)
        else:
            raise ValueError(f"Expected a run directory or JSONL file: {path}")
    if not found:
        raise ValueError("No raw trial files found")
    return sorted({p.resolve() for p in found})


def compare_rerun(original, repeat):
    """Compare outcomes exactly and retain any differences in saved input states."""
    def records(path):
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt") as stream:
            return {row["trial_id"]: row for row in map(json.loads, stream)}

    before, after = records(original), records(repeat)
    if before.keys() != after.keys():
        raise ValueError("Clean rerun has different trial IDs")
    ignored = {"run_id", "wall_seconds"}
    differences = {}
    for trial_id, old in before.items():
        new = after[trial_id]
        for key in sorted(old.keys() | new.keys()):
            if key in ignored or old.get(key) == new.get(key):
                continue
            change = differences.setdefault(key, {"trials": 0, "max_abs": None})
            change["trials"] += 1
            a, b = np.asarray(old.get(key)), np.asarray(new.get(key))
            if (a.shape == b.shape and np.issubdtype(a.dtype, np.number)
                    and np.issubdtype(b.dtype, np.number)):
                delta = float(np.max(np.abs(a - b)))
                change["max_abs"] = max(change["max_abs"] or 0, delta)
    old_manifest = json.loads((original.parent / "manifest.json").read_text())
    new_manifest = json.loads((repeat.parent / "manifest.json").read_text())
    return {
        "policy": repeat.parent.name, "original_run": old_manifest["run_id"], "repeat_run": new_manifest["run_id"],
        "trials": len(before), "git_sha": new_manifest["git_sha"], "git_dirty": new_manifest["git_dirty"],
        "original_raw_sha256": _sha(original), "repeat_raw_sha256": _sha(repeat),
        "config_equal": old_manifest["config"] == new_manifest["config"],
        "policy_metadata_equal": old_manifest["policy_metadata"] == new_manifest["policy_metadata"],
        "successes_original": sum(row["success"] for row in before.values()),
        "successes_repeat": sum(row["success"] for row in after.values()),
        "outcomes_equal": set(differences) <= {"initial_qpos", "initial_quaternion", "settled_qpos"},
        "ignored_fields": sorted(ignored), "differing_fields": differences,
    }


def load_trials(inputs, *, root=None):
    rows, provenance, seen = [], [], set()
    protocols, policy_hashes = {}, {}
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    for path in _files(inputs):
        if path.is_relative_to(root / "results/terrain"):
            continue
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("suite") == "terrain":
            continue
        invocation_path = path.parent.parent.parent / "manifest.json"
        invocation = json.loads(invocation_path.read_text()) if invocation_path.is_file() else {}
        if invocation.get("run_id") == manifest["run_id"] and (
                invocation.get("status") == "superseded" or invocation.get("superseded")):
            continue
        if manifest.get("status") == "superseded" or manifest.get("superseded"):
            continue
        source = f'{manifest["run_id"]}/{manifest["backend"]}/{path.parent.name}'
        provenance.append({"raw": f"{source}/{path.name}", "raw_sha256": _sha(path),
                           "manifest": f"{source}/manifest.json", "manifest_sha256": _sha(manifest_path),
                           "run_id": manifest["run_id"], "status": manifest["status"],
                           "git_sha": manifest["git_sha"], "git_dirty": manifest["git_dirty"],
                           "wall_seconds": manifest["wall_seconds"],
                           "modal_app_id": manifest.get("modal_app_id")})
        if invocation.get("repeats"):
            original = invocation_path.parent.parent / invocation["repeats"] / manifest["backend"] / path.parent.name / path.name
            provenance[-1]["rerun_comparison"] = compare_rerun(original, path)
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid raw row {path}:{line_number}: {exc}") from exc
                if row["suite"] == "terrain":
                    continue
                key = (row["backend"], row["trial_id"])
                if key in seen:
                    raise ValueError(f"Duplicate trial {key}; mark superseded runs explicitly")
                seen.add(key)
                if row["run_id"] != manifest["run_id"] or row["backend"] != manifest["backend"]:
                    raise ValueError(f"Trial/manifest mismatch: {path}:{line_number}")
                group = (row["backend"], _arm(row))
                config = manifest["config"]
                protocol = json.dumps({k: v for k, v in config.items() if k not in ("seeds", "suites")}, sort_keys=True)
                if group in protocols and protocols[group] != protocol:
                    raise ValueError(f"Cannot pool differing protocols for {group}")
                protocols[group] = protocol
                policy_key = (row["backend"], row["policy"])
                policy_hash = manifest["policy_metadata"][row["policy"]]["policy_sha256"]
                if policy_key in policy_hashes and policy_hashes[policy_key] != policy_hash:
                    raise ValueError(f"Cannot pool different checkpoints for {policy_key}")
                policy_hashes[policy_key] = policy_hash
                row = dict(row, arm=_arm(row), _config=config,
                           _standard=manifest.get("standard_protocol", False), _policy_hash=policy_hash,
                           _source=Path(os.path.relpath(path, root)).as_posix())
                for value in row.values():
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError(f"Non-finite raw scalar at {path}:{line_number}")
                if not isinstance(row["success"], bool):
                    raise ValueError(f"Success must be boolean at {path}:{line_number}")
                rows.append(row)
    if not rows:
        raise ValueError("No trials remain after excluding superseded runs")
    return rows, provenance


def survival_data(rows):
    """Pool accepted MuJoCo push counts, not fitted or interpolated curves."""
    policies = {}
    for arm in ("go1-walk", "go1-walk-push", "g1-walk"):
        selected = [r for r in rows if r["suite"] == "push"
                    and r["backend"] == "mujoco" and r["arm"] == arm]
        trials = []
        for impulse in sorted({r["impulse"] for r in selected}):
            group = [r for r in selected if r["impulse"] == impulse]
            survived = sum(r["success"] for r in group)
            trials.append({"impulse": impulse, "survived": survived, "total": len(group),
                           "seeds": sorted({r["seed"] for r in group}),
                           "survival_fraction": survived / len(group)})
        policies[f"{arm}-s0"] = {
            "source": sorted({r["_source"] for r in selected}),
            "impulse_unit": "N·s", "trials": trials,
        }
    return {"schema_version": 1, "policies": policies}


def load_terrain_trials(inputs, *, root):
    """Read terrain independently; retain incomplete and numerical-failure rows.

    Policy manifests use the evaluator's policy_metadata[policy].policy_sha256,
    plus run_id, backend, seeds=[N], status, config and git/wall-time provenance.
    Superseded policy or invocation manifests are excluded explicitly.
    """
    root = Path(root).resolve()
    rows, sources, seen = [], [], set()
    protocols, policy_hashes = {}, {}
    for path in _files(inputs) if inputs else []:
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        invocation_path = path.parent.parent / "manifest.json"
        invocation = json.loads(invocation_path.read_text()) if invocation_path.is_file() else {}
        if any(m.get("status") == "superseded" or m.get("superseded")
               for m in (manifest, invocation)):
            continue
        source = path.relative_to(root).as_posix()
        sources.append({
            "suite": "terrain", "raw": source, "raw_sha256": _sha(path),
            "manifest": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": _sha(manifest_path), "run_id": manifest["run_id"],
            "status": manifest["status"], "git_sha": manifest["git_sha"],
            "git_dirty": manifest["git_dirty"], "wall_seconds": manifest["wall_seconds"],
            "modal_app_id": manifest.get("modal_app_id"),
        })
        if invocation:
            sources[-1].update(invocation=invocation_path.relative_to(root).as_posix(),
                               invocation_sha256=_sha(invocation_path))
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line_number, line in enumerate(stream, 1):
                where = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid terrain row {where}: {exc}") from exc
                if (row["suite"] != "terrain" or row["backend"] != "mujoco"
                        or manifest["backend"] != row["backend"]
                        or row["policy"] != path.parent.name
                        or row["arm"] != _arm(row)
                        or row["arm"] not in ("go1-walk", "go1-walk-rough")
                        or row["terrain"] not in ("flat", "rough")
                        or manifest["seeds"] != [row["seed"]]
                        or row["policy"] != f'{row["arm"]}-s{row["seed"]}'):
                    raise ValueError(f"Terrain trial/manifest mismatch: {where}")
                if row["trial_id"] in seen:
                    raise ValueError(f"Duplicate terrain trial {row['trial_id']}: {where}")
                seen.add(row["trial_id"])
                if (not isinstance(row["success"], bool) or not isinstance(row["completed"], bool)
                        or not (row["failure"] is None or isinstance(row["failure"], str))
                        or (row["success"] and row["failure"] is not None)):
                    raise ValueError(f"Invalid terrain outcome: {where}")
                if (len(row["command"]) != 3 or any(
                        not isinstance(v, (int, float)) or not math.isfinite(v)
                        for v in row["command"])):
                    raise ValueError(f"Invalid terrain command: {where}")
                for key in ("forward_mae", "lateral_mae", "yaw_mae"):
                    value = row[key]
                    if value is None:
                        if row["completed"]:
                            raise ValueError(f"Completed terrain trial lacks {key}: {where}")
                    elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid terrain {key}: {where}")
                if not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
                    raise ValueError(f"Invalid terrain wall time: {where}")
                config = manifest["config"]
                protocol = json.dumps({k: v for k, v in config.items()
                                       if k not in ("seed", "seeds", "policies", "terrains")}, sort_keys=True)
                if row["arm"] in protocols and protocols[row["arm"]] != protocol:
                    raise ValueError(f"Cannot pool differing terrain protocols for {row['arm']}")
                protocols[row["arm"]] = protocol
                policy_hash = manifest["policy_metadata"][row["policy"]]["policy_sha256"]
                if row["policy"] in policy_hashes and policy_hashes[row["policy"]] != policy_hash:
                    raise ValueError(f"Cannot pool different terrain checkpoints for {row['policy']}")
                policy_hashes[row["policy"]] = policy_hash
                rows.append(dict(row, _source=source, _policy_hash=policy_hash,
                                 _manifest_complete=manifest["status"] == "complete"
                                 and invocation.get("status", "complete") == "complete"))
    return rows, sources


def terrain_tables(rows):
    """Episode-weighted errors include physical falls; missing metrics stay counted."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["terrain"], row["seed"])].append(row)

    def aggregate(trials):
        failures = defaultdict(int)
        for row in trials:
            if row["failure"] is not None:
                failures[row["failure"]] += 1
        falls = sum(r.get("fell", r["failure"] == "fall") for r in trials)
        result = {
            "backend": "mujoco", "seeds": sorted({r["seed"] for r in trials}),
            "trials": len(trials), "completed": sum(r["completed"] for r in trials),
            "successes": sum(r["success"] for r in trials), "falls": falls,
            "fall_rate": falls / len(trials), "failures": dict(sorted(failures.items())),
            "numerical_failures": sum(not r["completed"] for r in trials),
        }
        for metric in ("forward_mae", "lateral_mae", "yaw_mae"):
            values = [r[metric] for r in trials if r[metric] is not None]
            result[metric] = sum(values) / len(values) if values else None
            result[f"{metric}_trials"] = len(values)
        return result

    seed_rows = []
    command_grids = {}
    for (arm, terrain, seed), trials in sorted(grouped.items()):
        commands = {tuple(r["command"]) for r in trials}
        command_grids[(arm, terrain, seed)] = commands
        seed_rows.append({
            "arm": arm, "terrain": terrain, "seed": seed, **aggregate(trials),
            "complete_grid": len(trials) == len(commands) == 27
                             and all(r["completed"] and r["_manifest_complete"] for r in trials),
        })
    pooled = []
    for arm, terrain in sorted({(r["arm"], r["terrain"]) for r in rows}):
        trials = [r for r in rows if (r["arm"], r["terrain"]) == (arm, terrain)]
        seeds = [r for r in seed_rows if (r["arm"], r["terrain"]) == (arm, terrain)]
        grids = [command_grids[(arm, terrain, r["seed"])] for r in seeds]
        pooled.append({
            "arm": arm, "terrain": terrain, **aggregate(trials),
            "complete_grid": all(r["complete_grid"] for r in seeds)
                             and all(grid == grids[0] for grid in grids),
            "three_seed_protocol": [r["seed"] for r in seeds] == [0, 1, 2]
                                   and all(r["complete_grid"] for r in seeds)
                                   and all(grid == grids[0] for grid in grids),
        })
    return pooled, seed_rows


def terrain_readme_lines(table):
    """Publish the full two-arm/two-terrain comparison only, never a partial table."""
    cells = {(r["arm"], r["terrain"]): r for r in table}
    arms = ("go1-walk", "go1-walk-rough")
    if any((arm, terrain) not in cells or not cells[arm, terrain]["three_seed_protocol"]
           for arm in arms for terrain in ("flat", "rough")):
        return []
    example = cells[arms[0], "flat"]
    seed_count = len(example["seeds"])
    lines = ["", f"**Cross-terrain walking in MuJoCo.** {seed_count} trained seeds per arm; "
             f"{example['trials'] // seed_count} commands per seed and condition. "
             "Falls / trials (forward, lateral, yaw MAE):",
             "", "| Training terrain | Flat test | Rough test |", "|---|---|---|"]
    for arm, label in zip(arms, ("Flat", "Rough")):
        values = []
        for terrain in ("flat", "rough"):
            row = cells[arm, terrain]
            errors = " / ".join(f"{row[key]:.3f}" for key in ("forward_mae", "lateral_mae", "yaw_mae"))
            values.append(f"{row['falls']}/{row['trials']} ({errors})")
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines += ["", "Errors are episode means including time after falls: forward/lateral in m/s, yaw in rad/s. "
              "[Terrain counts and errors](results/report/terrain.csv) · "
              "[Per seed](results/report/terrain_seeds.csv) · [Raw trials](results/terrain/)"]
    return lines


def _csv(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def _bootstrap_outcomes(groups, rng, replicates):
    """Resample seeds, then Bernoulli counts equivalent to resampled raw trials."""
    seeds = sorted(groups)
    result = np.empty(replicates)
    for b in range(replicates):
        sampled = rng.choice(seeds, len(seeds), replace=True)
        rates = []
        for seed in sampled:
            values = groups[int(seed)]
            rates.append(rng.binomial(len(values), np.mean(values)) / len(values))
        result[b] = np.mean(rates)
    return result


def tracking_table(rows):
    grouped = defaultdict(list)
    for r in rows:
        if r["suite"] == "tracking":
            grouped[(r["backend"], r["arm"], r["seed"])].append(r)
    result, commands = [], []
    for (backend, arm, seed), trials in sorted(grouped.items()):
        def aggregate(selected):
            available = [r for r in selected if r.get("samples", 0) and r.get("forward_mae") is not None]
            n = sum(r["samples"] for r in available)
            return {"trials": len(selected), "completed": sum(r["success"] for r in selected),
                    "completion": np.mean([r["success"] for r in selected]).item(),
                    "samples": n,
                    **{metric: sum(r[metric] * r["samples"] for r in available) / n if n else None
                       for metric in ("forward_mae", "lateral_mae", "yaw_mae")}}
        base = {"backend": backend, "arm": arm, "seed": seed,
                "standard_protocol": all(r["_standard"] for r in trials),
                "complete_grid": len(trials) == trials[0]["_config"]["velocity_points"] ** 3}
        result.append(dict(base, **aggregate(trials)))
        by_command = defaultdict(list)
        for r in trials:
            by_command[tuple(r["command"])].append(r)
        for command, selected in sorted(by_command.items()):
            commands.append(dict(base, command=list(command), **aggregate(selected)))
    return result, commands


def _push_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["suite"] == "push":
            groups[(r["backend"], r["arm"])].append(r)
    return groups


def _push_arrays(trials):
    config = trials[0]["_config"]
    expected = config["go1_impulses"] if trials[0]["robot"] == "go1" else config["g1_impulses"]
    grid = sorted({float(r["impulse"]) for r in trials})
    seeds = sorted({r["seed"] for r in trials})
    indexed = defaultdict(dict)
    for r in trials:
        indexed[(r["seed"], float(r["impulse"]))][r["trial"]] = r
    complete_seeds = [s for s in seeds if all((s, j) in indexed for j in grid)]
    if complete_seeds != seeds:
        # Never estimate a seed-pooled curve whose seed mixture changes with J.
        return grid, seeds, indexed, False, False
    complete = (grid == list(map(float, expected)) and all(
        len(indexed[(s, j)]) == config["push_trials"] for s in seeds for j in grid))
    full = complete and seeds == [0, 1, 2] and all(r["_standard"] for r in trials)
    return grid, seeds, indexed, complete, full


def _trial_blocks(indexed, seeds, grid):
    """One sampled index carries its direction/phase through the whole grid."""
    blocks = {}
    for seed in seeds:
        ids = sorted(indexed[(seed, grid[0])])
        if any(set(indexed[(seed, impulse)]) != set(ids) for impulse in grid):
            return None
        first = indexed[(seed, grid[0])]
        for impulse in grid:
            for index in ids:
                row = indexed[(seed, impulse)][index]
                if (row["theta"], row["push_onset"]) != (first[index]["theta"], first[index]["push_onset"]):
                    raise ValueError("A trial index changes direction/phase across the impulse grid")
        blocks[seed] = np.asarray([[indexed[(seed, impulse)][i]["success"] for i in ids]
                                   for impulse in grid], dtype=float)
    return blocks


def push_tables(rows, rng, replicates):
    survival_rows, threshold_rows, fits = [], [], {}
    for (backend, arm), trials in sorted(_push_groups(rows).items()):
        grid, seeds, indexed, complete, full = _push_arrays(trials)
        balanced = all((s, j) in indexed for s in seeds for j in grid)
        raw_rates, weights = [], []
        for j in grid:
            present = [s for s in seeds if (s, j) in indexed]
            samples = [list(indexed[(s, j)].values()) for s in present]
            rate = float(np.mean([np.mean([r["success"] for r in group]) for group in samples]))
            weight = sum(len(group) for group in samples)
            raw_rates.append(rate)
            weights.append(weight)
        curve = monotone_survival(raw_rates, weights) if balanced else None
        blocks = _trial_blocks(indexed, seeds, grid) if balanced else None
        bootstrap = np.empty((replicates, len(grid))) if blocks is not None else None
        if blocks is not None:
            for b in range(replicates):
                sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
                samples = [blocks[int(s)][:, rng.integers(0, blocks[int(s)].shape[1], blocks[int(s)].shape[1])]
                           for s in sampled_seeds]
                rates = np.mean([sample.mean(axis=1) for sample in samples], axis=0)
                sample_weights = [sum(sample.shape[1] for sample in samples)] * len(grid)
                bootstrap[b] = monotone_survival(rates, sample_weights)
        for i, j in enumerate(grid):
            selected = [r for r in trials if float(r["impulse"]) == j]
            ci = _interval(bootstrap[:, i]) if bootstrap is not None else [None, None]
            survival_rows.append({"backend": backend, "arm": arm, "impulse": j,
                                  "trials": len(selected), "successes": sum(r["success"] for r in selected),
                                  "seeds": seeds, "survival": raw_rates[i],
                                  "monotone_survival": float(curve[i]) if balanced else None,
                                  "ci_low": ci[0], "ci_high": ci[1], "complete_grid": complete,
                                  "three_seed_protocol": full, "balanced_seeds": balanced})
        if balanced:
            estimate, censoring, bound = j50(grid, curve)
            draws = [j50(grid, p) for p in bootstrap] if bootstrap is not None else []
            censor_fraction = sum(c != "none" for _, c, _ in draws) / replicates if draws else None
            ci = _interval([v for v, _, _ in draws]) if censor_fraction == 0 else [None, None]
        else:
            estimate, censoring, bound, censor_fraction, ci = None, "missing_seed_grid", None, None, [None, None]
        threshold_rows.append({"backend": backend, "arm": arm, "seeds": seeds, "j50": estimate,
                               "censoring": censoring, "grid_bound": bound,
                               "grid_min": grid[0], "grid_max": grid[-1], "ci_low": ci[0], "ci_high": ci[1],
                               "bootstrap_censored_fraction": censor_fraction, "complete_grid": complete,
                               "three_seed_protocol": full})
        fits[(backend, arm)] = {"grid": grid, "seeds": seeds, "indexed": indexed, "curve": curve,
                               "j50": estimate, "censoring": censoring, "complete": complete, "full": full,
                               "protocol": {k: v for k, v in trials[0]["_config"].items() if k not in ("seeds", "suites")}}
    return survival_rows, threshold_rows, fits


def seed_push_tables(rows):
    curves, thresholds = [], []
    for (backend, arm), trials in sorted(_push_groups(rows).items()):
        for seed in sorted({r["seed"] for r in trials}):
            selected = [r for r in trials if r["seed"] == seed]
            grid, _, indexed, complete, _ = _push_arrays(selected)
            counts = [len(indexed[(seed, j)]) for j in grid]
            rates = [float(np.mean([r["success"] for r in indexed[(seed, j)].values()])) for j in grid]
            fit = monotone_survival(rates, counts)
            estimate, censoring, bound = j50(grid, fit)
            thresholds.append({"backend": backend, "arm": arm, "seed": seed, "j50": estimate,
                               "censoring": censoring, "grid_bound": bound, "complete_grid": complete})
            for j, rate, smooth, count in zip(grid, rates, fit, counts):
                curves.append({"backend": backend, "arm": arm, "seed": seed, "impulse": j,
                               "survival": rate, "monotone_survival": float(smooth), "trials": count})
    return curves, thresholds


def ablation_table(fits, rng, replicates):
    result = []
    for backend in sorted({key[0] for key in fits}):
        left, right = fits.get((backend, "go1-walk")), fits.get((backend, "go1-walk-push"))
        if left is None or right is None:
            continue
        if left["protocol"] != right["protocol"]:
            raise ValueError("Go1 ablation arms use different evaluation protocols")
        grid = left["grid"]
        seeds = sorted(set(left["seeds"]) & set(right["seeds"]))
        comparable = bool(seeds) and grid == right["grid"] and left["curve"] is not None and right["curve"] is not None
        paired = {}
        if comparable:
            for s in seeds:
                for j in grid:
                    a, b = left["indexed"][(s, j)], right["indexed"][(s, j)]
                    ids = sorted(set(a) & set(b))
                    for i in ids:
                        if (a[i]["theta"], a[i]["push_onset"]) != (b[i]["theta"], b[i]["push_onset"]):
                            raise ValueError("Go1 arms do not share trial direction/phase")
                    paired[(s, j)] = (a, b, ids)
                    comparable &= bool(ids)
        point_estimates = [(None, "unpaired_grid", None)] * 2
        if comparable:
            curves, weights = [[], []], []
            for j in grid:
                rates, count = [[], []], 0
                for s in seeds:
                    a, b, ids = paired[(s, j)]
                    rates[0].append(np.mean([a[i]["success"] for i in ids]))
                    rates[1].append(np.mean([b[i]["success"] for i in ids]))
                    count += len(ids)
                for arm_index in range(2):
                    curves[arm_index].append(np.mean(rates[arm_index]))
                weights.append(count)
            point_estimates = [j50(grid, monotone_survival(c, weights)) for c in curves]
        draws, censored = [], 0
        paired_blocks = None
        if comparable:
            paired_indices = [{key: {i: pair[arm][i] for i in pair[2]} for key, pair in paired.items()}
                              for arm in range(2)]
            paired_blocks = [_trial_blocks(indexed, seeds, grid) for indexed in paired_indices]
        can_bootstrap = paired_blocks is not None and all(block is not None for block in paired_blocks)
        if can_bootstrap:
            for _ in range(replicates):
                sampled = rng.choice(seeds, len(seeds), replace=True)
                samples = [[], []]
                weights = 0
                for seed in sampled:
                    n = paired_blocks[0][int(seed)].shape[1]
                    indices = rng.integers(0, n, n)
                    for arm in range(2):
                        samples[arm].append(paired_blocks[arm][int(seed)][:, indices].mean(axis=1))
                    weights += n
                curves = [np.mean(arm_samples, axis=0) for arm_samples in samples]
                estimates = [j50(grid, monotone_survival(c, [weights] * len(grid))) for c in curves]
                if any(c != "none" for _, c, _ in estimates):
                    censored += 1
                else:
                    draws.append(estimates[1][0] - estimates[0][0])
        delta = (point_estimates[1][0] - point_estimates[0][0]
                 if all(c == "none" for _, c, _ in point_estimates) else None)
        ci = _interval(draws) if can_bootstrap and not censored else [None, None]
        paired_count = sum(len(ids) for _, _, ids in paired.values())
        all_count = sum(len(v) for v in left["indexed"].values()) + sum(len(v) for v in right["indexed"].values())
        result.append({"backend": backend, "comparator": "go1-walk", "arm": "go1-walk-push", "seeds": seeds,
                       "j50_delta": delta, "ci_low": ci[0], "ci_high": ci[1],
                       "baseline_censoring": point_estimates[0][1], "push_arm_censoring": point_estimates[1][1],
                       "bootstrap_censored_fraction": censored / replicates if can_bootstrap else None,
                       "paired_trials": paired_count, "unpaired_trials": all_count - 2 * paired_count,
                       "three_seed_protocol": left["full"] and right["full"] and seeds == [0, 1, 2]})
    return result


def getup_start_pose(row):
    """Classify the saved settled pose; speed is not a pose criterion."""
    qpos = np.asarray(row.get("settled_qpos") or [], dtype=float)
    if len(qpos) < 7 or not np.isfinite(qpos[:7]).all():
        return "unclassified"
    q = qpos[3:7]
    norm2 = float(np.dot(q, q))
    if norm2 == 0:
        return "unclassified"
    cosine = 1 - 2 * float(q[1] * q[1] + q[2] * q[2]) / norm2
    tilt = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
    config = row["_config"]
    return ("already_upright" if tilt < config["getup_tilt_degrees"] and qpos[2] >= config["getup_height"]
            else "fallen")


def getup_table(rows, rng, replicates):
    groups = defaultdict(list)
    totals = defaultdict(list)
    for r in rows:
        if r["suite"] == "getup":
            totals[(r["backend"], r["arm"])].append(r)
            groups[(r["backend"], r["arm"], "all")].append(r)
            groups[(r["backend"], r["arm"], getup_start_pose(r))].append(r)
    result = []
    for (backend, arm, start_pose), trials in sorted(groups.items()):
        seeds = sorted({r["seed"] for r in trials})
        by_seed = {s: [r for r in trials if r["seed"] == s] for s in seeds}
        for seed in [*seeds, "all"]:
            selected = trials if seed == "all" else by_seed[seed]
            selected_seeds = seeds if seed == "all" else [seed]
            outcomes = {s: [r["success"] for r in by_seed[s]] for s in selected_seeds}
            ci = _interval(_bootstrap_outcomes(outcomes, rng, replicates))
            times = [r["recovery_time"] for r in selected if r["success"]]
            mean_times = []
            for _ in range(replicates):
                sampled_seeds = rng.choice(selected_seeds, len(selected_seeds), replace=True)
                sampled_times = []
                for s in sampled_seeds:
                    source = by_seed[int(s)]
                    for i in rng.integers(0, len(source), len(source)):
                        r = source[int(i)]
                        if r["success"]:
                            sampled_times.append(r["recovery_time"])
                if sampled_times:
                    mean_times.append(float(np.mean(sampled_times)))
            time_ci = _interval(mean_times)
            original = totals[(backend, arm)]
            successes = sum(r["success"] for r in selected)
            result.append({"backend": backend, "arm": arm, "start_pose": start_pose,
                           "seed": seed, "seeds": selected_seeds,
                           "trials": len(selected), "successes": successes,
                           "equal_seed_success": float(np.mean([np.mean(outcomes[s]) for s in selected_seeds])),
                           "count_success": successes / len(selected),
                           "ci_low": ci[0], "ci_high": ci[1],
                           "mean_recovery_seconds": float(np.mean(times)) if times else None,
                           "median_recovery_seconds": float(np.median(times)) if times else None,
                           "time_ci_low": time_ci[0], "time_ci_high": time_ci[1],
                           "complete": all(sum(r["seed"] == s for r in original) == original[0]["_config"]["getup_trials"]
                                           for s in selected_seeds),
                           "three_seed_protocol": selected_seeds == [0, 1, 2] and all(r["_standard"] for r in original)
                                                  and all(sum(r["seed"] == s for r in original) >= 200 for s in selected_seeds)})
    return result


def gap_table(rows, fits):
    indexed = {b: {r["trial_id"]: r for r in rows if r["backend"] == b and r["seed"] == 0 and r["suite"] != "getup"}
               for b in ("mujoco", "mjx")}
    if not all(indexed.values()):
        return []
    groups = defaultdict(list)
    all_ids = sorted(set(indexed["mujoco"]) | set(indexed["mjx"]))
    for identifier in all_ids:
        a, b = indexed["mujoco"].get(identifier), indexed["mjx"].get(identifier)
        representative = a or b
        key = (representative["arm"], representative["suite"], representative.get("impulse"))
        if a is not None and b is not None:
            for field in ("command", "theta", "push_onset", "_policy_hash"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"Backend pair differs in {field}: {identifier}")
            if a["_config"] != b["_config"]:
                # The budget-limited gap uses fewer paired push trials, not different physics.
                ac = {k: v for k, v in a["_config"].items() if k not in ("seeds", "suites", "push_trials")}
                bc = {k: v for k, v in b["_config"].items() if k not in ("seeds", "suites", "push_trials")}
                if ac != bc:
                    raise ValueError(f"Backend protocols differ: {identifier}")
        groups[key].append((a, b))
    result = []
    for (arm, suite, impulse), pairs in sorted(groups.items(), key=lambda item: str(item[0])):
        matched = [(a, b) for a, b in pairs if a is not None and b is not None]
        good = [(a, b) for a, b in matched if a.get("forward_mae") is not None and b.get("forward_mae") is not None]
        result.append({"arm": arm, "suite": suite, "impulse": impulse, "seed": 0,
                       "paired_trials": len(matched), "unpaired_trials": len(pairs) - len(matched),
                       "mjx_minus_mujoco_success": float(np.mean([int(b["success"]) - int(a["success"]) for a, b in matched])) if matched else None,
                       **{f"mjx_minus_mujoco_{metric}": float(np.mean([b[metric] - a[metric] for a, b in good])) if good else None
                          for metric in ("forward_mae", "lateral_mae", "yaw_mae")},
                       "mae_pairs": len(good), "j50_delta": None})
    for arm in sorted({r["arm"] for r in rows}):
        # Compare the same seed and trial IDs, including the reduced-count gap.
        trials = [r for r in rows if r["backend"] == "mujoco" and r["arm"] == arm and r["seed"] == 0
                  and r["suite"] == "push" and r["trial_id"] in indexed["mjx"]]
        mjx_fit = fits.get(("mjx", arm))
        if not trials or mjx_fit is None:
            continue
        grid, seeds, lookup, _, _ = _push_arrays(trials)
        rates = [np.mean([r["success"] for r in lookup[(0, j)].values()]) for j in grid]
        curve = monotone_survival(rates, [len(lookup[(0, j)]) for j in grid])
        estimate, censoring, bound = j50(grid, curve)
        same_grid = grid == mjx_fit["grid"]
        delta = mjx_fit["j50"] - estimate if same_grid and estimate is not None and mjx_fit["j50"] is not None else None
        count = mjx_fit["protocol"]["push_trials"]
        result.append({"arm": arm, "suite": "j50", "seed": 0, "j50_delta": delta,
                       "paired_trials": len(trials), "mujoco_paired_j50": estimate,
                       "mjx_paired_j50": mjx_fit["j50"],
                       "mujoco_censoring": censoring, "mujoco_grid_bound": bound,
                       "mjx_censoring": mjx_fit["censoring"], "same_grid": same_grid,
                       "complete_grid": same_grid and mjx_fit["complete"]
                                        and all(len(lookup[(0, j)]) == count for j in grid)})
    return result


COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2")


def _svg(path, title, series, x_label, y_label, *, probability=False):
    """Small dependency-free line/point plot, including available CI bands."""
    width, height = 880, 470
    left, top, right, bottom = 82, 45, 590, 400
    points = [p for _, values in series for p in values if p[1] is not None]
    escape = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#111827">',
             f'<text x="{left}" y="25" font-size="18">{escape(title)}</text>']
    if not points:
        parts.append(f'<text x="{left}" y="90">No measured trials for this figure.</text>')
    else:
        xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
        if xmin == xmax:
            xmin, xmax = xmin - 0.5, xmax + 0.5
        yvalues = [v for p in points for v in p[1:] if v is not None]
        ymin, ymax = (0.0, 1.0) if probability else (min(0.0, min(yvalues)), max(yvalues))
        if ymin == ymax:
            ymax = ymin + 1.0
        def xy(x, y):
            return left + (x - xmin) / (xmax - xmin) * (right - left), bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
        for i in range(6):
            y = ymin + (ymax - ymin) * i / 5
            _, pixel = xy(xmin, y)
            parts.append(f'<path d="M {left} {pixel:.2f} H {right}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{left - 10}" y="{pixel + 4:.2f}" text-anchor="end">{y:.3g}</text>')
            x = xmin + (xmax - xmin) * i / 5
            pixel, _ = xy(x, ymin)
            parts.append(f'<text x="{pixel:.2f}" y="{bottom + 22}" text-anchor="middle">{x:.3g}</text>')
        parts.append(f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#111827"/>')
        for index, (label, values) in enumerate(series):
            values = sorted(p for p in values if p[1] is not None)
            color = COLORS[index % len(COLORS)]
            if values:
                vertices = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(p[0], p[1]) for p in values))
                parts.append(f'<polyline points="{vertices}" fill="none" stroke="{color}" stroke-width="2"/>')
                for xvalue, value, low, high in values:
                    x, y = xy(xvalue, value)
                    if low is not None and high is not None:
                        _, lo = xy(xvalue, low)
                        _, hi = xy(xvalue, high)
                        parts.append(f'<path d="M {x:.2f} {lo:.2f} V {hi:.2f}" stroke="{color}" opacity="0.5"/>')
                    parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
            legend_y = top + index * 28
            parts.append(f'<path d="M 620 {legend_y} H 640" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="648" y="{legend_y + 4}">{escape(label)}</text>')
        parts.append(f'<text x="{(left + right) / 2}" y="{height - 15}" text-anchor="middle">{escape(x_label)}</text>')
        parts.append(f'<text transform="translate(20 {(top + bottom) / 2}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')
    parts.extend(["</g>", "</svg>"])
    path.write_text("\n".join(parts) + "\n")


def run_tables(root):
    training, benchmarks, sources = [], [], []
    ledger_path = root / "results/spend.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
    ends = {r["id"]: r for r in ledger if r["event"] == "end"}
    starts = [r for r in ledger if r["event"] == "start" and r["project"] == "pushable"]
    charges = {start["command"][start["command"].index("--run-id") + 1]: (start, ends[start["id"]])
               for start in starts if "--run-id" in start["command"] and start["id"] in ends}
    standalone = []
    for path in sorted((root / "results/runs").glob("*/manifest.json")):
        record = json.loads(path.read_text())
        if path.parent.name in charges:
            start, end = charges[path.parent.name]
            record.update(minutes=end["wall_minutes"], usd_upper=end["charged_upper"],
                          spend_guard={"start": start, "end": end})
        if "cost_estimate" in record:
            standalone.append((path, record))
        if record.get("superseded") or record.get("stage") not in ("train", "benchmark"):
            continue
        checkpoint = None
        if record.get("published_checkpoint"):
            checkpoint_path = path.parent.parent / record["published_checkpoint"] / "manifest.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint["status"] != "complete" or checkpoint["training_source"]["id"] != record["id"]:
                raise ValueError(f"Invalid published checkpoint for {record['id']}")
            sources.append({"path": str(checkpoint_path.relative_to(root)), "sha256": _sha(checkpoint_path)})
        if record.get("status") != "complete" and checkpoint is None:
            continue
        sources.append({"path": str(path.relative_to(root)), "sha256": _sha(path)})
        if record["stage"] == "train":
            source = checkpoint["training_source"] if checkpoint else record
            config = source["ppo_config"]
            update_steps = config["batch_size"] * config["unroll_length"] * config["num_minibatches"] * config["action_repeat"]
            epochs = max(config["num_evals"] - 1, 1) * max(config["num_resets_per_eval"], 1)
            epoch_steps = math.ceil(config["num_timesteps"] / (epochs * update_steps)) * update_steps
            reset_seconds = (epoch_steps / config["num_envs"] * source["environment_config"]["ctrl_dt"]
                             if config["num_resets_per_eval"] else None)
            row = {key: record.get(key) for key in
                   ("id", "env", "seed", "gpu", "minutes", "usd_upper", "final_reward", "trained_steps")}
            if checkpoint:
                row.update(final_reward=checkpoint["final_reward"], trained_steps=checkpoint["checkpoint_steps"])
            row.update(requested_steps=config["num_timesteps"], explicit_reset_interval_seconds=reset_seconds,
                       completion="interrupted_checkpoint" if checkpoint else "complete")
            training.append(row)
        else:
            if "cost_estimate" in record:
                per_second = record["cost_estimate"]["hourly_usd_with_margin"] / 3600
            else:
                booking = record["spend_guard"]["start"]
                per_second = booking["booked"] / (booking["minutes"] * 60)
            measurement = record["benchmark"]
            benchmarks.append({"id": record["id"], "gpu": record["gpu"],
                               "steps_per_second": measurement["steps_per_second"],
                               "measurement_seconds": measurement["seconds"],
                               "warmup_seconds": measurement["warmup_seconds"],
                               "minutes": record["minutes"], "usd_upper": record["usd_upper"],
                               "charged_usd_upper_per_million_steps": record["usd_upper"] * 1e6 / (
                                   measurement["steps_per_second"] * measurement["seconds"]),
                               "usd_upper_per_million_steps": per_second * 1e6 / measurement["steps_per_second"],
                               "scope": measurement["scope"]})
    invocations = []
    for start in starts:
        end = ends.get(start["id"])
        command = start["command"]
        arguments = {name: command[command.index(name) + 1] if name in command else None
                     for name in ("--run-id", "--stage")}
        manifest = root / "results/runs" / str(arguments["--run-id"]) / "manifest.json"
        invocations.append({
            "guard_id": start["id"], "id": arguments["--run-id"], "stage": arguments["--stage"],
            "gpu": start["gpu"], "planned_minutes": start["minutes"],
            "minutes": end["wall_minutes"] if end else None,
            "usd_upper": end["charged_upper"] if end else start["booked"],
            "accounting": "charged_upper" if end else "booked",
            "exit": end["exit"] if end else None,
            "manifest": str(manifest.relative_to(root)) if manifest.exists() else None,
        })
    recorded_manifests = {row["manifest"] for row in invocations}
    for path, record in standalone:
        relative = str(path.relative_to(root))
        if relative in recorded_manifests:
            continue
        invocations.append({
            "guard_id": None, "id": path.parent.name, "stage": record["stage"], "gpu": record["gpu"],
            "planned_minutes": record["launch_config"]["minutes"],
            "minutes": record.get("minutes", record.get("launch_wall_seconds", 0) / 60),
            "usd_upper": record["usd_upper"], "accounting": "list_price_upper",
            "exit": record["exit_code"], "manifest": relative,
        })
        sources.append({"path": relative, "sha256": _sha(path)})
    spend = {"usd_upper": sum(row["usd_upper"] for row in invocations),
             "finished_invocations": sum(row["exit"] is not None for row in invocations),
             "training_usd_upper": sum(r["usd_upper"] for r in training),
             "training_minutes": sum(r["minutes"] for r in training),
             "training_runs": len(training)}
    if ledger_path.exists():
        sources.append({"path": str(ledger_path.relative_to(root)), "sha256": _sha(ledger_path)})
    return training, benchmarks, invocations, spend, sources


def getup_diagnostic_table(root, rng, replicates):
    """Keep original and revised getup configurations separate, including history."""
    result, sources = [], []
    paths = set()
    for folder in ("eval", "getup-diagnostic"):
        paths.update((root / "results" / folder).glob("*/*/go1-getup-s0/trials.jsonl*"))
    for path in sorted(paths):
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        invocation = json.loads((path.parent.parent.parent / "manifest.json").read_text())
        complete = manifest.get("status") == "complete" and invocation.get("status") == "complete"
        if not complete and not invocation.get("keep_in_diagnostics"):
            continue
        if (manifest.get("superseded") or invocation.get("superseded")) and not invocation.get("keep_in_diagnostics"):
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            trials = [json.loads(line) for line in stream]
        trials = [r for r in trials if r["suite"] == "getup" and r["seed"] == 0]
        if not trials:
            continue
        name = trials[0]["policy"]
        metadata = manifest["policy_metadata"][name]
        outcomes = [r["success"] for r in trials]
        times = [r["recovery_time"] for r in trials if r["success"]]
        nominal = [r for r in trials if r.get("nominal_start", False)]
        dropped = [r for r in trials if not r.get("nominal_start", False)]
        ci = _interval(_bootstrap_outcomes({0: outcomes}, rng, replicates))
        result.append({
            "run_id": manifest["run_id"], "training_run_id": metadata["spec"]["run_id"],
            "complete": complete,
            "policy_sha256": metadata["policy_sha256"], "backend": manifest["backend"],
            "initialization": invocation.get("getup_initialization", "drop"), "seed": 0,
            "trials": len(trials), "successes": sum(outcomes), "success": float(np.mean(outcomes)),
            "nominal_trials": len(nominal), "nominal_successes": sum(r["success"] for r in nominal),
            "dropped_trials": len(dropped), "dropped_successes": sum(r["success"] for r in dropped),
            "ci_low": ci[0], "ci_high": ci[1],
            "median_recovery_seconds": float(np.median(times)) if times else None,
            "software_failures": sum(r.get("failure") == "simulator_error" for r in trials),
            "raw": str(path.relative_to(root)),
        })
        sources.append({"raw": str(path.relative_to(root)), "raw_sha256": _sha(path),
                        "manifest": str(manifest_path.relative_to(root)),
                        "manifest_sha256": _sha(manifest_path)})
    return result, sources


def experimental_getup_table(root, rng, replicates):
    """Keep each custom G1 configuration and its fallen-start denominator separate."""
    result, sources = [], []
    for path in sorted((root / "results/getup-diagnostic").glob("*/*/g1-getup-s0/trials.jsonl*")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        invocation = json.loads((path.parent.parent.parent / "manifest.json").read_text())
        if any(record.get("superseded") or record.get("status") == "superseded"
               for record in (manifest, invocation)):
            continue
        rows, provenance = load_trials([path], root=root)
        metadata = manifest["policy_metadata"]["g1-getup-s0"]
        for row in getup_table(rows, rng, replicates):
            if row["seed"] == "all":
                continue
            result.append({
                **row, "run_id": manifest["run_id"],
                "training_run_id": metadata["spec"]["run_id"],
                "policy_sha256": metadata["policy_sha256"],
                "complete": row["complete"] and manifest["status"] == "complete"
                            and invocation["status"] == "complete",
                "raw": path.relative_to(root).as_posix(),
            })
        sources.extend(provenance)
    return result, sources


def update_readme(root, tables, spend):
    """Regenerate measured publication text from report tables."""
    path = root / "README.md"
    text = path.read_text()
    ablation = next((r for r in tables["ablation"] if r["backend"] == "mujoco" and r["three_seed_protocol"]), None)
    getup = next((r for r in tables["getup"] if r["backend"] == "mujoco" and r["seed"] == "all"
                  and r["start_pose"] == "fallen" and r["three_seed_protocol"]), None)
    g1_completed = {r["seed"] for r in tables["training"]
                    if r["id"] in ("g1-walk-s1", "g1-walk-s2") and r["completion"] == "complete"
                    and r["trained_steps"] >= r["requested_steps"]}
    gap_tracking = next((r for r in tables["gap"] if r["arm"] == "go1-walk" and r["suite"] == "tracking"
                         and r["paired_trials"]), None)
    gap_j50 = next((r for r in tables["gap"] if r["arm"] == "go1-walk" and r["suite"] == "j50"
                    and r["complete_grid"]), None)
    published_ids = {f"{arm}-s{seed}{suffix}" for arm, suffix in (
        ("go1-walk", ""), ("go1-walk-push", ""), ("go1-getup", "-t200m"), ("g1-walk", ""))
        for seed in (0, 1, 2)}
    published = [r for r in tables["training"] if r["id"] in published_ids]
    if (not all((ablation, getup, gap_tracking, gap_j50)) or g1_completed != {1, 2}
            or {r["id"] for r in published} != published_ids):
        raise ValueError("Published README requires the full Go1 comparison, final getup, three G1 seeds and small gap")
    tracking = [r for r in tables["tracking"] if r["backend"] == "mujoco"]
    thresholds_by_arm = {}
    lines = ["| Policy | Seeds | Shove survived half the time, J50 (N·s), by seed | Tracking error: forward, lateral (m/s), turning (rad/s) |",
             "|---|---:|---|---|"]
    for arm, label in (("go1-walk", "Go1, trained without shoves"), ("go1-walk-push", "Go1, trained with shoves"),
                       ("g1-walk", "G1 humanoid")):
        errors = [r for r in tracking if r["arm"] == arm]
        thresholds = sorted((r for r in tables["j50_seeds"] if r["backend"] == "mujoco" and r["arm"] == arm),
                            key=lambda r: r["seed"])
        if ({r["seed"] for r in errors} != {0, 1, 2}
                or {r["seed"] for r in thresholds} != {0, 1, 2}
                or any(not r["complete_grid"] for r in errors + thresholds)
                or any(r["j50"] is None for r in thresholds)):
            raise ValueError(f"Missing complete uncensored three-seed result for {arm}")
        thresholds_by_arm[arm] = [r["j50"] for r in thresholds]
        values = " / ".join(f"{r['j50']:.1f}" for r in thresholds)
        mae = " / ".join(f"{np.mean([r[key] for r in errors]):.3f}"
                         for key in ("forward_mae", "lateral_mae", "yaw_mae"))
        lines.append(f"| {label} | {len(errors)} | {values} | {mae} |")
    if any(ablation[key] is None for key in ("j50_delta", "ci_low", "ci_high")) or gap_j50["j50_delta"] is None:
        raise ValueError("Published comparison requires uncensored estimates and bootstrap interval")
    words = {1: "one", 2: "two", 3: "three"}
    baseline, pushed = thresholds_by_arm["go1-walk"], thresholds_by_arm["go1-walk-push"]
    pooled = {r["arm"]: r["j50"] for r in tables["j50"]
              if r["backend"] == "mujoco" and r["three_seed_protocol"] and r["j50"] is not None}
    if not {"go1-walk", "go1-walk-push"} <= set(pooled):
        raise ValueError("Published headline requires uncensored pooled J50 for both Go1 arms")
    answer = "Yes." if ablation["ci_low"] > 0 else "No." if ablation["ci_high"] < 0 else "Not clearly."
    verb = "rises" if ablation["j50_delta"] > 0 else "falls"
    headline = (
        f"**Does training with random shoves make the dog harder to knock over?** {answer} "
        f"The shove it survives half the time (J50) {verb} from {pooled['go1-walk']:.1f} to "
        f"{pooled['go1-walk-push']:.1f} N·s with {words[len(ablation['seeds'])]} seeds per arm pooled "
        f"(per seed: {min(baseline):.1f}–{max(baseline):.1f} vs {min(pushed):.1f}–{max(pushed):.1f} N·s): "
        f"**{ablation['j50_delta']:+.1f} N·s, 95% interval {ablation['ci_low']:+.1f} to {ablation['ci_high']:+.1f}**. "
        f"[Comparison](results/report/ablation.csv)")
    getup_seeds = sorted((r for r in tables["getup"] if r["backend"] == "mujoco" and r["seed"] != "all"
                         and r["start_pose"] == "fallen"), key=lambda r: r["seed"])
    all_starts = sorted((r for r in tables["getup"] if r["backend"] == "mujoco" and r["seed"] != "all"
                        and r["start_pose"] == "all"), key=lambda r: r["seed"])
    perfect = [r for r in getup_seeds if r["successes"] == r["trials"]]
    weak = [r for r in getup_seeds if r["successes"] != r["trials"]]
    if (len(perfect) != 2 or len(weak) != 1 or weak[0]["count_success"] >= 0.1
            or {r["seed"] for r in all_starts} != {0, 1, 2}
            or any(r["start_pose"] == "unclassified" for r in tables["getup"])):
        raise ValueError("The approved get-up wording no longer describes the complete measured groups")
    counts = lambda r: f"{r['successes']}/{r['trials']}"
    max_gap = max(abs(gap_tracking[f"mjx_minus_mujoco_{axis}_mae"]) for axis in ("forward", "lateral", "yaw"))
    lines += [
        "",
        "[Per-seed tracking](results/report/tracking.csv) · [J50 by seed](results/report/j50_seeds.csv) · "
        "[survival curves](results/report/push_seeds.svg) · [raw trials](results/eval/)",
        "",
        f"**Getting up.** From fallen starts, {words[len(perfect)]} of the {words[len(getup_seeds)]} get-up seeds "
        f"recover in every trial ({counts(perfect[0])} and {counts(perfect[1])}); "
        f"the third almost never does ({counts(weak[0])}). "
        f"Median time back on its feet: {getup['median_recovery_seconds']:.1f} s. "
        f"Counting drops where the dog happened to land upright, the totals are "
        f"{counts(all_starts[0])}, {counts(all_starts[1])} and {counts(all_starts[2])}. [Per seed](results/report/getup.csv)",
        "",
        f"**Training simulator vs MuJoCo.** For one dog seed, tracking error differs by at most {max_gap:.3f} "
        f"between MJX (Warp), where it trained, and standard MuJoCo; J50 is {gap_j50['mjx_paired_j50']:.1f} vs "
        f"{gap_j50['mujoco_paired_j50']:.1f} N·s over {gap_j50['paired_trials']} matched shove trials. "
        "[Comparison](results/report/gap.csv)",
    ]
    g1_recovery = [r for r in tables.get("g1_getup", []) if r["start_pose"] == "fallen"
                   and r["complete"] and r["count_success"] >= 0.8]
    if g1_recovery:
        recovery = g1_recovery[-1]
        lines += ["", f"**Humanoid recovery.** The custom task's seed {recovery['seed']} gets up in "
                  f"{recovery['successes']}/{recovery['trials']} fallen-start trials; median recovery "
                  f"time {recovery['median_recovery_seconds']:.1f} s. "
                  "[All attempted configurations](results/report/g1_getup.csv)"]
    terrain_lines = terrain_readme_lines(tables.get("terrain", []))
    terrain_start, terrain_end = "<!-- generated:terrain:start -->", "<!-- generated:terrain:end -->"
    if terrain_lines:
        lines += ["", terrain_start, *terrain_lines, terrain_end]
    elif terrain_start in text and terrain_end in text:
        # An incomplete new run must not erase an already published comparison.
        previous = text.partition(terrain_start)[2].partition(terrain_end)[0]
        lines += ["", terrain_start + previous + terrain_end]
    walking = [r for r in published if r["env"] == "Go1JoystickFlatTerrain"]
    recovering = [r for r in published if r["env"] == "Go1Getup"]
    humanoid = [r for r in published if r["env"] == "G1JoystickFlatTerrain"]
    rough = [r for r in tables["training"] if r["env"] == "Go1JoystickRoughTerrain"]
    published += rough
    gpus = {r["gpu"] for r in published}
    if len(gpus) != 1:
        raise ValueError("Published cost wording requires one common GPU type")
    minute_range = lambda rows: f"{min(r['minutes'] for r in rows):.0f}–{max(r['minutes'] for r in rows):.0f}"
    cost_range = lambda rows: f"${min(r['usd_upper'] for r in rows):.2f}–{max(r['usd_upper'] for r in rows):.2f}"
    cost = (
        f"NVIDIA {next(iter(gpus))} training: flat-ground dog {minute_range(walking)} minutes "
        f"({cost_range(walking)}), dog get-up {minute_range(recovering)} ({cost_range(recovering)}), "
        f"walking humanoid {minute_range(humanoid)} ({cost_range(humanoid)}) per policy. "
    )
    if rough:
        cost += f"Rough-ground dog: {minute_range(rough)} minutes ({cost_range(rough)}). "
    cost += (
        f"The {len(published)} published policies cost ${sum(r['usd_upper'] for r in published):.2f}; "
        f"all cloud jobs, including failures, ${spend['usd_upper']:.2f}. "
        "Wall-time/list-price upper bounds, not invoices. "
        "[Runs](results/report/training.csv) · [All jobs](results/report/invocations.csv)"
    )
    for name, content in (("headline", headline), ("results", "\n".join(lines)), ("cost", cost)):
        start, end = f"<!-- generated:{name}:start -->", f"<!-- generated:{name}:end -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"README must contain one {name} result block")
        before, _, remainder = text.partition(start)
        _, _, after = remainder.partition(end)
        text = before + start + "\n" + content + "\n" + end + after
    limit_prefix = f"- {words[len(weak)].capitalize()} of {words[len(getup_seeds)]} get-up seeds "
    limits = [line for line in text.splitlines() if line.startswith(limit_prefix)]
    if len(limits) != 1:
        raise ValueError("README must contain one get-up limitation")
    text = text.replace(limits[0], limit_prefix +
                        f"rarely gets up ({weak[0]['successes']} of {weak[0]['trials']} fallen starts); "
                        "I report it rather than retrain it.")
    path.write_text(text)


def generate_report(inputs, output, *, bootstrap_replicates=BOOTSTRAP_REPLICATES,
                    bootstrap_seed=BOOTSTRAP_SEED, terrain_inputs=None, root=None):
    """Regenerate reports and web survival counts; terrain inputs stay independent.

    By default terrain raw files come from results/terrain/<invocation>/<policy>/.
    An explicit terrain_inputs list replaces that discovery, including [] to omit it.
    """
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    rows, provenance = load_trials(inputs, root=root)
    if terrain_inputs is None:
        terrain_inputs = sorted((root / "results/terrain").glob("*/*/trials.jsonl"))
        terrain_inputs += sorted((root / "results/terrain").glob("*/*/trials.jsonl.gz"))
    terrain_rows, terrain_sources = load_terrain_trials(terrain_inputs, root=root)
    terrain, terrain_seeds = terrain_tables(terrain_rows)
    provenance.extend(terrain_sources)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(bootstrap_seed)
    tracking, commands = tracking_table(rows)
    push, thresholds, fits = push_tables(rows, rng, bootstrap_replicates)
    seed_push, seed_thresholds = seed_push_tables(rows)
    ablation = ablation_table(fits, rng, bootstrap_replicates)
    getup = getup_table(rows, rng, bootstrap_replicates)
    getup_trials = [{**{key: r.get(key) for key in ("backend", "policy", "seed", "trial_id", "trial", "success", "recovery_time")},
                    "start_pose": getup_start_pose(r)}
                   for r in rows if r["suite"] == "getup"]
    gap = gap_table(rows, fits)
    training, benchmarks, invocations, spend, run_sources = run_tables(root)
    diagnostic, diagnostic_sources = getup_diagnostic_table(root, rng, bootstrap_replicates)
    g1_getup, g1_sources = experimental_getup_table(root, rng, bootstrap_replicates)
    provenance.extend(g1_sources)
    provenance.extend(diagnostic_sources)
    provenance.extend(run_sources)
    tables = {"tracking": tracking, "tracking_commands": commands, "push": push,
              "push_seeds": seed_push, "j50_seeds": seed_thresholds,
              "j50": thresholds, "ablation": ablation, "getup": getup, "gap": gap,
              "training": training, "benchmarks": benchmarks, "invocations": invocations,
              "getup_diagnostic": diagnostic, "g1_getup": g1_getup,
              "terrain": terrain, "terrain_seeds": terrain_seeds}
    tables["getup_trials"] = getup_trials
    tables["reruns"] = [source["rerun_comparison"] for source in provenance if "rerun_comparison" in source]
    columns = {
        "terrain": ("backend", "arm", "terrain", "seeds", "trials", "completed", "successes",
                    "falls", "fall_rate", "numerical_failures", "failures", "forward_mae",
                    "lateral_mae", "yaw_mae", "forward_mae_trials", "lateral_mae_trials",
                    "yaw_mae_trials", "complete_grid", "three_seed_protocol"),
        "terrain_seeds": ("backend", "arm", "terrain", "seed", "seeds", "trials", "completed",
                          "successes", "falls", "fall_rate", "numerical_failures", "failures",
                          "forward_mae", "lateral_mae", "yaw_mae", "forward_mae_trials",
                          "lateral_mae_trials", "yaw_mae_trials", "complete_grid"),
        "invocations": ("guard_id", "id", "stage", "gpu", "planned_minutes", "minutes",
                        "usd_upper", "accounting", "exit", "manifest"),
        "training": ("id", "env", "seed", "gpu", "minutes", "usd_upper", "final_reward",
                     "requested_steps", "trained_steps", "explicit_reset_interval_seconds", "completion"),
        "getup_diagnostic": ("run_id", "training_run_id", "complete", "policy_sha256", "backend", "initialization", "seed",
                            "trials", "successes", "success", "nominal_trials", "nominal_successes",
                            "dropped_trials", "dropped_successes", "ci_low", "ci_high", "median_recovery_seconds",
                            "software_failures", "raw"),
        "benchmarks": ("id", "gpu", "steps_per_second", "measurement_seconds", "warmup_seconds",
                       "minutes", "usd_upper", "charged_usd_upper_per_million_steps",
                       "usd_upper_per_million_steps", "scope"),
        "tracking": ("backend", "arm", "seed", "trials", "completed", "completion", "samples", "forward_mae", "lateral_mae", "yaw_mae", "complete_grid", "standard_protocol"),
        "tracking_commands": ("backend", "arm", "seed", "command", "trials", "completed", "completion", "samples", "forward_mae", "lateral_mae", "yaw_mae"),
        "push": ("backend", "arm", "impulse", "trials", "successes", "seeds", "survival", "monotone_survival", "ci_low", "ci_high", "complete_grid", "three_seed_protocol", "balanced_seeds"),
        "j50": ("backend", "arm", "seeds", "j50", "censoring", "grid_bound", "grid_min", "grid_max", "ci_low", "ci_high", "bootstrap_censored_fraction", "complete_grid", "three_seed_protocol"),
        "push_seeds": ("backend", "arm", "seed", "impulse", "trials", "survival", "monotone_survival"),
        "j50_seeds": ("backend", "arm", "seed", "j50", "censoring", "grid_bound", "complete_grid"),
        "ablation": ("backend", "comparator", "arm", "seeds", "j50_delta", "ci_low", "ci_high", "baseline_censoring", "push_arm_censoring", "bootstrap_censored_fraction", "paired_trials", "unpaired_trials", "three_seed_protocol"),
        "getup": ("backend", "arm", "start_pose", "seed", "seeds", "trials", "successes", "equal_seed_success", "count_success", "ci_low", "ci_high", "mean_recovery_seconds", "median_recovery_seconds", "time_ci_low", "time_ci_high", "complete", "three_seed_protocol"),
        "g1_getup": ("run_id", "training_run_id", "policy_sha256", "backend", "arm", "start_pose",
                     "seed", "seeds", "trials", "successes", "count_success", "ci_low", "ci_high",
                     "median_recovery_seconds", "complete", "raw"),
        "getup_trials": ("backend", "policy", "seed", "trial_id", "trial", "start_pose", "success", "recovery_time"),
        "reruns": ("policy", "original_run", "repeat_run", "trials", "git_sha", "git_dirty",
                   "original_raw_sha256", "repeat_raw_sha256", "config_equal", "policy_metadata_equal",
                   "successes_original", "successes_repeat", "outcomes_equal", "ignored_fields", "differing_fields"),
        "gap": ("arm", "suite", "impulse", "seed", "paired_trials", "unpaired_trials", "mjx_minus_mujoco_success", "mjx_minus_mujoco_forward_mae", "mjx_minus_mujoco_lateral_mae", "mjx_minus_mujoco_yaw_mae", "mae_pairs", "j50_delta", "mujoco_paired_j50", "mjx_paired_j50", "mujoco_censoring", "mujoco_grid_bound", "mjx_censoring", "same_grid", "complete_grid")}
    for name, table in tables.items():
        _csv(output / f"{name}.csv", table, columns[name])
    failures = defaultdict(int)
    for r in rows:
        if not r["success"]:
            failures[(r["backend"], r["arm"], r["suite"], r.get("failure") or "unspecified")] += 1
    failure_rows = [dict(zip(("backend", "arm", "suite", "failure"), key), trials=count) for key, count in sorted(failures.items())]
    _csv(output / "failures.csv", failure_rows, ("backend", "arm", "suite", "failure", "trials"))
    tables["failures"] = failure_rows
    push_series = defaultdict(list)
    for r in push:
        push_series[f'{r["backend"]}: {r["arm"]}'].append((r["impulse"], r["monotone_survival"], r["ci_low"], r["ci_high"]))
    _svg(output / "push.svg", "Push survival (simulation)", sorted(push_series.items()), "Impulse (N s)", "Survival", probability=True)
    seed_series = defaultdict(list)
    for r in seed_push:
        seed_series[f'{r["backend"]}: {r["arm"]} seed {r["seed"]}'].append((r["impulse"], r["monotone_survival"], None, None))
    _svg(output / "push_seeds.svg", "Per-seed push survival", sorted(seed_series.items()), "Impulse (N s)", "Survival", probability=True)
    for metric, filename, label in (
            ("forward_mae", "tracking.svg", "Forward velocity MAE (m/s)"),
            ("lateral_mae", "tracking_lateral.svg", "Lateral velocity MAE (m/s)"),
            ("yaw_mae", "tracking_yaw.svg", "Yaw-rate MAE (rad/s)")):
        series = defaultdict(list)
        for r in tracking:
            series[f'{r["backend"]}: {r["arm"]}'].append((r["seed"], r[metric], None, None))
        _svg(output / filename, "Conditional tracking error", sorted(series.items()), "Training seed", label)
    getup_series, time_series = defaultdict(list), defaultdict(list)
    for r in getup:
        if r["seed"] == "all":
            continue
        label = f'{r["backend"]}: {r["arm"]} ({r["start_pose"]})'
        getup_series[label].append((r["seed"], r["count_success"], r["ci_low"], r["ci_high"]))
        time_series[label].append((r["seed"], r["mean_recovery_seconds"], r["time_ci_low"], r["time_ci_high"]))
    _svg(output / "getup.svg", "Get-up success (simulation)", sorted(getup_series.items()), "Training seed", "Success", probability=True)
    _svg(output / "getup_time.svg", "Conditional get-up time", sorted(time_series.items()), "Training seed", "Recovery time (s)")
    threshold_series = defaultdict(list)
    arm_index = {arm: i for i, arm in enumerate(sorted({r["arm"] for r in thresholds}))}
    for r in thresholds:
        threshold_series[r["backend"]].append((arm_index[r["arm"]], r["j50"], r["ci_low"], r["ci_high"]))
    labels = "; ".join(f"{index}: {arm}" for arm, index in arm_index.items())
    _svg(output / "j50.svg", "J50 (censored values omitted; see CSV)", sorted(threshold_series.items()), labels or "Policy", "Impulse (N s)")
    gap_series = defaultdict(list)
    for r in gap:
        if r["suite"] == "push":
            gap_series[r["arm"]].append((r["impulse"], r["mjx_minus_mujoco_success"], None, None))
    _svg(output / "gap.svg", "One-seed MJX minus C MuJoCo survival", sorted(gap_series.items()), "Impulse (N s)", "Survival difference")
    summary = {"schema_version": 1, "inputs": provenance, "trial_count": len(rows),
               "spend": spend,
               "terrain_trial_count": len(terrain_rows),
               "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed,
                             "method": "hierarchical seeds then whole trial-index blocks across every impulse; paired Go1 arms",
                             "interval": "95% percentile; J50 interval withheld if any replicate is censored"},
               "tables": tables, "figures": ["push.svg", "push_seeds.svg", "tracking.svg", "tracking_lateral.svg", "tracking_yaw.svg", "getup.svg", "getup_time.svg", "j50.svg", "gap.svg"],
               "notes": ["Tracking per-axis MAE is conditional on available post-discard samples.",
                         "Terrain tracking errors average episodes including time after physical falls; "
                         "metric trial counts expose unavailable numerical-failure measurements.",
                         "Get-up times are conditional on success.",
                         "MJX comparisons use trained seed 0 only.",
                         "Missing suites and seeds are not imputed; see completeness columns."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    survival_path = root / "web/src/data/survival.json"
    survival_path.parent.mkdir(parents=True, exist_ok=True)
    survival_path.write_text(json.dumps(survival_data(rows), indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terrain-inputs", nargs="+", type=Path,
                        help="Terrain raw files/directories (default: results/terrain)")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--update-readme", action="store_true",
                        help="Update published README measurements; requires the complete experiment")
    args = parser.parse_args()
    summary = generate_report(args.inputs, args.output, bootstrap_replicates=args.bootstrap_replicates,
                              bootstrap_seed=args.bootstrap_seed, terrain_inputs=args.terrain_inputs)
    if args.update_readme:
        update_readme(Path(__file__).resolve().parents[1], summary["tables"], summary["spend"])
    print(f'{summary["trial_count"]} raw trials; report written to {args.output}')


if __name__ == "__main__":
    main()
