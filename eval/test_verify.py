"""Deliberately corrupted artifacts that `python -m eval.verify` must reject."""
import gzip
import hashlib
import json
import re
import shutil

import pytest

from eval.report import generate_report
from eval.verify import ROOT, readme_problems, verify


ARMS = ("go1-walk", "go1-walk-push", "g1-walk")
RAW = "results/eval/example/mujoco/{}/trials.jsonl.gz"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def _write_rows(path, rows):
    with gzip.open(path, "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)


def _repository(root):
    """A small committed state: model, exports, sidecar table, fixture, raw trials and report."""
    model = root / "assets/go1/walk.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>\n")
    (root / "assets/go1/walk.provenance.json").write_text(json.dumps({"files": {"walk.xml": _sha(model)}}))
    policies = root / "policies"
    policies.mkdir()
    table = ["| Policy | SHA-256 |", "|---|---|"]
    for arm in ARMS:
        for seed in range(3):
            name = f"{arm}-s{seed}"
            sidecar = hashlib.sha256(name.encode()).hexdigest()
            export = policies / f"{name}.json"
            export.write_text(json.dumps({
                "run_id": name, "model_xml": "assets/go1/walk.xml",
                "brax_params_file": f"{name}.brax", "brax_params_sha256": sidecar,
            }) + "\n")
            table.append(f"| {name}.brax | {sidecar} |")
            raw = root / RAW.format(name)
            raw.parent.mkdir(parents=True)
            (raw.parent / "manifest.json").write_text(json.dumps({
                "run_id": "example", "backend": "mujoco", "status": "complete", "seeds": [seed],
                "git_sha": "a" * 40, "git_dirty": False, "wall_seconds": 1, "modal_app_id": None,
                "config": {"seeds": [seed], "go1_impulses": [0, 2], "g1_impulses": [0, 2], "push_trials": 2},
                "policy_metadata": {name: {
                    "policy_sha256": _sha(export), "spec": {"run_id": name},
                    "model_xml": "assets/go1/walk.xml", "model_sha256": _sha(model)}},
            }))
            _write_rows(raw, [
                {"run_id": "example", "trial_id": f"{name}/push/{impulse}/{trial}", "backend": "mujoco",
                 "policy": name, "robot": arm.split("-")[0], "suite": "push", "seed": seed,
                 "trial": trial, "impulse": impulse, "theta": trial / 2, "push_onset": 2 + trial / 10,
                 "command": [0.5, 0, 0], "success": impulse == 0 or trial == 1, "wall_seconds": 0.1}
                for impulse in (0, 2) for trial in range(2)
            ])
    (policies / "README.md").write_text("\n".join(table) + "\n")
    fixture = root / "results/parity/go1-walk-s0.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(json.dumps({"policy": "policies/go1-walk-s0.json",
                                   "policy_sha256": _sha(policies / "go1-walk-s0.json")}))
    (root / "README.md").write_text("# example\n")
    generate_report([root / "results/eval"], root / "results/report", root=root, bootstrap_replicates=3)
    return root


def _edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value) + "\n")


def _edit_policy(root):
    _edit_json(root / "policies/go1-walk-s0.json", lambda spec: spec.update(control_dt=0.03))


def _edit_model(root):
    (root / "assets/go1/walk.xml").write_text('<mujoco><option timestep="0.005"/></mujoco>\n')


def _flip_outcome(root):
    path = root / RAW.format("go1-walk-s1")
    rows = _rows(path)
    rows[0]["success"] = not rows[0]["success"]
    _write_rows(path, rows)


def _truncate_raw(root):
    path = root / RAW.format("go1-walk-s1")
    path.write_bytes(path.read_bytes()[:40])


def _flip_raw_byte(root):
    path = root / RAW.format("go1-walk-s1")
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


def _rehash_manifest(root):
    # Hide a changed export by updating its evaluation manifest: the report hashes manifests.
    export = root / "policies/go1-walk-s1.json"
    _edit_json(export, lambda spec: spec.update(control_dt=0.03))
    manifest = root / RAW.format("go1-walk-s1").replace("trials.jsonl.gz", "manifest.json")
    _edit_json(manifest, lambda m: m["policy_metadata"]["go1-walk-s1"].update(policy_sha256=_sha(export)))


def _edit_table(root):
    path = root / "results/report/push.csv"
    path.write_text(path.read_text().replace("mujoco", "mjx", 1))


def _edit_survival(root):
    _edit_json(root / "web/src/data/survival.json",
               lambda data: data["policies"]["go1-walk-s0"]["trials"][0].update(survived=6))


def _edit_sidecar_table(root):
    path = root / "policies/README.md"
    path.write_text(re.sub(r"(go1-walk-s0\.brax \| )[0-9a-f]", r"\g<1>x", path.read_text()))


CORRUPTIONS = {
    "policy export edited after evaluation": (_edit_policy, "policies/go1-walk-s0.json has SHA-256"),
    "evaluated model edited": (_edit_model, "assets/go1/walk.xml has SHA-256"),
    "raw outcome flipped": (_flip_outcome, "results/report/push.csv differs from its regeneration"),
    "raw file truncated": (_truncate_raw, "Report regeneration failed"),
    "raw file byte flipped": (_flip_raw_byte, "Report regeneration failed"),
    "manifest rehashed to hide an edited export": (
        _rehash_manifest, "results/report/summary.json differs from its regeneration"),
    "report table edited": (_edit_table, "results/report/push.csv differs from its regeneration"),
    "report table deleted": (lambda root: (root / "results/report/j50.csv").unlink(),
                             "results/report/j50.csv is regenerated but not committed"),
    "stale report table": (lambda root: (root / "results/report/old.csv").write_text("x\n"),
                           "results/report/old.csv is committed but no longer generated"),
    "browser survival counts edited": (_edit_survival, "web/src/data/survival.json differs"),
    "sidecar hash table edited": (_edit_sidecar_table, "policies/README.md: go1-walk-s0.brax"),
}


def test_untouched_repository_passes(tmp_path):
    problems, counts = verify(_repository(tmp_path))
    assert problems == []
    assert counts["evaluated policy exports"] == 9
    assert counts["evaluated model files"] == 9
    assert counts["parity fixtures"] == 1
    assert counts["generated files"] == len(list((tmp_path / "results/report").iterdir())) + 1


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_corrupted_artifact_is_rejected(tmp_path, corruption):
    root = _repository(tmp_path)
    corrupt, expected = CORRUPTIONS[corruption]
    corrupt(root)
    problems, _ = verify(root)
    assert any(expected in problem for problem in problems), problems


def test_readme_numbers_must_match_the_report(tmp_path):
    summary = json.loads((ROOT / "results/report/summary.json").read_text())
    shutil.copyfile(ROOT / "README.md", tmp_path / "README.md")
    assert readme_problems(tmp_path, summary["tables"], summary["spend"]) == []
    head, marker, rest = (tmp_path / "README.md").read_text().partition("<!-- generated:headline:start -->")
    tampered = re.sub(r"\d+\.\d", lambda m: f"{float(m.group()) + 1:.1f}", rest, count=1)
    (tmp_path / "README.md").write_text(head + marker + tampered)
    assert readme_problems(tmp_path, summary["tables"], summary["spend"]) == [
        "README.md generated blocks differ from the regenerated report"]
