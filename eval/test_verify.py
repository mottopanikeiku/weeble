"""Deliberately corrupted artifacts that `python -m eval.verify` must reject."""
import gzip
import hashlib
import json
import re
import shutil

import pytest

from eval.atlas import _write_rows, read_rows
from eval.report import generate_report
from eval.verify import ROOT, atlas_problems, readme_problems, verify


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


@pytest.fixture
def small_readme(monkeypatch):
    """The small repository can't publish the README blocks; the real README is tested below."""
    from eval import verify as module

    calls = []
    monkeypatch.setattr(module, "readme_problems", lambda root, tables, spend: calls.append(root) or [])
    return calls


def test_untouched_repository_passes(tmp_path, small_readme):
    problems, counts = verify(_repository(tmp_path))
    assert problems == []
    assert small_readme == [tmp_path.resolve()]
    assert counts["evaluated policy exports"] == 9
    assert counts["evaluated model files"] == 9
    assert counts["parity fixtures"] == 1
    assert counts["generated files"] == len(list((tmp_path / "results/report").iterdir())) + 1


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_corrupted_artifact_is_rejected(tmp_path, corruption, small_readme):
    root = _repository(tmp_path)
    corrupt, expected = CORRUPTIONS[corruption]
    corrupt(root)
    problems, _ = verify(root)
    assert any(expected in problem for problem in problems), problems


def _readme(tmp_path, edit):
    summary = json.loads((ROOT / "results/report/summary.json").read_text())
    (tmp_path / "README.md").write_text(edit((ROOT / "README.md").read_text()))
    return readme_problems(tmp_path, summary["tables"], summary["spend"])


def test_readme_without_generated_markers_is_rejected(tmp_path):
    problems = _readme(tmp_path, lambda text: re.sub(r"<!-- generated:\w+:(start|end) -->", "", text))
    assert len(problems) == 8
    assert "README.md must contain <!-- generated:headline:start --> exactly once (found 0)" in problems


def test_readme_missing_one_block_is_rejected(tmp_path):
    cut = lambda text: re.sub(r"<!-- generated:cost:start -->.*?<!-- generated:cost:end -->", "", text, flags=re.S)
    assert _readme(tmp_path, cut) == [
        "README.md must contain <!-- generated:cost:start --> exactly once (found 0)",
        "README.md must contain <!-- generated:cost:end --> exactly once (found 0)"]


def test_readme_with_a_renamed_marker_is_rejected(tmp_path):
    rename = lambda text: text.replace("<!-- generated:terrain:end -->", "<!-- generated:terrain:stop -->")
    assert _readme(tmp_path, rename) == [
        "README.md must contain <!-- generated:terrain:end --> exactly once (found 0)"]


def test_readme_numbers_must_match_the_report(tmp_path):
    summary = json.loads((ROOT / "results/report/summary.json").read_text())
    shutil.copyfile(ROOT / "README.md", tmp_path / "README.md")
    assert readme_problems(tmp_path, summary["tables"], summary["spend"]) == []
    head, marker, rest = (tmp_path / "README.md").read_text().partition("<!-- generated:headline:start -->")
    tampered = re.sub(r"\d+\.\d", lambda m: f"{float(m.group()) + 1:.1f}", rest, count=1)
    (tmp_path / "README.md").write_text(head + marker + tampered)
    assert readme_problems(tmp_path, summary["tables"], summary["spend"]) == [
        "README.md generated blocks differ from the regenerated report"]


def _atlas_repository(root):
    """The committed atlas and README; exports and scenes are links into the checkout."""
    shutil.copytree(ROOT / "results/atlas", root / "results/atlas")
    shutil.copyfile(ROOT / "README.md", root / "README.md")
    (root / "policies").mkdir()
    for path in (ROOT / "policies").glob("*.json"):
        (root / "policies" / path.name).symlink_to(path)
    for scene in ("assets/go1/walk.xml", "assets/go1/getup.xml", "assets/g1/walk.xml"):
        (root / scene).parent.mkdir(parents=True, exist_ok=True)
        (root / scene).symlink_to(ROOT / scene)
    return root


def _atlas_rows(root, change, rehash=True):
    path = root / "results/atlas/go1-walk-s0.jsonl.gz"
    rows = read_rows(path)
    change(rows)
    _write_rows(path, rows)
    if rehash:
        _edit_json(root / "results/atlas/manifest.json",
                   lambda manifest: manifest["raw"]["go1-walk-s0"].update(sha256=_sha(path)))


def _fell(rows):
    next(r for r in rows if r["axis"] == "shove" and r["outcome"] == "never_fell")["outcome"] = "fell"


def _replace_export(root):
    path = root / "policies/go1-walk-s0.json"
    text = path.read_text()
    path.unlink()
    path.write_text(text + "\n")


def _edit_atlas_readme(root):
    path = root / "README.md"
    head, marker, rest = path.read_text().partition("<!-- generated:atlas:start -->")
    path.write_text(head + marker + re.sub(r"\d+", lambda m: str(int(m.group()) + 1), rest, count=1))


ATLAS_CORRUPTIONS = {
    "episode outcome flipped": (lambda root: _atlas_rows(root, _fell, rehash=False),
                                "go1-walk-s0.jsonl.gz does not match its recorded SHA-256"),
    "episode outcome flipped and rehashed": (lambda root: _atlas_rows(root, _fell),
                                             "results/atlas/report/atlas_seeds.csv differs"),
    "episode deleted and rehashed": (lambda root: _atlas_rows(root, lambda rows: rows.pop()),
                                     "not exactly one row per cell and trial"),
    "nominal episode changed": (lambda root: _atlas_rows(root, lambda rows: rows[0].update(outcome="fell")),
                                "Nominal episodes differ"),
    "configuration edited": (lambda root: _edit_json(root / "results/atlas/manifest.json",
                                                     lambda m: m["config"].update(trials=49)),
                             "configuration does not match its recorded hashes"),
    "atlas table edited": (lambda root: (root / "results/atlas/report/margins.csv").write_text("arm\n"),
                           "results/atlas/report/margins.csv differs"),
    "README atlas number edited": (_edit_atlas_readme, "README.md atlas block differs"),
    "export changed after the atlas ran": (_replace_export, "policies/go1-walk-s0.json changed after the atlas ran"),
}


def test_committed_atlas_regenerates():
    problems, counts = atlas_problems(ROOT)
    assert problems == [] and counts["atlas episodes"] > 0


@pytest.mark.parametrize("corruption", sorted(ATLAS_CORRUPTIONS))
def test_corrupted_atlas_is_rejected(tmp_path, corruption):
    root = _atlas_repository(tmp_path)
    corrupt, expected = ATLAS_CORRUPTIONS[corruption]
    corrupt(root)
    problems, _ = atlas_problems(root)
    assert any(expected in problem for problem in problems), problems
