"""Check committed artifacts against their recorded hashes, then regenerate the report byte for byte.

Run from the checkout: python -m eval.verify
It needs only the minimal evaluation dependencies and exits 1 listing every problem.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile

from eval.report import generate_report, render_readme


ROOT = Path(__file__).resolve().parents[1]
# The report publishes these inputs; the other directories hold diagnostics and checks.
PUBLISHED = ("results/eval", "results/terrain")
MANIFESTS = (
    "results/eval/*/*/*/manifest.json", "results/getup-diagnostic/*/*/*/manifest.json",
    "results/policy-checks/*/*/*/manifest.json", "results/terrain/*/*/manifest.json",
)
SIDECAR_ROW = re.compile(r"^\| (\S+\.brax) \| ([0-9a-f]{64}) \|$", re.MULTILINE)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check(problems, root, path, expected, label):
    path = Path(path)
    name = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    if not path.is_file():
        problems.append(f"{label}: {name} is missing")
    elif (actual := _sha(path)) != expected:
        problems.append(f"{label}: {name} has SHA-256 {actual}, recorded {expected}")


def _superseded(manifest, invocation):
    same_run = invocation.get("run_id") in (None, manifest["run_id"])
    return any(m.get("status") == "superseded" or m.get("superseded")
               for m in (manifest, invocation if same_run else {}))


def _retained_export(root, name, run_id):
    """The committed export of policy `name` from training run `run_id`, if any."""
    for path in (root / "results/runs" / run_id / "policies" / f"{name}.json",
                 root / "policies" / f"{name}.json"):
        if path.is_file() and json.loads(path.read_text()).get("run_id") == run_id:
            return path
    return None


def artifact_problems(root=ROOT):
    """Return (problems, counts) for asset, sidecar, fixture and evaluation-manifest hashes."""
    root = Path(root).resolve()
    problems, counts = [], Counter()
    for record_path in sorted((root / "assets").glob("*/*.provenance.json")):
        record = json.loads(record_path.read_text())
        label = record_path.relative_to(root).as_posix()
        for name, digest in record.get("files", {}).items():
            _check(problems, root, record_path.parent / name, digest, label)
            counts["asset files"] += 1
        if "model_sha256" in record:
            model = record_path.with_name(record_path.name.removesuffix(".provenance.json") + ".xml")
            _check(problems, root, model, record["model_sha256"], label)
            counts["asset files"] += 1
        if "source_model" in record:
            _check(problems, root, root / record["source_model"], record["source_sha256"], label)
            counts["asset files"] += 1

    registry = {}
    for path in sorted((root / "policies").glob("*.json")):
        spec = json.loads(path.read_text())
        registry[spec["brax_params_file"]] = spec["brax_params_sha256"]
    table = dict(SIDECAR_ROW.findall((root / "policies/README.md").read_text()))
    for name in sorted(registry.keys() | table.keys()):
        if registry.get(name) != table.get(name):
            problems.append(f"policies/README.md: {name} is {table.get(name)}, "
                            f"policy JSON records {registry.get(name)}")
        counts["sidecar registry entries"] += 1

    for path in sorted((root / "results/parity").glob("*.json")):
        fixture = json.loads(path.read_text())
        _check(problems, root, root / fixture["policy"], fixture["policy_sha256"],
               path.relative_to(root).as_posix())
        counts["parity fixtures"] += 1

    for path in sorted(p for pattern in MANIFESTS for p in root.glob(pattern)):
        manifest = json.loads(path.read_text())
        terrain = path.is_relative_to(root / "results/terrain")
        invocation_path = path.parents[1 if terrain else 2] / "manifest.json"
        invocation = json.loads(invocation_path.read_text()) if invocation_path.is_file() else {}
        if _superseded(manifest, invocation):
            counts["superseded manifests skipped"] += 1
            continue
        label = path.relative_to(root).as_posix()
        published = any(path.is_relative_to(root / p) for p in PUBLISHED)
        counts["evaluation manifests"] += 1
        for name, meta in manifest["policy_metadata"].items():
            run_id = meta["training_run"] if terrain else meta["spec"]["run_id"]
            export = _retained_export(root, name, run_id)
            if export is not None:
                _check(problems, root, export, meta["policy_sha256"], label)
                counts["evaluated policy exports"] += 1
            elif published:
                problems.append(f"{label}: no committed export of {name} from training run {run_id}")
            else:
                counts["historical exports not retained"] += 1
            models = meta["models"].values() if terrain else [
                {"path": meta["model_xml"], "sha256": meta["model_sha256"]}]
            for model in models:
                _check(problems, root, root / model["path"], model["sha256"], label)
                counts["evaluated model files"] += 1
    return problems, counts


def readme_problems(root, tables, spend):
    """Compare README.md with its generated blocks rendered from report tables."""
    text = (Path(root) / "README.md").read_text()
    if "<!-- generated:" not in text:
        return []
    try:
        expected = render_readme(Path(root), tables, spend)
    except ValueError as exc:
        return [f"README.md cannot be regenerated: {exc}"]
    return [] if expected == text else ["README.md generated blocks differ from the regenerated report"]


def report_problems(root=ROOT):
    """Regenerate results/report, survival.json and README blocks; return byte differences."""
    root = Path(root).resolve()
    committed = root / "results/report"
    summary_path = committed / "summary.json"
    if not summary_path.is_file():
        return [f"{summary_path.relative_to(root).as_posix()} is missing"], Counter()
    bootstrap = json.loads(summary_path.read_text())["bootstrap"]
    problems, counts = [], Counter()
    with tempfile.TemporaryDirectory(prefix="weeble-verify-") as temporary:
        output, survival = Path(temporary) / "report", Path(temporary) / "survival.json"
        try:
            summary = generate_report(
                [root / "results/eval"], output, root=root, survival_path=survival,
                bootstrap_replicates=bootstrap["replicates"], bootstrap_seed=bootstrap["seed"])
        except Exception as exc:  # Corrupt inputs fail in many ways; each is a finding.
            return [f"Report regeneration failed: {type(exc).__name__}: {exc}"], counts
        pairs = [(committed / name, output / name)
                 for name in sorted({p.name for p in committed.iterdir()} | {p.name for p in output.iterdir()})]
        pairs.append((root / "web/src/data/survival.json", survival))
        for published, regenerated in pairs:
            name = published.relative_to(root).as_posix()
            if not published.is_file():
                problems.append(f"{name} is regenerated but not committed")
            elif not regenerated.is_file():
                problems.append(f"{name} is committed but no longer generated")
            elif published.read_bytes() != regenerated.read_bytes():
                problems.append(f"{name} differs from its regeneration "
                                f"({_sha(published)} committed, {_sha(regenerated)} regenerated)")
            counts["generated files"] += 1
        problems += readme_problems(root, summary["tables"], summary["spend"])
    return problems, counts


def verify(root=ROOT):
    problems, counts = artifact_problems(root)
    report, report_counts = report_problems(root)
    return problems + report, counts + report_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    problems, counts = verify(args.root)
    for name, count in sorted(counts.items()):
        print(f"{count:6d} {name}")
    if problems:
        parser.exit(1, "".join(f"FAIL {problem}\n" for problem in problems))
    print("All recorded hashes match and the report regenerates byte for byte.")


if __name__ == "__main__":
    main()
