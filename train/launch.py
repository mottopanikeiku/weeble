"""Book each Modal invocation, then attach its measured ledger charge to the manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RATES = {"none": 0, "L4": 0.7992, "A100-40GB": 2.0988, "H100": 3.9492}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["assets", "smoke", "benchmark", "train", "eval", "parity", "fetch-eval", "fetch-training", "getup-diagnostic"])
    parser.add_argument("--gpu", choices=list(RATES), default="L4")
    parser.add_argument("--minutes", type=float, default=5)
    parser.add_argument("--cores", type=float, default=2)
    parser.add_argument("--mem-gib", type=int, default=8)
    parser.add_argument("--env", default="Go1JoystickFlatTerrain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pushes", action="store_true")
    parser.add_argument("--policy", default="")
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--remote-run", default="")
    parser.add_argument("--num-timesteps", type=int, default=0)
    parser.add_argument("--initialization", choices=["drop", "upstream"], default="drop")
    args = parser.parse_args()
    project = Path(os.environ.get("RL_WAVE_HOME", Path.home() / "Projects/rl-wave"))
    guard = project / "bin/rl-modal"
    ledger = project / "modal/ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    ends = {r["id"]: r for r in records if r["event"] == "end"}
    used = sum(ends.get(r["id"], {}).get("charged_upper", r["booked"])
               for r in records if r["event"] == "start" and r["project"] == "pushable")
    rate = RATES[args.gpu] + args.cores * 0.047160 + args.mem_gib * 0.007992
    booked = rate * args.minutes / 60 * 1.10
    if used + booked > 12:
        raise SystemExit(f"Preserving reserve: ${used:.4f} used, ${booked:.4f} booking exceeds $12 trainer limit")
    run_id = args.run_id or f"{args.stage}-{args.env}-s{args.seed}-{args.gpu}-{int(time.time())}"
    command = [str(guard), "run", "--project", "pushable", "--gpu", args.gpu,
               "--minutes", str(args.minutes), "--cores", str(args.cores), "--mem-gib", str(args.mem_gib),
               "--", "modal", "run", "train/app.py", "--stage", args.stage,
               "--env", args.env, "--seed", str(args.seed), "--run-id", run_id]
    if args.pushes:
        command.append("--pushes")
    if args.policy:
        command.extend(["--policy", args.policy])
    if args.backend != "mujoco":
        command.extend(["--backend", args.backend])
    if args.remote_run:
        command.extend(["--remote-run", args.remote_run])
    if args.num_timesteps:
        command.extend(["--num-timesteps", str(args.num_timesteps)])
    if args.initialization != "drop":
        command.extend(["--initialization", args.initialization])
    env = dict(os.environ, PUSHABLE_GPU=args.gpu, PUSHABLE_TIMEOUT=str(int(args.minutes * 60)),
               PUSHABLE_CPU=str(args.cores), PUSHABLE_MEMORY=str(args.mem_gib * 1024),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    source = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    }
    code = subprocess.call(command, cwd=ROOT, env=env)
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    starts = [r for r in records if r["event"] == "start" and r["project"] == "pushable"]
    start = next((r for r in reversed(starts) if run_id in r["command"]), None)
    if start:
        end = next((r for r in records if r["event"] == "end" and r["id"] == start["id"]), None)
        destination = ROOT / "results/runs" / run_id
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            "id": run_id, "stage": args.stage, "env": args.env, "seed": args.seed,
            "gpu": args.gpu, "status": "complete" if code == 0 else "failed",
            **source,
        }
        manifest["spend_guard"] = {"start": start, "end": end}
        if end:
            manifest["usd_upper"] = end["charged_upper"]
            manifest["minutes"] = end["wall_minutes"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (ROOT / "results/spend.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records
            if r.get("project") == "pushable" or any(s["id"] == r["id"] for s in starts)))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
