"""Run Modal with a function timeout and record wall time and list-price costs."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RATES = {"none": 0, "L4": 0.7992, "A100-40GB": 2.0988, "H100": 3.9492}
CPU_HOURLY_USD = 0.047160
MEMORY_GIB_HOURLY_USD = 0.007992
COST_MARGIN = 1.10


def write_manifest(path, source, config, code, seconds):
    """Preserve remote configuration while recording the complete invocation."""
    path = Path(path)
    manifest = json.loads(path.read_text()) if path.exists() else {}
    hourly = RATES[config["gpu"]] + config["cores"] * CPU_HOURLY_USD + config["mem_gib"] * MEMORY_GIB_HOURLY_USD
    manifest.update(
        id=config["run_id"], stage=config["stage"], env=config["env"], seed=config["seed"],
        gpu=config["gpu"], status="complete" if code == 0 else "failed", **source,
        launch_config=config, exit_code=code, minutes=seconds / 60,
        usd_upper=round(seconds / 3600 * hourly * COST_MARGIN, 4),
        cost_estimate={
            "source": "https://modal.com/pricing", "gpu_hourly_usd": RATES[config["gpu"]],
            "cpu_core_hourly_usd": CPU_HOURLY_USD, "memory_gib_hourly_usd": MEMORY_GIB_HOURLY_USD,
            "cpu_cores": config["cores"], "memory_gib": config["mem_gib"],
            "margin": COST_MARGIN, "hourly_usd_with_margin": hourly * COST_MARGIN,
            "basis": "Complete invocation wall time, including startup; not an invoice",
        },
    )
    manifest.setdefault("config", config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["assets", "smoke", "benchmark", "train", "eval", "parity", "fetch-eval", "fetch-training", "fetch-checkpoints", "getup-diagnostic", "fetch-getup-diagnostic", "export-checkpoint", "check-policy", "gap"])
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
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.minutes, args.cores, args.mem_gib)):
        parser.error("--minutes, --cores and --mem-gib must be finite and positive")
    run_id = args.run_id or f"{args.stage}-{args.env}-s{args.seed}-{args.gpu}-{time.time_ns()}"
    timeout_seconds = math.ceil(args.minutes * 60)
    command = ["modal", "run", "train/app.py", "--stage", args.stage,
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
    if args.checkpoint_steps:
        command.extend(["--checkpoint-steps", str(args.checkpoint_steps)])
    env = dict(os.environ, PUSHABLE_GPU=args.gpu, PUSHABLE_TIMEOUT=str(timeout_seconds),
               PUSHABLE_CPU=str(args.cores), PUSHABLE_MEMORY=str(args.mem_gib * 1024),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    source = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    }
    started = time.monotonic()
    code = 1
    try:
        code = subprocess.call(command, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        code = 130
    finally:
        config = {**vars(args), "run_id": run_id, "timeout_seconds": timeout_seconds}
        write_manifest(ROOT / "results/runs" / run_id / "manifest.json",
                       source, config, code, time.monotonic() - started)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
