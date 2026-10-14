"""Modal entry points for the public standalone launcher; no internal spend guard."""
from pathlib import Path
import csv
import io
import json
import os
import subprocess
import time
import zipfile

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("PUSHABLE_GPU", "L4")
TIMEOUT = int(os.environ.get("PUSHABLE_TIMEOUT", "300"))
CPU = float(os.environ.get("PUSHABLE_CPU", "2"))
MEMORY = int(os.environ.get("PUSHABLE_MEMORY", "8192"))
PLAYGROUND_COMMIT = "d59156e099a8785dd58fbce2e25a6ee1f7b5800c"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"
app = modal.App("pushable")
volume = modal.Volume.from_name("pushable-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libegl1", "libosmesa6")
    .uv_pip_install("uv==0.12.5")
    .add_local_file(ROOT / "pyproject.toml", "/opt/project/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/opt/project/uv.lock", copy=True)
    .run_commands("cd /opt/project && uv export --frozen "
                  + ("--extra cuda " if GPU != "none" else "")
                  + "--no-dev --no-hashes -o /opt/requirements.txt && uv pip install --system -r /opt/requirements.txt")
    .run_commands(
        "git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git /opt/menagerie && "
        "git -C /opt/menagerie sparse-checkout set unitree_go1 unitree_g1 && "
        f"git -C /opt/menagerie checkout {MENAGERIE_COMMIT}",
        f"git clone --depth 1 https://github.com/google-deepmind/mujoco_playground.git /opt/playground-license && "
        f"git -C /opt/playground-license fetch --depth 1 origin {PLAYGROUND_COMMIT} && "
        f"git -C /opt/playground-license checkout {PLAYGROUND_COMMIT} && "
        "mkdir -p /opt/licenses && cp /opt/playground-license/LICENSE /opt/licenses/LICENSE-Playground && rm -rf /opt/playground-license",
    )
    .env({
        "PUSHABLE_GPU": GPU, "PUSHABLE_TIMEOUT": str(TIMEOUT),
        "PUSHABLE_CPU": str(CPU), "PUSHABLE_MEMORY": str(MEMORY),
        "PYTHONPATH": "/workspace", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "MUJOCO_GL": "egl",
        "JAX_COMPILATION_CACHE_DIR": "/runs/cache/jax",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1",
        "WARP_CACHE_PATH": "/runs/cache/warp",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
    .add_local_dir(ROOT, "/workspace", ignore=[".git", "web", ".venv", "__pycache__", "*.pyc", "*.brax"])
)


def setup_playground():
    from etils import epath
    from mujoco_playground._src import mjx_env
    mjx_env.MENAGERIE_PATH = epath.Path("/opt/menagerie")


def packages():
    import importlib.metadata
    import platform
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in
        ("mujoco", "mujoco-mjx", "mujoco-warp", "jax", "jaxlib", "brax", "flax", "numpy", "playground", "warp-lang", "modal")}}


def archive_files(paths, root):
    root = Path(root).resolve()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.resolve().relative_to(root)))
    return buffer.getvalue()


@app.function(image=image, cpu=CPU, memory=MEMORY, timeout=TIMEOUT,
              max_containers=1, volumes={"/runs": volume})
def assets(provenance):
    started = time.monotonic()
    os.environ["JAX_PLATFORMS"] = "cpu"
    setup_playground()
    from mujoco_playground import registry
    from train.assets import export_model, MODEL_PATHS
    from train.export import _json
    out = Path("/runs") / provenance["run_id"] / "export"
    configs = {}
    for name in MODEL_PATHS:
        # Model parameters do not depend on the implementation; CPU avoids CUDA initialization.
        env = registry.load(name, config_overrides={"impl": "jax"})
        export_model(name, env, out)
        configs[name] = env._config.to_dict()
    manifest = {**provenance, "id": provenance["run_id"], "stage": "assets", "status": "complete",
                "modal_app_id": app.app_id, "gpu": "none", "versions": packages(),
                "environment_configs": configs, "wall_seconds": time.monotonic() - started}
    manifest_path = out / "results/runs" / provenance["run_id"] / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json(manifest), indent=2, allow_nan=False) + "\n")
    volume.commit()
    return archive_files([p for p in (out / "assets").rglob("*") if p.is_file()] + [manifest_path], out)


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=CPU,
              memory=MEMORY, timeout=TIMEOUT, max_containers=1, volumes={"/runs": volume})
def train_one(env_name, seed, run_id, stage, pushes, provenance, benchmark_seconds=120, num_timesteps=0):
    setup_playground()
    import functools
    import traceback
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks
    from mujoco_playground import registry
    from mujoco_playground._src import wrapper
    from mujoco_playground.config import locomotion_params
    from train.assets import MODEL_PATHS, export_model

    start = time.monotonic()
    out = Path("/runs") / run_id
    out.mkdir(parents=True, exist_ok=False)
    cfg = registry.get_default_config(env_name)
    if pushes:
        if env_name != "Go1JoystickFlatTerrain":
            raise ValueError("Push ablation is defined only for Go1 joystick")
        cfg.pert_config.enable = True
    env = registry.load(env_name, config=cfg)
    config = locomotion_params.brax_ppo_config(env_name).to_dict()
    original = json.loads(json.dumps(config))
    if num_timesteps:
        config["num_timesteps"] = num_timesteps
    if stage == "smoke":
        config.update(num_timesteps=1280, num_envs=64, batch_size=64,
                      num_minibatches=1, num_evals=2, num_resets_per_eval=1,
                      num_eval_envs=16)
    import inspect
    defaults = {name: parameter.default for name, parameter in inspect.signature(ppo.train).parameters.items()
                if parameter.default is not inspect.Parameter.empty and not callable(parameter.default)}
    manifest = {
        **provenance, "id": run_id, "env": env_name, "seed": seed, "stage": stage,
        "gpu": GPU, "cpu": CPU, "memory_mib": MEMORY,
        "devices": [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()],
        "playground_commit": PLAYGROUND_COMMIT, "menagerie_commit": MENAGERIE_COMMIT,
        "versions": packages(), "environment_config": cfg.to_dict(),
        "ppo_config": json.loads(json.dumps(config)), "upstream_ppo_config": original,
        "ppo_effective_config": {**defaults, **config, "seed": seed},
        "domain_randomization": False, "domain_randomization_reason": "Upstream train_jax_ppo CLI default is false",
        "modal_app_id": app.app_id, "status": "running",
    }
    curve = []

    def persist():
        from train.export import _json
        manifest["wall_seconds"] = time.monotonic() - start
        (out / "manifest.json").write_text(json.dumps(_json(manifest), indent=2, allow_nan=False) + "\n")
        if curve:
            keys = ["steps", "wall_seconds"] + sorted(set().union(*(r.keys() for r in curve)) - {"steps", "wall_seconds"})
            with (out / "curve.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(curve)
        volume.commit()

    def progress(steps, metrics):
        row = {"steps": int(steps), "wall_seconds": time.monotonic() - start}
        row.update({k: float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).size == 1})
        curve.append(row)
        print(json.dumps(row), flush=True)
        persist()

    persist()
    try:
        if stage == "benchmark":
            batch = config["num_envs"]
            wrapped = wrapper.wrap_for_brax_training(env, episode_length=config["episode_length"])
            reset = jax.jit(wrapped.reset)
            state = reset(jax.random.split(jax.random.PRNGKey(seed), batch))
            action = jnp.zeros((batch, env.action_size))

            @jax.jit
            def advance(s):
                return jax.lax.fori_loop(0, 20, lambda _, carry: wrapped.step(carry, action), s)

            state = advance(state)
            jax.block_until_ready(state)
            warmup = time.monotonic() - start
            begin = time.monotonic()
            steps = 0
            while time.monotonic() - begin < benchmark_seconds:
                state = advance(state)
                jax.block_until_ready(state)
                steps += batch * 20
            seconds = time.monotonic() - begin
            manifest.update(benchmark={"steps": steps, "seconds": seconds,
                "steps_per_second": steps / seconds, "warmup_seconds": warmup,
                "batch_size": batch, "control_steps_per_call": 20,
                "scope": "Environment stepping with zero policy actions, episode autoreset; excludes PPO and warmup"})
        else:
            network_config = config.pop("network_factory")
            make_network = functools.partial(networks.make_ppo_networks, **network_config)
            evaluation = registry.load(env_name, config=cfg.copy_and_resolve_references())
            make_policy, params, metrics = ppo.train(
                environment=env, eval_env=evaluation,
                network_factory=make_network, seed=seed, progress_fn=progress,
                save_checkpoint_path=str(out / "checkpoints"),
                wrap_env_fn=wrapper.wrap_for_brax_training, **config,
            )
            config["network_factory"] = network_config
            manifest["final_reward"] = float(metrics.get("eval/episode_reward", float("nan")))
            manifest["trained_steps"] = curve[-1]["steps"] if curve else 0
            if stage == "train":
                from train.export import export_policy
                export_model(env_name, env, out)
                robot = "go1" if env_name.startswith("Go1") else "g1"
                task = "getup" if env_name == "Go1Getup" else "walk-push" if pushes else "walk"
                destination = out / "policies" / f"{robot}-{task}-s{seed}.json"
                destination.parent.mkdir(exist_ok=True)
                export_policy(env_name, env, params, destination, seed, MODEL_PATHS[env_name], PLAYGROUND_COMMIT, run_id)
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        print(manifest["error"], flush=True)
    finally:
        persist()
    exported = [out / "manifest.json"]
    if (out / "curve.csv").exists():
        exported.append(out / "curve.csv")
    exported += list((out / "policies").glob("*.json"))
    # Checkpoints and raw Brax arrays stay in the Volume.
    return archive_files(exported, out)


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=CPU,
              memory=MEMORY, timeout=TIMEOUT, max_containers=1, volumes={"/runs": volume})
def export_checkpoint(provenance, remote_run, checkpoint_steps):
    """Export a specified persisted checkpoint without resuming optimization."""
    if not remote_run or Path(remote_run).name != remote_run or checkpoint_steps <= 0:
        raise ValueError("A training run ID and positive checkpoint step are required")
    setup_playground()
    import traceback
    from ml_collections import ConfigDict
    from brax.training.agents.ppo import checkpoint
    from mujoco_playground import registry
    from train.assets import MODEL_PATHS
    from train.export import _config_numbers, _json, export_policy

    started = time.monotonic()
    source = Path("/runs") / remote_run
    training = json.loads((source / "manifest.json").read_text())
    with (source / "curve.csv").open() as stream:
        row = next(row for row in csv.DictReader(stream) if int(row["steps"]) == checkpoint_steps)
    checkpoint_path = source / "checkpoints" / f"{checkpoint_steps:012d}"
    out = Path("/runs") / provenance["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        **provenance, "id": provenance["run_id"], "stage": "export-checkpoint",
        "env": training["env"], "seed": training["seed"], "gpu": GPU,
        "cpu": CPU, "memory_mib": MEMORY, "versions": packages(),
        "modal_app_id": app.app_id, "training_source": training,
        "checkpoint_path": str(checkpoint_path.relative_to(Path("/runs"))),
        "checkpoint_steps": checkpoint_steps,
        "planned_steps": training["ppo_config"]["num_timesteps"],
        "final_reward": float(row["eval/episode_reward"]), "status": "running",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(_json(manifest), indent=2) + "\n")
    volume.commit()
    try:
        cfg = ConfigDict(_config_numbers(training["environment_config"]))
        env = registry.load(training["env"], config=cfg)
        params = checkpoint.load(str(checkpoint_path))
        robot = "g1" if training["env"].startswith("G1") else "go1"
        task = "getup" if training["env"] == "Go1Getup" else "walk"
        destination = out / "policies" / f"{robot}-{task}-s{training['seed']}.json"
        spec = export_policy(training["env"], env, params, destination, training["seed"],
                             MODEL_PATHS[training["env"]], PLAYGROUND_COMMIT, provenance["run_id"])
        spec["training_checkpoint"] = {
            "run_id": remote_run, "steps": checkpoint_steps,
            "planned_steps": training["ppo_config"]["num_timesteps"],
            "selection": "Latest persisted checkpoint after the training time limit; not best-reward selection",
        }
        destination.write_text(json.dumps(_json(spec), indent=2) + "\n")
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
    manifest["wall_seconds"] = time.monotonic() - started
    manifest_path.write_text(json.dumps(_json(manifest), indent=2) + "\n")
    volume.commit()
    return archive_files([manifest_path, *list((out / "policies").glob("*.json"))], out)


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=CPU,
              memory=MEMORY, timeout=TIMEOUT, max_containers=1, volumes={"/runs": volume})
def evaluate(provenance, policy="", backend="mujoco", mode="full", initialization="drop", source_run=""):
    if GPU == "none":
        os.environ["JAX_PLATFORMS"] = "cpu"
    provenance = {**provenance, "modal_app_id": app.app_id, "gpu": GPU, "evaluation_mode": mode}
    setup_playground()
    from contextlib import ExitStack
    from tempfile import TemporaryDirectory
    import shutil
    from eval.run import SuiteConfig, run_suite
    out = Path("/runs") / {"full": "evaluation", "getup": "getup-diagnostic",
                          "tracking": "policy-checks", "gap": "evaluation"}[mode]
    out.mkdir(exist_ok=True)
    with ExitStack() as cleanup:
        root = Path("/workspace")
        if source_run:
            if Path(source_run).name != source_run or not policy or Path(policy).name != policy:
                raise ValueError("Historical evaluation requires one run ID and policy stem")
            root = Path(cleanup.enter_context(TemporaryDirectory()))
            (root / "assets").symlink_to("/workspace/assets", target_is_directory=True)
            (root / "policies").mkdir()
            shutil.copyfile(Path("/runs") / source_run / "policies" / f"{policy}.json",
                            root / "policies" / f"{policy}.json")
            provenance["source_training_run"] = source_run
        config = SuiteConfig(suites=("tracking",)) if mode == "tracking" else None
        if mode == "gap":
            config = SuiteConfig(seeds=(0,), suites=("tracking", "push"), push_trials=20)
        if mode == "getup":
            spec = json.loads((root / "policies" / f"{policy}.json").read_text())
            if spec["task"] != "getup" or spec["seed"] != 0 or initialization not in ("drop", "upstream"):
                raise ValueError("Getup diagnosis requires seed0 and drop/upstream initialization")
            settle = spec["training_config"]["settle_time"] if initialization == "upstream" else 1.0
            config = SuiteConfig(seeds=(0,), suites=("getup",), settle_seconds=settle)
        run_dir = run_suite(root=root, output=out, backend=backend, config=config,
                           policy_filter=policy or None, provenance=provenance,
                           getup_initialization=initialization)
    import gzip
    for path in list(run_dir.rglob("trials.jsonl")) + list(run_dir.rglob("trials.csv")):
        with path.open("rb") as source, gzip.open(path.with_suffix(path.suffix + ".gz"), "wb") as destination:
            shutil.copyfileobj(source, destination)
        path.unlink()
    volume.commit()
    return archive_files([p for p in run_dir.rglob("*") if p.is_file()], out)


@app.function(image=image, cpu=CPU, memory=MEMORY, timeout=TIMEOUT,
              max_containers=1, volumes={"/runs": volume})
def fetch_artifacts(remote_run, training=False, diagnostic=False):
    """Download committed diagnostics without repeating training or simulation."""
    if not remote_run or Path(remote_run).name != remote_run:
        raise ValueError("remote_run must name one run directory")
    if training:
        run_dir = Path("/runs") / remote_run
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(run_dir / "manifest.json", "source-manifest.json")
            if (run_dir / "curve.csv").exists():
                archive.write(run_dir / "curve.csv", "curve.csv")
            inventory = [{"path": str(p.relative_to(run_dir)), "bytes": p.stat().st_size}
                         for p in sorted((run_dir / "checkpoints").rglob("*")) if p.is_file()]
            archive.writestr("checkpoints.json", json.dumps(inventory, indent=2) + "\n")
        return buffer.getvalue()
    out = Path("/runs/getup-diagnostic" if diagnostic else "/runs/evaluation")
    run_dir = out / remote_run
    if not (run_dir / "manifest.json").is_file():
        matches = [p.parent for p in out.glob("*/manifest.json")
                   if json.loads(p.read_text()).get("provenance", {}).get("run_id") == remote_run]
        if len(matches) != 1:
            raise ValueError(f"Expected one evaluation for {remote_run}, found {len(matches)}")
        run_dir = matches[0]
    return archive_files([p for p in run_dir.rglob("*") if p.is_file()], out)


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=CPU,
              memory=MEMORY, timeout=TIMEOUT, max_containers=1, volumes={"/runs": volume})
def parity(provenance, policy=""):
    if GPU == "none":
        os.environ["JAX_PLATFORMS"] = "cpu"
    setup_playground()
    from eval.test_parity import check_policy
    from train.export import write_parity_fixture
    root = Path("/workspace")
    out = Path("/runs") / provenance["run_id"] / "parity"
    out.mkdir(parents=True, exist_ok=True)
    selected = sorted((root / "policies").glob(f"{policy or '*-s0'}.json"))
    if not selected:
        raise ValueError("No matching policy artifacts")
    started = time.monotonic()
    report = {**provenance, "modal_app_id": app.app_id, "versions": packages(), "checks": {}}
    for path in selected:
        spec = json.loads(path.read_text())
        raw = Path("/runs") / spec["run_id"] / "policies" / path.with_suffix(".brax").name
        impl = "warp" if GPU != "none" else "jax"
        report["checks"][path.stem] = check_policy(path, raw, root=root, impl=impl)
        write_parity_fixture(path, out / f"{path.stem}.json", root=root, steps=200)
    report["wall_seconds"] = time.monotonic() - started
    (out / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return archive_files([p for p in out.iterdir() if p.is_file()], out)


@app.function(image=image, cpu=CPU, memory=MEMORY, timeout=TIMEOUT,
              max_containers=1, volumes={"/runs": volume})
def fetch_checkpoints(provenance):
    """Retrieve and verify the original Brax sidecars referenced by policy JSON."""
    import hashlib

    root = Path("/workspace")
    out = Path("/runs") / provenance["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    files = []
    for spec_path in sorted((root / "policies").glob("*.json")):
        spec = json.loads(spec_path.read_text())
        sidecar_name = spec["brax_params_file"]
        if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".brax"):
            raise ValueError(f"Invalid sidecar name in {spec_path.name}: {sidecar_name}")
        source = Path("/runs") / spec["run_id"] / "policies" / sidecar_name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != spec["brax_params_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {sidecar_name}: expected {spec['brax_params_sha256']}, got {digest}")
        files.append((source, sidecar_name, digest))
    if not files:
        raise ValueError("No policy JSON files found")
    manifest = {
        **provenance, "id": provenance["run_id"], "stage": "fetch-checkpoints",
        "status": "complete", "gpu": "none", "cpu": CPU, "memory_mib": MEMORY,
        "modal_app_id": app.app_id, "versions": packages(),
        "policies": [{"file": name, "sha256": digest} for _, name, digest in files],
        "wall_seconds": time.monotonic() - started,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name, _ in files:
            archive.write(source, f"policies/{name}")
        archive.write(manifest_path, "manifest.json")
    return buffer.getvalue()




@app.local_entrypoint()
def main(stage: str = "smoke", env: str = "Go1JoystickFlatTerrain", seed: int = 0,
         run_id: str = "", pushes: bool = False, policy: str = "", backend: str = "mujoco",
         remote_run: str = "", num_timesteps: int = 0, initialization: str = "drop",
         checkpoint_steps: int = 0):
    if not run_id:
        run_id = f"{stage}-{env}-s{seed}-{int(time.time())}"
    provenance = {
        "run_id": run_id,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    }
    if stage == "assets":
        payload = assets.remote(provenance)
        target = ROOT
    elif stage == "export-checkpoint":
        payload = export_checkpoint.remote(provenance, remote_run, checkpoint_steps)
        target = ROOT / "results" / "runs" / run_id
    elif stage == "fetch-checkpoints":
        payload = fetch_checkpoints.remote(provenance)
        target = ROOT / "results" / "runs" / run_id
    elif stage == "check-policy":
        payload = evaluate.remote(provenance, policy, backend, mode="tracking", source_run=remote_run)
        target = ROOT / "results" / "policy-checks"
    elif stage == "eval":
        payload = evaluate.remote(provenance, policy, backend, source_run=remote_run)
        target = ROOT / "results" / "eval"
    elif stage == "gap":
        payload = evaluate.remote(provenance, policy, backend, mode="gap", source_run=remote_run)
        target = ROOT / "results" / "eval"
    elif stage == "getup-diagnostic":
        payload = evaluate.remote(provenance, policy, backend, "getup", initialization, remote_run)
        target = ROOT / "results" / "getup-diagnostic"
    elif stage == "fetch-eval":
        payload = fetch_artifacts.remote(remote_run)
        target = ROOT / "results" / "eval"
    elif stage == "fetch-training":
        payload = fetch_artifacts.remote(remote_run, True)
        target = ROOT / "results" / "runs" / remote_run
    elif stage == "fetch-getup-diagnostic":
        payload = fetch_artifacts.remote(remote_run, diagnostic=True)
        target = ROOT / "results" / "getup-diagnostic"
    elif stage == "parity":
        payload = parity.remote(provenance, policy)
        target = ROOT / "results" / "parity"
    else:
        payload = train_one.remote(env, seed, run_id, stage, pushes, provenance, num_timesteps=num_timesteps)
        target = ROOT / "results" / "runs" / run_id
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid archive path")
            destination = ROOT / relative if relative.parts[0] == "policies" else target / relative
            if stage == "parity" and relative.name == "checks.json":
                destination = ROOT / "results" / "parity-checks" / f"{run_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
    print(f"Saved {stage} outputs to {target}")
    manifest_path = target / "manifest.json"
    if stage != "fetch-training" and manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "failed":
        raise RuntimeError(f"Run failed; see {manifest_path}")
