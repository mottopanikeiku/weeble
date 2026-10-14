"""Checkpoint checks use real files; only the GitHub transport is substituted."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from train import fetch_checkpoints as fetch


def _policy(root, name, payload):
    policies = root / "policies"
    policies.mkdir(exist_ok=True)
    sidecar = policies / f"{name}.brax"
    (policies / f"{name}.json").write_text(json.dumps({
        "brax_params_file": sidecar.name,
        "brax_params_sha256": hashlib.sha256(payload).hexdigest(),
    }))
    return sidecar


def _release(monkeypatch, assets):
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        assert command[:6] == [
            "gh", "release", "download", "policies-2026-10-14",
            "--repo", "mottopanikeiku/pushable",
        ]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        directory = Path(command[command.index("--dir") + 1])
        for index, flag in enumerate(command):
            if flag == "--pattern":
                name = command[index + 1]
                if name in assets:
                    (directory / name).write_bytes(assets[name])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(fetch.subprocess, "run", download)
    return calls


def test_fetch_installs_verified_bytes_and_preserves_existing(tmp_path, monkeypatch):
    existing = _policy(tmp_path, "existing", b"existing checkpoint")
    existing.write_bytes(b"existing checkpoint")
    existing_stat = existing.stat()
    first = _policy(tmp_path, "first", b"first checkpoint")
    second = _policy(tmp_path, "second", b"second checkpoint")
    calls = _release(monkeypatch, {
        first.name: b"first checkpoint", second.name: b"second checkpoint",
    })

    assert fetch.fetch_checkpoints(tmp_path) == (2, 1)
    assert first.read_bytes() == b"first checkpoint"
    assert second.read_bytes() == b"second checkpoint"
    assert existing.read_bytes() == b"existing checkpoint"
    assert existing.stat().st_mtime_ns == existing_stat.st_mtime_ns
    assert existing.stat().st_ino == existing_stat.st_ino
    assert len(calls) == 1
    assert existing.name not in calls[0]
    download_dir = Path(calls[0][calls[0].index("--dir") + 1])
    assert not download_dir.is_relative_to(tmp_path)
    assert not download_dir.exists()
    assert not list((tmp_path / "policies").glob(".*.tmp"))
    assert fetch.fetch_checkpoints(tmp_path) == (0, 3)
    assert len(calls) == 1


def test_corrupt_download_prevents_entire_batch_install(tmp_path, monkeypatch):
    valid = _policy(tmp_path, "a-valid", b"valid checkpoint")
    corrupt = _policy(tmp_path, "z-corrupt", b"expected checkpoint")
    calls = _release(monkeypatch, {
        valid.name: b"valid checkpoint", corrupt.name: b"corrupt checkpoint",
    })

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch.fetch_checkpoints(tmp_path)
    assert not valid.exists()
    assert not corrupt.exists()
    download_dir = Path(calls[0][calls[0].index("--dir") + 1])
    assert not download_dir.exists()


def test_corrupt_existing_file_is_not_replaced(tmp_path, monkeypatch):
    corrupt = _policy(tmp_path, "corrupt", b"expected checkpoint")
    corrupt.write_bytes(b"corrupt checkpoint")
    calls = _release(monkeypatch, {corrupt.name: b"expected checkpoint"})

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch.fetch_checkpoints(tmp_path)
    assert corrupt.read_bytes() == b"corrupt checkpoint"
    assert calls == []


def test_missing_release_asset_is_not_installed(tmp_path, monkeypatch):
    sidecar = _policy(tmp_path, "missing", b"expected checkpoint")
    _release(monkeypatch, {})

    with pytest.raises(ValueError, match="did not provide required asset missing.brax"):
        fetch.fetch_checkpoints(tmp_path)
    assert not sidecar.exists()


@pytest.mark.parametrize("error", [
    FileNotFoundError("gh"),
    subprocess.CalledProcessError(1, ["gh"], stderr="HTTP 404: Not Found"),
])
def test_transport_errors_explain_private_repository_auth(tmp_path, monkeypatch, error):
    sidecar = _policy(tmp_path, "missing", b"expected checkpoint")

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(fetch.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="gh auth login") as caught:
        fetch.fetch_checkpoints(tmp_path)
    assert "read access to mottopanikeiku/pushable" in str(caught.value)
    assert not sidecar.exists()


def test_missing_policy_metadata_is_an_error(tmp_path):
    (tmp_path / "policies").mkdir()
    with pytest.raises(ValueError, match="No policy JSON files"):
        fetch.fetch_checkpoints(tmp_path)


def test_parity_fixture_skips_absence_before_loading_objects(tmp_path, monkeypatch):
    from eval import test_parity as parity

    _policy(tmp_path, "missing", b"expected checkpoint")
    monkeypatch.setattr(parity, "ROOT", tmp_path)

    def must_not_load(*args, **kwargs):
        pytest.fail("Missing sidecar must skip before loading policy or simulator")

    monkeypatch.setattr(parity, "_load_artifact", must_not_load)
    with pytest.raises(pytest.skip.Exception, match="python -m train.fetch_checkpoints"):
        parity.artifact.__wrapped__(SimpleNamespace(param="policies/missing.json"))


def test_parity_fixture_missing_policy_is_not_skipped(tmp_path, monkeypatch):
    from eval import test_parity as parity

    monkeypatch.setattr(parity, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="Missing trained policy artifact"):
        parity.artifact.__wrapped__(SimpleNamespace(param="policies/missing.json"))


def test_parity_fixture_corruption_still_reaches_real_hash_check(tmp_path, monkeypatch):
    from eval import test_parity as parity

    source = parity.ROOT / "policies" / "go1-walk-s0.json"
    policies = tmp_path / "policies"
    policies.mkdir()
    target = policies / source.name
    shutil.copyfile(source, target)
    spec = json.loads(target.read_text())
    (policies / spec["brax_params_file"]).write_bytes(b"corrupt checkpoint")
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        parity.artifact.__wrapped__(SimpleNamespace(param=f"policies/{source.name}"))
