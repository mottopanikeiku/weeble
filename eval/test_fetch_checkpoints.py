"""Checkpoint checks use real files; only the HTTPS transport is substituted."""

import hashlib
import io
import json
import shutil
from types import SimpleNamespace
import urllib.error

import pytest

from train import fetch_checkpoints as fetch


RELEASE_URL = "https://github.com/mottopanikeiku/weeble/releases/download/policies-2026-10-14/"


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
    """Serve `assets` at the public release URL; return requested URLs and download paths."""
    urls, destinations = [], []

    def urlopen(url, timeout):
        urls.append(url)
        assert url.startswith(RELEASE_URL) and timeout > 0
        name = url.removeprefix(RELEASE_URL)
        if name not in assets:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return io.BytesIO(assets[name])

    download = fetch._download

    def recording(name, destination):
        destinations.append(destination)
        return download(name, destination)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fetch, "_download", recording)
    return urls, destinations


def test_fetch_installs_verified_bytes_and_preserves_existing(tmp_path, monkeypatch):
    existing = _policy(tmp_path, "existing", b"existing checkpoint")
    existing.write_bytes(b"existing checkpoint")
    existing_stat = existing.stat()
    first = _policy(tmp_path, "first", b"first checkpoint")
    second = _policy(tmp_path, "second", b"second checkpoint")
    urls, destinations = _release(monkeypatch, {
        first.name: b"first checkpoint", second.name: b"second checkpoint",
    })

    assert fetch.fetch_checkpoints(tmp_path) == (2, 1)
    assert first.read_bytes() == b"first checkpoint"
    assert second.read_bytes() == b"second checkpoint"
    assert existing.read_bytes() == b"existing checkpoint"
    assert existing.stat().st_mtime_ns == existing_stat.st_mtime_ns
    assert existing.stat().st_ino == existing_stat.st_ino
    assert urls == [RELEASE_URL + first.name, RELEASE_URL + second.name]
    download_dir = destinations[0].parent
    assert not download_dir.is_relative_to(tmp_path)
    assert not download_dir.exists()
    assert not list((tmp_path / "policies").glob(".*.tmp"))
    assert fetch.fetch_checkpoints(tmp_path) == (0, 3)
    assert len(urls) == 2


def test_corrupt_download_prevents_entire_batch_install(tmp_path, monkeypatch):
    valid = _policy(tmp_path, "a-valid", b"valid checkpoint")
    corrupt = _policy(tmp_path, "z-corrupt", b"expected checkpoint")
    _, destinations = _release(monkeypatch, {
        valid.name: b"valid checkpoint", corrupt.name: b"corrupt checkpoint",
    })

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch.fetch_checkpoints(tmp_path)
    assert not valid.exists()
    assert not corrupt.exists()
    assert not destinations[0].parent.exists()


def test_corrupt_existing_file_is_not_replaced(tmp_path, monkeypatch):
    corrupt = _policy(tmp_path, "corrupt", b"expected checkpoint")
    corrupt.write_bytes(b"corrupt checkpoint")
    urls, _ = _release(monkeypatch, {corrupt.name: b"expected checkpoint"})

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch.fetch_checkpoints(tmp_path)
    assert corrupt.read_bytes() == b"corrupt checkpoint"
    assert urls == []


def test_missing_release_asset_is_not_installed(tmp_path, monkeypatch):
    sidecar = _policy(tmp_path, "missing", b"expected checkpoint")
    _release(monkeypatch, {})

    with pytest.raises(ValueError, match="did not provide required asset missing.brax"):
        fetch.fetch_checkpoints(tmp_path)
    assert not sidecar.exists()


@pytest.mark.parametrize("error, detail", [
    (urllib.error.URLError("Name or service not known"), "Name or service not known"),
    (urllib.error.HTTPError(RELEASE_URL + "missing.brax", 503, "Service Unavailable", {}, None),
     "HTTP 503 Service Unavailable"),
])
def test_transport_errors_name_the_public_release_url(tmp_path, monkeypatch, error, detail):
    sidecar = _policy(tmp_path, "missing", b"expected checkpoint")

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(fetch.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match=detail) as caught:
        fetch.fetch_checkpoints(tmp_path)
    assert f"Could not download {RELEASE_URL}missing.brax" in str(caught.value)
    assert not sidecar.exists()


def test_missing_policy_metadata_is_an_error(tmp_path):
    (tmp_path / "policies").mkdir()
    with pytest.raises(ValueError, match="No policy JSON files"):
        fetch.fetch_checkpoints(tmp_path)


def _parity():
    # The parity suite needs the JAX/Playground stack, absent from the minimal evaluation install.
    return pytest.importorskip("eval.test_parity")


def test_parity_fixture_skips_absence_before_loading_objects(tmp_path, monkeypatch):
    parity = _parity()

    _policy(tmp_path, "missing", b"expected checkpoint")
    monkeypatch.setattr(parity, "ROOT", tmp_path)

    def must_not_load(*args, **kwargs):
        pytest.fail("Missing sidecar must skip before loading policy or simulator")

    monkeypatch.setattr(parity, "_load_artifact", must_not_load)
    with pytest.raises(pytest.skip.Exception, match="python -m train.fetch_checkpoints"):
        parity.artifact.__wrapped__(SimpleNamespace(param="policies/missing.json"))


def test_parity_fixture_missing_policy_is_not_skipped(tmp_path, monkeypatch):
    parity = _parity()

    monkeypatch.setattr(parity, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="Missing trained policy artifact"):
        parity.artifact.__wrapped__(SimpleNamespace(param="policies/missing.json"))


def test_parity_fixture_corruption_still_reaches_real_hash_check(tmp_path, monkeypatch):
    parity = _parity()

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
