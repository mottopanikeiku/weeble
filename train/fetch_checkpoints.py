"""Fetch original PPO sidecars using the hashes in policies/*.json.

Run from the checkout: python -m train.fetch_checkpoints
The release is public: files download over HTTPS without a GitHub account or CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "mottopanikeiku/weeble"
RELEASE = "policies-2026-10-14"
RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}/"


def _checkpoint_hashes(policies):
    """Read the existing policy metadata, without loading networks or pickles."""
    paths = sorted(policies.glob("*.json"))
    if not paths:
        raise ValueError(f"No policy JSON files found in {policies}")
    hashes = {}
    for path in paths:
        spec = json.loads(path.read_text())
        name = spec["brax_params_file"]
        digest = spec["brax_params_sha256"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".brax"):
            raise ValueError(f"Invalid brax_params_file in {path}: {name!r}")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"Invalid brax_params_sha256 in {path}")
        if name in hashes and hashes[name] != digest:
            raise ValueError(f"Conflicting policy hashes for {name}")
        hashes[name] = digest
    return hashes


def _verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}. "
            "The file was not installed or replaced."
        )


def _install(source, target):
    # Downloads stay in system temp. This local staging file contains only
    # checked bytes and lets replacement be atomic across filesystem boundaries.
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as output:
            staging = Path(output.name)
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output)
        os.replace(staging, target)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def _download(name, destination):
    url = RELEASE_URL + urllib.parse.quote(name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"Release {RELEASE} did not provide required asset {name}") from exc
        raise RuntimeError(f"Could not download {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc.reason}") from exc


def fetch_checkpoints(root=ROOT):
    """Return (installed, preserved); reject corruption before installing files."""
    policies = Path(root) / "policies"
    hashes = _checkpoint_hashes(policies)
    missing = []
    for name, digest in hashes.items():
        target = policies / name
        if target.exists():
            _verify(target, digest)
        else:
            missing.append(name)
    if not missing:
        return 0, len(hashes)

    with tempfile.TemporaryDirectory(prefix="weeble-checkpoints-") as temporary:
        download_dir = Path(temporary)
        for name in missing:
            _download(name, download_dir / name)
        # Check the entire batch before installing any new file.
        for name in missing:
            _verify(download_dir / name, hashes[name])
        for name in missing:
            _install(download_dir / name, policies / name)
    return len(missing), len(hashes) - len(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        installed, preserved = fetch_checkpoints()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Checkpoint fetch failed: {exc}\n")
    print(f"Verified {installed + preserved} checkpoints: installed {installed}, preserved {preserved}.")


if __name__ == "__main__":
    main()
