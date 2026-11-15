#!/usr/bin/env python3
"""Download and verify selected release assets; archives are never extracted."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen


RELEASE_TAG = "v1.0.0-full40"
BASE_URL = (
    "https://github.com/Chloe-Jinlei/genesis-dual-franka-teddy-bag"
    f"/releases/download/{RELEASE_TAG}"
)
INDEX_PATH = Path(__file__).resolve().parents[1] / "environment" / "release-assets.json"
CHUNK_SIZE = 1024 * 1024
GROUP_KEYS = {
    "runtime": ("native_vendor", "genesis_source"),
    "evidence": ("latest_full_run", "video"),
    "full": ("full_scene_parts", "manifest"),
}


def load_assets(index_path, group):
    """Validate the selected index entries before creating or downloading files."""
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("tag") != RELEASE_TAG:
        raise ValueError(f"Release index must describe {RELEASE_TAG}")
    assets = []
    seen = set()
    for key in GROUP_KEYS[group]:
        if key not in index:
            raise ValueError(f"Release index is missing {key}")
        entries = index[key] if key == "full_scene_parts" else [index[key]]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Invalid release index entry: {key}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid release asset in {key}")
            name, size, digest = (entry.get(k) for k in ("file", "bytes", "sha256"))
            if (
                not isinstance(name, str)
                or name in (".", "..")
                or Path(name).name != name
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
            ):
                raise ValueError(f"Unsafe release asset filename in {key}")
            if name in seen:
                raise ValueError(f"Duplicate release asset filename: {name}")
            if type(size) is not int or size < 0:
                raise ValueError(f"Invalid byte count for {name}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"Invalid SHA256 for {name}")
            assets.append({"file": name, "bytes": size, "sha256": digest.lower()})
            seen.add(name)
    return assets


def file_matches(path, asset):
    if path.stat().st_size != asset["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset["sha256"]


def download_asset(asset, output):
    """Stream to a temporary .part file, verify it, then atomically replace."""
    destination = output / asset["file"]
    if destination.is_symlink():
        raise ValueError(f"Refusing symbolic-link destination: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Destination is not a regular file: {destination}")
        if file_matches(destination, asset):
            print(f"Verified existing file; skipping {asset['file']}", flush=True)
            return destination

    request = Request(
        f"{BASE_URL}/{quote(asset['file'], safe='')}",
        headers={"User-Agent": "genesis-dual-franka-teddy-bag-release-fetcher/1.0"},
    )
    temporary = None
    print(f"Downloading {asset['file']} ({asset['bytes']:,} bytes)", flush=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=asset["file"] + ".", suffix=".part", dir=output, delete=False
        ) as target:
            temporary = Path(target.name)
            digest = hashlib.sha256()
            received = 0
            with urlopen(request, timeout=60) as response:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > asset["bytes"]:
                        raise ValueError(f"Downloaded file exceeds expected size: {asset['file']}")
                    target.write(chunk)
                    digest.update(chunk)
            if received != asset["bytes"]:
                raise ValueError(f"Byte count mismatch for {asset['file']}")
            if digest.hexdigest() != asset["sha256"]:
                raise ValueError(f"SHA256 mismatch for {asset['file']}")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
        temporary = None
        print(f"Verified and saved {destination}", flush=True)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUP_KEYS, default="runtime")
    parser.add_argument("--output", type=Path, default=Path("release-assets"))
    args = parser.parse_args(argv)
    try:
        assets = load_assets(INDEX_PATH, args.group)
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            download_asset(asset, output)
    except (OSError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Download interrupted; unverified temporary file removed.", file=sys.stderr)
        return 130
    print("All selected files verified. Archives have not been extracted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
