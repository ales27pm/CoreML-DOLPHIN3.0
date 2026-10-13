#!/usr/bin/env python3
"""Download the pinned public Dolphin weights with verified, resumable HTTP ranges.

Large partials stay in /tmp. Never use HF snapshot_download(local_dir=workspace)
for these shards: workspace synchronization duplicated interrupted temp files and
exhausted the disk. Only a fully hashed shard is atomically renamed into source/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import requests

REPO = "dphn/Dolphin3.0-Llama3.2-3B"
REVISION = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
METADATA_URL = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
# Independently read from the authoritative HF API above on 2026-10-12.
SHARDS = {
    "model-00001-of-00002.safetensors": {
        "size": 4_965_811_384,
        "sha256": "30660d7b8dc6288690933fcad0609d250bd7cd8d88e633e5886c7f5a500236f2",
    },
    "model-00002-of-00002.safetensors": {
        "size": 1_459_729_952,
        "sha256": "ea93d07c32c36d1385491bf45f13188e1e08647c066de01e33fc20a262448bf5",
    },
}
CHUNK = 8 * 1024 * 1024
METADATA_FILES = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json", "README.md"]


def emit(**values: object) -> None:
    print(json.dumps(values, sort_keys=True), flush=True)


def verify(path: Path, expected: dict) -> str:
    if path.stat().st_size != expected["size"]:
        raise RuntimeError(f"Size mismatch for {path}: file preserved")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected['sha256']}, got {actual}; file preserved")
    emit(event="sha256_verified", name=path.name, bytes=path.stat().st_size, sha256=actual)
    return actual


def validate_response(response: requests.Response, offset: int, size: int) -> int:
    """Return exactly the permitted response-body length before opening a writer."""
    if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
        raise RuntimeError("Encoded range response rejected; no bytes appended")
    if offset and response.status_code != 206:
        raise RuntimeError(f"Resume requires HTTP 206, got {response.status_code}; partial preserved")
    if response.status_code == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match:
            raise RuntimeError("Malformed/missing Content-Range; no bytes appended")
        start, end, total = map(int, match.groups())
        if start != offset or total != size or end < start or end >= size:
            raise RuntimeError(f"Unexpected Content-Range {response.headers.get('Content-Range')!r}; no bytes appended")
        body_size = end - start + 1
    elif offset == 0 and response.status_code == 200:
        body_size = size
    else:
        raise RuntimeError(f"Unexpected HTTP {response.status_code}; no bytes appended")
    declared = response.headers.get("Content-Length")
    if declared is not None and (not declared.isdigit() or int(declared) != body_size):
        raise RuntimeError("Content-Length does not match the validated range; no bytes appended")
    return body_size


def download(session: requests.Session, name: str, source_dir: Path, partial_dir: Path) -> None:
    expected = SHARDS[name]
    final = source_dir / name
    if final.exists():
        verify(final, expected)
        return
    partial = partial_dir / (name + ".partial")
    if partial.exists() and partial.stat().st_size > expected["size"]:
        raise RuntimeError(f"Oversized partial {partial}; preserved for investigation")
    interruptions = 0
    while (offset := partial.stat().st_size if partial.exists() else 0) < expected["size"]:
        remaining = expected["size"] - offset
        if shutil.disk_usage(partial_dir).free < remaining + 512 * 1024 * 1024:
            raise RuntimeError("Insufficient free space for remaining bytes plus 512 MiB; partial preserved")
        emit(event="request_range", name=name, offset=offset, remaining=remaining)
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        try:
            response = session.get(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"},
                                   stream=True, timeout=(20, 30))
        except requests.RequestException as error:
            raise RuntimeError(f"HTTP request failed before validated response ({type(error).__name__}); no automatic retry") from error
        with response:
            body_size = validate_response(response, offset, expected["size"])
            emit(event="range_validated", name=name, http=response.status_code,
                 content_range=response.headers.get("Content-Range"), body_bytes=body_size)
            received = 0
            checkpoint = time.monotonic()
            try:
                with partial.open("ab") as target:
                    if target.tell() != offset:
                        raise RuntimeError("Partial changed since Range request; refusing append")
                    for chunk in response.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        if received + len(chunk) > body_size:
                            raise RuntimeError("Response exceeded validated range; extra bytes were not written")
                        target.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - checkpoint >= 10:
                            target.flush()
                            emit(event="progress", name=name, bytes=offset + received, total=expected["size"])
                            checkpoint = now
                    target.flush()
                    os.fsync(target.fileno())
            except requests.RequestException as error:
                # A connection interrupted after actual progress can resume at a new,
                # validated offset. No-progress or status failures require investigation.
                new_size = partial.stat().st_size
                emit(event="stream_interrupted", name=name, error_type=type(error).__name__,
                     old_offset=offset, new_offset=new_size)
                if new_size <= offset or interruptions >= 3:
                    raise RuntimeError("Interrupted stream made no progress or exhausted 3 resumptions; partial preserved") from error
                interruptions += 1
                continue
            if received != body_size:
                raise RuntimeError(f"Truncated HTTP body: received {received}, expected {body_size}; partial preserved")
            emit(event="range_complete", name=name, bytes=partial.stat().st_size)
    verify(partial, expected)
    if source_dir.stat().st_dev != partial_dir.stat().st_dev:
        raise RuntimeError("Source and partial directories must share a filesystem for an atomic rename")
    if final.exists():
        raise RuntimeError(f"Destination appeared while downloading {name}; refusing overwrite")
    os.replace(partial, final)
    emit(event="installed", name=name, bytes=final.stat().st_size, sha256=expected["sha256"])


def pinned_metadata(session: requests.Session) -> dict[str, dict]:
    response = session.get(METADATA_URL, timeout=(15, 30))
    if response.status_code != 200:
        raise RuntimeError(f"HF metadata returned HTTP {response.status_code}; no metadata files changed")
    metadata = response.json()
    if metadata.get("sha") != REVISION or metadata.get("id") != REPO:
        raise RuntimeError("HF metadata repository/revision mismatch")
    siblings = {item["rfilename"]: item for item in metadata.get("siblings", [])}
    for name, expected in SHARDS.items():
        item = siblings.get(name, {})
        if item.get("size") != expected["size"] or item.get("lfs", {}).get("sha256") != expected["sha256"]:
            raise RuntimeError(f"Authoritative shard metadata changed for immutable revision: {name}")
    for name in METADATA_FILES:
        item = siblings.get(name, {})
        if not isinstance(item.get("size"), int) or item["size"] < 0 or not re.fullmatch("[0-9a-f]{40}", item.get("blobId", "")):
            raise RuntimeError(f"Missing/invalid authoritative metadata for {name}")
        if "lfs" in item and not re.fullmatch("[0-9a-f]{64}", item["lfs"].get("sha256", "")):
            raise RuntimeError(f"Missing/invalid LFS identity for {name}")
    return siblings


def verify_metadata_file(path: Path, expected: dict) -> dict:
    size = path.stat().st_size
    if size != expected["size"]:
        raise RuntimeError(f"Metadata size mismatch for {path.name}; existing file preserved")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            sha256.update(chunk); git_blob.update(chunk)
    expected_digest = expected["lfs"]["sha256"] if "lfs" in expected else expected["blobId"]
    actual_digest = sha256.hexdigest() if "lfs" in expected else git_blob.hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(f"Metadata identity mismatch for {path.name}; existing file preserved")
    return {"size": size, "sha256": sha256.hexdigest(), "git_blob_id": expected["blobId"],
            "verification_algorithm": "sha256" if "lfs" in expected else "git-blob-sha1",
            "expected_digest": expected_digest, "actual_digest": actual_digest, "verified": True}


def metadata_file(session: requests.Session, name: str, expected: dict, source_dir: Path, partial_dir: Path) -> dict:
    final = source_dir / name
    if final.exists():
        record = verify_metadata_file(final, expected)
        emit(event="metadata_verified", name=name, sha256=record["sha256"], preserved=True)
        return record
    partial = partial_dir / (name + ".metadata.partial")
    # Metadata transfers are small; a previous incomplete metadata file stays untouched
    # until a new HTTP response has been checked. Large weight partials are never reset.
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    with session.get(url, headers={"Accept-Encoding": "identity"}, stream=True, timeout=(15, 30)) as response:
        if response.status_code != 200 or response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
            raise RuntimeError(f"Unexpected metadata response for {name}: HTTP {response.status_code}; no file written")
        declared = response.headers.get("Content-Length")
        if declared is not None and (not declared.isdigit() or int(declared) != expected["size"]):
            raise RuntimeError(f"Metadata response length mismatch for {name}; no file written")
        count = 0
        with partial.open("wb") as target:
            for chunk in response.iter_content(chunk_size=CHUNK):
                if count + len(chunk) > expected["size"]:
                    raise RuntimeError(f"Metadata response exceeded authoritative size for {name}")
                target.write(chunk); count += len(chunk)
            target.flush(); os.fsync(target.fileno())
    record = verify_metadata_file(partial, expected)
    if source_dir.stat().st_dev != partial_dir.stat().st_dev or final.exists():
        raise RuntimeError(f"Cannot atomically install metadata {name}; verified temporary file preserved")
    os.replace(partial, final)
    emit(event="metadata_verified", name=name, sha256=record["sha256"], preserved=False)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "source")
    parser.add_argument("--partials", type=Path, default=Path("/tmp/dolphin3-ane-download-392a6f"))
    args = parser.parse_args()
    source_dir, partial_dir = args.source.resolve(), args.partials.resolve()
    # A hard guard prevents reproducing the workspace-sync failure described above.
    if not partial_dir.is_relative_to(Path("/tmp").resolve()):
        raise RuntimeError("Partials must be below /tmp, outside the synchronized workspace")
    source_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    with (partial_dir / "download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another downloader owns this partial directory") from error
        with requests.Session() as session:
            identities = pinned_metadata(session)
            files = {name: metadata_file(session, name, identities[name], source_dir, partial_dir) for name in METADATA_FILES}
            # Verify completed shards first, then append only the missing suffix.
            for name in sorted(SHARDS, key=lambda n: not (source_dir / n).exists()):
                download(session, name, source_dir, partial_dir)
                files[name] = {"size": SHARDS[name]["size"], "sha256": SHARDS[name]["sha256"],
                               "git_blob_id": identities[name]["blobId"], "verification_algorithm": "sha256",
                               "expected_digest": SHARDS[name]["sha256"], "actual_digest": SHARDS[name]["sha256"], "verified": True}
        report = {"repo": REPO, "revision": REVISION, "metadata_url": METADATA_URL,
                  "verified_at_utc": datetime.now(timezone.utc).isoformat(), "files": files, "all_verified": True}
        temporary = partial_dir / "source-verification.json.partial"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path = source_dir.parent / "source-verification.json"
        os.replace(temporary, report_path)
        emit(event="all_source_weights_verified", repo=REPO, revision=REVISION, metadata_url=METADATA_URL)


if __name__ == "__main__":
    main()
