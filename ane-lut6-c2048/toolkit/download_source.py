#!/usr/bin/env python3
"""Download the pinned public Dolphin weights with verified, resumable HTTP ranges.

Large partials stay in a disk-backed cache outside the workspace. Never use HF snapshot_download(local_dir=workspace)
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
import stat
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
# SHA-256 is pinned in code as well as recorded in source-verification.json.
# Otherwise a caller could change both a metadata file (especially the weight
# map) and its unsigned verification report while leaving the two official
# shards untouched.
PINNED_METADATA = {
    "config.json": {"size": 935, "sha256": "e21ff53ea39726f972362beba869807216775d5e308bc2f531784846c06a0249"},
    "generation_config.json": {"size": 206, "sha256": "e627b5a8b2dc371f90388947ada64fa6e71de0f991c04c835f0c0bc97e305a4f"},
    "model.safetensors.index.json": {"size": 20_919, "sha256": "e6322ad005d27134ba439f7ddd572e632e5b2116ebfbdc88186596dada904c75"},
    "tokenizer.json": {"size": 17_210_298, "sha256": "e40b93124a3e29f62d5f4ff41be56cb2af34ecacf9239acd9da53a98860380b5"},
    "tokenizer_config.json": {"size": 53_573, "sha256": "51ad9580aba8d00016efda43357185a0d8ff9884584dcc82ab58ca552afd14e1"},
    "special_tokens_map.json": {"size": 444, "sha256": "2df2c4620bb1a9eb877bc7c90c7fa04608bda9fa7c0cf2cdcc0a17b849649683"},
    "README.md": {"size": 5_840, "sha256": "0c4007169c0ebeb8928283543a8f5115ef85414f437aa7cfa5ad2b85095c3336"},
}


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


def pinned_source_weight_identity(report: dict) -> dict:
    """Return the canonical identity of every pinned source input in *report*.

    The downloader report is useful evidence, but it is not itself proof that the
    files still contain the bytes that were downloaded. This helper validates the
    report against the immutable constants above; ``verify_pinned_source_shards``
    performs the separate, current-byte verification.
    """
    if (report.get("all_verified") is not True or report.get("repo") != REPO
            or report.get("revision") != REVISION):
        raise RuntimeError("Pinned source verification repository/revision mismatch")
    files = report.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Pinned source verification has no file inventory")
    shards = []
    for name in sorted(SHARDS):
        authoritative = SHARDS[name]
        record = files.get(name)
        if (not isinstance(record, dict) or record.get("verified") is not True
                or record.get("size") != authoritative["size"]
                or record.get("sha256") != authoritative["sha256"]):
            raise RuntimeError(f"Pinned source verification shard identity mismatch: {name}")
        shards.append({"file": name, "bytes": authoritative["size"],
                       "sha256": authoritative["sha256"]})
    metadata = []
    for name in sorted(PINNED_METADATA):
        authoritative = PINNED_METADATA[name]
        record = files.get(name)
        if (not isinstance(record, dict) or record.get("verified") is not True
                or record.get("size") != authoritative["size"]
                or record.get("sha256") != authoritative["sha256"]):
            raise RuntimeError(f"Pinned source metadata identity mismatch: {name}")
        metadata.append({"file": name, "bytes": authoritative["size"],
                         "sha256": authoritative["sha256"]})
    value = {"repo": REPO, "revision": REVISION, "shards": shards, "metadata": metadata}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    value["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    recorded = report.get("source_weight_identity")
    if recorded is not None and recorded != value:
        raise RuntimeError("Recorded source weight identity differs from the pinned shard inventory")
    return value


def _safe_source_path(source_dir: Path, name: str) -> Path:
    path = source_dir / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source_dir):
        raise RuntimeError(f"Pinned source file is missing, unsafe, or not a regular file: {path}")
    return path


def _stat_record(path: Path) -> dict:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"Pinned source path is not a regular file: {path}")
    return {"device": value.st_dev, "inode": value.st_ino, "mode": value.st_mode,
            "bytes": value.st_size, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def capture_pinned_source_snapshot(source_dir: Path, verification_path: Path | None = None) -> dict:
    """Capture cheap mutation evidence for use around a long worker phase."""
    source_dir = Path(source_dir).resolve()
    verification_path = (Path(verification_path).resolve() if verification_path is not None
                         else source_dir.parent / "source-verification.json")
    names = sorted(set(SHARDS) | set(PINNED_METADATA))
    files = {name: _stat_record(_safe_source_path(source_dir, name)) for name in names}
    directory = source_dir.stat()
    return {"source_directory": {"device": directory.st_dev, "inode": directory.st_ino,
                                  "mtime_ns": directory.st_mtime_ns, "ctime_ns": directory.st_ctime_ns},
            "verification_file": {"path": str(verification_path), **_stat_record(verification_path)},
            "files": files}


def assert_pinned_source_snapshot(source_dir: Path, snapshot: dict,
                                  verification_path: Path | None = None) -> None:
    """Fail if a pinned input was replaced or mutated, even if bytes were restored."""
    current = capture_pinned_source_snapshot(source_dir, verification_path)
    if current != snapshot:
        raise RuntimeError("Pinned source inputs changed during the worker phase")


def verify_pinned_source_snapshot(source_dir: Path, verification_path: Path | None = None) -> tuple[dict, dict]:
    """Re-hash all pinned inputs and return identity plus a stable stat snapshot."""
    source_dir = Path(source_dir).resolve()
    verification_path = (Path(verification_path).resolve() if verification_path is not None
                         else source_dir.parent / "source-verification.json")
    before = capture_pinned_source_snapshot(source_dir, verification_path)
    report = json.loads(verification_path.read_text(encoding="utf-8"))
    identity = pinned_source_weight_identity(report)
    for item in (*identity["shards"], *identity["metadata"]):
        path = _safe_source_path(source_dir, item["file"])
        actual = verify(path, {"size": item["bytes"], "sha256": item["sha256"]})
        if actual != item["sha256"]:  # ``verify`` already fails; keep the invariant explicit.
            raise RuntimeError(f"Pinned source SHA256 mismatch: {path}")

    index_name = "model.safetensors.index.json"
    if index_name in PINNED_METADATA:
        index = json.loads((source_dir / index_name).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("Pinned safetensors index has no nonempty weight_map")
        referenced = set(weight_map.values())
        if (any(not isinstance(name, str) for name in weight_map.values())
                or referenced != set(SHARDS)):
            raise RuntimeError("Pinned safetensors index does not reference exactly the verified shards")
    after = capture_pinned_source_snapshot(source_dir, verification_path)
    if before != after:
        raise RuntimeError("Pinned source inputs changed while they were being verified")
    return identity, after


def verify_pinned_source_shards(source_dir: Path, verification_path: Path | None = None) -> dict:
    """Re-hash every pinned source input and return its canonical identity.

    Callers use this at the export/reference boundary rather than trusting the
    earlier download event. The returned value contains no paths or timestamps,
    so it can be embedded verbatim in manifests and compared deterministically.
    """
    return verify_pinned_source_snapshot(source_dir, verification_path)[0]


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


def prepare_download_directories(source_dir: Path, partial_dir: Path,
                                 workspace_dir: Path | None = None) -> None:
    """Reject unsafe staging before requesting any large response body."""
    workspace_dir = (workspace_dir or Path(__file__).resolve().parent).resolve()
    source_dir, partial_dir = source_dir.resolve(), partial_dir.resolve()
    for protected in (workspace_dir, source_dir):
        if partial_dir.is_relative_to(protected) or protected.is_relative_to(partial_dir):
            raise RuntimeError("Partials must be separate from the synchronized workspace and source directory")
    source_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.stat().st_dev != partial_dir.stat().st_dev:
        raise RuntimeError("Source and partial directories must share the same filesystem for an atomic rename; "
                           "choose a disk-backed cache with --partials (not a separate /tmp tmpfs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "source")
    parser.add_argument("--partials", type=Path,
                        default=Path.home() / ".cache" / "dolphin3-ane-download-392a6f")
    args = parser.parse_args()
    source_dir, partial_dir = args.source.resolve(), args.partials.resolve()
    prepare_download_directories(source_dir, partial_dir)
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
        report["source_weight_identity"] = pinned_source_weight_identity(report)
        temporary = partial_dir / "source-verification.json.partial"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path = source_dir.parent / "source-verification.json"
        os.replace(temporary, report_path)
        emit(event="all_source_weights_verified", repo=REPO, revision=REVISION, metadata_url=METADATA_URL)


if __name__ == "__main__":
    main()
