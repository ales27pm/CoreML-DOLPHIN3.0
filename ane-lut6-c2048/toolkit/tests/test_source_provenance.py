"""Pinned source-shard revalidation and export-report binding regressions."""

import hashlib
import json
import os

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

import download_source
import run_export


def _verification(source, name):
    path = source / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {"size": path.stat().st_size, "sha256": digest}
    report = {"repo": download_source.REPO, "revision": download_source.REVISION,
              "all_verified": True, "files": {name: {**expected, "verified": True}}}
    verification = source.parent / "source-verification.json"
    verification.write_text(json.dumps(report), encoding="utf-8")
    return expected, verification


def test_same_size_safetensors_payload_mutation_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    name = "model.safetensors"
    path = source / name
    save_file({"weight": torch.arange(32, dtype=torch.float32)}, str(path))
    expected, verification = _verification(source, name)
    monkeypatch.setattr(download_source, "SHARDS", {name: expected})
    monkeypatch.setattr(download_source, "PINNED_METADATA", {})

    identity = download_source.verify_pinned_source_shards(source, verification)
    assert identity["shards"] == [{"file": name, "bytes": expected["size"],
                                    "sha256": expected["sha256"]}]

    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1  # Safetensors payload only: header and byte count stay valid.
    path.write_bytes(contents)
    assert path.stat().st_size == expected["size"]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert handle.get_tensor("weight").shape == (32,)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        download_source.verify_pinned_source_shards(source, verification)


def test_changed_weight_map_and_unsigned_report_cannot_redirect_verified_shards(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    shards = {}
    for name, value in (("official-a.safetensors", b"official-a"),
                        ("official-b.safetensors", b"official-b"),
                        ("unverified.safetensors", b"unverified")):
        (source / name).write_bytes(value)
        if name.startswith("official"):
            shards[name] = {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    index_path = source / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"a": "official-a.safetensors",
                                                       "b": "official-b.safetensors"}}))
    pinned_index = {"size": index_path.stat().st_size,
                    "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()}
    monkeypatch.setattr(download_source, "SHARDS", shards)
    monkeypatch.setattr(download_source, "PINNED_METADATA", {index_path.name: pinned_index})
    files = {name: {**value, "verified": True} for name, value in shards.items()}
    files[index_path.name] = {**pinned_index, "verified": True}
    verification = tmp_path / "source-verification.json"
    verification.write_text(json.dumps({"repo": download_source.REPO, "revision": download_source.REVISION,
                                         "all_verified": True, "files": files}))
    download_source.verify_pinned_source_shards(source, verification)

    index_path.write_text(json.dumps({"weight_map": {"a": "unverified.safetensors",
                                                       "b": "official-b.safetensors"}}))
    forged = json.loads(verification.read_text())
    forged_record = {"size": index_path.stat().st_size,
                     "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(), "verified": True}
    forged["files"][index_path.name] = forged_record
    verification.write_text(json.dumps(forged))
    with pytest.raises(RuntimeError, match="metadata identity mismatch"):
        download_source.verify_pinned_source_shards(source, verification)


def test_stat_snapshot_detects_mutate_then_restore(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    shard = source / "model.safetensors"
    original = b"original pinned bytes"
    shard.write_bytes(original)
    expected = {"size": len(original), "sha256": hashlib.sha256(original).hexdigest()}
    monkeypatch.setattr(download_source, "SHARDS", {shard.name: expected})
    monkeypatch.setattr(download_source, "PINNED_METADATA", {})
    verification = tmp_path / "source-verification.json"
    verification.write_text(json.dumps({"repo": download_source.REPO, "revision": download_source.REVISION,
                                         "all_verified": True,
                                         "files": {shard.name: {**expected, "verified": True}}}))
    _, snapshot = download_source.verify_pinned_source_snapshot(source, verification)
    previous = shard.stat()
    shard.write_bytes(b"temporary evil bytes")
    shard.write_bytes(original)
    # A distinct two-second timestamp also works on coarse filesystems; this
    # must not depend on two adjacent writes crossing a kernel clock tick.
    os.utime(shard, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))
    with pytest.raises(RuntimeError, match="changed during"):
        download_source.assert_pinned_source_snapshot(source, snapshot, verification)


@pytest.fixture
def completed_export(tmp_path):
    package = tmp_path / "embedding.mlpackage"
    package.mkdir()
    payload = package / "payload.bin"
    payload.write_bytes(b"package")
    report_path = tmp_path / "embedding.json"
    identity = {"repo": download_source.REPO, "revision": download_source.REVISION,
                "shards": [{"file": "tiny.safetensors", "bytes": 4, "sha256": "a" * 64}],
                "identity_sha256": "b" * 64}
    report = {"source_revision": download_source.REVISION, "part": "embedding",
              "layer_start": 0, "layer_end": 0, "context_length": 8,
              "stage": "fp16_conversion", "source_weight_identity": identity,
              "files": [{"path": payload.name, "bytes": payload.stat().st_size,
                          "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}]}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return package, report_path, identity, report


def test_completed_export_requires_exact_source_weight_identity(completed_export):
    package, report_path, identity, report = completed_export
    assert run_export.validate_completed(package, report_path, "embedding", 0, 0, 8,
                                         "fp16_conversion", identity)

    report["source_weight_identity"] = {**identity, "identity_sha256": "c" * 64}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source_weight_identity"):
        run_export.validate_completed(package, report_path, "embedding", 0, 0, 8,
                                      "fp16_conversion", identity)


@pytest.mark.parametrize("mutation", ["empty", "traversal", "absolute", "duplicate", "unlisted", "symlink"])
def test_completed_export_requires_exact_confined_inventory(completed_export, mutation):
    package, report_path, identity, report = completed_export
    if mutation == "empty":
        report["files"] = []
    elif mutation in {"traversal", "absolute"}:
        external = package.parent / "external.bin"
        external.write_bytes(b"package")
        report["files"][0]["path"] = "../external.bin" if mutation == "traversal" else str(external)
    elif mutation == "duplicate":
        report["files"] *= 2
    elif mutation == "unlisted":
        (package / "unlisted.bin").write_bytes(b"unverified")
    else:
        (package / "linked.bin").symlink_to(package / "payload.bin")
    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="integrity check failed"):
        run_export.validate_completed(package, report_path, "embedding", 0, 0, 8,
                                      "fp16_conversion", identity)


def test_worker_report_cannot_be_retroactively_blessed_with_source_identity(tmp_path):
    report_path = tmp_path / "stage.json"
    report_path.write_text(json.dumps({"stage": "fp16_conversion"}), encoding="utf-8")
    identity = {"identity_sha256": "a" * 64}

    with pytest.raises(RuntimeError, match="missing its required source identity"):
        run_export.verify_stage_source_identity(report_path, identity)
    assert "source_weight_identity" not in json.loads(report_path.read_text())

    report_path.write_text(json.dumps({"source_weight_identity": identity}), encoding="utf-8")
    run_export.verify_stage_source_identity(report_path, identity)


@pytest.mark.parametrize("dangling", ["model", "report"])
def test_dangling_prior_output_symlinks_are_not_treated_as_absent(tmp_path, dangling):
    package, report = tmp_path / "embedding.mlpackage", tmp_path / "embedding.json"
    (package if dangling == "model" else report).symlink_to(tmp_path / "missing-target")
    with pytest.raises(RuntimeError, match="Incomplete prior output requires inspection"):
        run_export.validate_completed(package, report, "embedding", 0, 0, 8, "fp16_conversion", {})


def test_absent_prior_output_can_still_be_created(tmp_path):
    assert not run_export.validate_completed(tmp_path / "embedding.mlpackage", tmp_path / "embedding.json",
                                            "embedding", 0, 0, 8, "fp16_conversion", {})
