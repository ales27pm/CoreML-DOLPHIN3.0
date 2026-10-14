"""Synthetic nine-package orchestration; no large checkpoint or Apple runtime."""

import hashlib
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from pathlib import Path

import pytest

import download_source
import verify_export as verify
from dolphin_ane.output_lock import exclusive_output_directory, output_lock_path


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _synthetic_export(tmp_path, monkeypatch, mutate=None):
    models, reference, source = (tmp_path / name for name in ("models", "reference", "source"))
    source.mkdir()
    shard = source / "weights.bin"
    shard.write_bytes(b"source weights")
    pinned = tmp_path / "source-verification.json"
    pinned.write_text("{}")
    monkeypatch.setattr(download_source, "SHARDS", {shard.name: {}})
    monkeypatch.setattr(download_source, "PINNED_METADATA", {})
    identity = {"identity_sha256": "a" * 64}
    context, bits = 8, 6
    parts, serialized, reconstructed = {}, {}, {}
    for index, part in enumerate(verify.PART_IDS):
        kind = "embedding" if part == "embedding" else "head" if part == "head" else "block"
        start, end = (0, 0) if kind == "embedding" else (28, 28) if kind == "head" else verify.BLOCK_RANGES[index - 1]
        shapes = verify.expected_shapes(part)
        parts[part], weights = {}, []
        for name, shape in shapes.items():
            parts[part][name] = {"shape": shape, "source_fp16_sha256": "b" * 64,
                "quantized": True, "index_bits": bits, "dtype": "float16", "byte_order": "little",
                "order": "C", "group_size": 16, "channel_axis": 0, "dense_reference_sha256": "c" * 64}
            weights.append({"parameter_name": name, "shape": shape, "index_dtype": "uint6",
                "index_bits": bits, "lut_shape": [shape[0] // 16, *([1] * (len(shape) - 1)), 64, 1],
                "dense_fp16_sha256": "c" * 64})
        norms = {"norm.weight": [3072]} if kind == "head" else ({
            f"layers.{layer}.{norm}.weight": [3072] for layer in range(4)
            for norm in ("input_layernorm", "post_attention_layernorm")} if kind == "block" else {})
        for name, shape in norms.items():
            parts[part][name] = {"shape": shape, "source_fp16_sha256": "b" * 64, "quantized": False}
        states = ({"keyCache": [4, 1024, 1, context], "valueCache": [4, 1024, 1, context]}
                  if kind == "block" else {})
        description = {"inputs": {}, "outputs": {}, "input_dtypes": {}, "output_dtypes": {},
            "states": states, "state_dtypes": {name: "FLOAT16" for name in states},
            "operation_counts": {"constexpr_lut_to_dense": len(shapes)},
            "metadata": {"source_revision": verify.REVISION, "part": kind,
                "layer_start": str(start), "layer_end": str(end), "context_length": str(context),
                "weight_format": "LUT6 grouped-channel-16", "weight_index_bits": "6"}}
        package = models / (part + ".mlpackage")
        package.mkdir(parents=True)
        payload = package / "weight.bin"
        payload.write_bytes(b"serialized payload")
        inventory = [{"path": payload.name, "bytes": payload.stat().st_size,
                      "sha256": verify.file_hash(payload)}]
        _write(models / "evidence" / (part + ".json"), {**description,
            "stage": "lut6_compression", "part": kind, "layer_start": start, "layer_end": end,
            "context_length": context, "source_revision": verify.REVISION,
            "source_weight_identity": identity, "files": inventory, "weights": weights,
            "compression": {"large_weight_tensors_compressed": len(shapes),
                "large_weight_tensors_expected": len(shapes), "index_bits": bits,
                "group_size": 16, "channel_axis": 0}})
        serialized[part], reconstructed[part] = description, weights
    source_parameters = {part: {name: {key: value[key] for key in ("shape", "source_fp16_sha256")}
                              for name, value in weights.items()} for part, weights in parts.items()}
    signature = hashlib.sha256(json.dumps(source_parameters, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _write(reference / "reference-weight-hashes.json", {"schema_version": 3, "complete": True,
        "parts": parts, "index_bits": bits, "group_size": 16, "channel_axis": 0,
        "source_weight_identity": identity, "source_fp16_parameter_identity_sha256": signature})
    _write(reference / "reference-report.json", {"schema_version": 3, "complete": True,
        "context_length": context, "source_revision_requested": verify.REVISION,
        "results": {"lut6_reference": [True]}, "index_bits": bits,
        "compression": {"group_size": 16, "channel_axis": 0},
        "source_identity": {"source_fp16_parameter_identity_sha256": signature}})
    for name in ("reference-fixtures.json", "reference-logits.npz"):
        (reference / name).write_bytes(b"synthetic reference")
    _write(reference / "reference-progress.json", {"complete": True})
    _write(models / "manifest.json", {"format_version": 3, "source_revision": verify.REVISION,
        "source_weight_identity": identity, "block_names": verify.PART_IDS[1:-1],
        "block_layer_ranges": verify.BLOCK_RANGES,
        "model_files": [part + ".mlpackage" for part in verify.PART_IDS], "hidden_size": 3072,
        "head_dim": 128, "vocab_size": 128258, "query_length": 1, "head_sizes": [16032] * 7 + [16034],
        "states": "float16", "activations": "float16", "context_length": context,
        "state_bytes": 28 * 2 * 8 * 128 * context * 2,
        "weights": {"index_bits": bits, "format": "LUT6", "group_size": 16,
                    "channel_axis": 0, "lut_dtype": "float16"}})
    (models / "evidence/unit-tests.xml").write_text("<testsuites/>")
    calls = []

    def sidecars(*args, source_snapshot_out, **kwargs):
        verification_path = kwargs.get("source_verification") or pinned
        source_snapshot_out.update(download_source.capture_pinned_source_snapshot(source, verification_path))
        return [], {"source_weight_identity": identity, "sha256": verify.file_hash(verification_path)}

    def reconstruct(package, **kwargs):
        part = package.stem
        calls.append(part)
        if part == "head" and mutate is not None:
            targets = {"shard": shard, "verification": pinned,
                "reference_report": reference / "reference-report.json",
                "reference_hashes": reference / "reference-weight-hashes.json",
                "reference_logits": reference / "reference-logits.npz",
                "reference_fixtures": reference / "reference-fixtures.json",
                "reference_progress": reference / "reference-progress.json",
                "part_evidence": models / "evidence/embedding.json",
                "unit_tests": models / "evidence/unit-tests.xml"}
            if mutate in ("add_reference", "add_evidence"):
                target = (reference if mutate == "add_reference" else models / "evidence") / "unexpected.txt"
                target.write_text("unverified addition")
                return reconstructed[part]
            target = targets[mutate]
            before = target.stat()
            target.write_bytes(b"changed bytes")
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return reconstructed[part]

    monkeypatch.setattr(verify, "verify_sidecars", sidecars)
    monkeypatch.setattr(verify, "verify_fixtures", lambda *args, **kwargs: {
        "numerical_quality_gate": {"passed": True}, "numerical_quality_gate_passed": True,
        "cases": [{}, {}, {}]})
    monkeypatch.setattr(verify, "inspect_serialized_spec", lambda package: serialized[package.stem])
    monkeypatch.setattr(verify, "reconstruct_serialized_luts", reconstruct)
    return models, reference, source, pinned, calls


@pytest.mark.parametrize("mutate", ["shard", "verification"])
def test_source_mutated_during_last_reconstruction_cannot_commit_success(tmp_path, monkeypatch, mutate):
    models, reference, source, pinned, calls = _synthetic_export(tmp_path, monkeypatch, mutate)
    output = models / "evidence/verification.json"
    with pytest.raises(RuntimeError, match="changed during the worker phase"):
        verify.verify_export(models, reference, output, source, source_verification=pinned)
    assert calls == verify.PART_IDS
    result = json.loads(output.read_text())
    assert not result["artifact_integrity_and_weight_parity_passed"]
    assert not result["serialized_lut_reconstruction_verified"]
    assert "validation" not in json.loads((models / "manifest.json").read_text())
    assert not (models / "evidencefixtures.json").exists()


def test_success_reports_exact_serialized_digest_and_complete_supplemental_evidence(tmp_path, monkeypatch):
    models, reference, source, pinned, calls = _synthetic_export(tmp_path, monkeypatch)
    output = models / "evidence/verification.json"
    result = verify.verify_export(models, reference, output, source, source_verification=pinned)
    assert calls == verify.PART_IDS
    assert result["verification_report_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    records = result["model_evidence_files"]
    assert {item["path"] for item in records} == {
        *["evidence/" + part + ".json" for part in verify.PART_IDS], "evidence/unit-tests.xml"}
    for item in records:
        content = (models / item["path"]).read_bytes()
        assert item["bytes"] == len(content)
        assert item["sha256"] == hashlib.sha256(content).hexdigest()
    assert "verification_report_sha256" not in json.loads(output.read_text())
    assert {item["path"] for item in result["reference_files"]} == {
        "reference-report.json", "reference-weight-hashes.json", "reference-fixtures.json",
        "reference-logits.npz", "reference-progress.json"}


def test_atomic_report_digest_is_from_written_bytes_not_a_later_reread(tmp_path):
    output = tmp_path / "verification.json"
    expected = (json.dumps({"verified": True}, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    actual = verify.atomic_json(output, {"verified": True})
    assert actual == hashlib.sha256(expected).hexdigest()
    assert output.read_bytes() == expected


@pytest.mark.parametrize("mutate", ["reference_report", "reference_hashes", "reference_logits",
    "reference_fixtures", "reference_progress", "part_evidence", "unit_tests", "add_reference", "add_evidence"])
def test_evidence_changed_after_validation_cannot_be_blessed_by_a_late_inventory(tmp_path, monkeypatch, mutate):
    models, reference, source, pinned, calls = _synthetic_export(tmp_path, monkeypatch, mutate)
    output = models / "evidence/verification.json"
    with pytest.raises(RuntimeError, match="evidence.*changed during verification"):
        verify.verify_export(models, reference, output, source, source_verification=pinned)
    assert calls == verify.PART_IDS
    result = json.loads(output.read_text())
    assert not result["artifact_integrity_and_weight_parity_passed"]
    assert "validation" not in json.loads((models / "manifest.json").read_text())


def test_atomic_json_concurrent_writers_publish_their_own_bytes_without_shared_temp(tmp_path, monkeypatch):
    output = tmp_path / "verification.json"
    barrier, guard = Barrier(2), Lock()
    published = {}
    replace = Path.replace

    def coordinated_replace(temporary, destination):
        # Both writers have finished writing before either replacement occurs.
        barrier.wait(timeout=5)
        with guard:
            contents = temporary.read_bytes()
            value = json.loads(contents)
            replace(temporary, destination)
            published[value["writer"]] = contents
        return destination

    monkeypatch.setattr(Path, "replace", coordinated_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {writer: pool.submit(verify.atomic_json, output, {"writer": writer}) for writer in ("first", "second")}
        result = {writer: task.result(timeout=10) for writer, task in pending.items()}
    assert set(published) == set(result)
    for writer, digest in result.items():
        assert digest == hashlib.sha256(published[writer]).hexdigest()
    assert list(tmp_path.iterdir()) == [output]


def test_atomic_json_cleans_its_unique_temporary_file_when_replace_fails(tmp_path, monkeypatch):
    output = tmp_path / "verification.json"
    output.write_text("previous evidence")

    def fail_replace(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        verify.atomic_json(output, {"writer": "failed"})
    assert output.read_text() == "previous evidence"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("reference_name", ["reference", "cpu-reference"])
def test_nested_reference_lock_is_not_artifact_evidence_even_when_it_changes(tmp_path, monkeypatch, reference_name):
    import build_release

    models, reference, source, pinned, calls = _synthetic_export(tmp_path, monkeypatch)
    nested_reference = models / "evidence" / reference_name
    reference.rename(nested_reference)
    lock = output_lock_path(nested_reference)
    reconstruct = verify.reconstruct_serialized_luts

    def update_coordination_file(package, **kwargs):
        if package.stem == "head":
            lock.write_text("coordination data is not a verified model artifact")
        return reconstruct(package, **kwargs)

    monkeypatch.setattr(verify, "reconstruct_serialized_luts", update_coordination_file)
    with exclusive_output_directory(nested_reference):
        result = verify.verify_export(models, nested_reference, models / "evidence/verification.json",
                                      source, source_verification=pinned)
    assert calls == verify.PART_IDS
    assert lock.is_file()  # The persistent lock inode must not be removed.
    assert not any("dolphin-writer.lock" in item["path"] for item in result["model_evidence_files"])
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    build_release.verify_current_inputs(models, nested_reference, result)


@pytest.mark.parametrize("explicit", [False, True])
def test_verifier_resolves_default_or_explicit_report_from_a_relative_source_path(tmp_path, monkeypatch, explicit):
    models, reference, source, pinned, calls = _synthetic_export(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    if explicit:
        override = tmp_path / "explicit-verification.json"
        pinned.rename(override)
        pinned = override
    seen = []
    check = download_source.assert_pinned_source_snapshot

    def assert_resolved_inputs(source_directory, snapshot, verification_path):
        seen.append((source_directory, verification_path))
        return check(source_directory, snapshot, verification_path)

    monkeypatch.setattr(download_source, "assert_pinned_source_snapshot", assert_resolved_inputs)
    result = verify.verify_export(models, reference, models / "evidence/verification.json", Path("source"),
                                 source_verification=Path(pinned.name) if explicit else None)
    assert result["artifact_integrity_and_weight_parity_passed"]
    assert seen and all(pair == (source, pinned) for pair in seen)


@pytest.mark.parametrize("mutation", ["change", "replace", "add_file", "remove_file", "add_package", "remove_package", "symlink"])
def test_package_changed_after_its_own_reconstruction_cannot_commit_success(tmp_path, monkeypatch, mutation):
    models, reference, source, pinned, calls = _synthetic_export(tmp_path, monkeypatch)
    reconstruct = verify.reconstruct_serialized_luts

    def change_earlier_package(package, **kwargs):
        result = reconstruct(package, **kwargs)
        if package.stem == "head":
            earlier = models / "embedding.mlpackage"
            target = earlier / "weight.bin"
            before = target.stat()
            if mutation == "change":
                original = target.read_bytes()
                target.write_bytes(b"X" + original[1:])
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
            elif mutation == "replace":
                replacement = earlier / "replacement.bin"
                replacement.write_bytes(target.read_bytes())
                replacement.replace(target)
            elif mutation == "add_file":
                (earlier / "unexpected.bin").write_bytes(b"unverified")
            elif mutation == "remove_file":
                target.unlink()
            elif mutation == "add_package":
                (models / "unexpected.mlpackage").mkdir()
            else:
                moved = tmp_path / "moved-package"
                earlier.rename(moved)
                if mutation == "symlink":
                    earlier.symlink_to(moved, target_is_directory=True)
        return result

    monkeypatch.setattr(verify, "reconstruct_serialized_luts", change_earlier_package)
    output = models / "evidence/verification.json"
    with pytest.raises((RuntimeError, ValueError), match="changed during verification|package set"):
        verify.verify_export(models, reference, output, source, source_verification=pinned)
    assert calls == verify.PART_IDS
    assert not json.loads(output.read_text())["artifact_integrity_and_weight_parity_passed"]
    assert "validation" not in json.loads((models / "manifest.json").read_text())


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644, 0o664])
def test_atomic_json_preserves_existing_destination_mode(tmp_path, mode):
    output = tmp_path / "shared-evidence.json"
    output.write_text("old report")
    output.chmod(mode)
    verify.atomic_json(output, {"verified": True})
    assert stat.S_IMODE(output.stat().st_mode) == mode


@pytest.mark.parametrize("mask,expected", [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_new_atomic_json_obeys_umask_without_changing_process_global_umask(tmp_path, mask, expected):
    # A subprocess isolates the test's setup; production must never mutate or
    # temporarily clear the process-global umask in a multithreaded verifier.
    script = """
import os, pathlib, stat, sys
import verify_export
os.umask(int(sys.argv[2]))
def reject_umask_mutation(*args):
    raise AssertionError('atomic_json must not inspect or mutate global umask')
os.umask = reject_umask_mutation
path = pathlib.Path(sys.argv[1])
verify_export.atomic_json(path, {'verified': True})
print(stat.S_IMODE(path.stat().st_mode))
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "new-report.json"), str(mask)],
                            cwd=Path(verify.__file__).parent, capture_output=True, text=True, check=True, timeout=10)
    assert int(result.stdout.strip()) == expected


def test_atomic_json_incomplete_contents_remain_private_until_flush(tmp_path, monkeypatch):
    output = tmp_path / "shared.json"
    output.write_text("previous")
    output.chmod(0o664)
    fdopen = os.fdopen

    class InspectWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def write(self, contents):
            assert stat.S_IMODE(os.fstat(self.fileno()).st_mode) == 0o600
            return self.stream.write(contents)

        def flush(self):
            return self.stream.flush()

    monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: InspectWriter(fdopen(*args, **kwargs)))
    verify.atomic_json(output, {"verified": True})
    assert stat.S_IMODE(output.stat().st_mode) == 0o664


def test_package_mutation_during_result_preparation_cannot_publish_success(tmp_path, monkeypatch):
    models, reference, source, pinned, _ = _synthetic_export(tmp_path, monkeypatch)
    atomic_json = verify.atomic_json

    def change_package_after_manifest_write(path, value):
        digest = atomic_json(path, value)
        if path == models / "manifest.json":
            target = models / "embedding.mlpackage/weight.bin"
            before = target.stat()
            target.write_bytes(b"changed after first package recheck")
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return digest

    monkeypatch.setattr(verify, "atomic_json", change_package_after_manifest_write)
    output = models / "evidence/verification.json"
    with pytest.raises(RuntimeError, match="changed during verification"):
        verify.verify_export(models, reference, output, source, source_verification=pinned)
    report = json.loads(output.read_text())
    assert not report["artifact_integrity_and_weight_parity_passed"]
    assert not report["serialized_lut_reconstruction_verified"]


@pytest.mark.parametrize("name", ["tokenizer.json", "rope_cos.fp16"])
def test_sidecar_changed_after_early_validation_cannot_publish_success(tmp_path, monkeypatch, name):
    models, reference, source, pinned, _ = _synthetic_export(tmp_path, monkeypatch)
    target = models / name
    original = b"verified sidecar bytes"
    target.write_bytes(original)
    record = {"path": name, "bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}
    sidecars = verify.verify_sidecars
    reconstruct = verify.reconstruct_serialized_luts

    def verified_sidecar(*args, **kwargs):
        _, identity = sidecars(*args, **kwargs)
        return [record], identity

    def mutate_after_sidecar_validation(package, **kwargs):
        result = reconstruct(package, **kwargs)
        if package.stem == "head":
            target.write_bytes(b"X" + original[1:])
        return result

    monkeypatch.setattr(verify, "verify_sidecars", verified_sidecar)
    monkeypatch.setattr(verify, "reconstruct_serialized_luts", mutate_after_sidecar_validation)
    output = models / "evidence/verification.json"
    with pytest.raises(RuntimeError, match="Model sidecar changed during verification"):
        verify.verify_export(models, reference, output, source, source_verification=pinned)
    assert not json.loads(output.read_text())["artifact_integrity_and_weight_parity_passed"]


def test_earlier_package_cannot_change_while_later_package_is_rehashed(tmp_path, monkeypatch):
    models, reference, source, pinned, _ = _synthetic_export(tmp_path, monkeypatch)
    file_hash = verify.file_hash
    head_reads = 0

    def mutate_previous_package_during_final_rehash(path):
        nonlocal head_reads
        digest = file_hash(path)
        if Path(path) == models / "head.mlpackage/weight.bin":
            head_reads += 1
            # Initial capture, normal file validation, then first commit guard.
            if head_reads == 3:
                target = models / "embedding.mlpackage/weight.bin"
                before = target.stat()
                target.write_bytes(b"changed while later package is checked")
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return digest

    monkeypatch.setattr(verify, "file_hash", mutate_previous_package_during_final_rehash)
    output = models / "evidence/verification.json"
    with pytest.raises(RuntimeError, match="package changed during verification"):
        verify.verify_export(models, reference, output, source, source_verification=pinned)
    assert "validation" not in json.loads((models / "manifest.json").read_text())
