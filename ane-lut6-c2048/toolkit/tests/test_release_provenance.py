"""Release packaging must not accept pre-provenance or mixed identities."""

import hashlib
import json
import os
import sys
import zipfile

import pytest

import build_release


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path, identity, version=3):
    models = tmp_path / "models"
    reference = tmp_path / "reference"
    _write(models / "manifest.json", {"format_version": version,
           "source_weight_identity": identity})
    _write(reference / "reference-report.json", {"schema_version": version,
           "source_identity": {"source_weight_identity": identity}})
    _write(reference / "reference-weight-hashes.json", {"schema_version": version,
           "source_weight_identity": identity})
    verification = {"schema_version": version, "source_weight_identity": identity}
    return models, reference, verification


@pytest.mark.parametrize("version", [2, 4])
def test_release_rejects_legacy_or_unknown_schema_before_packaging(tmp_path, monkeypatch, version):
    identity = {"identity_sha256": "a" * 64}
    models, reference, verification = _inputs(tmp_path, identity, version=version)
    monkeypatch.setattr(build_release, "verify_pinned_source_snapshot",
                        lambda *args, **kwargs: pytest.fail("legacy reports must fail before source verification"))
    with pytest.raises(RuntimeError, match="expected schema 3"):
        build_release.verify_release_source_binding(models, reference, verification,
                                                    tmp_path / "source", tmp_path / "verification.json")


@pytest.mark.parametrize("changed_report", ["manifest", "reference", "reference_hashes", "verification"])
def test_release_requires_one_fresh_identity_across_every_report(tmp_path, monkeypatch, changed_report):
    identity = {"identity_sha256": "a" * 64}
    models, reference, verification = _inputs(tmp_path, identity)
    snapshot = {"verified": True}
    monkeypatch.setattr(build_release, "verify_pinned_source_snapshot", lambda *args, **kwargs: (identity, snapshot))
    manifest, report, actual_snapshot = build_release.verify_release_source_binding(
        models, reference, verification, tmp_path / "source", tmp_path / "verification.json")
    assert manifest["source_weight_identity"] == report["source_identity"]["source_weight_identity"] == identity
    assert actual_snapshot == snapshot

    changed_identity = {"identity_sha256": "b" * 64}
    if changed_report == "verification":
        verification["source_weight_identity"] = changed_identity
    else:
        path, nested = {
            "manifest": (models / "manifest.json", False),
            "reference": (reference / "reference-report.json", True),
            "reference_hashes": (reference / "reference-weight-hashes.json", False),
        }[changed_report]
        value = json.loads(path.read_text())
        (value["source_identity"] if nested else value)["source_weight_identity"] = changed_identity
        _write(path, value)
    with pytest.raises(RuntimeError, match="do not share"):
        build_release.verify_release_source_binding(models, reference, verification,
                                                    tmp_path / "source", tmp_path / "verification.json")


def test_release_rehashes_metadata_bytes_stored_in_archive(tmp_path):
    content = b"pinned config bytes"
    records = [{"file": "config.json", "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}]
    archive_path = tmp_path / "release.zip"
    member = "Release/toolkit/source/config.json"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, content)
    build_release.verify_archived_source_metadata(archive_path, "Release/toolkit", records)

    with zipfile.ZipFile(archive_path, "w") as archive:
        changed = b"altered config byte"
        assert len(changed) == len(content)
        archive.writestr(member, changed)
    with pytest.raises(RuntimeError, match="Archived source metadata differs"):
        build_release.verify_archived_source_metadata(archive_path, "Release/toolkit", records)

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, content)
        archive.writestr("Release/toolkit/source/unverified.bin", b"unexpected")
    with pytest.raises(RuntimeError, match="metadata inventory differs"):
        build_release.verify_archived_source_metadata(archive_path, "Release/toolkit", records)


def _release_archive_inputs():
    def record(path, content):
        return {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    weight = record("Data/weight.bin", b"verified weights")
    sidecar = record("config.json", b"verified config")
    reference = record("reference-report.json", b"verified reference")
    manifest = record("manifest.json", b"verified manifest")
    fixtures = record("evidencefixtures.json", b"verified fixtures")
    source = record("source-verification.json", b"verified source")
    verification = {
        "packages": {"embedding": {"package_files_verified": [weight]}},
        "model_sidecar_files": [sidecar], "reference_files": [reference],
        "manifest_sha256": manifest["sha256"],
        "swift_fixtures": {"file": fixtures["path"], "sha256": fixtures["sha256"]},
        "pinned_source_verification": {"sha256": source["sha256"]},
    }
    members = {
        "Release/models/embedding.mlpackage/Data/weight.bin": b"verified weights",
        "Release/models/config.json": b"verified config",
        "Release/models/evidence/reference/reference-report.json": b"verified reference",
        "Release/models/manifest.json": b"verified manifest",
        "Release/models/evidencefixtures.json": b"verified fixtures",
        "Release/models/evidence/source-verification.json": b"verified source",
        "Release/toolkit/source-verification.json": b"verified source",
        "Release/models/evidence/verification.json": json.dumps(verification).encode(),
    }
    return verification, members


def _archive(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None


def test_release_accepts_exact_verified_archive_bytes(tmp_path):
    verification, members = _release_archive_inputs()
    path = tmp_path / "release.zip"
    _archive(path, members)
    build_release.verify_archived_release_inputs(path, "Release", verification)


@pytest.mark.parametrize("extra", ["Release/models/unlisted.txt",
    "Release/models/evidence/reference/unlisted.txt", "Release/models/evidence/unit-tests.xml",
    "Release/toolkit/secret.txt", "outside.txt"])
def test_release_rejects_unbound_archive_members(tmp_path, extra):
    verification, members = _release_archive_inputs()
    members[extra] = b"unbound bytes"
    path = tmp_path / "release.zip"
    _archive(path, members)
    with pytest.raises(RuntimeError, match="Archived.*inventory"):
        build_release.verify_archived_release_inputs(path, "Release", verification)


@pytest.mark.parametrize("member", list(_release_archive_inputs()[1]))
def test_release_rejects_same_size_archive_mutation_with_valid_crc(tmp_path, member):
    verification, members = _release_archive_inputs()
    contents = bytearray(members[member])
    contents[-1] ^= 1
    members[member] = bytes(contents)
    path = tmp_path / "release.zip"
    _archive(path, members)  # ZIP CRC remains correct: it does not bind verified bytes.
    with pytest.raises(RuntimeError, match="Archived"):
        build_release.verify_archived_release_inputs(path, "Release", verification)


@pytest.mark.parametrize("change", ["missing", "extra", "extra_package", "duplicate"])
def test_release_rejects_changed_archived_package_inventory(tmp_path, change):
    verification, members = _release_archive_inputs()
    weight = "Release/models/embedding.mlpackage/Data/weight.bin"
    if change == "missing":
        del members[weight]
    elif change == "extra":
        members["Release/models/embedding.mlpackage/extra.bin"] = b"extra"
    elif change == "extra_package":
        members["Release/models/unknown.mlpackage/weight.bin"] = b"extra"
    path = tmp_path / "release.zip"
    _archive(path, members)
    if change == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr(weight, members[weight])
    with pytest.raises(RuntimeError, match="Archived"):
        build_release.verify_archived_release_inputs(path, "Release", verification)


def test_release_rejects_changed_valid_archived_verification_report(tmp_path):
    verification, members = _release_archive_inputs()
    changed = {**verification, "unverified_claim": True}
    members["Release/models/evidence/verification.json"] = json.dumps(changed).encode()
    path = tmp_path / "release.zip"
    _archive(path, members)
    with pytest.raises(RuntimeError, match="Archived verification report differs"):
        build_release.verify_archived_release_inputs(path, "Release", verification)


@pytest.mark.parametrize("change", [None, "mutated", "missing"])
def test_small_toolkit_archive_preserves_pinned_source_report(tmp_path, change):
    verification, _ = _release_archive_inputs()
    members = {"Toolkit/source-verification.json": b"verified source"}
    if change == "mutated":
        members["Toolkit/source-verification.json"] = b"modified source"
    elif change == "missing":
        members.clear()
    path = tmp_path / "toolkit.zip"
    _archive(path, members)
    if change is None:
        build_release.verify_archived_release_inputs(path, "Toolkit", verification, toolkit_only=True)
    else:
        with pytest.raises(RuntimeError, match="Archived"):
            build_release.verify_archived_release_inputs(path, "Toolkit", verification, toolkit_only=True)


def test_edited_inventory_cannot_replace_the_independently_pinned_report(tmp_path):
    verification, members = _release_archive_inputs()
    report = tmp_path / "verification.json"
    original = json.dumps(verification).encode()
    report.write_bytes(original)
    trusted_digest = hashlib.sha256(original).hexdigest()
    assert build_release.read_pinned_verification(report, trusted_digest) == verification
    # A consistent replacement inventory and package cannot satisfy the retained
    # report digest, even though local package/report comparisons alone would.
    changed = b"modified weights"
    assert len(changed) == len(members["Release/models/embedding.mlpackage/Data/weight.bin"])
    verification["packages"]["embedding"]["package_files_verified"][0]["sha256"] = hashlib.sha256(changed).hexdigest()
    report.write_text(json.dumps(verification))
    with pytest.raises(RuntimeError, match="separately pinned SHA-256"):
        build_release.read_pinned_verification(report, trusted_digest)


@pytest.mark.parametrize("pin", [None, "", "abcd", "A" * 64])
def test_release_pin_requires_an_explicit_valid_digest(tmp_path, pin):
    with pytest.raises(ValueError, match="trusted SHA-256"):
        build_release.read_pinned_verification(tmp_path / "unused.json", pin)


def _member_inventory(members):
    return [{"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for path, content in members.items()]


@pytest.mark.parametrize("target", ["Release/toolkit/runtime.py", "Release/models/evidence/unit-tests.xml"])
def test_complete_archive_inventory_binds_supplemental_bytes(tmp_path, target):
    verification, members = _release_archive_inputs()
    content = b"supplemental bytes"
    members[target] = content
    if target.startswith("Release/models/"):
        verification["model_evidence_files"] = [{"path": target.removeprefix("Release/models/"),
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]
        members["Release/models/evidence/verification.json"] = json.dumps(verification).encode()
    inventory = _member_inventory(members)
    path = tmp_path / "release.zip"
    _archive(path, members)
    build_release.verify_archived_release_inputs(path, "Release", verification, complete_inventory=inventory)
    members[target] = b"X" + content[1:]
    _archive(path, members)
    with pytest.raises(RuntimeError, match="Archived file differs"):
        build_release.verify_archived_release_inputs(path, "Release", verification, complete_inventory=inventory)


def test_packaging_inventory_cannot_bless_unverified_model_evidence(tmp_path):
    verification, members = _release_archive_inputs()
    members["Release/models/evidence/unverified.log"] = b"new evidence"
    path = tmp_path / "release.zip"
    _archive(path, members)
    with pytest.raises(RuntimeError, match="inventory contains unverified"):
        build_release.verify_archived_release_inputs(path, "Release", verification,
            complete_inventory=_member_inventory(members))


@pytest.mark.parametrize("mutation", ["in_place", "replace"])
def test_archive_changed_between_payload_validation_and_digest_is_rejected(tmp_path, monkeypatch, mutation):
    verification, members = _release_archive_inputs()
    path = tmp_path / "release.zip"
    _archive(path, members)
    validate = build_release.verify_archived_release_inputs

    def validate_then_mutate(*args, **kwargs):
        validate(*args, **kwargs)
        changed = dict(members)
        changed["Release/models/config.json"] = b"modified config"
        replacement = path if mutation == "in_place" else tmp_path / "replacement.zip"
        _archive(replacement, changed)
        if mutation == "replace":
            replacement.replace(path)
        else:
            # Deterministic even on filesystems with coarse timestamp clocks.
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 2_000_000_000))

    monkeypatch.setattr(build_release, "verify_archived_release_inputs", validate_then_mutate)
    with pytest.raises(RuntimeError, match="Archive changed during"):
        build_release.verified_archive_record(path, "Release", verification, [], _member_inventory(members))


def test_archive_digest_binds_validated_bytes_and_late_change_is_not_rehashed(tmp_path):
    verification, members = _release_archive_inputs()
    path = tmp_path / "release.zip"
    _archive(path, members)
    record, snapshot = build_release.verified_archive_record(
        path, "Release", verification, [], _member_inventory(members))
    assert record == {"file": path.name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    members["Release/models/config.json"] = b"modified config"
    _archive(path, members)
    os.utime(path, ns=(path.stat().st_atime_ns, snapshot[4] + 2_000_000_000))
    with pytest.raises(RuntimeError, match="Archive changed after validation"):
        build_release.assert_archive_unchanged(path, snapshot)


def test_nonempty_destination_is_rejected_before_reading_or_writing_inputs(tmp_path, monkeypatch):
    destination = tmp_path / "release"
    destination.mkdir()
    preserved = destination / "Dolphin3_ANE_Conversion_Report.md"
    preserved.write_text("existing user report")
    monkeypatch.setattr(sys, "argv", ["build_release.py", "--models", str(tmp_path / "absent-models"),
        "--reference", str(tmp_path / "absent-reference"), "--destination", str(destination),
        "--verification-sha256", "a" * 64])
    with pytest.raises(FileExistsError, match="absent or empty"):
        build_release.main()
    assert list(destination.iterdir()) == [preserved]
    assert preserved.read_text() == "existing user report"


def test_missing_and_empty_release_destinations_are_accepted(tmp_path):
    destination = tmp_path / "release"
    build_release.require_fresh_destination(destination)
    destination.mkdir()
    build_release.require_fresh_destination(destination)


@pytest.mark.parametrize("extra", [None, "models/unlisted.txt", "models/evidence/unlisted.log",
    "reference/unlisted.txt"])
def test_prepackaging_requires_exact_model_and_reference_inventories(tmp_path, monkeypatch, extra):
    verification, members = _release_archive_inputs()
    verification["model_evidence_files"] = []
    models, reference, toolkit = (tmp_path / name for name in ("models", "reference", "toolkit"))
    for member, content in members.items():
        if member.startswith("Release/models/evidence/reference/"):
            path = reference / member.removeprefix("Release/models/evidence/reference/")
        elif member.startswith("Release/models/"):
            path = models / member.removeprefix("Release/models/")
        else:
            path = toolkit / member.removeprefix("Release/toolkit/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(build_release, "ROOT", toolkit)
    if extra is None:
        build_release.verify_current_inputs(models, reference, verification)
    else:
        path = tmp_path / extra
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unverified data")
        with pytest.raises(RuntimeError, match="inventory"):
            build_release.verify_current_inputs(models, reference, verification)


def _packaging_inputs(tmp_path, monkeypatch):
    """Small real filesystem/ZIP integration; source hashing is a separate seam."""
    toolkit, models, reference = (tmp_path / name for name in ("toolkit", "models", "reference"))
    for name in ("dolphin_ane", "tests", "swift", "licenses"):
        path = toolkit / name / "included.txt"
        path.parent.mkdir(parents=True)
        path.write_text("toolkit content")
    for name in ("README.md", "requirements.txt", "environment-lock.txt"):
        (toolkit / name).write_text("toolkit metadata")
    metadata = []
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "README.md"):
        path = toolkit / "source" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"pinned metadata")
        metadata.append({"file": name, "bytes": path.stat().st_size, "sha256": build_release.digest(path)})
    identity = {"identity_sha256": "a" * 64, "metadata": metadata}
    _write(toolkit / "source-verification.json", {"source_weight_identity": identity})
    _write(models / "manifest.json", {"format_version": 3, "source_weight_identity": identity,
        "weights": {"index_bits": 6}, "context_length": 8})
    _write(models / "evidencefixtures.json", {"cases": []})
    _write(models / "config.json", {})
    _write(models / "embedding.mlpackage" / "weight.bin", {"synthetic": True})
    _write(models / "evidence/unit-tests.xml", {"synthetic": True})
    _write(reference / "reference-report.json", {"schema_version": 3,
        "source_identity": {"source_weight_identity": identity}, "numerical_quality_gate_passed": True,
        "source_vs_lut6": [{"greedy_token_equal": True}] * 3})
    _write(reference / "reference-weight-hashes.json", {"schema_version": 3,
        "source_weight_identity": identity})
    _write(reference / "reference-progress.json", {"complete": True})

    def record(root, name):
        path = root / name
        return {"path": name, "bytes": path.stat().st_size, "sha256": build_release.digest(path)}

    verification = {"schema_version": 3, "source_weight_identity": identity,
        "artifact_integrity_and_weight_parity_passed": True, "serialized_lut_reconstruction_verified": True,
        "numerical_quality_gate_passed": True, "manifest_sha256": build_release.digest(models / "manifest.json"),
        "packages": {"embedding": {"package_files_verified": [record(models / "embedding.mlpackage", "weight.bin")]}},
        "model_sidecar_files": [record(models, "config.json")],
        "model_evidence_files": [record(models, "evidence/unit-tests.xml")],
        "reference_files": [record(reference, path.name) for path in sorted(reference.iterdir())],
        "swift_fixtures": {"file": "evidencefixtures.json", "sha256": build_release.digest(models / "evidencefixtures.json")},
        "pinned_source_verification": {"source_weight_identity": identity,
                                      "sha256": build_release.digest(toolkit / "source-verification.json")}}
    report = models / "evidence/verification.json"
    _write(report, verification)
    destination = tmp_path / "release"
    monkeypatch.setattr(build_release, "ROOT", toolkit)
    monkeypatch.setattr(build_release, "verify_pinned_source_snapshot", lambda *args: (identity, {}))
    monkeypatch.setattr(build_release, "assert_pinned_source_snapshot", lambda *args: None)
    monkeypatch.setattr(build_release, "report", lambda *args: "Synthetic conversion report")
    monkeypatch.setattr(sys, "argv", ["build_release.py", "--models", str(models), "--reference", str(reference),
        "--destination", str(destination), "--verification-sha256", build_release.digest(report)])
    return destination


def test_small_release_round_trip_covers_every_delivered_member(tmp_path, monkeypatch, capsys):
    destination = _packaging_inputs(tmp_path, monkeypatch)
    build_release.main()
    status = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert status["release_ready"] is True
    records = json.loads((destination / "release-files.json").read_text())
    assert status["files"] == records and len(records) == 3
    for record in records:
        path = destination / record["file"]
        assert path.stat().st_size == record["bytes"]
        assert build_release.digest(path) == record["sha256"]
    full = destination / "Dolphin3_ANE_Stateful_LUT6_C8.zip"
    with zipfile.ZipFile(full) as archive:
        assert any(name.endswith("reference-progress.json") for name in archive.namelist())
        assert any(name.endswith("unit-tests.xml") for name in archive.namelist())


def test_release_ready_is_not_emitted_if_archive_changes_after_its_verified_digest(tmp_path, monkeypatch, capsys):
    destination = _packaging_inputs(tmp_path, monkeypatch)
    verify_record = build_release.verified_archive_record
    first = None

    def mutate_first_after_second_digest(path, *args, **kwargs):
        nonlocal first
        result = verify_record(path, *args, **kwargs)
        if first is None:
            first = path
        else:
            before = first.stat()
            with zipfile.ZipFile(first, "a") as archive:
                archive.writestr("unverified.txt", "changed after verification")
            os.utime(first, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return result

    monkeypatch.setattr(build_release, "verified_archive_record", mutate_first_after_second_digest)
    with pytest.raises(RuntimeError, match="Archive changed after validation"):
        build_release.main()
    assert "release_ready" not in capsys.readouterr().out
    assert not (destination / "release-files.json").exists()
