"""Release packaging must not accept pre-provenance or mixed identities."""

import hashlib
import json
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


def test_release_requires_one_fresh_identity_across_every_report(tmp_path, monkeypatch):
    identity = {"identity_sha256": "a" * 64}
    models, reference, verification = _inputs(tmp_path, identity)
    snapshot = {"verified": True}
    monkeypatch.setattr(build_release, "verify_pinned_source_snapshot", lambda *args, **kwargs: (identity, snapshot))
    manifest, report, actual_snapshot = build_release.verify_release_source_binding(
        models, reference, verification, tmp_path / "source", tmp_path / "verification.json")
    assert manifest["source_weight_identity"] == report["source_identity"]["source_weight_identity"] == identity
    assert actual_snapshot == snapshot

    verification["source_weight_identity"] = {"identity_sha256": "b" * 64}
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
        archive.writestr(member, b"tampered config bytes")
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
