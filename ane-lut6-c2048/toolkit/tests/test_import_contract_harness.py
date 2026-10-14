import json
import subprocess
import sys
import zipfile

import validate_import_contract


def test_harness_consumes_archived_bytes_and_distinguishes_missing_from_null(tmp_path, monkeypatch):
    output = tmp_path / "evidence"
    observed = []

    def run(command, **_kwargs):
        if "-swift-version" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        data = (output / "extracted" / "manifest.json").read_bytes()
        observed.append(data)
        # Only testing the harness transport, not replacing the real Swift contract test.
        return subprocess.CompletedProcess(command, 0 if len(observed) <= 2 else 1, "", "")

    monkeypatch.setattr(validate_import_contract.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["validate_import_contract.py", "--swarmer", str(tmp_path), "--output", str(output)])
    validate_import_contract.main()
    with zipfile.ZipFile(output / "SYNTHETIC-contract.zip") as archive:
        assert observed[0] == archive.read("manifest.json")
    report = json.loads((output / "results.json").read_text())
    cases = {case["case"]: json.loads(data) for case, data in zip(report["cases"], observed)}
    assert "source_weight_identity" not in cases["missing-identity"]
    assert cases["null-identity"]["source_weight_identity"] is None
