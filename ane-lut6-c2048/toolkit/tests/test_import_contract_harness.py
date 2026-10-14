import json
import subprocess
import sys
import zipfile

import pytest

import validate_import_contract


def test_harness_consumes_archived_bytes_and_distinguishes_missing_from_null(tmp_path, monkeypatch):
    output = tmp_path / "evidence"
    observed = []
    timeouts = []

    def run(command, **_kwargs):
        timeouts.append(_kwargs.get("timeout"))
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
    assert report["passed"]
    assert report["compilation"]["passed"]
    assert not report["compilation"]["timed_out"]
    assert timeouts == [120, *([15] * len(observed))]
    assert all(not case["timed_out"] for case in report["cases"])
    cases = {case["case"]: json.loads(data) for case, data in zip(report["cases"], observed)}
    assert "source_weight_identity" not in cases["missing-identity"]
    assert cases["null-identity"]["source_weight_identity"] is None


def test_compile_timeout_is_reported_as_a_completed_failure(tmp_path, monkeypatch, capsys):
    output = tmp_path / "evidence"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 120),
                                        output=b"compiler partial output\n", stderr=b"compiler partial error\n")

    monkeypatch.setattr(validate_import_contract.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["validate_import_contract.py", "--swarmer", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        validate_import_contract.main()
    assert error.value.code == 1
    assert len(commands) == 1
    report = json.loads((output / "results.json").read_text())
    assert not report["passed"]
    assert report["cases"] == []
    compilation = report["compilation"]
    assert compilation["timed_out"]
    assert compilation["returncode"] is None
    assert compilation["timeout_seconds"] == 120
    assert not compilation["passed"]
    log = (output / "compile.log").read_text()
    assert "compiler partial output" in log and "compiler partial error" in log
    assert "timed out after 120 seconds" in log
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("timed_out_case", [0, 2])
def test_reader_timeout_fails_even_an_expected_rejection_and_continues(tmp_path, monkeypatch, timed_out_case):
    output = tmp_path / "evidence"
    calls = []

    def run(command, **kwargs):
        if "-swift-version" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        index = len(calls)
        calls.append(command)
        if index == timed_out_case:
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 15),
                                            output="reader partial output\n", stderr=b"reader partial error\n")
        return subprocess.CompletedProcess(command, 0 if index < 2 else 1, "", "")

    monkeypatch.setattr(validate_import_contract.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["validate_import_contract.py", "--swarmer", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        validate_import_contract.main()
    assert error.value.code == 1
    report = json.loads((output / "results.json").read_text())
    assert not report["passed"]
    assert len(calls) == len(report["cases"]) == 19
    failed = report["cases"][timed_out_case]
    assert failed["expected_accept"] is (timed_out_case == 0)
    assert failed["accepted"] is None and failed["returncode"] is None
    assert failed["timed_out"] and not failed["passed"]
    assert failed["timeout_seconds"] == 15
    assert "reader partial output" in failed["stdout"]
    assert "reader partial error" in failed["stderr"]
    assert "timed out after 15 seconds" in failed["stderr"]
    assert all(case["passed"] for index, case in enumerate(report["cases"]) if index != timed_out_case)
    assert (output / "extracted/manifest.json").read_bytes() == (output / "synthetic-models/manifest.json").read_bytes()


def test_compile_error_also_writes_a_failed_report_without_running_cases(tmp_path, monkeypatch):
    output = tmp_path / "evidence"
    monkeypatch.setattr(validate_import_contract.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, "", "compiler error\n"))
    monkeypatch.setattr(sys, "argv", ["validate_import_contract.py", "--swarmer", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        validate_import_contract.main()
    assert error.value.code == 1
    report = json.loads((output / "results.json").read_text())
    assert not report["passed"] and not report["compilation"]["passed"]
    assert not report["compilation"]["timed_out"]
    assert report["compilation"]["returncode"] == 1
    assert report["cases"] == []
    assert (output / "compile.log").read_text() == "compiler error\n"
