"""Writer exclusion is enforced across processes and at both public entry points."""

from pathlib import Path
import subprocess
import sys

import pytest

from dolphin_ane.output_lock import exclusive_output_directory
from dolphin_ane import reference
import run_export


def test_output_lock_excludes_another_process_and_releases_after_failure(tmp_path):
    output = tmp_path / "output"
    script = ("import sys; from dolphin_ane.output_lock import exclusive_output_directory; "
              "\nwith exclusive_output_directory(sys.argv[1]): print('acquired')")

    def competing_writer():
        return subprocess.run([sys.executable, "-c", script, str(output)],
                              cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=10)

    with pytest.raises(ValueError, match="failure"):
        with exclusive_output_directory(output):
            result = competing_writer()
            assert result.returncode != 0
            assert "Another job is writing" in result.stderr
            assert "acquired" not in result.stdout
            raise ValueError("failure")
    result = competing_writer()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "acquired"
    assert not output.exists()  # The lock does not pollute output artifacts.


def test_export_entrypoint_rejects_concurrent_writer_before_processing(tmp_path, monkeypatch):
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(sys, "argv", ["run_export.py", "--output", str(output)])
    monkeypatch.setattr(run_export, "export", lambda *args: calls.append(args))
    with exclusive_output_directory(output), pytest.raises(RuntimeError, match="Another job is writing"):
        run_export.main()
    assert calls == []


def test_reference_entrypoint_rejects_concurrent_writer_before_processing(tmp_path, monkeypatch):
    output = tmp_path / "reference"
    calls = []
    monkeypatch.setattr(reference, "_run_reference_locked", lambda *args: calls.append(args))
    with exclusive_output_directory(output), pytest.raises(RuntimeError, match="Another job is writing"):
        reference.run_reference(tmp_path / "source", output)
    assert calls == []
