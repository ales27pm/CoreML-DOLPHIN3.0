"""Writer exclusion is enforced across processes and at both public entry points."""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

import pytest

from dolphin_ane.output_lock import exclusive_output_directory, output_lock_path
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


def test_explicit_build_directory_excludes_exports_with_other_outputs(tmp_path):
    output, build = tmp_path / "a-output", tmp_path / "z-shared-build"
    script = ("import run_export; run_export.export = lambda *args: print('processed'); "
              "run_export.main()")
    with exclusive_output_directory(build):
        result = subprocess.run([sys.executable, "-c", script, "--output", str(output),
                                 "--build-dir", str(build)], cwd=Path(__file__).parents[1],
                                capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Another job is writing" in result.stderr
    assert "processed" not in result.stdout
    # If build-lock acquisition fails, the previously acquired output lock is
    # released as well; a retry must not inherit a stranded lock.
    with exclusive_output_directory(output):
        pass


@pytest.mark.parametrize("use_alias", [False, True])
def test_equal_canonical_output_and_build_paths_are_rejected_before_locking(tmp_path, monkeypatch, capsys, use_alias):
    output = tmp_path / "output"
    output.mkdir()
    alias = tmp_path / "build-alias"
    alias.symlink_to(output, target_is_directory=True)
    monkeypatch.setattr(run_export, "exclusive_output_directory", lambda _: pytest.fail("invalid paths must not create locks"))
    monkeypatch.setattr(sys, "argv", ["run_export.py", "--output", str(output),
                                     "--build-dir", str(alias if use_alias else output)])
    with pytest.raises(SystemExit) as error:
        run_export.main()
    assert error.value.code == 2
    assert "--build-dir must differ from --output" in capsys.readouterr().err
    assert list(output.iterdir()) == []
    assert not output_lock_path(output).exists()
    assert output_lock_path(alias) == output_lock_path(output)


@pytest.mark.parametrize("entrypoint", ["reference", "export", "export-build"])
def test_symlink_retarget_does_not_redirect_the_locked_output(tmp_path, monkeypatch, entrypoint):
    locked_target, other_target = tmp_path / "locked", tmp_path / "other"
    locked_target.mkdir()
    other_target.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(locked_target, target_is_directory=True)
    seen = []

    @contextmanager
    def retarget_after_acquiring(directory):
        with exclusive_output_directory(directory) as resolved:
            alias.unlink()
            alias.symlink_to(other_target, target_is_directory=True)
            yield resolved

    if entrypoint == "reference":
        monkeypatch.setattr(reference, "exclusive_output_directory", retarget_after_acquiring)
        monkeypatch.setattr(reference, "_run_reference_locked", lambda _, output, *args: seen.append(output.resolve()))
        reference.run_reference(tmp_path / "source", alias)
    else:
        monkeypatch.setattr(run_export, "exclusive_output_directory", retarget_after_acquiring)
        monkeypatch.setattr(run_export, "export", lambda args, _: seen.append(
            (args.build_dir if entrypoint == "export-build" else args.output).resolve()))
        arguments = (["--output", str(tmp_path / "export-output"), "--build-dir", str(alias)]
                     if entrypoint == "export-build" else ["--output", str(alias)])
        monkeypatch.setattr(sys, "argv", ["run_export.py", *arguments])
        run_export.main()
    assert alias.resolve() == other_target
    assert seen == [locked_target]


@pytest.mark.parametrize("options", [
    {"mode": "invalid"}, {"bits": 3}, {"palette_workers": 0},
    {"mode": "source", "source_reference": "prior"},
    {"source_reference": "same-output"}, {"context_length": 0}, {"threads": "invalid"},
])
def test_invalid_reference_arguments_do_not_create_lock_or_directories(tmp_path, options):
    output = tmp_path / "not-created" / "reference"
    options = dict(options)
    if options.get("source_reference") == "same-output":
        options["source_reference"] = output
    with pytest.raises(ValueError):
        reference.run_reference(tmp_path / "source", output, **options)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("option", ["--context", "--layers-per-block", "--threads", "--palette-workers"])
def test_invalid_export_arguments_do_not_create_lock_or_directories(tmp_path, monkeypatch, option):
    output = tmp_path / "not-created" / "output"
    monkeypatch.setattr(sys, "argv", ["run_export.py", "--output", str(output), option, "0"])
    with pytest.raises(SystemExit) as error:
        run_export.main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
