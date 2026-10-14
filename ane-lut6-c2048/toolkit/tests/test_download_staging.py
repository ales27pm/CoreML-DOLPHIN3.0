import os
import sys
from pathlib import Path

import pytest

import download_source
from download_source import prepare_download_directories


def test_staging_accepts_external_same_filesystem_cache(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "source"
    partials = tmp_path / "cache"
    prepare_download_directories(source, partials, workspace)
    assert source.is_dir() and partials.is_dir()


@pytest.mark.parametrize("git_marker", ["directory", "worktree-file"])
def test_default_workspace_protects_entire_checkout(tmp_path, monkeypatch, git_marker):
    checkout = tmp_path / "checkout"
    toolkit = checkout / "ane-lut6-c2048" / "toolkit"
    toolkit.mkdir(parents=True)
    if git_marker == "directory":
        (checkout / ".git").mkdir()
    else:
        (checkout / ".git").write_text("gitdir: /not-used")
    monkeypatch.setattr(download_source, "__file__", str(toolkit / "download_source.py"))
    with pytest.raises(RuntimeError, match="separate"):
        prepare_download_directories(toolkit / "source", checkout / "other" / "partials")
    with pytest.raises(RuntimeError, match="separate"):
        prepare_download_directories(toolkit / "source", checkout / "other" / "partials", toolkit)
    assert not (checkout / "other").exists()


def test_explicit_workspace_guard_can_cover_a_larger_sync_root(tmp_path):
    workspace = tmp_path / "synchronized"
    with pytest.raises(RuntimeError, match="separate"):
        prepare_download_directories(workspace / "checkout" / "source", workspace / "another-checkout", workspace)


@pytest.mark.parametrize("relative", ["workspace/partial", "workspace/source", "workspace/source/partial", "."])
def test_staging_rejects_workspace_overlap_before_creating_files(tmp_path, relative):
    workspace = tmp_path / "workspace"
    with pytest.raises(RuntimeError, match="separate"):
        prepare_download_directories(workspace / "source", tmp_path / relative, workspace)
    assert not workspace.exists()


def test_staging_rejects_cross_filesystem_before_download(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    source = workspace / "source"
    partials = tmp_path / "cache"
    source.mkdir(parents=True)
    partials.mkdir()
    original = Path.stat

    def stat(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == partials:
            fields = list(value)
            fields[2] += 1
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(RuntimeError, match="same filesystem"):
        prepare_download_directories(source, partials, workspace)
    assert not list(partials.iterdir())


def test_cli_default_partials_uses_home_cache_not_tmpdir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    tmpdir = tmp_path / "tmpfs"
    source = tmp_path / "workspace" / "source"
    captured = {}

    class StagingCaptured(Exception):
        pass

    def capture(source_dir, partial_dir, workspace_dir=None):
        captured["source"] = source_dir
        captured["partials"] = partial_dir
        raise StagingCaptured

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setattr(download_source, "prepare_download_directories", capture)
    monkeypatch.setattr(sys, "argv", ["download_source.py", "--source", str(source)])

    with pytest.raises(StagingCaptured):
        download_source.main()

    assert captured["source"] == source.resolve()
    assert captured["partials"] == (home / ".cache" / "dolphin3-ane-download-392a6f").resolve()
    assert not captured["partials"].is_relative_to(tmpdir.resolve())
