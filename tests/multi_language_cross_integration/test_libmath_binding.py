from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake required")
def test_pybind11_module_builds_and_runs(tmp_path: Path) -> None:
    pybind11 = pytest.importorskip("pybind11")
    source_dir = REPO_ROOT / "tasks" / "multi_language_cross_integration" / "libmath"
    build_dir = tmp_path / "build"

    env = dict(os.environ)
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    artifacts = list(build_dir.glob("libmath*.so")) or list(build_dir.glob("Release/libmath*.dll"))
    if not artifacts:
        artifacts = list(build_dir.glob("libmath*.dylib"))
    if not artifacts:
        raise AssertionError("libmath module was not produced by the build")

    module_path = artifacts[0]
    spec = importlib.util.spec_from_file_location("libmath", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.add(2, 3) == 5
    assert module.mul(4, 6) == 24

    metadata = {
        "module": str(module_path),
        "doc": module.__doc__,
    }
    report_path = tmp_path / "libmath_report.json"
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert report_path.exists()
