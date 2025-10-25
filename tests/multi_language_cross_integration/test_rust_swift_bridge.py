from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _artifact_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libffi_bridge.dylib"
    if system == "Windows":
        return "ffi_bridge.dll"
    return "libffi_bridge.so"


def _build_bridge() -> Path:
    env = dict(os.environ)
    env.setdefault("RUSTFLAGS", "-Cdebuginfo=0")
    subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(REPO_ROOT / "Cargo.toml"),
            "-p",
            "ffi_bridge",
            "--release",
        ],
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )
    artifact = REPO_ROOT / "target" / "release" / _artifact_name()
    if not artifact.exists():
        raise FileNotFoundError(f"Failed to locate built bridge at {artifact}")
    return artifact


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo toolchain required")
def test_rust_bridge_roundtrip(tmp_path: Path) -> None:
    library_path = _build_bridge()
    copied = tmp_path / library_path.name
    copied.write_bytes(library_path.read_bytes())

    bridge = ctypes.CDLL(str(copied))
    bridge.rust_echo.restype = ctypes.c_void_p
    bridge.rust_echo.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

    bridge.rust_free.restype = None
    bridge.rust_free.argtypes = [ctypes.c_void_p]

    payload = b"swift ffi roundtrip"
    buffer = (ctypes.c_uint8 * len(payload))(*payload)
    pointer = bridge.rust_echo(buffer, len(payload))
    assert pointer, "rust_echo returned NULL"

    echoed = ctypes.string_at(pointer).decode("utf-8")
    assert echoed == "swift ffi roundtrip"

    bridge.rust_free(pointer)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo toolchain required")
def test_rust_bridge_invalid_utf8(tmp_path: Path) -> None:
    library_path = _build_bridge()
    copied = tmp_path / library_path.name
    copied.write_bytes(library_path.read_bytes())

    bridge = ctypes.CDLL(str(copied))
    bridge.rust_echo.restype = ctypes.c_void_p
    bridge.rust_echo.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

    invalid = (ctypes.c_uint8 * 1)(0xFF)
    pointer = bridge.rust_echo(invalid, 1)
    assert pointer in (None, 0)
