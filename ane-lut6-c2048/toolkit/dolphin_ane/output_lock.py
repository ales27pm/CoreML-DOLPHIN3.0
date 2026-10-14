"""Serialize writers to an output directory on supported Linux/macOS hosts."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path


@contextmanager
def exclusive_output_directory(directory):
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock outside the artifact inventory. Never unlink it: another
    # writer may already have opened this inode while waiting to acquire it.
    lock_path = directory.with_name(f".{directory.name}.dolphin-writer.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another job is writing this output directory: {directory}") from error
        yield directory
    finally:
        os.close(descriptor)
