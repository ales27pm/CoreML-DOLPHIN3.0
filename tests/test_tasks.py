from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tasks as tasks_module  # noqa: E402  -- imported after path mutation for pytest
from tasks import (  # noqa: E402  -- imported after path mutation for pytest
    DependencyCheckResult,
    assert_dependencies,
    check_python_dependencies,
)


def test_check_python_dependencies_reports_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str):  # noqa: ANN202 - simple sentinel helper
        return object()

    monkeypatch.setattr(tasks_module, "_find_spec", fake_find_spec)
    result = check_python_dependencies({"rich": "rich"})

    assert isinstance(result, DependencyCheckResult)
    assert result.is_satisfied()
    assert result.present == ("rich",)
    assert result.missing == ()


def test_check_python_dependencies_records_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str):  # noqa: ANN202 - simple sentinel helper
        return None

    monkeypatch.setattr(tasks_module, "_find_spec", fake_find_spec)
    result = check_python_dependencies(["torch", "numpy"])

    assert not result.is_satisfied()
    assert result.present == ()
    assert result.missing == ("torch", "numpy")


def test_assert_dependencies_raises_with_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fake_find_spec(name: str):  # noqa: ANN202 - simple sentinel helper
        return None

    monkeypatch.setattr(tasks_module, "_find_spec", fake_find_spec)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        assert_dependencies(["torch"], logger=events.append)

    assert "Missing optional dependencies" in str(excinfo.value)
    assert events
    assert "torch" in events[-1]


def test_assert_dependencies_logs_success(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def fake_find_spec(name: str):  # noqa: ANN202 - simple sentinel helper
        return object()

    monkeypatch.setattr(tasks_module, "_find_spec", fake_find_spec)

    assert_dependencies(["pytest"], logger=events.append)

    assert events
    assert "pytest" in events[-1]
