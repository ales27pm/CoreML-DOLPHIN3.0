"""Utility helpers for CI tasks and dependency validation.

This module keeps lightweight logic that does not depend on the heavy
Core ML conversion stack.  The helpers are exercised in CI to ensure we
catch dependency regressions without importing optional packages such as
PyTorch during linting.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec as _find_spec
from typing import Callable, Iterable, Mapping

__all__ = [
    "DependencyCheckResult",
    "check_python_dependencies",
    "assert_dependencies",
]


@dataclass(frozen=True, slots=True)
class DependencyCheckResult:
    """Represents the outcome of checking optional Python dependencies."""

    required: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]

    def is_satisfied(self) -> bool:
        """Return ``True`` when all required dependencies are importable."""

        return not self.missing


def _normalise_dependencies(
    dependencies: Mapping[str, str] | Iterable[str],
) -> Mapping[str, str]:
    """Normalise a dependency specification.

    Args:
        dependencies: Either a mapping of package name -> import target or an
            iterable of package names where the import target matches the
            package name.

    Returns:
        An ordered mapping of package names to import targets without empty
        entries.
    """

    if isinstance(dependencies, Mapping):
        items = list(dependencies.items())
    else:
        items = [(name, name) for name in dependencies]

    normalised: dict[str, str] = {}
    for package, module_name in items:
        package = package.strip()
        module_name = module_name.strip()
        if not package or not module_name:
            msg = "Package names and module targets must be non-empty strings"
            raise ValueError(msg)
        # Preserve the last occurrence to keep behaviour predictable.
        normalised[package] = module_name
    return normalised


def check_python_dependencies(
    dependencies: Mapping[str, str] | Iterable[str],
) -> DependencyCheckResult:
    """Inspect optional dependencies using ``importlib``.

    The helper only reports whether modules are importable—it does not attempt
    to import them.  This keeps CI runs lightweight while still producing
    actionable feedback for contributors.
    """

    normalised = _normalise_dependencies(dependencies)
    present: list[str] = []
    missing: list[str] = []
    for package, module_name in normalised.items():
        if _find_spec(module_name) is None:
            missing.append(package)
        else:
            present.append(package)

    required = tuple(normalised.keys())
    return DependencyCheckResult(
        required=required,
        present=tuple(present),
        missing=tuple(missing),
    )


def assert_dependencies(
    dependencies: Mapping[str, str] | Iterable[str],
    *,
    logger: Callable[[str], None] | None = print,
) -> None:
    """Raise ``ModuleNotFoundError`` when dependencies cannot be imported."""

    result = check_python_dependencies(dependencies)
    if not result.missing:
        if logger is not None:
            logger(
                "All optional dependencies resolved: "
                + ", ".join(sorted(result.present))
            )
        return

    message = (
        "Missing optional dependencies: "
        + ", ".join(sorted(result.missing))
        + ". Install them before running the conversion pipeline."
    )
    if logger is not None:
        logger(message)
    raise ModuleNotFoundError(message)
