"""Infrastructure compliance scanning utilities (Task 50)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping

logger = logging.getLogger(__name__)


class ControlCheckError(RuntimeError):
    """Raised when a compliance control cannot be evaluated."""


@dataclass(frozen=True)
class Control:
    """Represents a single compliance control check."""

    name: str
    description: str
    check: Callable[[Mapping[str, object]], bool]

    def evaluate(self, resource: Mapping[str, object]) -> bool:
        """Evaluate control against a resource, logging failures."""

        try:
            result = bool(self.check(resource))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Control '%s' raised an exception for resource %s", self.name, resource
            )
            raise ControlCheckError(
                f"Control '{self.name}' failed with error: {exc}"
            ) from exc
        if not result:
            logger.debug(
                "Control '%s' failed for resource %s", self.name, resource
            )
        return result


DEFAULT_CONTROLS: tuple[Control, ...] = (
    Control(
        name="encryption_at_rest",
        description="All resources must enable encryption at rest.",
        check=lambda resource: bool(resource.get("encrypted", False)),
    ),
    Control(
        name="multi_az",
        description="Resources must span at least two availability zones.",
        check=lambda resource: int(resource.get("availability_zones", 1)) > 1,
    ),
)


def load_resources(path: Path) -> list[Mapping[str, object]]:
    """Load infrastructure resources from a JSON file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc

    if isinstance(payload, MutableMapping):
        resources = payload.get("resources", payload)
    else:
        resources = payload

    if not isinstance(resources, Iterable):
        raise ValueError("Resources payload must be an iterable of mappings")

    normalized: list[Mapping[str, object]] = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping):
            raise ValueError(f"Resource at index {index} must be a mapping")
        normalized.append(dict(resource))
    return normalized


def evaluate(
    resources: Iterable[Mapping[str, object]],
    *,
    controls: Iterable[Control] = DEFAULT_CONTROLS,
) -> dict[str, bool]:
    """Evaluate resources against compliance controls."""

    controls = tuple(controls)
    if not controls:
        raise ValueError("At least one control must be provided")

    results: dict[str, bool] = {control.name: True for control in controls}
    for resource in resources:
        for control in controls:
            compliant = control.evaluate(resource)
            results[control.name] = results[control.name] and compliant
    return results


def write_report(results: Mapping[str, bool], path: Path) -> None:
    """Write compliance results to ``path`` as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(results), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate infrastructure resources against compliance controls",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON file describing resources",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write compliance report JSON",
    )
    parser.add_argument(
        "--fail-on-noncompliant",
        action="store_true",
        help="Exit with non-zero status when any control fails",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for execution",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))

    try:
        resources = load_resources(args.input)
        results = evaluate(resources)
        write_report(results, args.output)
    except ControlCheckError as exc:
        logger.error("Compliance evaluation aborted: %s", exc)
        return 3
    except Exception as exc:  # noqa: BLE001
        logger.error("Compliance evaluation failed: %s", exc)
        return 2

    all_passed = all(results.values())
    if all_passed:
        logger.info("All controls satisfied across %d resources", len(resources))
    else:
        failed = [name for name, status in results.items() if not status]
        logger.warning(
            "Non-compliant controls detected: %s", ", ".join(sorted(failed))
        )
        if args.fail_on_noncompliant:
            return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
