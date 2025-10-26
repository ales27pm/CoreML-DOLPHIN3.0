from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance_scanner import (
    Control,
    ControlCheckError,
    evaluate,
    load_resources,
    main,
    write_report,
)


def test_evaluate_all_controls_pass() -> None:
    resources = [
        {"id": "db-1", "encrypted": True, "availability_zones": 3},
        {"id": "queue-1", "encrypted": True, "availability_zones": 2},
    ]
    results = evaluate(resources)
    assert results == {"encryption_at_rest": True, "multi_az": True}


def test_evaluate_detects_failures_and_continues() -> None:
    resources = [
        {"id": "bucket", "encrypted": False, "availability_zones": 1},
        {"id": "db", "encrypted": True, "availability_zones": 1},
    ]
    results = evaluate(resources)
    assert results["encryption_at_rest"] is False
    assert results["multi_az"] is False


def test_evaluate_raises_on_control_exception() -> None:
    failing_control = Control(
        name="unstable",
        description="raises",
        check=lambda resource: 1 / resource["denominator"] > 0,
    )
    with pytest.raises(ControlCheckError):
        evaluate([{"denominator": 0}], controls=(failing_control,))


def test_load_resources_accepts_wrapped_payload(tmp_path: Path) -> None:
    payload = {"resources": [{"id": "svc", "encrypted": True, "availability_zones": 2}]}
    source = tmp_path / "resources.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    resources = load_resources(source)
    assert resources == payload["resources"]


def test_load_resources_invalid_payload(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text(json.dumps([123]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_resources(source)


def test_write_report_persists_results(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    write_report({"control": True}, destination)
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data == {"control": True}


def test_cli_success(tmp_path: Path) -> None:
    resources = tmp_path / "resources.json"
    report = tmp_path / "report.json"
    resources.write_text(
        json.dumps(
            [
                {"id": "svc", "encrypted": True, "availability_zones": 3},
                {"id": "cache", "encrypted": True, "availability_zones": 2},
            ]
        ),
        encoding="utf-8",
    )
    exit_code = main([
        "--input",
        str(resources),
        "--output",
        str(report),
        "--log-level",
        "DEBUG",
    ])
    assert exit_code == 0
    assert report.exists()


def test_cli_failure_on_noncompliant(tmp_path: Path) -> None:
    resources = tmp_path / "resources.json"
    report = tmp_path / "report.json"
    resources.write_text(
        json.dumps([
            {"id": "svc", "encrypted": True, "availability_zones": 1},
        ]),
        encoding="utf-8",
    )
    exit_code = main([
        "--input",
        str(resources),
        "--output",
        str(report),
        "--fail-on-noncompliant",
    ])
    assert exit_code == 3
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["multi_az"] is False
