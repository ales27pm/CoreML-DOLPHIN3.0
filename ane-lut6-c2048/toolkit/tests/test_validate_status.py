"""Keep the CLI wired to the separately Swift-tested exit-status resolver."""

from pathlib import Path


def test_validate_cli_delegates_status_priority_to_tested_resolver():
    root = Path(__file__).parents[1] / "swift"
    cli = (root / "Sources" / "Validate" / "Validate.swift").read_text()
    assert "ValidationExitStatus.resolve(" in cli
