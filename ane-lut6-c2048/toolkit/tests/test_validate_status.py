"""Keep the CLI wired to the separately Swift-tested exit-status resolver."""

from pathlib import Path


def test_validate_cli_delegates_status_priority_to_tested_resolver():
    root = Path(__file__).parents[1] / "swift"
    cli = (root / "Sources" / "Validate" / "Validate.swift").read_text()
    policy = (root / "Sources" / "DolphinANE" / "ValidationExitStatus.swift").read_text()
    swift_tests = (root / "Tests" / "DolphinANETests" / "ValidationExitStatusTests.swift").read_text()
    assert "ValidationExitStatus.resolve(" in cli
    assert "status = 2" not in cli and "status = 3" not in cli
    assert policy.index("if runtimeFailed { return 1 }") < policy.index("if numericalParityFailed { return 3 }")
    assert policy.index("if numericalParityFailed { return 3 }") < policy.index("if strictANERequested && !anePassed { return 2 }")
    for expected in ("), 0)", "), 1)", "), 2)", "), 3)"):
        assert expected in swift_tests
