from __future__ import annotations

from pathlib import Path

import hcl2
import pytest

ROOT = (
    Path(__file__).resolve().parents[2] / "tasks" / "devops" / "terraform_multi_region"
)


def _load_hcl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return hcl2.load(handle)


def test_main_defines_primary_and_secondary_modules() -> None:
    config = _load_hcl(ROOT / "main.tf")
    modules: dict[str, dict] = {}
    for entry in config.get("module", []):
        modules.update(entry)
    assert "primary" in modules
    assert "secondary" in modules
    primary = modules["primary"]
    secondary = modules["secondary"]
    assert primary["source"] == "./modules/app"
    assert primary["providers"]["aws"] == "${aws}"
    assert secondary["providers"]["aws"] == "${aws.secondary}"


def test_module_outputs_expose_health_checks() -> None:
    module_dir = ROOT / "modules" / "app"
    config = _load_hcl(module_dir / "main.tf")
    outputs: dict[str, dict] = {}
    for entry in config.get("output", []):
        outputs.update(entry)
    assert "alb_dns_name" in outputs
    assert "health_check_ids" in outputs


@pytest.mark.parametrize(
    "variable", ["primary_region", "secondary_region", "service_name"]
)
def test_variables_defined(variable: str) -> None:
    raw_variables = _load_hcl(ROOT / "variables.tf").get("variable", [])
    flattened: dict[str, dict] = {}
    for entry in raw_variables:
        flattened.update(entry)
    assert variable in flattened
