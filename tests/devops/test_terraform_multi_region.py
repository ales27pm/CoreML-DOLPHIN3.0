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
    assert primary["tls_certificate_arn"] == "${var.tls_certificate_arn}"
    assert secondary["tls_certificate_arn"] == "${var.tls_certificate_arn}"


def test_module_outputs_expose_health_checks() -> None:
    module_dir = ROOT / "modules" / "app"
    config = _load_hcl(module_dir / "main.tf")
    outputs: dict[str, dict] = {}
    for entry in config.get("output", []):
        outputs.update(entry)
    assert "alb_dns_name" in outputs
    assert "health_check_ids" in outputs


@pytest.mark.parametrize(
    "variable",
    ["primary_region", "secondary_region", "service_name", "tls_certificate_arn"],
)
def test_variables_defined(variable: str) -> None:
    raw_variables = _load_hcl(ROOT / "variables.tf").get("variable", [])
    flattened: dict[str, dict] = {}
    for entry in raw_variables:
        flattened.update(entry)
    assert variable in flattened


def test_module_alb_uses_multi_az_configuration() -> None:
    module_dir = ROOT / "modules" / "app"
    config = _load_hcl(module_dir / "main.tf")
    resources = config.get("resource", [])
    lb_blocks: dict[str, dict] = {}
    for resource in resources:
        lb_block = resource.get("aws_lb")
        if lb_block:
            lb_blocks.update(lb_block)
    alb = lb_blocks["this"]
    assert len(alb["subnets"]) == 2
    assert alb["drop_invalid_header_fields"] is True
