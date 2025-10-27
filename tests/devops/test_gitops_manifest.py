from __future__ import annotations

from pathlib import Path

import yaml

APPLICATION = Path(__file__).resolve().parents[2] / "tasks" / "devops" / "gitops" / "backend-application.yaml"


def test_manifest_contains_required_fields() -> None:
    with APPLICATION.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    assert manifest["apiVersion"] == "argoproj.io/v1alpha1"
    assert manifest["kind"] == "Application"
    spec = manifest["spec"]
    assert spec["destination"]["namespace"] == "production"
    assert spec["syncPolicy"]["automated"]["selfHeal"] is True
    assert "CreateNamespace=true" in spec["syncPolicy"]["syncOptions"]


def test_manifest_includes_retry_configuration() -> None:
    with APPLICATION.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    retry = manifest["spec"]["syncPolicy"]["retry"]
    assert retry["limit"] == 5
    assert retry["backoff"]["factor"] == 2
