from __future__ import annotations

from pathlib import Path

import yaml

MANIFEST_PATH = Path("tasks/systems_backend_engineering/deployment.yaml")


def test_deployment_manifest_structure() -> None:
    document = yaml.safe_load(MANIFEST_PATH.read_text())
    assert document["kind"] == "Deployment"
    container = document["spec"]["template"]["spec"]["containers"][0]
    assert container["name"] == "web"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health/live"
    assert container["startupProbe"]["failureThreshold"] >= 10
    assert container["resources"]["requests"]["cpu"] == "100m"


def test_deployment_manifest_labels_are_consistent() -> None:
    document = yaml.safe_load(MANIFEST_PATH.read_text())
    metadata_labels = document["metadata"]["labels"]
    template_labels = document["spec"]["template"]["metadata"]["labels"]
    selector_labels = document["spec"]["selector"]["matchLabels"]
    assert metadata_labels == template_labels == selector_labels
