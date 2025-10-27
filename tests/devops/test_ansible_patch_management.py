from __future__ import annotations

from pathlib import Path

import yaml

PLAYBOOK = (
    Path(__file__).resolve().parents[2]
    / "tasks"
    / "devops"
    / "ansible"
    / "patch_management.yml"
)


def test_playbook_structure() -> None:
    with PLAYBOOK.open("r", encoding="utf-8") as handle:
        documents = list(yaml.safe_load_all(handle))
    assert documents, "Expected at least one play"
    play = documents[0][0]
    assert play["become"] is True
    assert play["hosts"] == "all"
    tasks = play["tasks"]
    names = [task["name"] for task in tasks]
    assert "Apply security updates" in names
    assert any(task.get("notify") == "Restart host when required" for task in tasks)


def test_handler_reboot_configuration() -> None:
    with PLAYBOOK.open("r", encoding="utf-8") as handle:
        playbook = yaml.safe_load(handle)
    play = playbook[0]
    handlers = play.get("handlers", [])
    reboot_handler = next(
        handler
        for handler in handlers
        if handler["name"] == "Restart host when required"
    )
    module_args = reboot_handler["ansible.builtin.reboot"]
    assert module_args["reboot_timeout"] == 600
    assert module_args["test_command"] == "whoami"
