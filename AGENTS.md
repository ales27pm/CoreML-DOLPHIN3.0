# Repository Automation & Contribution Protocol

This repository uses dynamic `AGENTS.md` files to communicate directory-specific contribution
rules. These guidelines are **mandatory** for every change within the repository.

## Session Requirements
1. Run `python tools/manage_agents.py sync` immediately after cloning or before starting work
   to ensure all scoped `AGENTS.md` files are up to date. Re-run the command before committing
   to refresh instructions and clean up outdated agent files.
2. Respect the scope hierarchy defined by generated `AGENTS.md` files. More deeply nested files
   override parent scopes.
3. **Absolutely no placeholders, stubs, mocks, incomplete flows, or simplified code examples are
   permitted.** Deliver only advanced, fully implemented, production-ready logic with complete
   error handling, tests, and documentation relevant to the change.
4. Align with repository tooling (formatters, linters, tests) described in scoped instructions.
   If a tool is unavailable, document the limitation and provide remediation steps.
5. Keep this root file as the source of truth for the automation manifest below. Modify the
   manifest when adding, updating, or removing directory-specific instructions.

## Automation Manifest
The block below drives automatic creation, update, and cleanup of scoped `AGENTS.md` files.
Maintain valid JSON and avoid comments.

<!--AGENTS_MANIFEST_BEGIN-->
{
  "directories": [
    {
      "path": "tools",
      "content": "# Tooling Automation Instructions\n\nThis scope covers files within the `tools/` directory, including automation and maintenance\nscripts. Apply the following rules:\n\n- Implement scripts in Python 3.10+ with comprehensive type hints and docstrings for every\n  public function.\n- Use the standard library whenever possible. If third-party dependencies are required, update\n  the manifest and document installation steps in the root README before usage.\n- Provide robust error handling with actionable messages. Prefer `logging` over bare prints for\n  reusable modules.\n- Write deterministic unit tests under `tests/tools` when modifying logic. Ensure they run via\n  `pytest` or the repository's preferred test runner.\n- Remember: no placeholders, simplified logic, or half implementations—ship production-grade\n  automation code only."
    }
  ]
}
<!--AGENTS_MANIFEST_END-->

## Maintenance Workflow
- Use `python tools/manage_agents.py check` in CI or pre-commit hooks to verify that all agent
  files are synchronized without mutating the working tree.
- The sync script deletes unmanaged `AGENTS.md` files, preventing stale instructions.
- Document any scope addition or removal in commit messages for traceability.

## Escalations
If new directories require custom guidance, update the manifest above with their scoped
instructions and run the sync command. Never leave directories without explicit guidance when the
changes introduce new technologies or workflows.
