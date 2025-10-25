# Tooling Automation Instructions

This scope covers files within the `tools/` directory, including automation and maintenance
scripts. Apply the following rules:

- Implement scripts in Python 3.10+ with comprehensive type hints and docstrings for every
  public function.
- Use the standard library whenever possible. If third-party dependencies are required, update
  the manifest and document installation steps in the root README before usage.
- Provide robust error handling with actionable messages. Prefer `logging` over bare prints for
  reusable modules.
- Write deterministic unit tests under `tests/tools` when modifying logic. Ensure they run via
  `pytest` or the repository's preferred test runner.
- Remember: no placeholders, simplified logic, or half implementations—ship production-grade
  automation code only.
