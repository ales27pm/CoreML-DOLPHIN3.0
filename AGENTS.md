# Repository Automation & Contribution Protocol

This repository uses dynamic `AGENTS.md` files to communicate directory-specific contribution
rules. These guidelines are **mandatory** for every change within the repository.

## Repository Knowledge Snapshot
- **Core export pipeline** lives in `dolphin2coreml_full.py` and provides a full end-to-end
  conversion flow (Rich console UX, dependency bootstrapping, LoRA merge, LLM2Vec attachment,
  Core ML conversion, compression, and optional validation). Preserve the production-grade logging
  strategy (`rich` for UX, explicit error propagation) and dependency management via
  `ensure_packages` when evolving this script.
- **Swift runtime bindings** exist under `Sources/App/LLM/DolphinCoreML.swift`. They expose a thin
  wrapper around the multifunction `.mlpackage` the pipeline exports, including compute-unit
  selection, KV-cache streaming, embedding inference, and greedy sampling helpers.
- **Benchmark harness** functionality lives under `Sources/App/Bench/BenchmarkHarness.swift`,
  providing latency instrumentation, cache trimming safeguards, and tokenizer abstractions for
  integration testing within iOS/macOS apps.
- **Automation tooling** resides in `tools/manage_agents.py`, which enforces manifest-driven scoped
  instructions. Always update the manifest block below and run the sync workflow whenever scopes
  change so downstream instructions stay authoritative.

## Session Requirements
1. Treat this file as a **dynamic contract**: it is the single source of truth that drives
   creation, updates, and cleanup of every scoped `AGENTS.md` in the repository.
2. Run `python tools/manage_agents.py sync` immediately after cloning or before starting work
   to materialize any new scopes defined in the manifest below. Re-run the command at the end of
   every session—before committing—to refresh instructions and clean up unmanaged agent files.
  3. Respect the scope hierarchy defined by generated `AGENTS.md` files. More deeply nested files
   override parent scopes.
4. **Absolutely no placeholders, stubs, mocks, incomplete flows, or simplified code examples are
   permitted.** Deliver only advanced, fully implemented, production-ready logic with complete
   error handling, tests, and documentation relevant to the change.
5. Align with repository tooling (formatters, linters, tests) described in scoped instructions.
   If a tool is unavailable, document the limitation and provide remediation steps.
6. Keep this root file as the source of truth for the automation manifest below. Modify the
   manifest when adding, updating, or removing directory-specific instructions and then run the
   sync command to apply the changes immediately.

## Automation Manifest
The block below drives automatic creation, update, and cleanup of scoped `AGENTS.md` files.
Maintain valid JSON and avoid comments.

<!--AGENTS_MANIFEST_BEGIN-->
{
  "directories": [
    {
      "path": "tools",
      "content": "# Tooling Automation Instructions\n\nThis scope covers files within the `tools/` directory, including automation and maintenance\nscripts. Apply the following rules:\n\n- Implement scripts in Python 3.10+ with comprehensive type hints and docstrings for every\n  public function.\n- Use the standard library whenever possible. If third-party dependencies are required, update\n  the manifest and document installation steps in the root README before usage.\n- Provide robust error handling with actionable messages. Prefer `logging` over bare prints for\n  reusable modules.\n- Write deterministic unit tests under `tests/tools` when modifying logic. Ensure they run via\n  `pytest` or the repository's preferred test runner.\n- Remember: no placeholders, simplified logic, or half implementations—ship production-grade\n  automation code only."
    },
    {
      "path": "Sources",
      "content": "# Swift Sources Guidance\n\nThis scope applies to all Swift code under `Sources/`.\n\n- Target Swift 5.9+ with compatibility for iOS 18 and macOS 15 as reflected in the Core ML\n  packages produced by `dolphin2coreml_full.py`.\n- Keep modules free of placeholder implementations—mirror the production utilities already in\n  place (Core ML wrappers, benchmark harnesses) and ensure public APIs expose thorough\n  documentation comments when behaviour is non-obvious.\n- Maintain Core ML imports and avoid introducing platform-specific APIs without `#if canImport`\n  guards.\n- Prefer deterministic performance measurements and guard against unsafe pointer usage by\n  validating tensor ranks and data types, as demonstrated in existing files.\n- When introducing new Swift sources, accompany them with integration notes in the README or\n  inline doc comments so app developers understand how to wire them into Xcode projects."
    },
    {
      "path": "Sources/App/LLM",
      "content": "# DolphinCoreML Integration Guidance\n\nThis scope covers the `Sources/App/LLM/` directory, including `DolphinCoreML.swift`.\n\n- Preserve compatibility with multifunction `.mlpackage` artifacts that expose `init`, `decode`,\n  and `encode` entry points. Keep function names (`input_ids`, `attention_mask`, `past_k_*`,\n  `out_v_*`, etc.) synchronized with the exporter.\n- Keep `ComputeUnitSelection` and metadata-driven initialisers exhaustive. If new compute-unit\n  options or Core ML targets emerge, extend the enums and configuration mapping with full test\n  coverage.\n- Maintain robust error handling—propagate descriptive `NSError` payloads when Core ML outputs are\n  missing, and document all thrown errors.\n- Sampling helpers such as `greedySample` must continue to support both Float16 and Float32 logits\n  without introducing temporary allocations that would regress performance.\n- When adding new functionality (e.g., temperature sampling, batching), include unit or integration\n  tests exercising the logic with synthetic `MLMultiArray` inputs and update benchmarking guidance\n  accordingly."
    },
    {
      "path": "Sources/App/Bench",
      "content": "# Benchmark Harness Guidance\n\nThis scope covers the `Sources/App/Bench/` directory, including `BenchmarkHarness.swift`.\n\n- Retain the production-friendly benchmarking workflow: prompt encoding, warmup decode steps,\n  timed decode loop, and embedding latency measurement.\n- `trimCacheToSeqLen` must continue to validate tensor ranks and data types before performing\n  pointer arithmetic. Any modifications should add regression tests that exercise Float16 and\n  Float32 caches with varying sequence lengths.\n- Do not remove tokenizer protocol abstractions—extend `YourTokenizerProtocol` with documented\n  methods if the runtime contract evolves, and provide migration notes in doc comments.\n- Emit structured logs (e.g., formatted strings with timings) that are safe to consume in release\n  builds and avoid `print` spam unless reporting final benchmark summaries.\n- Keep the harness deterministic and make warmup/measurement token counts configurable through\n  method parameters when exposing new benchmarking scenarios."
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
