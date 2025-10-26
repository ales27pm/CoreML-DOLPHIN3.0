# Repository Operations Playbook

This playbook governs code, documentation, and automation across the entire
CoreML-DOLPHIN3.0 workspace. Treat it as the canonical agreement for how
contributors interact with the manifest-driven documentation fabric, automation
scripts, and scoped contribution rules.

## Documentation Fabric
- `docs/documentation_manifest.json` defines every markdown surface maintained by
  `tools/session_finalize.py`. Update the manifest before introducing new logs or
  histories, then rerun the finalizer to materialize templates.
- Auto-managed sections (README timeline, Codex session journal, roadmap snapshot,
  session notes, session history) must be updated via the finalizer—never edit
  the generated blocks by hand.
- Keep the Codex ledger, roadmap, and session logs in sync with code changes;
  reviews must confirm all three moved together.

## Automation Lifecycle
- Begin sessions by running `python tools/manage_agents.py sync` to materialize
  scoped `AGENTS.md` files described in the manifest below.
- End sessions with `python tools/session_finalize.py --session-name "<date>" --summary "<work>" --include-git-status`.
  The finalizer now enforces manifest-driven document creation, README pruning,
  roadmap regeneration, and ledger updates.
- Never ship placeholder code or half implementations. Tests, documentation, and
  automation updates are mandatory for every change.

## Automation Manifest
The JSON block below drives automatic creation, update, and cleanup of scoped
`AGENTS.md` files. Maintain valid JSON and keep descriptions current with the
repository’s architecture.

<!--AGENTS_MANIFEST_BEGIN-->
{
  "directories": [
    {
      "path": "tools",
      "content": "# Tooling Automation Instructions\n\nThis scope covers files within the `tools/` directory, including automation and maintenance\nscripts. Apply the following rules:\n\n- Implement scripts in Python 3.10+ with comprehensive type hints and docstrings for every\n  public function.\n- Use the standard library whenever possible. If third-party dependencies are required, update\n  the manifest and document installation steps in the root README before usage.\n- Provide robust error handling with actionable messages. Prefer `logging` over bare prints for\n  reusable modules.\n- Write deterministic unit tests under `tests/tools` when modifying logic. Ensure they run via\n  `pytest` or the repository's preferred test runner.\n- Remember: no placeholders, simplified logic, or half implementations—ship production-grade\n  automation code only."
    },
    {
      "path": "docs",
      "content": "# Documentation Guidance\n\nThis scope governs all files under `docs/`.\n\n- Treat documentation as code: changes require clear narrative structure, tables kept in sync\n  with automation outputs, and explicit references to the manifest-driven workflow.\n- Auto-managed sections must remain compatible with `tools/session_finalize.py`; avoid manual\n  edits to generated timelines or snapshots.\n- Provide contextual introductions for new documents and include cross-links back to the Codex\n  ledger or roadmap when relevant.\n- Use semantic headings (H1/H2/H3) and fenced code blocks to aid navigation.\n- When adding templates, ensure they live under `docs/templates/` with descriptive names."
    },
    {
      "path": "docs/history",
      "content": "# Session History Guidance\n\nThis scope covers `docs/history/` including `SESSION_LOG.md`.\n\n- Never edit session entry blocks by hand—run `tools/session_finalize.py` to append updates.\n- Keep introductory context accurate and reference the automation manifest when adjusting structure.\n- Use this directory for canonical logs only; additional playbooks belong elsewhere in `docs/`."
    },
    {
      "path": "docs/templates",
      "content": "# Documentation Template Guidance\n\nThis scope governs reusable markdown templates stored under `docs/templates/`.\n\n- Templates must render valid Markdown with placeholders avoided entirely.\n- Keep instructions within templates concise and ensure they describe how automation scripts populate the sections.\n- When updating a template, note the change in `docs/documentation_manifest.json` if new files should be auto-created."
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
    },
    {
      "path": "tasks",
      "content": "# Tasks & Session Notes Guidance\n\nThis scope applies to everything under `tasks/`.\n\n- Keep task implementations production-ready with exhaustive tests and documentation.\n- For `tasks/SESSION_NOTES.md`, never edit the auto-managed session blocks manually—use the\n  session finalizer to append entries.\n- When authoring new tasks, document validation commands and cross-reference the Codex ledger\n  section that tracks the task.\n- Maintain parity between Python, Rust, Go, and TypeScript implementations referenced in the\n  Codex dashboard."
    }
  ]
}
<!--AGENTS_MANIFEST_END-->

## Maintenance Workflow
- Use `python tools/manage_agents.py check` in CI to validate manifest sync.
- The sync script deletes unmanaged `AGENTS.md` files, preventing stale instructions.
- Document any scope addition or removal in commit messages for traceability.

## Escalations
Update the manifest with new directory scopes as the architecture evolves. Never
introduce new technologies without corresponding scoped instructions.
