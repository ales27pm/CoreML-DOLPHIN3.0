# CoreML Dolphin 3.0 Engineering Handbook

This repository packages everything required to export **Dolphin3.0-Llama3.1-8B**
into a multifunction Core ML `.mlpackage`, integrate the model inside Apple
platform apps, and verify runtime performance. The codebase is production-ready:
it contains the full Python conversion pipeline, Swift runtime bindings, a
benchmark harness, cross-language task implementations, and an automation layer
that keeps documentation synchronized with every engineering session.

## Platform Overview

- **Single-command export:** `dolphin2coreml_full.py` orchestrates dependency
  bootstrapping, LoRA merging, tokenizer validation, Core ML conversion,
  palettization/compression, and optional evaluation loops.
- **Multifunction runtime:** the exported `.mlpackage` exposes `init`, `decode`,
  and `encode_for_llm2vec` entry points for chat + embedding workloads.
- **Swift bindings:** `Sources/App/LLM/DolphinCoreML.swift` wraps the model with
  compute-unit selection, KV-cache streaming, greedy sampling, and metadata
  helpers tuned for iOS 18 / macOS 15.
- **Benchmark harness:** `Sources/App/Bench/BenchmarkHarness.swift` provides
  deterministic latency profiling with warmup control, tokenizer abstractions,
  and CSV export for regression monitoring.

## Conversion Pipeline

```bash
python dolphin2coreml_full.py \
  --model dphn/Dolphin3.0-Llama3.1-8B \
  --lora-checkpoint path/to/lora_dir \
  --llm2vec-checkpoint path/to/llm2vec_dir \
  --seq-len 2048 \
  --output out/Dolphin3_1_8B_W4.mlpackage \
  --minimum-deployment-target iOS18 \
  --profile-validate \
  --clean-tmp
```

Key behaviours:

1. Resolves Python dependencies with `ensure_packages`, ensuring deterministic
   environments for CI and local runs.
2. Validates tokenizer compatibility and sequence-length limits before
   exporting the model graph.
3. Applies 4-bit palettization with optional joint compression and writes the
   compressed package to disk.
4. Optionally benchmarks the converted model to confirm regression budgets.

## Runtime Integration

1. Add `DolphinCoreML.swift` to your Xcode target and ship the `.mlpackage`
   inside the application bundle.
2. Instantiate the wrapper with the exported metadata:

   ```swift
   let dolphin = try DolphinCoreML(
       modelURL: url,
       computeUnits: .all,
       metadata: (vocabSize: 32000, hiddenSize: 4096, numLayers: 32,
                  numHeads: 32, headDim: 128, seqLen: 2048)
   )
   ```

3. Use `initPass` + `decodeStep` for streaming chat inference, and
   `encodeEmbedding` for LLM2Vec workloads. The helper guarantees Float16/Float32
   compatibility and low-allocation cache updates.
4. Run the benchmark harness inside your app to capture `init`, per-token, and
   embedding latency using device-appropriate compute units.

## Documentation System

The repository now uses a manifest-driven documentation architecture. The
manifest (`docs/documentation_manifest.json`) declares every markdown file the
finalizer manages, the headers used for session journaling, retention limits,
and optional templates for auto-created files. The automation layer keeps the
following artefacts in sync:

- **README timeline:** Short-form view of the last three sessions.
- **Codex Master Task Ledger:** Operational status dashboard and canonical task
  specifications.
- **docs/history/SESSION_LOG.md:** Full chronological record for every session.
- **docs/ROADMAP.md:** Executive roadmap snapshot derived from the Codex
  dashboard.
- **tasks/SESSION_NOTES.md:** Long-form engineering notes and git status
  summaries.

## Automation Playbooks

Run the session finalizer whenever you finish a work block:

```bash
python tools/session_finalize.py \
  --session-name "Session 2025-10-27" \
  --summary "Refined documentation automation" \
  --note "manage_agents synced" \
  --include-git-status
```

The command will:

1. Load `docs/documentation_manifest.json` to determine which files require
   updates and create templated history files if they are missing.
2. Synchronize scoped `AGENTS.md` instructions via `tools/manage_agents.py`.
3. Append structured entries to the README timeline, Codex ledger, canonical
   session log, and session notes while pruning the README to the configured
   retention window.
4. Regenerate `docs/ROADMAP.md` using the latest `## Status Dashboard` section in
   the ledger.
5. Emit a detailed report summarizing which documents changed.

Use `python tools/manage_agents.py sync` after editing any manifest-controlled
`AGENTS.md` file to keep scoped guidance authoritative.

## Build & Validation Checklist

- `python -m pytest` — Python task validation.
- `cargo test` and `cargo bench -p parallel_csv_reader parallel_csv` — Rust
  artefacts and benchmarks.
- `go test ./...` within `tasks/multi_language_cross_integration/go_dot` — Go
  vector math module validation.
- `npm install && npm run lint && npm test` — TypeScript utilities and Vitest
  coverage.

Run the commands relevant to your change before finalizing a session.

## Session Timeline

<!-- session-log:session-2025-10-27-063000:2025-10-27T06:30:00+00:00 -->
### Session 2025-10-27 (2025-10-27T06:30:00+00:00)

**Summary:** Implemented Task 9 parallel CSV checksum crate, Task 10 GoDoc enrichment, and Task 11 docstring rewriter CLI

**Notes:**
- Added Rayon-backed checksum crate with Criterion bench output persisted to `benchmarks/parallel_csv.json`
- Authored Go `vectormath` package with Markdown GoDoc and deterministic arithmetic tests
- Delivered Google-style docstring CLI with logging, recursive traversal, and pytest coverage
- git status changes:
  - M `Codex_Master_Task_Results.md`
  - M Cargo.toml
  - M Cargo.lock
  - M `docs/ROADMAP.md`
  - M `tasks/SESSION_NOTES.md`
  - A `benchmarks/parallel_csv.json`
  - A `tasks/core_algorithmic_foundations/parallel_csv_reader/`
  - A `tasks/documentation/docstring_rewriter.py`
  - A `tasks/multi_language_cross_integration/go_dot/`
  - A `tests/documentation/test_docstring_rewriter.py`
  - Updated `tasks/core_algorithmic_foundations/parallel_csv_reader/benches/parallel_csv.rs`
- Tests & benches executed:
  - `cargo test`
  - `cargo bench -p parallel_csv_reader parallel_csv`
  - `go test ./...` (within `tasks/multi_language_cross_integration/go_dot`)
  - `pytest`
  - `npm run lint`

<!-- session-log:session-2025-10-27-073000:2025-10-27T07:30:00+00:00 -->
### Session 2025-10-27 (2025-10-27T07:30:00+00:00)

**Summary:** Reconciled ledger statuses for Tasks 7–11

**Notes:**
- Updated task ledger entries with implementation artifacts for Tasks 7–11
- Narrowed outstanding follow-up checklist to remaining unimplemented tasks
- git status changes:
  - M `Codex_Master_Task_Results.md`
  - M `tasks/SESSION_NOTES.md`

<!-- session-log:2025-10-27t12-00-00-00-00:2025-10-26T22:40:37+00:00 -->
### 2025-10-27T12:00:00+00:00 (2025-10-26T22:40:37+00:00)

**Summary:** Implemented tasks 12-15 with JSDoc enrichment, Flask integration coverage, React snapshots, and Rust proptest crate.

**Notes:**
- git status changes:
- M Cargo.lock
- M Cargo.toml
- M Codex_Master_Task_Results.md
- M eslint.config.js
- M package-lock.json
- M package.json
- M tsconfig.json
- M vitest.config.ts
- ?? requirements-dev.txt
- ?? tasks/core_algorithmic_foundations/safe_add/
- ?? tasks/documentation/flask_app.py
- ?? tasks/documentation/jsdoc_enricher.ts
- ?? tasks/multi_language_cross_integration/react_snapshot/
- ?? tests/documentation/test_flask_app.py
- ?? tests_ts/documentation/
- ?? tests_ts/react/

