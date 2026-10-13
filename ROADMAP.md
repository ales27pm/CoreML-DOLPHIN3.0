# CoreML Dolphin 3.0 Roadmap

The existing 8B roadmap aligns ongoing development with this objective:
ship a production-ready Core ML export of **Dolphin3.0-Llama3.1-8B** that merges
LoRA adapters via Unsloth, applies 4-bit quantization (with optional sweeps),
and delivers multifunction chat + LLM2Vec capabilities through a Swift runtime
wrapper.

## Separately versioned 3B stateful release

[`ane-lut6-c2048/`](ane-lut6-c2048/README.md) adds Dolphin3.0-Llama3.2-3B as a
separate nine-package pipeline and Swift runtime. The iOS 18 / macOS 15 export
uses LUT6/group16, FP16 activations and native KV states, a fixed 2048-token
context and query length 1. The existing 8B pipeline and `Sources/App` APIs
retain their multifunction contract.

- Completed: package export, all 205 serialized weight reconstruction checks,
  the recorded 84-test Python suite, and 3/3 fixed-prompt next-token agreement
  with the FP16 source. This evidence has limited numerical scope.
- Publication target: GitHub code/evidence plus model files under
  `ales27pm/Dolphin3.0-CoreML/ane-lut6-c2048/`; the version README documents the
  layout and separate dependency environment.
- Pending: Apple SDK compilation, Core ML device parity, ANE plan inspection
  and runtime tracing, followed by measured device performance and broader
  generation/context evaluation. No Apple result is inferred from Linux export.

The remaining principles and milestones describe the existing 8B workstream.

## Guiding Principles

- **Single-command conversion** – Maintain `dolphin2coreml_full.py` as the 8B
  entry point for end-to-end export, dependency bootstrap, and artifact cleanup.
- **Feature completeness** – Every 8B release must validate LoRA merging, 4-bit
  palettization, multifunction Core ML outputs (init/decode/encode), and
  embedding parity with the upstream LLM2Vec checkpoint.
- **Apple platform readiness** – Ensure exported packages, Swift runtime code,
  and documentation remain compatible with current iOS, macOS, and visionOS
  toolchains.

## Current State (Q2 FY25)

- Conversion pipeline merges LoRA weights with Unsloth, exports
  init/decode/encode graphs, and persists multifunction `.mlpackage` bundles.
- Quantization defaults to 4-bit grouped-channel palettization, with
  CLI-configurable group sizes, mixed-precision overrides, backend-specific
  guard rails, and an automated sweep mode that benchmarks multiple
  configurations while emitting Rich/JSON summaries for CI consumption.
- `--profile-validate` executes golden transcript comparisons, reports decode
  latency percentiles + KV-cache residency, and enforces ≥0.99 cosine similarity
  for LLM2Vec embeddings.
- Swift runtime (`Sources/App/LLM`) includes streaming decode helpers,
  background-task patterns, and batched embedding APIs documented in the README.

## Near-Term Objectives (0-2 sprints)

1. **Sweep automation in CI** (✅ Sprint 3)
   - Nightly matrix sweeps now run via `.github/workflows/quantization-sweep.yml`
     and publish JSON artefacts for regression tracking.
   - `tasks/sweep_guard.py` enforces configurable latency and package-size
     guardrails in pull requests.
2. **Validation extensibility** (✅ Sprint 3)
   - Golden prompts and embedding suites can be sourced from external
     YAML/JSON, and validation runs emit machine-readable JSON alongside Rich
     tables.
3. **Swift runtime ergonomics** (✅ Sprint 3)
   - Async/await wrappers and KV-cache utilities simplify concurrency-aware
     integrations with the runtime.

## Mid-Term Initiatives (2-4 sprints)

1. **Model variant support** (🚧 Sprint 4)
   - Exporter auto-tags Dolphin 3B/8B/70B checkpoints, emits structured
     `dolphin-metadata.json`, and the Swift runtime can load metadata directly
     from the package for configuration.
   - Document adapter-merging strategies when stacking multiple task-specific
     LoRAs.
2. **Telemetry & benchmarking automation**
   - Run the benchmarking harness on Apple Silicon CI, storing historical
     latency trends and surfacing regressions in pull requests.
3. **Distribution hardening**
   - Publish signed SwiftPM artifacts from CI and attach SBOM/provenance data to
     exported `.mlpackage` bundles.

## Long-Term Goals (4+ sprints)

1. **Operational excellence**
   - Embed monitoring hooks for production deployments (memory footprint,
     decode throughput) and feed insights back into quantization decisions.
2. **Tooling ecosystem integration**
   - Integrate with internal registries for reproducible downloads of Core ML
     packages and Swift runtime releases.
   - Offer sample apps showcasing chat + embedding hybrid workflows using
     Apple-native UI frameworks across iOS, macOS, and visionOS.
3. **Research collaboration**
   - Provide documented hooks for importing external evaluation sets and sharing
     validation results across research + product teams.

## Checkpoints & Milestones

- **Milestone A (✅):** Quantization sweep mode with CI-ready reports (delivered
  Sprint 2).
- **Milestone B:** Externalised validation suites with JSON artefacts (target:
  Sprint 3).
- **Milestone C:** Multi-variant export support + signed artifact publication
  (target: Sprint 6+).

Progress should be reviewed at the end of each sprint, verifying that new work
continues to reinforce the core mission: high-quality, quantized Core ML exports
of Dolphin3.0 with first-class chat and LLM2Vec functionality.
