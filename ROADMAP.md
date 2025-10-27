# CoreML Dolphin 3.0 Roadmap

This roadmap aligns ongoing development with the repository's single objective:
ship a production-ready Core ML export of **Dolphin3.0-Llama3.1-8B** that merges
LoRA adapters via Unsloth, applies 4-bit quantization, and delivers multifunction
chat + LLM2Vec capabilities through a Swift runtime wrapper.

## Guiding Principles
- **Single-command conversion** – Maintain `dolphin2coreml_full.py` as the only
  entry point for end-to-end export, dependency bootstrap, and artifact cleanup.
- **Feature completeness** – Every release must validate LoRA merging, 4-bit
  palettization, multifunction Core ML outputs (init/decode/encode), and
  embedding parity with the upstream LLM2Vec checkpoint.
- **Apple platform readiness** – Ensure exported packages, Swift runtime code,
  and documentation remain compatible with current iOS, macOS, and visionOS
  toolchains.

## Current State (Q1 FY25)
- Conversion pipeline successfully merges LoRA weights with Unsloth and exports
  init/decode/encode graphs to Core ML.
- Quantization defaults to 4-bit palettization with grouped channels (size 16)
  and optional profile validation.
- Swift runtime (`Sources/App/LLM`) exposes streaming decode, embedding
  generation, and benchmarking utilities.
- Documentation reduced to README + this roadmap; contributor flow relies on
  `python -m compileall` for pre-commit checks.

## Near-Term Objectives (0-2 sprints)
1. **Deterministic validation suite**
   - Add golden transcripts comparing Core ML decode outputs against PyTorch for
     representative prompts.
   - Extend `--profile-validate` to surface latency percentiles and KV-cache
     residency metrics.
2. **LLM2Vec feature parity**
   - Confirm embedding cosine similarity against upstream checkpoints exceeds
     0.99 on benchmark sentences.
   - Ship sample Swift code demonstrating batched embedding extraction.
3. **Pipeline resilience**
   - Harden temporary directory management and error handling around dependency
     installation, ensuring retries and actionable logging.

## Mid-Term Initiatives (2-4 sprints)
1. **Quantization flexibility**
   - Parameterize group size and bit width via CLI with guardrails for supported
     Core ML backends.
   - Investigate mixed-precision schemes (e.g., 6-bit attention, 4-bit MLP)
     while preserving decode latency.
2. **Swift runtime distribution**
   - Package the runtime as a Swift Package Manager artifact with CI validation
     on iOS, macOS, and visionOS targets.
   - Document integration patterns for background execution and streaming UI.
3. **Telemetry & benchmarking**
   - Automate benchmarking runs on Apple Silicon CI to produce historical
     latency/regression charts.

## Long-Term Goals (4+ sprints)
1. **Model variants**
   - Generalize the pipeline to additional Dolphin sizes (3B, 70B) while
     preserving the core feature set.
   - Provide guidance for adapter merging when multiple task-specific Loras are
     chained.
2. **Tooling ecosystem integration**
   - Publish the Core ML package and Swift runtime in an internal registry with
     reproducible build metadata and provenance tracking.
   - Offer sample apps showcasing chat + embedding hybrid workflows using
     Apple-native UI frameworks.
3. **Operational excellence**
   - Establish monitoring hooks for production deployments (memory footprint,
     decode throughput) and feed insights back into quantization decisions.

## Checkpoints & Milestones
- **Milestone A:** Deterministic validation + enhanced profile report ready for
  inclusion in CI (target: Sprint 2).
- **Milestone B:** SPM distribution with multi-platform Swift CI (target:
  Sprint 4).
- **Milestone C:** Multi-variant export support with registry publishing
  pipeline (target: Sprint 6+).

Progress should be reviewed at the end of each sprint, verifying that new work
continues to reinforce the core mission: high-quality, quantized Core ML exports
of Dolphin3.0 with first-class chat and LLM2Vec functionality.
