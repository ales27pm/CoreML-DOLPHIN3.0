# Benchmark Harness Guidance

This scope covers the `Sources/App/Bench/` directory, including `BenchmarkHarness.swift`.

- Retain the production-friendly benchmarking workflow: prompt encoding, warmup decode steps,
  timed decode loop, and embedding latency measurement that stays aligned with the exporter’s
  quantization sweep reports.
- `trimCacheToSeqLen` must continue to validate tensor ranks and data types before performing
  pointer arithmetic. Any modifications should add regression tests that exercise Float16 and
  Float32 caches with varying sequence lengths, and keep the output summaries aligned with the
  decode latency metrics emitted by `--profile-validate`.
- Do not remove tokenizer protocol abstractions—extend `YourTokenizerProtocol` with documented
  methods if the runtime contract evolves, and provide migration notes in doc comments.
- Emit structured logs (e.g., formatted strings with timings) that are safe to consume in release
  builds and avoid `print` spam unless reporting final benchmark summaries. Maintain compatibility
  with the logging categories referenced in README background/streaming sections and ensure any new
  metrics can be correlated with JSON sweep artefacts.
- Keep the harness deterministic and make warmup/measurement token counts configurable through
  method parameters when exposing new benchmarking scenarios.
