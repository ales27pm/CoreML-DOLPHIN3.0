# DolphinCoreML Integration Guidance

This scope covers the `Sources/App/LLM/` directory, including `DolphinCoreML.swift`.

- Preserve compatibility with multifunction `.mlpackage` artifacts that expose `init`, `decode`,
  and `encode` entry points. Keep function names (`input_ids`, `attention_mask`, `past_k_*`,
  `out_v_*`, etc.) synchronized with the exporter.
- Keep `ComputeUnitSelection` and metadata-driven initialisers exhaustive. If new compute-unit
  options or Core ML targets emerge, extend the enums and configuration mapping with full test
  coverage.
- Maintain robust error handling—propagate descriptive `NSError` payloads when Core ML outputs are
  missing, and document all thrown errors.
- Sampling helpers such as `greedySample` must continue to support both Float16 and Float32 logits
  without introducing temporary allocations that would regress performance.
- When adding new functionality (e.g., temperature sampling, batching), include unit or integration
  tests exercising the logic with synthetic `MLMultiArray` inputs and update benchmarking guidance
  accordingly.
