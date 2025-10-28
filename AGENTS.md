# CoreML-DOLPHIN3.0 Contributor Guide

This repository now focuses solely on the Dolphin3.0 Core ML export pipeline and
its companion Swift runtime. Follow these rules for any contribution:

## Python Conversion Pipeline

- Keep `dolphin2coreml_full.py` production ready. Changes must preserve LoRA
  merging (via Unsloth), LLM2Vec attachment, mixed-precision aware quantization,
  and the deterministic validation suite triggered by `--profile-validate`.
- Prefer standard-library modules; if third-party packages are required, update
  `requirements-dev.txt` and document the usage in the README.
- Provide descriptive logging and actionable error messages. Avoid placeholders
  or partially implemented functions.
- When touching the golden prompts or validation logic, update the README's
  command-reference section so operators understand the latest CLI surface.

## Swift Runtime

- Update the Swift sources under `Sources/App` with care. Maintain compatibility
  with multifunction `.mlpackage` exports that expose `init`, `decode`, and
  `encode_for_llm2vec` entry points.
- Keep documentation comments current so application developers understand how to
  integrate the runtime wrappers.

## Documentation & Testing

- The README is the single source of project documentation—ensure it stays in
  sync with the codebase (CLI options, validation reports, Swift integration)
  when behaviour changes.
- Run `python -m compileall dolphin2coreml_full.py Sources/App/LLM Sources/App/Bench`
  and keep `ROADMAP.md` aligned with implemented features before committing
  changes.
