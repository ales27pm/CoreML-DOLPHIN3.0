# CoreML-DOLPHIN3.0 Contributor Guide

This repository now focuses solely on the Dolphin3.0 Core ML export pipeline and
its companion Swift runtime. Follow these rules for any contribution:

## Python Conversion Pipeline
- Keep `dolphin2coreml_full.py` production ready. Changes must preserve LoRA
  merging (via Unsloth), LLM2Vec attachment, and 4-bit quantization support.
- Prefer standard-library modules; if third-party packages are required, update
  `requirements-dev.txt` and document the usage in the README.
- Provide descriptive logging and actionable error messages. Avoid placeholders
  or partially implemented functions.

## Swift Runtime
- Update the Swift sources under `Sources/App` with care. Maintain compatibility
  with multifunction `.mlpackage` exports that expose `init`, `decode`, and
  `encode_for_llm2vec` entry points.
- Keep documentation comments current so application developers understand how to
  integrate the runtime wrappers.

## Documentation & Testing
- The README is the single source of project documentation—ensure it stays in
  sync with the codebase when behaviour changes.
- Run `python -m compileall dolphin2coreml_full.py Sources/App/LLM Sources/App/Bench`
  (or richer validation when applicable) before committing changes.
