# CoreML Dolphin 3.0 Conversion Toolkit

This repository contains the production pipeline for exporting
**Dolphin3.0-Llama3.1-8B** to a multifunction Core ML `.mlpackage`. The tooling
focuses exclusively on the original goals of the project:

- Merge LoRA adapters into the base model using Unsloth before export.
- Quantize the Core ML weights with configurable palettization bit-width and
  grouped-channel compression tuned per backend.
- Attach an LLM2Vec encoder head so the resulting package serves both chat and
  embedding workloads.
- Ship a Swift runtime wrapper that wires the Core ML model into Apple
  applications.

## Repository Layout

```text
dolphin2coreml_full.py   # End-to-end conversion script
Sources/App/LLM/         # Swift runtime wrapper for chat + embeddings
Sources/App/Bench/       # Optional benchmarking harness for Core ML packages
requirements-dev.txt     # Lightweight development dependencies
pyproject.toml           # Packaging metadata for the conversion script
```

## Conversion Workflow

Run the pipeline to produce a compressed `.mlpackage` that exposes `init`,
`decode`, and `encode` entry points.

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

Key stages:

1. **Dependency bootstrap** – Ensures `torch`, `transformers`, `coremltools`,
   `peft`, `llm2vec`, and related packages are available before conversion.
2. **Unsloth LoRA merge** – Applies the LoRA adapters to the base weights so the
   merged checkpoint exactly mirrors the fine-tuned Dolphin3.0 model.
3. **Tokenizer + config validation** – Checks compatibility and verifies the
   requested sequence length is supported by the target architecture.
4. **Core ML export** – Builds PyTorch wrapper modules for init/decode/encode and
   converts them into a multifunction `mlprogram`.
5. **Configurable quantization** – Executes palettization followed by linear
   quantization. `--wbits` and `--palett-group-size` tune the global compression
   level, with guard rails keeping Neural Engine / GPU builds on supported group
   sizes {8, 16, 32, 64}. Provide `--mixed-precision attention=6,mlp=4` to keep
   attention projections at 6-bit while compressing MLP blocks to 4-bit without
   sacrificing decode latency.
6. **Optional validation** – When `--profile-validate` is set the script now
   compares deterministic golden transcripts between the PyTorch model and the
   exported Core ML package, reporting decode latency percentiles and KV-cache
   residency/eviction metrics before cleaning temporary artifacts. Runs exit
   non-zero when any transcript diverges, and you can tweak the prompts by
   editing the `GOLDEN_PROMPTS` list in `dolphin2coreml_full.py`.

## Swift Runtime Integration

Use the Swift wrapper under `Sources/App/LLM/` to integrate the exported package
inside an iOS or macOS application.

```swift
let dolphin = try DolphinCoreML(
    modelURL: bundledModelURL,
    computeUnits: .all,
    metadata: (vocabSize: 32000, hiddenSize: 4096, numLayers: 32,
               numHeads: 32, headDim: 128, seqLen: 2048)
)

let (promptIds, promptMask) = tokenizer.encodeToMultiArray(prompt, seqLen: dolphin.seqLen)
let initResult = try dolphin.initPass(inputIds: promptIds, attentionMask: promptMask)
let decodeStep = try dolphin.decodeStep(
    nextId: tokenizer.firstNextIdArray(),
    nextMask: tokenizer.oneMaskArray(),
    pastK: initResult.pastK,
    pastV: initResult.pastV
)

let prompts: [String] = ["security audit checklist", "wireless intrusion detection"]
let batchedInputs = prompts.map { tokenizer.encodeToMultiArray($0, seqLen: dolphin.seqLen) }
let embeddings = try dolphin.encodeEmbeddingBatch(batchedInputs)
```

The wrapper streams KV-cache updates, supports Float16 and Float32 logits, and
propagates descriptive `NSError` payloads when Core ML outputs are missing. The
optional benchmarking harness in `Sources/App/Bench/` measures init, per-token,
and embedding latencies with warm-up control for regression tracking.

When the conversion pipeline is invoked with `--profile-validate`, it now
verifies that LLM2Vec embeddings from the exported Core ML package stay within
0.99 cosine similarity of the upstream checkpoint across a set of benchmark
sentences.

### Quantization knobs

- `--wbits` selects the global palettization bit-width (2/4/6/8).
- `--palett-group-size` adjusts grouped-channel LUT size; Neural Engine and GPU
  builds are clamped to {8, 16, 32, 64} to match Core ML hardware support.
- `--mixed-precision attention=6,mlp=4` applies mixed schemes, keeping
  attention projections at a higher precision while compressing MLP blocks more
  aggressively to preserve decode latency on Apple Silicon.

## Development Environment

Install the full helper dependencies for linting, local validation, and parity
with the production export stack:

```bash
python -m pip install -r requirements-dev.txt
```

The development requirements now pull in the same heavyweight packages that the
pipeline uses at runtime—PyTorch, Transformers, Accelerate, PEFT, Unsloth,
LLM2Vec, Core ML Tools, and SentencePiece—so CI mirrors the production
environment without relying on stubs. Use `python -m compileall` to perform a
quick syntax check before committing changes:

```bash
python -m compileall dolphin2coreml_full.py Sources/App/LLM Sources/App/Bench
```

## License

The tooling is provided for legitimate research and application development.
Refer to the accompanying license information for usage guidelines.
