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
requirements-dev.txt     # Heavier dev/test dependencies (Torch, Transformers, Core ML Tools, etc.)
pyproject.toml           # Packaging metadata for the conversion script
```
To benchmark multiple quantization settings in one run, provide a directory as
`--output` and enable the sweep mode:

## Conversion Workflow

Run the pipeline to produce a compressed `.mlpackage` that exposes `init`,
`decode`, and `encode` entry points. The script bootstraps its own
dependencies – if `torch`, `transformers`, `coremltools`, or `rich` are missing
they are installed on-demand with retry/backoff and actionable error messages –
so fresh environments can execute the conversion without manual prep work.

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

To benchmark multiple quantization settings in one run, provide a directory as
`--output` and enable the sweep mode:

```bash
python dolphin2coreml_full.py \
  --model dphn/Dolphin3.0-Llama3.1-8B \
  --lora-checkpoint path/to/lora_dir \
  --llm2vec-checkpoint path/to/llm2vec_dir \
  --seq-len 2048 \
  --output sweeps/ \
  --quant-sweep \
  --sweep-report artifacts/dolphin_sweep.json
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
5. **Configurable quantization & sweeps** – Executes palettization followed by
   linear quantization. `--wbits` and `--palett-group-size` tune the global
   compression level, with guard rails keeping Neural Engine / GPU builds on
   supported group sizes {8, 16, 32, 64} while CPU-only exports accept any
   positive group size. Provide `--mixed-precision attention=6,mlp=4` to keep
   attention projections at 6-bit while compressing MLP blocks to 4-bit without
   sacrificing decode latency. Mixed overrides are summarised in the console so
   you can verify the number of affected operators per component. Enable
   `--quant-sweep` to iterate through multiple bit-width/group-size pairs,
   collect decode/residency metrics for each variant, and surface a summary
   table plus an optional JSON report consumable by CI dashboards.
6. **Optional validation** – When `--profile-validate` is set the script now
   compares deterministic golden transcripts between the PyTorch model and the
   exported Core ML package, reporting decode latency percentiles, KV-cache
   residency/eviction metrics, and LLM2Vec embedding cosine similarity. Runs
   exit non-zero when any transcript diverges or the minimum cosine similarity
   drops below the 0.99 threshold.

### Command-line reference

- `--model`, `--revision`, `--hf-token`, `--cache-dir` – Locate the base checkpoint on Hugging Face (supports private repos via token).
- `--lora-checkpoint`, `--llm2vec-checkpoint` – Provide directories for the mandatory LoRA adapters and LLM2Vec head.
- `--seq-len` – Maximum context window used for export and validation.
- `--output` – Destination `.mlpackage` path or directory.
- `--tmp` – Scratch directory for intermediate PyTorch modules and Core ML artifacts.
- `--wbits`, `--palett-granularity`, `--palett-group-size` – Configure palettization behaviour with backend-aware validation.
- `--mixed-precision` – Apply per-component bit-width overrides such as `attention=6,mlp=4`.
- `--quant-sweep` – Export a baseline model plus additional bit-width/group-size combinations, emitting a Rich summary table and optional JSON report. Requires `--output` to reference a directory.
- `--sweep-wbits`, `--sweep-group-sizes` – Limit the sweep to specific bit-widths or palettization group sizes (e.g., `--sweep-wbits 2,4,6`). Defaults cover all supported values for the requested compute units.
- `--sweep-report` – Path to a JSON artefact containing size/latency metrics for each variant; useful for CI dashboards.
- `--compute-units` – Choose Core ML compute units (`ALL`, `CPU_AND_GPU`, `CPU_ONLY`) for validation runs.
- `--minimum-deployment-target` – Stamp the exported model with the minimum iOS/macOS version you intend to support.
- `--profile-validate` – Enable golden transcript + embedding parity checks with latency reporting.
- `--clean-tmp` – Delete the temporary working directory after a successful run.

Golden prompts for validation live in the `GOLDEN_PROMPTS` tuple within
`dolphin2coreml_full.py`. Adjust the prompts, maximum new tokens, or expected
behaviour there to tailor the suite for your domain. The validation report is
rendered as Rich tables so deviations are obvious even in CI logs.

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
sentences, surfacing the minimum cosine in the validation summary alongside the
per-prompt transcript comparison.

### Swift Package Manager distribution

The runtime ships as a Swift Package so you can consume it directly from Xcode
or as an archive artifact. Point your application `Package.swift` at the root
of this repository when working with the sources:

```swift
// In your application's Package.swift
dependencies: [
    .package(url: "https://github.com/your-org/CoreML-DOLPHIN3.0.git", branch: "main")
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "DolphinCoreMLRuntime", package: "CoreML-DOLPHIN3.0"),
            .product(name: "DolphinBenchmarkHarness", package: "CoreML-DOLPHIN3.0")
        ]
    )
]
```

Every CI run on `main` publishes a `DolphinCoreMLRuntime.swiftpm.artifactbundle`
containing a signed source archive suitable for Swift Package Registry
distribution. Download the bundle from the workflow run and upload the embedded
`DolphinCoreMLRuntime.swiftpm.zip` to your registry of choice. Consumers can
then depend on the package through their registry instead of targeting this Git
repository directly.

To reproduce the SwiftPM source bundle locally, run the archiver script on
macOS:

```bash
swift package archive-source \
  --output ./artifacts/DolphinCoreMLRuntime.swiftpm.artifactbundle \
  --allow-writing-to-directory ./artifacts
```

The generated bundle is compatible with Xcode 16 and later across iOS 18,
macOS 15, and visionOS 2 targets.

### Background execution patterns

For background token generation (for example, summarising a capture while the
app is suspended) wire the runtime through `BGProcessingTaskRequest` and
`BackgroundTasks`:

1. Register the task in `application(_:didFinishLaunchingWithOptions:)`.
2. Schedule the task after each successful run.
3. Instantiate `DolphinCoreML` inside the task handler, execute the workload,
   and call the completion handler with `setTaskCompleted(success:)`.

```swift
import BackgroundTasks

BGTaskScheduler.shared.register(forTaskWithIdentifier: "ai.dolphin.runtime.background") { task in
    Task { @MainActor in
        guard let processingTask = task as? BGProcessingTask else { return }
        do {
            let runtime = try DolphinCoreML(
                modelURL: ModelLocator.shared.url,
                metadata: ModelLocator.shared.metadata
            )
            try await RuntimePipelines.shared.summariseCaptures(using: runtime)
            processingTask.setTaskCompleted(success: true)
        } catch {
            Logger.background.error("Background run failed: \(error.localizedDescription)")
            processingTask.setTaskCompleted(success: false)
        }
    }
}

func scheduleBackgroundDecode() {
    let request = BGProcessingTaskRequest(identifier: "ai.dolphin.runtime.background")
    request.requiresNetworkConnectivity = false
    request.requiresExternalPower = false
    try? BGTaskScheduler.shared.submit(request)
}
```

On macOS you can achieve the same behaviour with `NSBackgroundActivityScheduler`
for command-line tooling. When running on iOS ensure the background task owns a
`BGProcessingTaskRequest` entitlement and keep the Core ML model in a shared
container (e.g. via App Group) so the task can access it even when launched in
the background.

### Streaming UI integration

Combine `AsyncStream` with `Observable` view models to surface incremental
tokens or embeddings in SwiftUI. The pattern keeps decoding off the main actor
and yields updates as soon as `decodeStep` returns:

```swift
import os.log

@MainActor
final class StreamedChatViewModel: ObservableObject {
    @Published private(set) var transcript: [String] = []
    private let runtime: DolphinCoreML
    private let tokenizer: Tokenizer
    private let logger = Logger(subsystem: "ai.dolphin.runtime", category: "StreamedChat")

    init(runtime: DolphinCoreML, tokenizer: Tokenizer) {
        self.runtime = runtime
        self.tokenizer = tokenizer
    }

    func streamResponse(for prompt: String) {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }

            do {
                let (ids, mask) = tokenizer.encodeToMultiArray(prompt, seqLen: runtime.seqLen)
                let initOut = try runtime.initPass(inputIds: ids, attentionMask: mask)
                let nextSeedId = try tokenizer.firstNextIdArray()
                let nextSeedMask = try tokenizer.oneMaskArray()

                let stream = AsyncStream<String> { continuation in
                    Task.detached {
                        var cacheK = initOut.pastK
                        var cacheV = initOut.pastV
                        var localNextId = nextSeedId
                        var localNextMask = nextSeedMask

                        while !Task.isCancelled {
                            do {
                                let step = try runtime.decodeStep(
                                    nextId: localNextId,
                                    nextMask: localNextMask,
                                    pastK: cacheK,
                                    pastV: cacheV
                                )
                                cacheK = step.outK
                                cacheV = step.outV
                                let token = runtime.greedySample(from: step.logits)
                                continuation.yield(tokenizer.decode(token: token))
                                (localNextId, localNextMask) = tokenizer.nextMaskArrays(tokenId: token)
                            } catch {
                                continuation.finish()
                                return
                            }
                        }
                        continuation.finish()
                    }
                }

                for await piece in stream {
                    await MainActor.run {
                        self.transcript.append(piece)
                    }
                }
            } catch {
                await MainActor.run {
                    logger.error("Streaming decode failed: \(error.localizedDescription)")
                }
            }
        }
    }
}
```

The `StreamedChatViewModel` publishes each decoded token to the UI, enabling
SwiftUI views to animate in real time. When combined with `@ScenePhase` you can
pause the `AsyncStream` when the app moves to the background and resume it when
the scene becomes active again, ensuring resources are released while the user
is away.

### Quantization knobs

- `--wbits` selects the global palettization bit-width (2/4/6/8).
- `--palett-group-size` adjusts grouped-channel LUT size; Neural Engine and GPU
  builds are clamped to {8, 16, 32, 64} to match Core ML hardware support while
  `CPU_ONLY` accepts any positive value.
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
