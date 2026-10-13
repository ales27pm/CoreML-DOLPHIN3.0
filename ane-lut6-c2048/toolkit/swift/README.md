# Dolphin ANE LUT6 stateful runtime

For this GitHub edition, first follow the [version download guide](../../README.md).
Run its download from the repository root, then use `ane-lut6-c2048/` as the
full-release root below. The downloaded `models/` and this `toolkit/` are siblings.

Target: iOS 18+ / macOS 15+, Xcode 16+ (Swift tools 6, Swift 5 language mode).
The source was reviewed against Apple documentation but **not compiled with an
Apple SDK or executed on Apple hardware in the export environment**. Successful
export alone does not establish ANE placement, numerical parity, memory fit or
iPhone performance.

## Build and run on an Apple silicon Mac

Start at the root of the extracted `Dolphin3_ANE_Stateful_LUT6_C2048/` release,
which contains `models/` and `toolkit/`:

```sh
cd toolkit/swift
swift build -c release
.build/release/dolphin-ane-validate \
  --models ../../models \
  --fixtures ../../models/evidencefixtures.json \
  --strict-ane \
  --output device-parity-and-plan.json
```

If you are already in this `swift/` directory, omit `cd toolkit/swift`. If using
the smaller toolkit archive, first run the conversion and reference/verifier
commands from the main README; from `swift/`, their model output is `../output`
and their generated fixtures are `../output/evidencefixtures.json`. The toolkit
archive itself contains no converted model packages.

The selected export uses six-bit palette indices and FP16 activations/KV states.
The Python conversion and reference commands use `--bits 6`; the verifier reads
the precision from the manifest. This Swift CLI needs no bit-width flag because
Core ML interprets the compressed model.
The initial four-bit strategy was rejected by numerical checks, and the six-bit
candidate must satisfy the recorded numerical acceptance gate. Its three short
reference prompts do not establish general model quality or iPhone performance.

The model directory must contain `manifest.json`, the two RoPE tables and all
nine model packages. The runtime prefers a sibling `.mlmodelc` and
otherwise compiles the `.mlpackage` asynchronously on the destination device.
Keep only compiled siblings corresponding to the exact package revision being
tested; this loader does not establish their provenance. `loadSeconds` includes
any compilation, model loading and compute-plan analysis.

`--tokens` accepts already-tokenized IDs, including the exact Dolphin chat
template and special tokens. The library intentionally does not guess a chat
template or download a tokenizer. Use the tokenizer shipped with the export.
The main release README provides a Python 3.12 tokenizer-only environment and
`tokenize_prompt.py` commands, including the required Jinja2 template dependency.
For a manual generation run, replace `--fixtures ...` with `--tokens ID,ID,...`
and optionally `--max-new-tokens 8`. Do not pass a JSON array to `--tokens`.

The embedding output is `hiddenStates`. Each transformer block accepts
`hiddenStates` and returns `outputHiddenStates`; its input and output use distinct
names required by the exported graph. The head accepts `hiddenStates` and returns
`logits_0` through `logits_7`, with channel sizes from the manifest.

## Integrate in an iOS app

Add this directory as a local Swift package and link the `DolphinANE` library.
Add each of the nine `.mlpackage` files individually to the app target for Xcode
compilation, and copy the manifest and two RoPE files as bundle resources. The
directory supplied to the engine must contain all nine compiled `.mlmodelc`
directories and those three files directly beneath it. Verify that layout in
the built app; a project group alone does not define the runtime directory.
Alternatively, a directory of original `.mlpackage` files can be loaded and
compiled on the destination device. That adds compilation work to initial load.
The tokenizer files are needed separately by the host app's tokenizer.

The command-line target is a macOS driver. Call the library from an iOS app to
obtain results on the actual iPhone; passing a Mac run does not validate iPhone
execution.

```swift
import DolphinANE

let engine = try await DolphinANEEngine(directory: modelDirectory)
let result = try await engine.generate(tokens: promptIDs, maxNewTokens: 32)
// Decode result.generatedIDs with the matching tokenizer.
let plans = await engine.computePlans
// To stop an in-flight generation: await engine.cancel(), or cancel its Task.
```

The engine reserves the complete split pipeline before its first suspension.
Each generation creates one `MLState` per block and shares those states only
across that generation's token prefill and decode. Cancellation is checked after
each awaited prediction returns; all states are then discarded together.
There is no public API to mutate or reset an in-flight state.

The final generated token (including EOS) is not fed back into the cache. Thus
`consumedTokens` can differ from prompt length plus generated token count. Every
new generation replays its supplied prompt into fresh states. Retaining chat
history is the caller's responsibility.

## Evidence and exit status

The report separates these properties:

| Field | Meaning |
|---|---|
| `compiledModelsLoaded` | All compiled models loaded on this device. |
| `runtimeValidated` | The requested prediction sequence completed and every inspected logit was finite. |
| `numericalParityValidated` | All provided sparse-logit fixture tolerances and expected next-token IDs passed. False when no fixtures were supplied. |
| `strictANEComputePlanPassed` | All nonconstant, non-bookkeeping block/head operations have ANE as their *anticipated preferred* device. Unknown or missing plans fail this gate. Embedding is recorded but excluded from this gate. |
| `aneRuntimeTraceValidated` | Always false in this harness: a compute plan is not an observed runtime trace. Use Core ML Instruments to collect that additional evidence. |

The strict plan gate excludes `const`, `constexpr_*`, `identity`, `read_state`,
`write_state` and `coreml_update_state` as bookkeeping; their placement is still
recorded. `slice_update` and mathematical operations remain subject to the gate.
These exemptions do not remove the CPU fallback check for transformer math.

Exit 0 means the requested checks passed; exit 1 means an input, model, inference
or report error; exit 2 means `--strict-ane` failed; exit 3 means fixture parity
failed. If both strict placement and parity fail, exit 2 takes precedence and the
report retains both failed results. Without `--strict-ane`, exit 0 does not assert
ANE placement. `.cpuAndNeuralEngine` permits CPU fallback and excludes GPU.

Fixture schema (the exporter must supply real reference values, not this example):

```json
{"reference_backend":"LUT6 reconstruction in PyTorch","cases":[
  {"name":"short-prompt","input_ids":[128000,9906],"expected_next_token":123,
   "reference_logits":[{"id":123,"logit":11.2}],"max_abs_logit_error":1.0}
]}
```

Sparse logits and a next-token match are a limited parity check. They do not
establish task quality or whole-vocabulary parity. Run a separately chosen prompt
suite, long-context cases, repeated generations, cancellation, memory-pressure
and thermal tests before app release. The single-token prefill path evaluates
the head only at the last prompt token. Timings include Swift mask preparation,
Core ML calls, state transitions and output scanning; they are not isolated ANE
kernel timings. This harness does not measure peak memory or thermal stability.

## Primary API references checked

- [MLModel.compileModel(at:completionHandler:)](https://developer.apple.com/documentation/coreml/mlmodel/compilemodel(at:completionhandler:))
- [MLModel.load(contentsOf:configuration:)](https://developer.apple.com/documentation/coreml/mlmodel/load(contentsof:configuration:))
- [MLModel.makeState()](https://developer.apple.com/documentation/coreml/mlmodel/makestate())
- [Asynchronous stateful prediction](https://developer.apple.com/documentation/coreml/mlmodel/prediction(from:using:options:)-8b4qa)
- [MLState serialization contract](https://developer.apple.com/documentation/coreml/mlstate)
- [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)
- [MLComputePlan.DeviceUsage preferred/supported](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n/deviceusage)
- [Program.Operation and nested blocks](https://developer.apple.com/documentation/coreml/mlmodelstructure-swift.enum/program/operation)

The documentation's `MLState` overview contains an older `newState()` Swift
snippet. The actual Swift declaration used here is `makeState()`.
