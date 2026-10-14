# Dolphin 3.0 Llama 3.2 3B — LUT6 stateful Core ML export

**GitHub edition:** Start with the [version index and model download](../README.md).
The converter, tests and Swift sources are maintained here. Converted
packages and the large `source/tokenizer.json` are fetched separately; small
metadata and evidence are kept in GitHub. Once `models/` has been downloaded,
every “full-release root” below means the enclosing `ane-lut6-c2048/` directory,
which contains both `models/` and `toolkit/`. Keep the accompanying
[Notice](../Notice.txt), [license](licenses/LLAMA_3_2_LICENSE.txt), and
[use policy](licenses/LLAMA_3_2_USE_POLICY.md) with redistributed copies.

This toolkit converts the actual public Dolphin checkpoint into an iOS 18 / macOS 15 ML Program pipeline with persistent KV states and an architecture intended for the Apple Neural Engine. It includes the generated packages, a reproducible converter, numerical references, integrity evidence, and a Swift runtime.

**Apple SDK compilation, Core ML device inference, and ANE placement must be tested on Apple hardware. An export or a `.cpuAndNeuralEngine` configuration alone does not prove ANE execution.** The delivered manifest records those hardware checks as unverified. No device speed, power, or peak-memory result is claimed.

## Model contract

| Setting | Value |
| --- | --- |
| Source | `dphn/Dolphin3.0-Llama3.2-3B` |
| Source revision | `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364` |
| Minimum OS | iOS 18 / macOS 15 |
| Format | ML Program, `.mlpackage` |
| Pipeline | embedding, seven four-layer transformer blocks, final norm and vocabulary head |
| Context | 2,048 consumed tokens, fixed at export time |
| Query length | 1 token per invocation |
| Prefill | Token by token; head evaluated only for the final prompt token |
| Projections | 1×1 convolutions in BC1S layout |
| Attention | Separate head matmuls, grouped-query KV heads, explicit RoPE and causal masking |
| Weight representation | Scalar LUT6, 64 palette entries per group of 16 output channels |
| Activations and KV state | FP16 |
| Vocabulary | All 128,258 IDs |
| KV allocation | 224 MiB total at context 2,048, excluding weights and runtime buffers |
| Runtime compute units | `.cpuAndNeuralEngine`; CPU fallback is still possible |

Each block owns `keyCache` and `valueCache` with shape `[4,1024,1,2048]`. That is 32 MiB per block. The states are actual Core ML state features, with state read/write operations in the serialized program; they are not ordinary cache tensors copied through the public output interface.

The source model advertises a longer context. This artifact deliberately fixes a 2,048-token cache. It does not roll or evict old tokens. The supplied Swift runtime creates fresh block states for each `generate` call and replays the supplied prompt; the host application retains the conversation history.

## Important checkpoint details preserved

The source vocabulary has two extra ChatML tokens: ID 128256 is `<|im_end|>`, and ID 128257 is `<|im_start|>`. The eight true logit widths are `16032,16032,16032,16032,16032,16032,16032,16034`. The final projection has padded weight rows for group-16 palettization, but the graph slices off those padded logits. Both final vocabulary IDs remain available to sampling.

The embedding and vocabulary projection use the checkpoint's tied embedding weights. Because they are separate Core ML packages, their compressed copies are stored separately. The final RMS normalization occurs once, inside the head.

RoPE uses the source checkpoint's Llama 3 frequency scaling equations, matching Transformers 4.47.1. The exported cosine and sine tables contain little-endian FP16 values with shape `[2048,128]`. A simple multiplication of the rotary base by the scaling factor would produce different frequencies and is not used.

The original tokenizer, special tokens, generation configuration and tool-aware ChatML template are supplied unchanged. Use that template when producing token IDs.

## Layout of the full release

The full archive extracts to:

```text
Dolphin3_ANE_Stateful_LUT6_C2048/
  README.md
  CONVERSION_REPORT.md
  models/
    embedding.mlpackage/
    block_00.mlpackage/ ... block_06.mlpackage/
    head.mlpackage/
    manifest.json
    rope_cos.fp16
    rope_sin.fp16
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    evidencefixtures.json
    evidence/
  toolkit/
    download_source.py
    run_export.py
    verify_export.py
    tokenize_prompt.py
    dolphin_ane/
    tests/
    swift/
    requirements.txt
    source-verification.json
    source/  # verified tokenizer/configuration files; no source weight shards
    licenses/
```

The smaller toolkit archive contains the same conversion/runtime code and the verified metadata needed to reproduce the conversion. It does not contain the original 6.43 GB weight shards or the converted model packages.

## Validate on a Mac

Use a Mac running macOS 15 or later with Xcode 16 or later. The Swift package targets Swift 6 tooling and Swift 5 language mode.

From the root of the extracted full release:

```sh
cd toolkit/swift
swift build -c release

.build/release/dolphin-ane-validate \
  --models ../../models \
  --fixtures ../../models/evidencefixtures.json \
  --strict-ane \
  --output device-validation.json
```

The loader uses an existing `.mlmodelc` when available. Otherwise, it asks Core ML to compile the `.mlpackage`, then loads it with `.cpuAndNeuralEngine`.

The validation report separates:

1. Model compilation/loading and successful stateful predictions.
2. Sparse numerical parity against the exported LUT6 reference.
3. ANE preference in the device's Core ML compute plan.
4. Actual runtime tracing, which this CLI does not collect and leaves unverified.

The fixture's maximum absolute logit error of 1.0 is an explicit provisional acceptance threshold. It has not been calibrated against Apple-device measurements. The comparison also checks the expected greedy next token. Inspect the recorded errors and device plan; do not convert a threshold pass into a broad quality or performance claim.

Strict plan inspection records every operation. It excludes the embedding, `const`, `constexpr_*`, `identity`, `read_state`, `write_state` and `coreml_update_state` from its ANE requirement. Mathematical operations and cache `slice_update` remain subject to the strict requirement. A plan-level pass reports anticipated preferred devices; use Instruments/Core ML profiling to confirm runtime behavior on the target iPhone.

## Integrate into an iOS application

Add the local Swift package under `toolkit/swift` to the app and link its `DolphinANE` library product. Add all nine `.mlpackage` files from `models/` as individual model resources with app-target membership so Xcode processes them. Add `manifest.json`, `rope_cos.fp16` and `rope_sin.fp16` to the target's Copy Bundle Resources phase.

The example below uses the app bundle's resource root. Inspect the built `.app` to confirm that `embedding.mlmodelc`, `block_00.mlmodelc` through `block_06.mlmodelc`, `head.mlmodelc`, `manifest.json` and both RoPE tables are direct children of that same directory. A project group name does not by itself establish a runtime directory. If resources are deliberately stored in a subdirectory, pass that directory instead.

The loader also accepts uncompiled `.mlpackage` siblings and compiles them on the destination device, which adds first-load work. Copying a raw `models/` folder is therefore different from having Xcode compile individual model resources. Include the tokenizer files separately if the host app's tokenizer reads them from its bundle.

```swift
import DolphinANE

let directory = Bundle.main.resourceURL!
let engine = try await DolphinANEEngine(directory: directory)
let result = try await engine.generate(
    tokens: promptTokenIDs,
    maxNewTokens: 32
)
```

`promptTokenIDs` is `[Int32]` produced with the supplied tokenizer and ChatML template. The Swift package accepts token IDs and returns token IDs; it does not implement the full Hugging Face BPE tokenizer. Use a compatible tokenizer in the host app.

For test prompts, use the bundled Python utility. Start this next block from the **root of the extracted full release**, not from `toolkit/swift`. It requires Python 3.12. This separate environment installs tokenizer and template dependencies without the conversion toolchain:

```sh
python3.12 -m venv toolkit/.tokenizer-venv
toolkit/.tokenizer-venv/bin/python -m pip install \
  'transformers==4.47.1' 'numpy==1.26.4' 'jinja2==3.1.6'

toolkit/.tokenizer-venv/bin/python toolkit/tokenize_prompt.py \
  --model-dir models \
  --prompt 'What is 2 + 2? Answer with the number only.'
```

The utility reads tokenizer files locally and prints comma-separated IDs. `--json` prints a JSON array; that form is for inspection and is not the CLI's `--tokens` format. A Transformers message about unavailable model backends is expected in a tokenizer-only environment. Jinja2 supplies the ChatML template renderer.

After building the Swift executable as above, run a generated prompt from the full-release root:

```sh
DOLPHIN_PROMPT_IDS="$(toolkit/.tokenizer-venv/bin/python toolkit/tokenize_prompt.py \
  --model-dir models --prompt 'What is 2 + 2? Answer with the number only.')"

toolkit/swift/.build/release/dolphin-ane-validate \
  --models models \
  --tokens "$DOLPHIN_PROMPT_IDS" \
  --max-new-tokens 8 \
  --output generated-example.json
```

The result contains generated token IDs, not decoded text. The runtime owns one coherent set of seven `MLState` objects, acquires exclusive generation ownership before any asynchronous call, preserves all caches through prefill/decode, and discards them together on completion, error or cancellation.

The last sampled token is not needlessly fed back into the cache. For a prompt of length P and a maximum of N generated tokens, the maximum number of consumed positions is `P + N - 1`. The runtime rejects an invalid context budget and stops on the checkpoint's EOS IDs.

## Reproduce the export

Run the conversion in a dedicated Python 3.12 environment. The original run used Python 3.12, PyTorch 2.5.0 CPU and Core ML Tools 9.0 on Linux. Linux can write these packages; Apple's runtime is still required to compile and execute them. The following block starts at the full-release root. In the smaller toolkit archive, start in the directory containing `run_export.py` and omit `cd toolkit`.

```sh
cd toolkit
python3.12 -m venv .venv
if [ "$(uname -s)" = "Linux" ]; then
  .venv/bin/python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python download_source.py --source source

.venv/bin/python run_export.py \
  --source source \
  --output output \
  --bits 6 \
  --palette-workers 4 \
  --context 2048 \
  --layers-per-block 4 \
  --threads 2

.venv/bin/python -m dolphin_ane.reference \
  --source source \
  --output validation/reference \
  --bits 6 \
  --palette-workers 4 \
  --context 2048 \
  --threads 4

.venv/bin/python verify_export.py \
  --models output \
  --reference validation/reference \
  --source source \
  --output output/evidence/verification.json
```

The conditional install selects CPU-only PyTorch wheels on Linux before resolving the remaining requirements. Apple Silicon macOS uses its normal PyTorch wheel.

The downloader pins the commit and verifies both model shards using the authoritative Hugging Face SHA-256 values. It checks metadata/tokenizer Git blob or LFS identities, preserves already matching files, and resumes an incomplete shard only after validating HTTP 206 and its exact `Content-Range`. Large in-progress downloads and builds stay outside the synchronized workspace; only complete files are moved into their destinations. Download partials default to `~/.cache/dolphin3-ane-download-392a6f`. Before networking, staging must be separate from the workspace and on the same filesystem as `source/` for atomic installation. Use `--partials` to select another disk-backed cache; a separate `/tmp` tmpfs is rejected before downloading.

The entire Git checkout (including worktrees) is protected, not only the toolkit
subdirectory. For a standalone toolkit the toolkit root is protected; pass
`--workspace /path/to/synchronized-root` if synchronization covers a larger tree.
If the home cache and `source/` are on different volumes (for example a mounted
CI workspace), **`--partials` is required**: choose a writable disk-backed
directory outside that protected tree on the source volume. The default does
not silently fall back to a non-atomic copy or to workspace staging. For example:

```sh
.venv/bin/python download_source.py --source /mnt/models/checkout/toolkit/source \
  --workspace /mnt/models/checkout --partials /mnt/models/dolphin-download-cache
```

The converter creates each part with meta parameters, reads only its source tensors, and runs conversion and compression in separate subprocesses. Completed packages have integrity reports and can be reused after their hashes and configuration are checked. It refuses to silently overwrite incomplete or incompatible outputs. Manifest/reference schema 3 binds the pinned shard and metadata hashes. A schema 2 reference may be migrated only by a fresh source re-hash plus verification of every loaded FP16 parameter; schema 2 manifests and stage reports require a fresh export into a new output directory. Release packaging accepts only independently reverified schema 3 evidence.

Export and reference runs hold an exclusive process lock for their output
directory. Exports also lock an explicit `--build-dir`, which must differ from
`--output`, including when paths are aliases. A second writer fails before
processing; the persistent sibling
`.<output-dir-name>.dolphin-writer.lock` file (for example,
`.output.dolphin-writer.lock`) is deliberately retained to avoid lock-inode
races and is not part of that output's artifact inventory.

The exporter, numerical reference and independent verifier re-hash the pinned source inputs before use; a former `all_verified: true` flag is not sufficient. Existing published schema 2 models and evidence remain historical and are not upgraded by editing JSON. Download the full tokenizer and source shards, then regenerate the export and its evidence before building a new schema 3 release.

After a successful independent verification, retain its printed
`verification_report_sha256` in a trusted job log or other protected location
outside the mutable export directory. Set `VERIFICATION_SHA256` to that exact
64-character digest, then package into a new or empty destination. **Do not
derive this pin by hashing the mutable report at packaging time**: changing
both an archive input and its inventory must not create a new trust anchor.

```sh
.venv/bin/python build_release.py \
  --models output \
  --reference validation/reference \
  --destination release-schema3 \
  --verification-sha256 "$VERIFICATION_SHA256"
```

Before reporting `release_ready`, the packager re-hashes the bytes actually stored in the ZIP against the verified model packages, sidecars, reference files, manifest, Swift fixtures and source-verification report. It also checks the copied source metadata in both archives, rejects duplicate members and changed package inventories, and emits `release-files.json` plus `SHA256SUMS.txt`. A valid ZIP CRC alone is not accepted as provenance. These checks bind the delivered files to the local verification result; they are not a publisher signature or evidence of Apple-device execution.

Every member, including toolkit sources and supplemental logs, is covered by
the complete archive inventory. Model/reference bytes are additionally bound
to the independently pinned verification report; inventorying toolkit code or
logs does not claim numerical parity for those files. Archive validation and
digest calculation use the same open file, with mutation checks before the
checksum files and final success result. Older reports without the complete
evidence inventory must be regenerated by the independent verifier.

The complete original source shards occupy about 6.43 GB. Allow additional space for the Python environment, intermediate FP16 packages, compressed output and archives. The converter's measured peak process memory is recorded per part in the evidence. The conversion environment has an 8 GiB memory limit; this is not an iPhone runtime memory claim.


### Cross-repository import contract check

From `ane-lut6-c2048/toolkit`, with a sibling Swarmer checkout and Swift 6:

```sh
.venv/bin/python validate_import_contract.py --swarmer /path/to/swarmer \
  --swiftc /path/to/swiftc --output /path/to/new-contract-evidence
```

On macOS, select Xcode and set `SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"`
when invoking a toolchain's `swiftc` directly, so it can find the host standard library.

This invokes the production manifest builder, serializes a synthetic bundle,
ZIPs and extracts it, then compiles and executes Swarmer's actual
`DolphinCoreMLBundle` reader. It checks v3 acceptance, legacy v2 compatibility
and rejection of unknown schemas or changed source declarations. `results.json`
and `compile.log` retain the evidence. The fixture has no executable model
weights and is explicitly **not** a complete export, an iOS `LocalModelStore`
transaction, a source-byte rehash, or Core ML inference. Swarmer accepts the v3
source declaration only when its full pinned identity matches; that declaration
alone does not authenticate converted package bytes. Full qualification still
requires the real schema 3 export, reference, verifier, release packager and
Apple/device runs described above.

## Compression and numerical evidence

The initial four-bit palette strategy was rejected after numerical checks showed unacceptable differences from the uncompressed FP16 export architecture. The default was changed to six bits to favor numerical fidelity. The rejection and the selected candidate's measured results belong in `CONVERSION_REPORT.md`; successful serialization or matching compressed-weight hashes alone do not establish a usable model.

LUT6 uses 64 palette entries for each group of 16 output channels. The fitting routine uses the weighted scalar `kmeans1d` implementation bundled with Core ML Tools 9, operating on unique FP16 source values and their occurrence counts, then stores the palette in FP16. It does not use the rejected local Lloyd recipe or its uniform-palette fallback. This fits weight values without activation calibration; projection checks and a full-model numerical gate must assess its effect on model outputs.

Independent groups can be fitted concurrently with `--palette-workers 4`, the default. Work is submitted in bounded batches to control memory. Use `--palette-workers 1` for sequential execution. Threaded and sequential fitting use the same solver and are checked for identical palette values, indices and reconstructed weight hashes. This option affects conversion/reference preparation, not the iOS runtime.

All 205 large embedding/projection tensors must become `constexpr_lut_to_dense` operations with genuine `uint6` indices. The export fails if the complete expected set is absent. Norm weights remain FP16. Use `--bits 6` for both conversion and reference generation. The verifier infers the precision from the model manifest and cross-checks the reference and serialized weights; it needs no bit-width flag. Use fresh output/reference directories when changing the bit width.

The verifier reconstructs the actual exported LUTs and indices, hashes their dense FP16 values, and compares every compressed tensor with the weights used by the PyTorch reference. It also checks file sizes and SHA-256, actual serialized state shapes, public interfaces, and the total cache allocation. The reference records complete next-token logits on three short templated prompts for the uncompressed FP16 export architecture and for its LUT6 reconstruction. The numerical gate requires greedy next-token agreement with the FP16 reference on each of those three fixed prompts; full-logit comparison metrics are recorded separately. This gate is distinct from package-integrity and reconstructed-weight checks.

The Python test suite compares the architecture with an independent Transformers Llama implementation, including rotary scaling, state updates, chunk boundaries, reset/isolation, future masking, final normalization and both final vocabulary IDs. Small real Core ML conversions additionally test the state features, model part types, selected palette width and reconstruction hashes.

These checks establish the conversion-level and CPU numerical evidence described in `CONVERSION_REPORT.md`. Three next-token probes do not establish general answer quality, perplexity, long-context quality, or device speed. The Swift package remains source-reviewed until it is actually built with the Apple SDK.

## Sources and notices

- [Original Dolphin checkpoint](https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B/tree/392a6f57223e7ccfe6ef4ebdb2ff101a42d57364)
- [Apple: stateful Core ML models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Apple: palettization performance and grouped channels](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
- [Transformers 4.47.1 Llama implementation](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llama/modeling_llama.py)
- [Transformers 4.47.1 RoPE implementation](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/modeling_rope_utils.py)
- [ANEMLL reference](https://github.com/Anemll/Anemll/tree/f4ad26d061dc2e426faa19f81c367343bdeb9f0d). Its use of the `concat(x,-x)` LayerNorm identity informed the RMSNorm construction; attribution is retained in the source.

The supplied model weights retain the source model's Llama 3.2 Community License. The original model card and copies of the Llama 3.2 license and use policy are included. Built with Llama.
