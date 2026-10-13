# Llama 3.2 Dolphin 3B — stateful Core ML, LUT6, context 2048

This version contains the reproducible converter, a separate Swift runtime, and
the recorded validation evidence for
[`dphn/Dolphin3.0-Llama3.2-3B`](https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B/tree/392a6f57223e7ccfe6ef4ebdb2ff101a42d57364).
The source revision is `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.

**The nine packages were exported and their serialized weights verified. Apple
SDK compilation, Core ML device execution, and ANE placement remain unverified.**
Selecting `.cpuAndNeuralEngine` permits CPU fallback and does not prove ANE
execution. No iPhone speed, energy, or memory-fit result is claimed.

## Release contract

| Setting | Value |
| --- | --- |
| Minimum OS | iOS 18 / macOS 15 |
| Packages | Embedding, seven four-layer blocks, final norm/head: nine `.mlpackage` files |
| Weight compression | LUT6, 64 FP16 palette entries per group of 16 output channels |
| Activations and native KV states | FP16 |
| Context / query length | Fixed 2048 / 1; token-by-token prefill |
| KV state allocation | 224 MiB total; each block owns two `[4,1024,1,2048]` states |
| Vocabulary | All 128,258 IDs, including ChatML IDs 128256 and 128257 |
| Package size | 2,718,527,235 bytes across nine packages |

The exact contract is in [manifest.json](manifest.json). This multipart runtime
uses `DolphinANEEngine` from [toolkit/swift](toolkit/swift/README.md). The existing
8B multifunction pipeline and `Sources/App` APIs have a different model contract;
their documentation remains in the [repository README](../README.md).

## Download the model files

The publication target is
[`ales27pm/Dolphin3.0-CoreML/ane-lut6-c2048`](https://huggingface.co/ales27pm/Dolphin3.0-CoreML/tree/main/ane-lut6-c2048).
These instructions describe that target layout; availability depends on the
publication commit. GitHub contains no converted weights, source weight shards,
or full-release ZIP. The 17 MB `tokenizer.json` is also fetched with the models.

Run this from the **GitHub repository root**. The Hugging Face CLI preserves the
remote prefix, creating `ane-lut6-c2048/models/` beside this version's `toolkit/`:

```sh
python3.12 -m venv ane-lut6-c2048/.download-venv
ane-lut6-c2048/.download-venv/bin/python -m pip install 'huggingface-hub==0.36.0'
ane-lut6-c2048/.download-venv/bin/hf download \
  ales27pm/Dolphin3.0-CoreML \
  --include 'ane-lut6-c2048/models/**' \
  --local-dir .
```

The download contains the nine packages, manifest, RoPE tables, matching
tokenizer/configuration, Swift fixtures and full evidence. Retain this version's
[Notice](Notice.txt), [Llama 3.2 license](toolkit/licenses/LLAMA_3_2_LICENSE.txt),
and [use policy](toolkit/licenses/LLAMA_3_2_USE_POLICY.md) with redistributed copies.

On an Apple silicon Mac with macOS 15+ and Xcode 16+, start again from the GitHub
repository root:

```sh
cd ane-lut6-c2048/toolkit/swift
swift build -c release
.build/release/dolphin-ane-validate \
  --models ../../models \
  --fixtures ../../models/evidencefixtures.json \
  --strict-ane \
  --output device-validation.json
```

This command performs the pending device checks. Its strict compute-plan result
is a preferred-device assessment; actual ANE execution still requires runtime
profiling. The fixture tolerance of 1.0 absolute logit error is provisional and
has not been calibrated on Apple hardware.

For app integration and tokenization, follow the
[detailed toolkit guide](toolkit/README.md#integrate-into-an-ios-application).
After downloading, its “full-release root” is this `ane-lut6-c2048/` directory.
Pass `models/` to `tokenize_prompt.py --model-dir`; the metadata-only
`toolkit/source/` in GitHub does not contain the complete tokenizer.

## Recorded validation

[verification.json](evidence/verification.json) records all 205 compressed
matrices reconstructed from the actual saved LUT/index payloads, with hashes
matching the FP16 CPU reference reconstruction. It also checks package bytes,
state shapes, source sidecars and the exact RoPE tables. The prior
[Python test run](evidence/unit-tests.xml) passed all 84 tests.

The [CPU reference report](evidence/reference/reference-report.json) compares the
FP16 source with the LUT6 candidate on three fixed next-token probes:

| Probe | Matching next token | KL(source ∥ LUT6), nats | Logit RMSE |
| --- | --- | ---: | ---: |
| English greeting | `Hello` — 9906 | 0.003005 | 0.184519 |
| 2 + 2 | `4` — 19 | 0.009955 | 0.369218 |
| French greeting | `Bonjour` — 82681 | 0.014317 | 0.196018 |

All three produced finite logits and matched the source's greedy next token.
This is a **3/3 fixed-prompt sanity check**, without a broad quality, generated
answer, long-context, or Apple-runtime claim. The original LUT4 attempt failed
these checks and was rejected; its
[diagnosis](evidence/rejected_lut4_diagnostics/quantization-diagnosis.json) is kept
as evidence. The selected LUT6 uses the pinned weighted one-dimensional k-means
solver. [Parallel fitting evidence](evidence/parallel-fitting-equivalence.json)
records byte-identical serial and parallel results for the full real embedding.

The [French conversion report](CONVERSION_REPORT.md) gives the complete result
and limitations. `evidence/reference/` includes the full measured logits in NPZ
form, prompt identities, weight hashes and JSON reports. These are measurements,
not model weights.

## Reproduce or inspect

The [toolkit guide](toolkit/README.md#reproduce-the-export) documents the complete
download/export/reference/verification commands. Use its **separate Python 3.12
environment and [pinned requirements](toolkit/requirements.txt)**; the repository
root's 8B requirements use different Torch, Transformers and Core ML Tools
versions.

From `ane-lut6-c2048/toolkit/`, the conversion stages are `download_source.py`,
`run_export.py --bits 6 --palette-workers 4 --context 2048 --layers-per-block 4`,
`python -m dolphin_ane.reference --bits 6 --palette-workers 4 --context 2048`,
and `verify_export.py`. The downloader restores the omitted tokenizer and checks
the pinned source identities before loading original weight shards. These steps
run locally; this release adds no GitHub Actions workflow.

The conversion, test and Swift source files are unchanged from the verified
toolkit. Publication edits adapt documentation, add this index and attribution,
and omit the large tokenizer and model payloads from GitHub.

**Built with Llama.** See the [original model card](toolkit/source/README.md),
[license](toolkit/licenses/LLAMA_3_2_LICENSE.txt),
[use policy](toolkit/licenses/LLAMA_3_2_USE_POLICY.md), and [Notice](Notice.txt).
