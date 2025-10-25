# Dolphin 3.1 8B → Core ML (W4, multifunction) — Chat + LLM2Vec

This repo contains a **turn-key pipeline** to export **Dolphin3.0-Llama3.1-8B** to a single Core ML `.mlpackage` with **two entry points**:
- `init` + `decode` (with KV-cache) for token streaming
- `encode_for_llm2vec` for embeddings

It also includes a **Swift helper** and a **benchmark harness** for iOS/macOS.

## 1) End-to-end export

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

What you get
•out/Dolphin3_1_8B_W4.mlpackage — one artifact with:
•init   → logits + KV cache tensors (first tokens)
•decode → logits + updated KV (one token per step)
•encode_for_llm2vec → embedding [1, hidden]

Compression
•Palettization to W4 (4-bit LUT+indices), per_grouped_channel, group size 16
•Linear quant over palettized tables with joint_compression=True

Target: iOS 18 / macOS 15+ (ANE/GPU/CPU).

2) App integration (Swift)
•Add DolphinCoreML.swift to your Xcode target.
•Put .mlpackage into your app bundle (e.g. via Xcode “Add Files to …”).
•Instantiate the wrapper:

```swift
let url = Bundle.main.url(forResource: "Dolphin3_1_8B_W4", withExtension: "mlpackage")!
let dolphin = try DolphinCoreML(
  modelURL: url,
  computeUnits: .all,
  metadata: (vocabSize: 32000, hiddenSize: 4096, numLayers: 32, numHeads: 32, headDim: 128, seqLen: 2048)
)
```

Use your model’s actual dims if they differ.

Token loop sketch

```swift
// 1) Prepare ids/mask [1, T] for the prompt
let (ids, mask) = YourTokenizer.encode(prompt: "Hello")

// 2) First pass to build KV
let initOut = try dolphin.initPass(inputIds: ids, attentionMask: mask)

// 3) Stream tokens with decode step
var k = initOut.pastK
var v = initOut.pastV
for _ in 0..<128 {
  let nextId = try tokenizer.lastTokenAsArray(fromLogits: initOut.logits) // or use greedySample
  let nextMask = try makeMaskOne() // [1,1] of 1s
  let step = try dolphin.decodeStep(nextId: nextId, nextMask: nextMask, pastK: k, pastV: v)
  let tok = dolphin.greedySample(from: step.logits)
  // append token to output, update KV
  k = step.outK; v = step.outV
}
```

Embeddings

```swift
let (ids, mask) = YourTokenizer.encode(text: "Search document …")
let emb = try dolphin.encodeEmbedding(inputIds: ids, attentionMask: mask)  // [1, hidden]
```

3) Benchmarking
•Tokens/sec depends on device + compute units + sequence length + cache I/O.
•Use BenchmarkHarness.swift (below) inside your app to measure:
•Init pass latency
•Per-token step latency (average over N tokens)
•Embedding latency

4) Troubleshooting
•Op not supported: ensure you exported with minimum_deployment_target=iOS18.
•Memory pressure: lower --seq-len, or try .cpuAndGPU on older phones.
•Quality drop at 4-bit: try per_channel or increase group_size → higher fidelity.
•Tokenizer mismatch: verify vocab + special tokens exactly match Dolphin’s tokenizer.

5) License / Credits
•Dolphin model weights per original license.
•Conversion scripts and helpers MIT.

---

# `Sources/App/Bench/BenchmarkHarness.swift`

```swift
//
//  BenchmarkHarness.swift
//  Simple tok/s + embedding latency measurement for DolphinCoreML
//

import Foundation
import CoreML

public final class BenchmarkHarness {
    private let dolphin: DolphinCoreML
    private let tokenizer: YourTokenizerProtocol   // replace with your tokenizer impl
    private let warmupTokens = 16

    public init(dolphin: DolphinCoreML, tokenizer: YourTokenizerProtocol) {
        self.dolphin = dolphin
        self.tokenizer = tokenizer
    }

    @discardableResult
    public func run(prompt: String, genTokens: Int = 64) throws -> (initMs: Double, tokPerSec: Double, embedMs: Double) {
        // Encode prompt
        let (ids, mask) = tokenizer.encodeToMultiArray(prompt, seqLen: dolphin.seqLen)

        // INIT
        let t0 = CFAbsoluteTimeGetCurrent()
        let initOut = try dolphin.initPass(inputIds: ids, attentionMask: mask)
        let t1 = CFAbsoluteTimeGetCurrent()
        let initMs = (t1 - t0) * 1000.0

        // Warmup a few decode steps
        var k = initOut.pastK
        var v = initOut.pastV
        var stepCountWarmup = 0
        while stepCountWarmup < warmupTokens {
            let (nid, nmask) = tokenizer.nextMaskArrays(tokenId: Int32(1)) // dummy 1 or last sampled token
            _ = try dolphin.decodeStep(nextId: nid, nextMask: nmask, pastK: k, pastV: v)
            stepCountWarmup += 1
        }

        // Timed decoding
        let steps = max(1, genTokens)
        var produced = 0
        var nextId = try tokenizer.firstNextIdArray() // provide your first next token array
        var nextMask = try tokenizer.oneMaskArray()

        let t2 = CFAbsoluteTimeGetCurrent()
        while produced < steps {
            let step = try dolphin.decodeStep(nextId: nextId, nextMask: nextMask, pastK: k, pastV: v)
            let tok = dolphin.greedySample(from: step.logits)
            (nextId, nextMask) = tokenizer.nextMaskArrays(tokenId: tok)
            k = step.outK; v = step.outV
            produced += 1
        }
        let t3 = CFAbsoluteTimeGetCurrent()
        let stepMs = ((t3 - t2) * 1000.0) / Double(steps)
        let tokPerSec = 1000.0 / stepMs

        // Embedding latency
        let tE0 = CFAbsoluteTimeGetCurrent()
        _ = try dolphin.encodeEmbedding(inputIds: ids, attentionMask: mask)
        let tE1 = CFAbsoluteTimeGetCurrent()
        let embedMs = (tE1 - tE0) * 1000.0

        print(String(format: "Init: %.1f ms • Decode: %.2f tok/s • Embed: %.1f ms", initMs, tokPerSec, embedMs))
        return (initMs, tokPerSec, embedMs)
    }
}

/// Example tokenizer protocol you can implement with your own tokenizer.
public protocol YourTokenizerProtocol {
    /// Build MLMultiArray [1, T] int32 ids and [1, T] int32 mask. Must match export seqLen.
    func encodeToMultiArray(_ text: String, seqLen: Int) -> (MLMultiArray, MLMultiArray)
    /// Return nextId [1,1] and nextMask [1,1] (both int32)
    func nextMaskArrays(tokenId: Int32) -> (MLMultiArray, MLMultiArray)
    /// Helper variations
    func firstNextIdArray() throws -> MLMultiArray
    func oneMaskArray() throws -> MLMultiArray
}
```

---

Quick glue for a basic tokenizer (placeholder)

You can wire any tokenizer (e.g., Hugging Face tokenizers via a small Swift port or ship precomputed SentencePiece tables). If you prefer, keep tokenization in Python and send IDs to the app; otherwise I can generate a tiny Swift tokenizer adapter if you share which Dolphin tokenizer vocab/SentencePiece model you’re using.
