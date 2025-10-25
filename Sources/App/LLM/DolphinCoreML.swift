//
//  DolphinCoreML.swift
//  One-artifact integration for chat (init/step + KV-cache) and LLM2Vec embeddings.
//  Works with the multifunction .mlpackage produced by dolphin2coreml_full.py
//
//  Notes
//  - If your Core ML toolchain emitted separate packages instead of a single multi-function package,
//    see the fallback loader at the bottom of this file.
//
//  Minimum: iOS 18 / macOS 15
//

import Foundation
import CoreML

public enum ComputeUnitSelection: String {
    case all = "ALL"              // ANE + GPU + CPU on iPhone/iPad, GPU+CPU on Mac
    case cpuAndGPU = "CPU_AND_GPU"
    case cpuOnly = "CPU_ONLY"

    var coreML: MLComputeUnits {
        switch self {
        case .all:       return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly:   return .cpuOnly
        }
    }
}

/// Thin wrapper around a multifunction Core ML LLM package that exposes:
///  - init pass (first tokens → logits + KV cache)
///  - decode step (next token with KV cache → logits + updated KV)
///  - encode_for_llm2vec (sequence → embedding)
public final class DolphinCoreML {
    public struct InitOutput {
        public let logits: MLMultiArray          // [1, T, vocab]
        public let lastHidden: MLMultiArray      // [1, T, hidden]
        public let pastK: [MLMultiArray]         // per-layer [1, nH, T, headDim]
        public let pastV: [MLMultiArray]         // per-layer [1, nH, T, headDim]
    }

    public struct DecodeOutput {
        public let logits: MLMultiArray          // [1, 1, vocab]
        public let lastHidden: MLMultiArray      // [1, 1, hidden]
        public let outK: [MLMultiArray]          // per-layer [1, nH, T+1, headDim]
        public let outV: [MLMultiArray]          // per-layer [1, nH, T+1, headDim]
    }

    public let model: MLModel
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let headDim: Int
    public let seqLen: Int

    /// Load a multifunction mlpackage from app bundle or file URL.
    public init(modelURL: URL,
                computeUnits: ComputeUnitSelection = .all,
                metadata: (vocabSize: Int, hiddenSize: Int, numLayers: Int, numHeads: Int, headDim: Int, seqLen: Int)) throws
    {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits.coreML
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)

        self.vocabSize  = metadata.vocabSize
        self.hiddenSize = metadata.hiddenSize
        self.numLayers  = metadata.numLayers
        self.numHeads   = metadata.numHeads
        self.headDim    = metadata.headDim
        self.seqLen     = metadata.seqLen
    }

    // MARK: - Public API

    /// First forward pass (no KV cache yet).
    /// - Parameters:
    ///   - inputIds: [1, T]
    ///   - attentionMask: [1, T]
    public func initPass(inputIds: MLMultiArray,
                         attentionMask: MLMultiArray) throws -> InitOutput
    {
        let inputs: [String: Any] = [
            "input_ids": inputIds,
            "attention_mask": attentionMask
        ]
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))

        guard let logits = out.featureValue(for: "logits")?.multiArrayValue,
              let lastHidden = out.featureValue(for: "last_hidden")?.multiArrayValue
        else { throw NSError(domain: "DolphinCoreML", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing logits/last_hidden"]) }

        var pastK: [MLMultiArray] = []
        var pastV: [MLMultiArray] = []
        for i in 0..<numLayers {
            guard let k = out.featureValue(for: "past_k_\(i)")?.multiArrayValue,
                  let v = out.featureValue(for: "past_v_\(i)")?.multiArrayValue
            else { throw NSError(domain: "DolphinCoreML", code: -2, userInfo: [NSLocalizedDescriptionKey: "Missing past KV layer \(i)"]) }
            pastK.append(k); pastV.append(v)
        }
        return .init(logits: logits, lastHidden: lastHidden, pastK: pastK, pastV: pastV)
    }

    /// Decode one token step with KV cache.
    /// - Parameters:
    ///   - nextId: [1, 1]
    ///   - nextMask: [1, 1]
    ///   - pastK/pastV: previous cache (per-layer arrays)
    public func decodeStep(nextId: MLMultiArray,
                           nextMask: MLMultiArray,
                           pastK: [MLMultiArray],
                           pastV: [MLMultiArray]) throws -> DecodeOutput
    {
        var dict: [String: Any] = [
            "input_ids": nextId,
            "attention_mask": nextMask
        ]
        for i in 0..<numLayers {
            dict["in_k_\(i)"] = pastK[i]
            dict["in_v_\(i)"] = pastV[i]
        }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))

        guard let logits = out.featureValue(for: "logits")?.multiArrayValue,
              let lastHidden = out.featureValue(for: "last_hidden")?.multiArrayValue
        else { throw NSError(domain: "DolphinCoreML", code: -3, userInfo: [NSLocalizedDescriptionKey: "Missing logits/last_hidden"]) }

        var outK: [MLMultiArray] = []
        var outV: [MLMultiArray] = []
        for i in 0..<numLayers {
            guard let k = out.featureValue(for: "out_k_\(i)")?.multiArrayValue,
                  let v = out.featureValue(for: "out_v_\(i)")?.multiArrayValue
            else { throw NSError(domain: "DolphinCoreML", code: -4, userInfo: [NSLocalizedDescriptionKey: "Missing out KV layer \(i)"]) }
            outK.append(k); outV.append(v)
        }
        return .init(logits: logits, lastHidden: lastHidden, outK: outK, outV: outV)
    }

    /// Embedding forward pass for LLM2Vec head (single shot).
    /// - Parameters:
    ///   - inputIds: [1, T]
    ///   - attentionMask: [1, T]
    /// - Returns: embedding [1, hidden]
    public func encodeEmbedding(inputIds: MLMultiArray,
                                attentionMask: MLMultiArray) throws -> MLMultiArray {
        let inputs: [String: Any] = [
            "input_ids": inputIds,
            "attention_mask": attentionMask
        ]
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        guard let emb = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw NSError(domain: "DolphinCoreML", code: -5, userInfo: [NSLocalizedDescriptionKey: "Missing embedding"])
        }
        return emb
    }

    // MARK: - Convenience samplers

    /// Greedy sample argmax from logits [1, *, vocab] → Int32 token id.
    public func greedySample(from logits: MLMultiArray) -> Int32 {
        precondition(logits.dataType == .float16 || logits.dataType == .float32)
        let vocab = vocabSize
        // take last time-step
        let lastOffset = logits.count - vocab
        if logits.dataType == .float16 {
            let ptr = logits.dataPointer.assumingMemoryBound(to: UInt16.self).advanced(by: lastOffset)
            var bestIdx = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for i in 0..<vocab {
                let value = floatValue(fromHalfBits: ptr[i])
                if value > bestVal {
                    bestVal = value
                    bestIdx = i
                }
            }
            return Int32(bestIdx)
        } else {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: lastOffset)
            var bestIdx = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for i in 0..<vocab {
                let value = ptr[i]
                if value > bestVal {
                    bestVal = value
                    bestIdx = i
                }
            }
            return Int32(bestIdx)
        }
    }

    /// Convert IEEE 754 half (UInt16) to Float (utility).
    private func floatValue(fromHalfBits bits: UInt16) -> Float {
        return Float(Float16(bitPattern: bits))
    }
}

// MARK: - Fallback loader for three-package export (if your toolchain didn’t stitch functions)
// Usage: create DolphinCoreMLFallback with URLs to init/decode/llm2vec packages and call same methods,
// mapping names as generated by your script (Dolphin_init, Dolphin_decode, Dolphin_llm2vec).
public final class DolphinCoreMLFallback {
    public let initModel: MLModel
    public let decodeModel: MLModel
    public let embedModel: MLModel

    public init(initURL: URL, decodeURL: URL, embedURL: URL,
                computeUnits: ComputeUnitSelection = .all) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits.coreML
        initModel   = try MLModel(contentsOf: initURL, configuration: cfg)
        decodeModel = try MLModel(contentsOf: decodeURL, configuration: cfg)
        embedModel  = try MLModel(contentsOf: embedURL, configuration: cfg)
    }
}
