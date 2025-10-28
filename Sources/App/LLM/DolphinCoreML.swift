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
    case cpuAndNeuralEngine = "CPU_AND_NEURAL_ENGINE"
    case cpuOnly = "CPU_ONLY"

    var coreML: MLComputeUnits {
        switch self {
        case .all:                return .all
        case .cpuAndGPU:          return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuOnly:            return .cpuOnly
        }
    }
}

/// Thin wrapper around a multifunction Core ML LLM package that exposes:
///  - init pass (first tokens → logits + KV cache)
///  - decode step (next token with KV cache → logits + updated KV)
///  - encode (sequence → embedding)
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
    private let decodeModel: MLModel
    private let encodeModel: MLModel
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
        self.model = try DolphinCoreML.loadFunctionModel(at: modelURL,
                                                         functionName: "init",
                                                         configuration: cfg)
        self.decodeModel = try DolphinCoreML.loadFunctionModel(at: modelURL,
                                                               functionName: "decode",
                                                               configuration: cfg)
        self.encodeModel = try DolphinCoreML.loadFunctionModel(at: modelURL,
                                                               functionName: "encode",
                                                               configuration: cfg)

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
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let out = try model.prediction(from: provider)

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
        guard pastK.count == numLayers, pastV.count == numLayers else {
            throw NSError(domain: "DolphinCoreML", code: -8,
                          userInfo: [NSLocalizedDescriptionKey: "pastK/pastV must contain \(numLayers) layers. Got K=\(pastK.count), V=\(pastV.count)."])
        }
        var dict: [String: Any] = [
            "input_ids": nextId,
            "attention_mask": nextMask
        ]
        for i in 0..<numLayers {
            dict["in_k_\(i)"] = pastK[i]
            dict["in_v_\(i)"] = pastV[i]
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        let out = try decodeModel.prediction(from: provider)

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
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let out = try encodeModel.prediction(from: provider)
        guard let emb = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw NSError(domain: "DolphinCoreML", code: -5, userInfo: [NSLocalizedDescriptionKey: "Missing embedding"])
        }
        return emb
    }

    /// Convenience helper that evaluates ``encode`` for multiple sequences in order.
    /// - Parameter batches: Sequence of (input IDs, attention mask) tuples with shape ``[1, seqLen]`` each.
    /// - Returns: ``MLMultiArray`` embeddings in the same order as the input.
    /// - Throws: ``NSError`` when Core ML fails to produce an embedding for any element.
    ///
    /// Example:
    /// ```swift
    /// let prompts: [String] = ["packet capture", "wireless audit"]
    /// let requests = prompts.map { tokenizer.encodeToMultiArray($0, seqLen: dolphin.seqLen) }
    /// let embeddings = try dolphin.encodeEmbeddingBatch(requests)
    /// ```
    public func encodeEmbeddingBatch(
        _ batches: [(inputIds: MLMultiArray, attentionMask: MLMultiArray)]
    ) throws -> [MLMultiArray] {
        if batches.isEmpty {
            return []
        }

        var providers: [MLFeatureProvider] = []
        providers.reserveCapacity(batches.count)

        for (index, batch) in batches.enumerated() {
            if batch.inputIds.count != batch.attentionMask.count {
                throw NSError(
                    domain: "DolphinCoreML",
                    code: -6,
                    userInfo: [
                        NSLocalizedDescriptionKey: "input_ids count (\(batch.inputIds.count)) does not match attention_mask count (\(batch.attentionMask.count)) for element \(index)."
                    ]
                )
            }
            if batch.inputIds.shape.count != 2 || batch.attentionMask.shape.count != 2 ||
                batch.inputIds.shape[0].intValue != 1 || batch.attentionMask.shape[0].intValue != 1 ||
                batch.inputIds.shape[1].intValue != seqLen || batch.attentionMask.shape[1].intValue != seqLen {
                throw NSError(
                    domain: "DolphinCoreML",
                    code: -9,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Expected [1, seqLen=\(seqLen)] input tensors for element \(index)."
                    ]
                )
            }
            let validInputTypes: Set<MLMultiArrayDataType> = [.int32]
            if !validInputTypes.contains(batch.inputIds.dataType) ||
                !validInputTypes.contains(batch.attentionMask.dataType) {
                throw NSError(
                    domain: "DolphinCoreML",
                    code: -10,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Expected Int32 input_ids and attention_mask for element \(index)."
                    ]
                )
            }
            let provider = try MLDictionaryFeatureProvider(
                dictionary: [
                    "input_ids": batch.inputIds,
                    "attention_mask": batch.attentionMask,
                ]
            )
            providers.append(provider)
        }

        var outputs: [MLFeatureProvider] = []
        outputs.reserveCapacity(providers.count)
        for provider in providers {
            let prediction = try encodeModel.prediction(from: provider)
            outputs.append(prediction)
        }

        var embeddings: [MLMultiArray] = []
        embeddings.reserveCapacity(outputs.count)
        if outputs.count != providers.count {
            throw NSError(
                domain: "DolphinCoreML",
                code: -11,
                userInfo: [
                    NSLocalizedDescriptionKey: "Batched result count \(outputs.count) does not match input count \(providers.count)."
                ]
            )
        }
        for (index, provider) in outputs.enumerated() {
            guard let embedding = provider.featureValue(for: "embedding")?.multiArrayValue else {
                throw NSError(
                    domain: "DolphinCoreML",
                    code: -7,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Missing embedding in batched result at index \(index).",
                    ]
                )
            }
            if embedding.shape.count != 2 || embedding.shape[0].intValue != 1 ||
                embedding.shape[1].intValue != hiddenSize {
                throw NSError(
                    domain: "DolphinCoreML",
                    code: -12,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Unexpected embedding shape \(embedding.shape) at index \(index); expected [1, hiddenSize=\(hiddenSize)]."
                    ]
                )
            }
            let validEmbeddingTypes: Set<MLMultiArrayDataType> = [.float16, .float32, .double]
            if !validEmbeddingTypes.contains(embedding.dataType) {
                throw NSError(
                    domain: "DolphinCoreML",
                    code: -13,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Unsupported embedding dtype \(embedding.dataType) at index \(index)."
                    ]
                )
            }
            embeddings.append(embedding)
        }
        return embeddings
    }

    private static func loadFunctionModel(at baseURL: URL,
                                          functionName: String,
                                          configuration: MLModelConfiguration) throws -> MLModel
    {
        let fm = FileManager.default

        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: baseURL.path, isDirectory: &isDirectory) else {
            throw NSError(
                domain: "DolphinCoreML",
                code: -20,
                userInfo: [
                    NSLocalizedDescriptionKey: "Model package not found at \(baseURL.path)."
                ]
            )
        }

        if baseURL.pathExtension == "mlmodelc" {
            return try MLModel(contentsOf: baseURL, configuration: configuration)
        }

        let expectedName = "\(functionName).mlmodelc"
        var discovered: URL?
        if let enumerator = fm.enumerator(at: baseURL,
                                          includingPropertiesForKeys: [.isDirectoryKey],
                                          options: [.skipsHiddenFiles],
                                          errorHandler: nil) {
            for case let url as URL in enumerator {
                if url.pathExtension == "mlmodelc" &&
                    url.lastPathComponent.caseInsensitiveCompare(expectedName) == .orderedSame {
                    discovered = url
                    break
                }
            }
        }

        guard let resolvedURL = discovered else {
            throw NSError(
                domain: "DolphinCoreML",
                code: -21,
                userInfo: [
                    NSLocalizedDescriptionKey: "Unable to locate function \(functionName) within package at \(baseURL.path)."
                ]
            )
        }

        return try MLModel(contentsOf: resolvedURL, configuration: configuration)
    }

    // MARK: - Convenience samplers

    /// Greedy sample argmax from logits [1, *, vocab] → Int32 token id.
    public func greedySample(from logits: MLMultiArray) -> Int32 {
        precondition(logits.dataType == .float16 || logits.dataType == .float32)
        let vocab = vocabSize
        precondition(logits.count >= vocab, "Logits tensor smaller than vocab.")
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
