//
//  BenchmarkHarness.swift
//  Simple tok/s + embedding latency measurement for DolphinCoreML
//

import Foundation
import CoreML
import DolphinCoreMLRuntime
#if canImport(os)
import os.log
#endif

public final class BenchmarkHarness {
    private let dolphin: DolphinCoreML
    private let tokenizer: YourTokenizerProtocol
    private let defaultWarmupTokens: Int
#if canImport(os)
    private let logger = Logger(subsystem: "ai.dolphin.bench", category: "BenchmarkHarness")
#endif

    public init(dolphin: DolphinCoreML, tokenizer: YourTokenizerProtocol, warmupTokens: Int = 16) {
        precondition(warmupTokens >= 0, "Warmup tokens must be non-negative.")
        self.dolphin = dolphin
        self.tokenizer = tokenizer
        self.defaultWarmupTokens = warmupTokens
    }

    /// Executes a single benchmarking pass using the provided prompt.
    ///
    /// The harness performs an optional warmup phase to prime the KV cache,
    /// measures end-to-end initialization latency, streams `genTokens` decode
    /// steps to derive a tokens-per-second score, and records embedding
    /// encoding latency.
    /// - Parameters:
    ///   - prompt: Input text to tokenize and feed into the Core ML runtime.
    ///   - genTokens: Number of decode iterations to time for throughput.
    ///   - warmupTokens: Overrides the default warmup token budget when
    ///     specified; otherwise `defaultWarmupTokens` is used.
    /// - Returns: A tuple containing initialization milliseconds, produced
    ///   tokens per second, and embedding encoding milliseconds.
    @discardableResult
    public func run(
        prompt: String,
        genTokens: Int = 64,
        warmupTokens: Int? = nil
    ) throws -> (initMs: Double, tokPerSec: Double, embedMs: Double) {
        let (ids, mask) = tokenizer.encodeToMultiArray(prompt, seqLen: dolphin.seqLen)

        let t0 = CFAbsoluteTimeGetCurrent()
        let initOut = try dolphin.initPass(inputIds: ids, attentionMask: mask)
        let t1 = CFAbsoluteTimeGetCurrent()
        let initMs = (t1 - t0) * 1000.0

        var k = initOut.pastK
        var v = initOut.pastV
        let warmupBudget = warmupTokens ?? defaultWarmupTokens
        var stepCountWarmup = 0
        while stepCountWarmup < warmupBudget {
            let (nid, nmask) = tokenizer.nextMaskArrays(tokenId: Int32(1))
            let warmupStep = try dolphin.decodeStep(nextId: nid, nextMask: nmask, pastK: k, pastV: v)
            k = try trimCacheToSeqLen(warmupStep.outK)
            v = try trimCacheToSeqLen(warmupStep.outV)
            stepCountWarmup += 1
        }

        let steps = max(1, genTokens)
        var produced = 0
        var nextId = try tokenizer.firstNextIdArray()
        var nextMask = try tokenizer.oneMaskArray()

        let t2 = CFAbsoluteTimeGetCurrent()
        while produced < steps {
            let step = try dolphin.decodeStep(nextId: nextId, nextMask: nextMask, pastK: k, pastV: v)
            let tok = dolphin.greedySample(from: step.logits)
            (nextId, nextMask) = tokenizer.nextMaskArrays(tokenId: tok)
            k = try trimCacheToSeqLen(step.outK)
            v = try trimCacheToSeqLen(step.outV)
            produced += 1
        }
        let t3 = CFAbsoluteTimeGetCurrent()
        let stepMs = ((t3 - t2) * 1000.0) / Double(steps)
        let tokPerSec = 1000.0 / stepMs

        let tE0 = CFAbsoluteTimeGetCurrent()
        _ = try dolphin.encodeEmbedding(inputIds: ids, attentionMask: mask)
        let tE1 = CFAbsoluteTimeGetCurrent()
        let embedMs = (tE1 - tE0) * 1000.0

        print(String(format: "Init: %.1f ms • Decode: %.2f tok/s • Embed: %.1f ms", initMs, tokPerSec, embedMs))
#if canImport(os)
        logger.log(
            "Single benchmark iteration — init: \(initMs, privacy: .public) ms, decode: \(tokPerSec, privacy: .public) tok/s, embed: \(embedMs, privacy: .public) ms"
        )
#endif
        return (initMs, tokPerSec, embedMs)
    }

    /// Runs multiple benchmark iterations and persists the throughput metrics to CSV.
    /// - Parameters:
    ///   - prompt: Prompt to feed into the model.
    ///   - genTokens: Number of tokens to decode per iteration.
    ///   - iterations: Number of benchmark passes to execute (must be > 0).
    ///   - outputURL: Destination CSV file URL.
    ///   - minimumTokensPerSecond: Guardrail for minimum acceptable throughput per iteration.
    ///   - warmupTokens: Optional override for warmup decode steps (defaults to the harness configuration).
    ///   - clock: Dependency-injected clock for deterministic testing.
    /// - Returns: The recorded benchmark samples written to disk.
    @discardableResult
    public func runAndExport(
        prompt: String,
        genTokens: Int = 64,
        iterations: Int,
        outputURL: URL,
        minimumTokensPerSecond: Double = 33.0,
        warmupTokens: Int? = nil,
        clock: () -> Date = Date.init
    ) throws -> [BenchmarkCSVRow] {
        precondition(iterations > 0, "Iterations must be positive.")
        var samples: [BenchmarkCSVRow] = []
        samples.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let metrics = try run(prompt: prompt, genTokens: genTokens, warmupTokens: warmupTokens)
            let decodeMsPerToken = 1000.0 / metrics.tokPerSec
            let sample = BenchmarkCSVRow(
                timestamp: clock(),
                tokensPerSecond: metrics.tokPerSec,
                initMilliseconds: metrics.initMs,
                decodeMillisecondsPerToken: decodeMsPerToken,
                embedMilliseconds: metrics.embedMs
            )
            samples.append(sample)
        }

        let summary = try ThroughputRegressor.validate(samples, minimumRate: minimumTokensPerSecond)
        let writer = BenchmarkCSVWriter()
        try writer.write(samples: samples, to: outputURL)

#if canImport(os)
        logger.log(
            "Persisted \(samples.count, privacy: .public) benchmark samples to \(outputURL.absoluteString, privacy: .public); minimum observed \(summary.minimumObserved, privacy: .public) tok/s"
        )
#endif

        return samples
    }
}

extension BenchmarkHarness {
    private enum CacheError: Error, LocalizedError {
        case invalidRank(actual: Int)
        case insufficientSequenceLength
        case unsupportedDataType(MLMultiArrayDataType)

        var errorDescription: String? {
            switch self {
            case .invalidRank(let actual):
                return "Expected 4D KV cache tensor but received rank \(actual)."
            case .insufficientSequenceLength:
                return "KV cache tensor does not contain enough timesteps to trim."
            case .unsupportedDataType(let type):
                return "Unsupported MLMultiArray data type \(type)."
            }
        }
    }

    private func trimCacheToSeqLen(_ arrays: [MLMultiArray]) throws -> [MLMultiArray] {
        return try arrays.map { array in
            let shape = array.shape.map { $0.intValue }
            guard shape.count == 4 else {
                throw CacheError.invalidRank(actual: shape.count)
            }

            let seqPlusOne = shape[2]
            guard seqPlusOne > 1 else {
                throw CacheError.insufficientSequenceLength
            }

            let trimmedSeqLen = seqPlusOne - 1
            let headDim = shape[3]
            let outer = shape[0] * shape[1]
            let newShape = [shape[0], shape[1], trimmedSeqLen, headDim].map { NSNumber(value: $0) }
            let trimmed = try MLMultiArray(shape: newShape, dataType: array.dataType)

            let sourceStride = seqPlusOne * headDim
            let destStride = trimmedSeqLen * headDim

            switch array.dataType {
            case .float16:
                let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
                let dst = trimmed.dataPointer.bindMemory(to: UInt16.self, capacity: trimmed.count)
                for outerIndex in 0..<outer {
                    let srcStart = outerIndex * sourceStride + headDim
                    let dstStart = outerIndex * destStride
                    dst.advanced(by: dstStart).assign(from: src.advanced(by: srcStart), count: destStride)
                }
            case .float32:
                let src = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
                let dst = trimmed.dataPointer.bindMemory(to: Float.self, capacity: trimmed.count)
                for outerIndex in 0..<outer {
                    let srcStart = outerIndex * sourceStride + headDim
                    let dstStart = outerIndex * destStride
                    dst.advanced(by: dstStart).assign(from: src.advanced(by: srcStart), count: destStride)
                }
            default:
                throw CacheError.unsupportedDataType(array.dataType)
            }

            return trimmed
        }
    }
}

public protocol YourTokenizerProtocol {
    func encodeToMultiArray(_ text: String, seqLen: Int) -> (MLMultiArray, MLMultiArray)
    func nextMaskArrays(tokenId: Int32) -> (MLMultiArray, MLMultiArray)
    func firstNextIdArray() throws -> MLMultiArray
    func oneMaskArray() throws -> MLMultiArray
}
