//
//  BenchmarkHarness.swift
//  Simple tok/s + embedding latency measurement for DolphinCoreML
//

import Foundation
import CoreML

public final class BenchmarkHarness {
    private let dolphin: DolphinCoreML
    private let tokenizer: YourTokenizerProtocol
    private let warmupTokens = 16

    public init(dolphin: DolphinCoreML, tokenizer: YourTokenizerProtocol) {
        self.dolphin = dolphin
        self.tokenizer = tokenizer
    }

    @discardableResult
    public func run(prompt: String, genTokens: Int = 64) throws -> (initMs: Double, tokPerSec: Double, embedMs: Double) {
        let (ids, mask) = tokenizer.encodeToMultiArray(prompt, seqLen: dolphin.seqLen)

        let t0 = CFAbsoluteTimeGetCurrent()
        let initOut = try dolphin.initPass(inputIds: ids, attentionMask: mask)
        let t1 = CFAbsoluteTimeGetCurrent()
        let initMs = (t1 - t0) * 1000.0

        var k = initOut.pastK
        var v = initOut.pastV
        var stepCountWarmup = 0
        while stepCountWarmup < warmupTokens {
            let (nid, nmask) = tokenizer.nextMaskArrays(tokenId: Int32(1))
            _ = try dolphin.decodeStep(nextId: nid, nextMask: nmask, pastK: k, pastV: v)
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
            k = step.outK
            v = step.outV
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
        return (initMs, tokPerSec, embedMs)
    }
}

public protocol YourTokenizerProtocol {
    func encodeToMultiArray(_ text: String, seqLen: Int) -> (MLMultiArray, MLMultiArray)
    func nextMaskArrays(tokenId: Int32) -> (MLMultiArray, MLMultiArray)
    func firstNextIdArray() throws -> MLMultiArray
    func oneMaskArray() throws -> MLMultiArray
}
