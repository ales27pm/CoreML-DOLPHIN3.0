import Foundation
@preconcurrency import CoreML

public struct Manifest: Decodable, Sendable {
    public let context_length: Int, vocab_size: Int, hidden_size: Int, head_dim: Int
    public let eos_token_ids: [Int32], head_sizes: [Int], block_names: [String]
}
public struct TokenLogit: Codable, Sendable { public let id: Int32; public let logit: Float }
public struct GenerationResult: Codable, Sendable {
    public let generatedIDs: [Int32], consumedTokens: Int, usedStateCount: Int
    public let prefillSeconds: Double, decodeSeconds: Double, tokenSeconds: [Double]
    public let prefillLogits: [TokenLogit], stopReason: String
}
public struct OperationPlan: Codable, Sendable {
    public let path: String, operation: String, preferred: String, supported: [String], requiresANE: Bool
}
public struct ModelPlan: Codable, Sendable {
    public let model: String, operations: [OperationPlan], error: String?
    public var strictANEPass: Bool {
        let relevant = operations.filter(\.requiresANE)
        return error == nil && !relevant.isEmpty && relevant.allSatisfy { $0.preferred == "ane" }
    }
}
public enum DolphinError: Error { case invalid(String), busy }

/// IEEE 754 binary16 helpers that also work on Intel macOS, where Swift.Float16 is unavailable.
enum Binary16 {
    static let zero: UInt16 = 0x0000
    static let one: UInt16 = 0x3c00
    static let negativeGreatestFiniteMagnitude: UInt16 = 0xfbff

    static func isFinite(_ bits: UInt16) -> Bool { bits & 0x7c00 != 0x7c00 }

    static func float(from bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = Int((bits >> 10) & 0x001f)
        var fraction = UInt32(bits & 0x03ff)
        let floatBits: UInt32

        switch exponent {
        case 0 where fraction == 0:
            floatBits = sign
        case 0:
            var unbiasedExponent = -14
            while fraction & 0x0400 == 0 {
                fraction <<= 1
                unbiasedExponent -= 1
            }
            fraction &= 0x03ff
            floatBits = sign | (UInt32(unbiasedExponent + 127) << 23) | (fraction << 13)
        case 0x1f:
            floatBits = sign | 0x7f80_0000 | (fraction << 13)
        default:
            floatBits = sign | (UInt32(exponent - 15 + 127) << 23) | (fraction << 13)
        }
        return Float(bitPattern: floatBits)
    }

    static func bits(from bytes: UnsafeRawBufferPointer, elementOffset: Int) throws -> UInt16 {
        let byteOffset = elementOffset * MemoryLayout<UInt16>.size
        guard elementOffset >= 0, byteOffset + 1 < bytes.count else {
            throw DolphinError.invalid("Truncated FP16 storage")
        }
        return UInt16(bytes[byteOffset]) | (UInt16(bytes[byteOffset + 1]) << 8)
    }

    static func multiArray(_ values: [UInt16], shape: [Int]) throws -> MLMultiArray {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }), shape.reduce(1, *) == values.count else {
            throw DolphinError.invalid("Invalid FP16 input shape")
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        try array.withUnsafeMutableBytes { bytes, strides in
            guard strides.count == shape.count else { throw DolphinError.invalid("Invalid FP16 input strides") }
            var logicalStrides = [Int](repeating: 1, count: shape.count)
            if shape.count > 1 {
                for axis in stride(from: shape.count - 2, through: 0, by: -1) {
                    logicalStrides[axis] = logicalStrides[axis + 1] * shape[axis + 1]
                }
            }
            for (linearIndex, value) in values.enumerated() {
                var elementOffset = 0
                for axis in shape.indices {
                    let coordinate = (linearIndex / logicalStrides[axis]) % shape[axis]
                    elementOffset += coordinate * strides[axis]
                }
                let byteOffset = elementOffset * MemoryLayout<UInt16>.size
                guard byteOffset + 1 < bytes.count else { throw DolphinError.invalid("Invalid FP16 input storage") }
                bytes[byteOffset] = UInt8(truncatingIfNeeded: value)
                bytes[byteOffset + 1] = UInt8(truncatingIfNeeded: value >> 8)
            }
        }
        return array
    }
}

/// One owner for every state in the split pipeline; a fresh coherent state set per generation.
public actor DolphinANEEngine {
    public let manifest: Manifest, computePlans: [ModelPlan]
    private let models: [MLModel], ropeCos: [UInt16], ropeSin: [UInt16]
    private var busy = false, cancelRequested = false

    public init(directory: URL) async throws {
        let m = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard m.context_length > 0, m.hidden_size == 3072, m.head_dim == 128,
              m.vocab_size == 128258, m.head_sizes == Array(repeating: 16032, count: 7) + [16034],
              !m.block_names.isEmpty, Set(m.block_names).count == m.block_names.count,
              m.block_names.allSatisfy({ $0.range(of: "^block_[0-9]{2}$", options: .regularExpression) != nil })
        else { throw DolphinError.invalid("Unsupported manifest contract") }
        let config = MLModelConfiguration(); config.computeUnits = .cpuAndNeuralEngine
        var loaded: [MLModel] = [], plans: [ModelPlan] = []
        for name in ["embedding"] + m.block_names + ["head"] {
            try Task.checkCancellation()
            var url = directory.appendingPathComponent(name + ".mlmodelc")
            if !FileManager.default.fileExists(atPath: url.path) {
                let source = directory.appendingPathComponent(name + ".mlpackage")
                url = try await withCheckedThrowingContinuation { (c: CheckedContinuation<URL, Error>) in
                    MLModel.compileModel(at: source) { c.resume(with: $0) }
                }
            }
            loaded.append(try await MLModel.load(contentsOf: url, configuration: config))
            do { plans.append(try await Self.plan(name: name, url: url, config: config)) }
            catch { plans.append(ModelPlan(model: name, operations: [], error: String(describing: error))) }
        }
        for (i, block) in loaded.dropFirst().dropLast().enumerated() {
            guard Set(block.modelDescription.stateDescriptionsByName.keys) == Set(["keyCache", "valueCache"])
            else { throw DolphinError.invalid("Missing KV states in \(m.block_names[i])") }
        }
        manifest = m; models = loaded; computePlans = plans
        ropeCos = try Self.rope(directory.appendingPathComponent("rope_cos.fp16"), count: m.context_length * m.head_dim)
        ropeSin = try Self.rope(directory.appendingPathComponent("rope_sin.fp16"), count: m.context_length * m.head_dim)
    }

    /// Cancellation never releases the lease while a Core ML prediction is still in flight.
    public func cancel() { if busy { cancelRequested = true } }

    public func generate(tokens: [Int32], maxNewTokens: Int = 32, probeIDs: [Int32] = []) async throws -> GenerationResult {
        guard !busy else { throw DolphinError.busy }
        guard !tokens.isEmpty, maxNewTokens > 0, tokens.count <= manifest.context_length,
              maxNewTokens - 1 <= manifest.context_length - tokens.count,
              (tokens + probeIDs).allSatisfy({ $0 >= 0 && Int($0) < manifest.vocab_size })
        else { throw DolphinError.invalid("Tokens, probe IDs or context budget invalid") }
        busy = true; cancelRequested = false // Acquired before the first await; actor reentrancy is permitted.
        defer { busy = false; cancelRequested = false }
        let states = models.dropFirst().dropLast().map { $0.makeState() }
        var consumed = 0, timings: [Double] = [], next: Int32 = 0, reference: [TokenLogit] = []
        let start = ProcessInfo.processInfo.systemUptime
        for (i, token) in tokens.enumerated() {
            let t = ProcessInfo.processInfo.systemUptime
            if let logits = try await step(token, position: consumed, states: states,
                                           needLogits: i == tokens.count - 1, probeIDs: Set(probeIDs)) {
                next = logits.0; reference = logits.1
            }
            consumed += 1; timings.append(ProcessInfo.processInfo.systemUptime - t)
        }
        let prefillEnd = ProcessInfo.processInfo.systemUptime
        var generated: [Int32] = []; var stop = "length"
        for index in 0..<maxNewTokens {
            try checkCancellation()
            generated.append(next) // These logits already follow the last consumed token.
            if manifest.eos_token_ids.contains(next) { stop = "eos"; break }
            if index + 1 == maxNewTokens { break } // The final sampled token has not entered KV state.
            let t = ProcessInfo.processInfo.systemUptime
            guard let logits = try await step(next, position: consumed, states: states, needLogits: true, probeIDs: [])
            else { throw DolphinError.invalid("Missing decode logits") }
            next = logits.0; consumed += 1; timings.append(ProcessInfo.processInfo.systemUptime - t)
        }
        try checkCancellation()
        return GenerationResult(generatedIDs: generated, consumedTokens: consumed, usedStateCount: states.count,
                                prefillSeconds: prefillEnd - start, decodeSeconds: ProcessInfo.processInfo.systemUptime - prefillEnd,
                                tokenSeconds: timings, prefillLogits: reference, stopReason: stop)
        // On success, error or cancellation all states leave scope together, after awaited predictions return.
    }

    private func checkCancellation() throws {
        try Task.checkCancellation()
        if cancelRequested { throw CancellationError() }
    }
    private func step(_ token: Int32, position: Int, states: [MLState], needLogits: Bool,
                      probeIDs: Set<Int32>) async throws -> (Int32, [TokenLogit])? {
        try checkCancellation()
        guard position < manifest.context_length else { throw DolphinError.invalid("KV cache full") }
        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32); ids[0] = NSNumber(value: token)
        let embedded = try await models[0].prediction(from: MLDictionaryFeatureProvider(dictionary: ["inputIds": ids]))
        try checkCancellation()
        var hidden = try Self.output(embedded, "hiddenStates", width: manifest.hidden_size)
        let range = (position * manifest.head_dim)..<((position + 1) * manifest.head_dim)
        let cos = try Self.half(Array(ropeCos[range]), shape: [1, manifest.head_dim, 1, 1])
        let sin = try Self.half(Array(ropeSin[range]), shape: [1, manifest.head_dim, 1, 1])
        var write = [UInt16](repeating: Binary16.zero, count: manifest.context_length)
        write[position] = Binary16.one
        let causal = (0..<manifest.context_length).map {
            $0 <= position ? Binary16.zero : Binary16.negativeGreatestFiniteMagnitude
        }
        let writeArray = try Self.half(write, shape: [1, 1, 1, manifest.context_length])
        let causalArray = try Self.half(causal, shape: [1, 1, 1, manifest.context_length])
        for i in states.indices {
            let input = try MLDictionaryFeatureProvider(dictionary: ["hiddenStates": hidden, "ropeCos": cos,
                       "ropeSin": sin, "writeMask": writeArray, "causalMask": causalArray])
            let output = try await models[i + 1].prediction(from: input, using: states[i])
            try checkCancellation()
            hidden = try Self.output(output, "outputHiddenStates", width: manifest.hidden_size)
        }
        if !needLogits { return nil }
        let output = try await models[models.count - 1].prediction(from: MLDictionaryFeatureProvider(dictionary: ["hiddenStates": hidden]))
        try checkCancellation()
        var best = -Float.infinity, bestID: Int32 = 0, offset = 0, selected: [TokenLogit] = []
        for (head, width) in manifest.head_sizes.enumerated() {
            let array = try Self.output(output, "logits_\(head)", width: width)
            try array.withUnsafeBytes { bytes in
                let stride = array.strides[1].intValue
                for j in 0..<width {
                    let bits = try Binary16.bits(from: bytes, elementOffset: j * stride)
                    let value = Binary16.float(from: bits), id = Int32(offset + j)
                    guard value.isFinite else { throw DolphinError.invalid("Nonfinite logit at \(id)") }
                    if value > best { best = value; bestID = id } // Earliest vocabulary index wins exact ties.
                    if probeIDs.contains(id) { selected.append(TokenLogit(id: id, logit: value)) }
                }
            }
            offset += width
        }
        return (bestID, selected)
    }
    private static func output(_ provider: any MLFeatureProvider, _ name: String, width: Int) throws -> MLMultiArray {
        guard let a = provider.featureValue(for: name)?.multiArrayValue,
              a.dataType == .float16, a.shape.map(\.intValue) == [1, width, 1, 1]
        else { throw DolphinError.invalid("Unexpected output \(name)") }
        return a
    }
    private static func half(_ values: [UInt16], shape: [Int]) throws -> MLMultiArray {
        try Binary16.multiArray(values, shape: shape)
    }
    private static func rope(_ url: URL, count: Int) throws -> [UInt16] {
        let d = try Data(contentsOf: url)
        guard d.count == count * 2 else { throw DolphinError.invalid("Invalid RoPE table length") }
        let values = (0..<count).map { UInt16(d[$0 * 2]) | (UInt16(d[$0 * 2 + 1]) << 8) }
        guard values.allSatisfy(Binary16.isFinite) else { throw DolphinError.invalid("Nonfinite RoPE") }
        return values
    }
    private static func device(_ d: MLComputeDevice) -> String {
        switch d { case .cpu: return "cpu"; case .gpu: return "gpu"; case .neuralEngine: return "ane"; @unknown default: return "unknown" }
    }
    private static func plan(name: String, url: URL, config: MLModelConfiguration) async throws -> ModelPlan {
        let p = try await MLComputePlan.load(contentsOf: url, configuration: config)
        guard case .program(let program) = p.modelStructure else { throw DolphinError.invalid("Not an ML Program") }
        let bookkeeping: Set<String> = ["const", "read_state", "write_state", "coreml_update_state", "identity"]
        var records: [OperationPlan] = []
        func walk(_ block: MLModelStructure.Program.Block, path: String) {
            for (i, op) in block.operations.enumerated() {
                let usage = p.deviceUsage(for: op), here = "\(path)/\(i)"
                records.append(OperationPlan(path: here, operation: op.operatorName,
                    preferred: usage.map { device($0.preferred) } ?? "unknown", supported: usage?.supported.map(device) ?? [],
                    requiresANE: name != "embedding" && !bookkeeping.contains(op.operatorName) && !op.operatorName.hasPrefix("constexpr_")))
                for (j, nested) in op.blocks.enumerated() { walk(nested, path: "\(here)/block\(j)") }
            }
        }
        for (name, function) in program.functions.sorted(by: { $0.key < $1.key }) { walk(function.block, path: name) }
        return ModelPlan(model: name, operations: records, error: nil)
    }
}
