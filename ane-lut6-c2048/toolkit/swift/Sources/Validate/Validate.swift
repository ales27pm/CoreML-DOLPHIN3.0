import Foundation
import Darwin
import DolphinANE

private struct Fixture: Decodable {
    let name: String, input_ids: [Int32], expected_next_token: Int32
    let reference_logits: [TokenLogit], max_abs_logit_error: Float
}
private struct Fixtures: Decodable { let cases: [Fixture] }

@main struct Validate {
    static func main() async {
        var report: [String: Any] = ["compiledModelsLoaded": false, "runtimeValidated": false, "numericalParityValidated": false,
            "aneRuntimeTraceValidated": false, "computeUnits": "cpuAndNeuralEngine",
            "os": ProcessInfo.processInfo.operatingSystemVersionString]
        var size = 0
        if sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 0 {
            var bytes = [CChar](repeating: 0, count: size)
            if sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 { report["device"] = String(cString: bytes) }
        }
        let args = Array(CommandLine.arguments.dropFirst())
        func option(_ key: String) -> String? {
            guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }; return args[i + 1]
        }
        var status: Int32 = 0
        do {
            guard let path = option("--models") else {
                throw DolphinError.invalid("Usage: dolphin-ane-validate --models PATH [--tokens 128000,9906] [--max-new-tokens 8] [--fixtures PATH] [--strict-ane] [--output PATH]")
            }
            let start = ProcessInfo.processInfo.systemUptime
            let engine = try await DolphinANEEngine(directory: URL(fileURLWithPath: path, isDirectory: true))
            report["loadSeconds"] = ProcessInfo.processInfo.systemUptime - start
            report["compiledModelsLoaded"] = true
            let plans = await engine.computePlans
            report["computePlan"] = try json(plans)
            let relevant = plans.filter { $0.model != "embedding" }
            let anePass = !relevant.isEmpty && relevant.allSatisfy(\.strictANEPass)
            report["strictANEComputePlanPassed"] = anePass
            if let fixturePath = option("--fixtures") {
                let fixtures = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))
                guard !fixtures.cases.isEmpty else { throw DolphinError.invalid("Empty fixture suite") }
                var results: [[String: Any]] = [], allPassed = true
                for f in fixtures.cases {
                    guard f.max_abs_logit_error.isFinite, f.max_abs_logit_error >= 0, !f.reference_logits.isEmpty,
                          Set(f.reference_logits.map(\.id)).count == f.reference_logits.count,
                          f.reference_logits.allSatisfy({ $0.logit.isFinite }) else { throw DolphinError.invalid("Invalid fixture \(f.name)") }
                    let run = try await engine.generate(tokens: f.input_ids, maxNewTokens: 1, probeIDs: f.reference_logits.map(\.id))
                    let observed = Dictionary(uniqueKeysWithValues: run.prefillLogits.map { ($0.id, $0.logit) })
                    let errors = f.reference_logits.map { abs((observed[$0.id] ?? .infinity) - $0.logit) }
                    let maxError = errors.max() ?? .infinity
                    let pass = run.generatedIDs.first == f.expected_next_token && maxError <= f.max_abs_logit_error
                    allPassed = allPassed && pass
                    results.append(["name": f.name, "passed": pass, "maxAbsLogitError": maxError,
                                    "allowedAbsLogitError": f.max_abs_logit_error, "run": try json(run)])
                }
                report["fixtureResults"] = results; report["numericalParityValidated"] = allPassed
                if !allPassed { status = 3 }
            } else {
                guard let tokenText = option("--tokens"), let n = Int(option("--max-new-tokens") ?? "8")
                else { throw DolphinError.invalid("Supply --tokens or --fixtures; max-new-tokens must be an integer") }
                let pieces = tokenText.split(separator: ",", omittingEmptySubsequences: false)
                let tokens = pieces.compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
                guard tokens.count == pieces.count else { throw DolphinError.invalid("Invalid token list") }
                let generation = try await engine.generate(tokens: tokens, maxNewTokens: n)
                report["generation"] = try json(generation)
            }
            report["runtimeValidated"] = true
            if args.contains("--strict-ane") && !anePass { status = 2 }
        } catch { report["error"] = String(describing: error); status = 1 }
        report["exitStatus"] = status
        do {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            if let output = option("--output") { try data.write(to: URL(fileURLWithPath: output), options: .atomic) }
            FileHandle.standardOutput.write(data); FileHandle.standardOutput.write(Data([10]))
        } catch { FileHandle.standardError.write(Data("Report write failed: \(error)\n".utf8)); status = 1 }
        exit(status)
    }
    static func json<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
}
