import Foundation

/// Represents a single benchmark observation captured during Dolphin token streaming.
public struct BenchmarkCSVRow: Equatable, Codable {
    /// Timestamp associated with the benchmark iteration.
    public let timestamp: Date
    /// Achieved tokens per second for the iteration.
    public let tokensPerSecond: Double
    /// Time spent in the model initialisation pass in milliseconds.
    public let initMilliseconds: Double
    /// Average decode latency per generated token in milliseconds.
    public let decodeMillisecondsPerToken: Double
    /// Time spent producing an embedding vector in milliseconds.
    public let embedMilliseconds: Double

    public init(
        timestamp: Date,
        tokensPerSecond: Double,
        initMilliseconds: Double,
        decodeMillisecondsPerToken: Double,
        embedMilliseconds: Double
    ) {
        self.timestamp = timestamp
        self.tokensPerSecond = tokensPerSecond
        self.initMilliseconds = initMilliseconds
        self.decodeMillisecondsPerToken = decodeMillisecondsPerToken
        self.embedMilliseconds = embedMilliseconds
    }
}

/// Serialises benchmark samples to CSV with deterministic formatting suitable for regression tracking.
public struct BenchmarkCSVWriter {
    private let dateFormatter: ISO8601DateFormatter
    private let numberFormatter: NumberFormatter

    public init(timeZone: TimeZone = TimeZone(secondsFromGMT: 0)!) {
        let formatter = ISO8601DateFormatter()
        formatter.timeZone = timeZone
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        self.dateFormatter = formatter

        let numberFormatter = NumberFormatter()
        numberFormatter.locale = Locale(identifier: "en_US_POSIX")
        numberFormatter.minimumFractionDigits = 2
        numberFormatter.maximumFractionDigits = 4
        numberFormatter.minimumIntegerDigits = 1
        self.numberFormatter = numberFormatter
    }

    /// Produces a CSV string using a stable column ordering and POSIX locale.
    /// - Parameter samples: Benchmark samples ordered chronologically.
    /// - Returns: A CSV string containing a header and one line per sample.
    public func render(samples: [BenchmarkCSVRow]) -> String {
        var lines = ["timestamp,tokens_per_second,init_ms,decode_ms_per_token,embed_ms"]
        lines.reserveCapacity(samples.count + 1)
        for sample in samples {
            let timestamp = dateFormatter.string(from: sample.timestamp)
            let tokensPerSecond = numberFormatter.string(from: sample.tokensPerSecond as NSNumber) ?? "0"
            let initMs = numberFormatter.string(from: sample.initMilliseconds as NSNumber) ?? "0"
            let decodeMs = numberFormatter.string(from: sample.decodeMillisecondsPerToken as NSNumber) ?? "0"
            let embedMs = numberFormatter.string(from: sample.embedMilliseconds as NSNumber) ?? "0"
            lines.append("\(timestamp),\(tokensPerSecond),\(initMs),\(decodeMs),\(embedMs)")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Writes benchmark samples to disk, ensuring the parent directory exists.
    /// - Parameters:
    ///   - samples: The samples to serialise.
    ///   - url: Destination filesystem URL (file://) for the CSV payload.
    public func write(samples: [BenchmarkCSVRow], to url: URL) throws {
        let directoryURL = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let csv = render(samples: samples)
        try csv.write(to: url, atomically: true, encoding: .utf8)
    }
}

/// Validates benchmark throughput to guard against regressions.
public enum ThroughputRegressor {
    /// Summary statistics emitted after validation.
    public struct Summary: Equatable {
        public let minimumObserved: Double
        public let averageTokensPerSecond: Double
    }

    public struct ValidationError: Error, LocalizedError {
        public let message: String

        public var errorDescription: String? { message }
    }

    /// Ensures the provided benchmark samples satisfy the minimum throughput requirement.
    /// - Parameters:
    ///   - samples: Samples to validate (must be non-empty).
    ///   - minimumRate: Minimum acceptable tokens per second for each sample.
    /// - Returns: Summary statistics describing the dataset.
    @discardableResult
    public static func validate(_ samples: [BenchmarkCSVRow], minimumRate: Double) throws -> Summary {
        guard !samples.isEmpty else {
            throw ValidationError(message: "Benchmark sample set cannot be empty.")
        }
        guard minimumRate > 0 else {
            throw ValidationError(message: "Minimum throughput requirement must be positive.")
        }

        var total = 0.0
        var minimumObserved = Double.greatestFiniteMagnitude
        for (index, sample) in samples.enumerated() {
            let rate = sample.tokensPerSecond
            if rate < minimumRate {
                let formattedRate = String(format: "%.2f", rate)
                throw ValidationError(
                    message: "Iteration \(index) produced \(formattedRate) tok/s below the minimum of \(minimumRate)."
                )
            }
            minimumObserved = min(minimumObserved, rate)
            total += rate
        }
        let average = total / Double(samples.count)
        return Summary(minimumObserved: minimumObserved, averageTokensPerSecond: average)
    }
}
