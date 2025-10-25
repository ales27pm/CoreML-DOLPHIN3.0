import Foundation

@_silgen_name("rust_echo")
private func rustEchoBytes(_ pointer: UnsafePointer<UInt8>?, _ length: Int) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rust_echo_c")
private func rustEchoCString(_ pointer: UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rust_free")
private func rustFree(_ pointer: UnsafeMutablePointer<CChar>?)

/// Swift entry point exported for Rust callers.
@_cdecl("swift_echo")
public func swift_echo(_ pointer: UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>? {
    guard let pointer else { return nil }
    let string = String(cString: pointer)
    return strdup(string)
}

/// Errors surfaced when the Rust bridge reports failures.
public enum RustBridgeError: Error {
    /// Returned when the Rust layer produced a `NULL` pointer.
    case nullPointer
    /// Returned when the resulting string cannot be decoded as UTF-8.
    case invalidUTF8
}

/// Convenience helpers that wrap the Rust bridge in Swift-native types.
public enum RustBridge {
    /// Forwards a Swift `String` through Rust's `rust_echo_c` implementation.
    /// - Parameter text: The input text that should be echoed by Rust.
    /// - Returns: The echoed string value.
    public static func echo(_ text: String) throws -> String {
        return try text.withCString { basePointer -> String in
            guard let response = rustEchoCString(basePointer) else {
                throw RustBridgeError.nullPointer
            }
            defer { rustFree(response) }
            guard let swiftString = String(validatingUTF8: response) else {
                throw RustBridgeError.invalidUTF8
            }
            return swiftString
        }
    }

    /// Invokes `rust_echo` with raw bytes to support payloads sourced from `Data`.
    /// - Parameter data: The binary payload to echo.
    /// - Returns: The echoed payload encoded as UTF-8 data.
    public static func echoBytes(_ data: Data) throws -> Data {
        if data.isEmpty {
            return Data()
        }

        return try data.withUnsafeBytes { rawBuffer -> Data in
            let pointer = rawBuffer.bindMemory(to: UInt8.self).baseAddress
            guard let pointer else {
                throw RustBridgeError.nullPointer
            }
            guard let response = rustEchoBytes(pointer, data.count) else {
                throw RustBridgeError.nullPointer
            }
            defer { rustFree(response) }
            guard let string = String(validatingUTF8: response) else {
                throw RustBridgeError.invalidUTF8
            }
            guard let echoedData = string.data(using: .utf8) else {
                throw RustBridgeError.invalidUTF8
            }
            return echoedData
        }
    }
}
