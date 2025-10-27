//
//  WiFiScanner.swift
//  CoreML-DOLPHIN3.0
//
//  Created as part of Task 38 – iOS Wi-Fi Scanner Module.
//

import Foundation
import OSLog

#if canImport(CoreWLAN)
import CoreWLAN
#endif

/// Describes a Wi-Fi network discovered during a scan.
public struct WiFiNetwork: Codable, Hashable, Sendable {
    /// Service set identifier broadcast by the access point. Hidden SSIDs are
    /// surfaced as "<hidden>" to distinguish them from empty values.
    public let ssid: String

    /// Basic service set identifier used to uniquely identify the access point.
    public let bssid: String?

    /// Received signal strength indicator measured in dBm.
    public let rssi: Int

    /// Noise floor reported by the interface in dBm when available.
    public let noise: Int?

    /// IEEE 802.11 channel number reported by the radio.
    public let channel: Int?

    /// Human-readable band description (e.g., 2.4 GHz, 5 GHz).
    public let channelBand: String?

    /// Channel width representation (e.g., 20 MHz).
    public let channelWidth: String?

    /// Security capabilities advertised by the network.
    public let security: [String]

    /// Indicates whether the network hides its SSID.
    public let isHidden: Bool

    /// Timestamp recorded when the scan results were processed.
    public let lastSeen: Date

    /// Country code advertised by the network if exposed by the driver.
    public let countryCode: String?

    public init(
        ssid: String,
        bssid: String?,
        rssi: Int,
        noise: Int?,
        channel: Int?,
        channelBand: String?,
        channelWidth: String?,
        security: [String],
        isHidden: Bool,
        lastSeen: Date,
        countryCode: String?
    ) {
        self.ssid = ssid
        self.bssid = bssid
        self.rssi = rssi
        self.noise = noise
        self.channel = channel
        self.channelBand = channelBand
        self.channelWidth = channelWidth
        self.security = security
        self.isHidden = isHidden
        self.lastSeen = lastSeen
        self.countryCode = countryCode
    }
}

/// Errors thrown by ``WiFiScanner``.
public enum WiFiScannerError: LocalizedError, Sendable {
    case interfaceUnavailable(String?)
    case platformNotSupported
    case scanFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .interfaceUnavailable(name):
            if let name {
                return "No Wi-Fi interface available for name \(name)."
            }
            return "No active Wi-Fi interface available."
        case .platformNotSupported:
            return "Wi-Fi scanning is not supported on this platform."
        case let .scanFailed(reason):
            return "Failed to scan Wi-Fi networks: \(reason)"
        }
    }
}

/// Production-ready Wi-Fi scanner that supports macOS and gracefully degrades on
/// unsupported platforms.
public final class WiFiScanner: @unchecked Sendable {
    private let logger = Logger(subsystem: "com.securityresearch.wifi", category: "scanner")

    /// Creates a new scanner instance.
    public init() {}

    /// Perform a Wi-Fi scan and return discovered networks ordered by signal strength.
    /// - Parameters:
    ///   - interfaceName: Optional BSD interface name (for example, ``en0``). When omitted the
    ///     system default interface is used.
    ///   - includeHiddenNetworks: When `true`, hidden SSIDs are represented in the result set
    ///     with the placeholder "<hidden>". When `false`, hidden networks are omitted entirely.
    /// - Throws: ``WiFiScannerError`` describing why the scan could not be completed.
    /// - Returns: Array of ``WiFiNetwork`` objects sorted by descending RSSI.
    public func scan(
        interfaceName: String? = nil,
        includeHiddenNetworks: Bool = false
    ) throws -> [WiFiNetwork] {
#if canImport(CoreWLAN)
        guard let client = CWWiFiClient.shared() else {
            logger.error("Unable to obtain shared CoreWLAN client")
            throw WiFiScannerError.interfaceUnavailable(interfaceName)
        }

        let interface: CWInterface?
        if let interfaceName {
            interface = client.interface(withName: interfaceName)
        } else {
            interface = client.interface()
        }

        guard let activeInterface = interface else {
            logger.error("Requested Wi-Fi interface not available")
            throw WiFiScannerError.interfaceUnavailable(interfaceName)
        }

        do {
            let networks = try activeInterface.scanForNetworks(withSSID: nil)
            let now = Date()
            let results = networks.compactMap { network -> WiFiNetwork? in
                let converted = WiFiNetwork(coreWLANNetwork: network, timestamp: now)
                if converted.isHidden && !includeHiddenNetworks {
                    return nil
                }
                return converted
            }
            logger.info("Discovered \(results.count, privacy: .public) Wi-Fi networks")
            return results.sorted { lhs, rhs in
                if lhs.rssi == rhs.rssi {
                    return lhs.ssid < rhs.ssid
                }
                return lhs.rssi > rhs.rssi
            }
        } catch {
            logger.error("Wi-Fi scan failed: \(error.localizedDescription, privacy: .public)")
            throw WiFiScannerError.scanFailed(error.localizedDescription)
        }
#else
        logger.error("Wi-Fi scanning attempted on unsupported platform")
        throw WiFiScannerError.platformNotSupported
#endif
    }

    /// Perform a scan and export the results to disk for offline analysis.
    /// - Parameters:
    ///   - destination: File URL where JSON data will be written.
    ///   - interfaceName: Optional BSD interface name to scope the scan to.
    ///   - includeHiddenNetworks: Controls whether hidden SSIDs should appear in the export.
    ///   - encoder: Custom JSON encoder configuration. Defaults to an encoder configured for ISO8601 timestamps.
    /// - Returns: Tuple containing the serialized JSON payload and the list of ``WiFiNetwork``
    ///   objects for in-memory processing.
    @discardableResult
    public func scanAndWrite(
        to destination: URL,
        interfaceName: String? = nil,
        includeHiddenNetworks: Bool = false,
        encoder: JSONEncoder = WiFiScanner.makeEncoder()
    ) throws -> (data: Data, networks: [WiFiNetwork]) {
        let networks = try scan(
            interfaceName: interfaceName,
            includeHiddenNetworks: includeHiddenNetworks
        )
        let payload = try encoder.encode(networks)
        try payload.write(to: destination, options: [.atomic])
        logger.info("Persisted Wi-Fi scan results to \(destination.path, privacy: .public)")
        return (payload, networks)
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

#if canImport(CoreWLAN)
private extension WiFiNetwork {
    init(coreWLANNetwork network: CWNetwork, timestamp: Date) {
        let ssidValue = network.ssid ?? "<hidden>"
        let channel = network.wlanChannel
        self.init(
            ssid: ssidValue,
            bssid: network.bssid,
            rssi: network.rssiValue,
            noise: network.noiseMeasurement,
            channel: channel?.channelNumber,
            channelBand: channel?.bandDescription,
            channelWidth: channel?.widthDescription,
            security: network.securityDescriptions(),
            isHidden: network.ssid == nil,
            lastSeen: timestamp,
            countryCode: network.countryCode
        )
    }
}

private extension CWChannel {
    var bandDescription: String {
        switch channelBand {
        case .band2GHz:
            return "2.4 GHz"
        case .band5GHz:
            return "5 GHz"
        case .band6GHz:
            return "6 GHz"
        @unknown default:
            return "Unknown"
        }
    }

    var widthDescription: String {
        switch channelWidth {
        case .width20MHz:
            return "20 MHz"
        case .width40MHz:
            return "40 MHz"
        case .width80MHz:
            return "80 MHz"
        case .width160MHz:
            return "160 MHz"
        case .width80MHzUpper:
            return "80+80 MHz"
        @unknown default:
            return "Unknown"
        }
    }
}

private extension CWNetwork {
    func securityDescriptions() -> [String] {
        var descriptors: [String] = []
        if supportsSecurity(.none) {
            descriptors.append("Open")
        }
        if supportsSecurity(.wep) {
            descriptors.append("WEP")
        }
        if supportsSecurity(.wpaPersonal) {
            descriptors.append("WPA Personal")
        }
        if supportsSecurity(.wpa2Personal) {
            descriptors.append("WPA2 Personal")
        }
        if supportsSecurity(.wpa3Personal) {
            descriptors.append("WPA3 Personal")
        }
        if supportsSecurity(.wpaEnterprise) {
            descriptors.append("WPA Enterprise")
        }
        if supportsSecurity(.wpa2Enterprise) {
            descriptors.append("WPA2 Enterprise")
        }
        if supportsSecurity(.wpa3Enterprise) {
            descriptors.append("WPA3 Enterprise")
        }
        if descriptors.isEmpty {
            descriptors.append("Unknown")
        }
        return descriptors
    }
}
#endif
