# Security Module Guidance

This scope covers `Sources/App/Security/` where platform security integrations live.

- Use public, entitlement-compatible frameworks (e.g., CoreWLAN, NetworkExtension) behind
  `#if canImport` guards so builds remain portable across Apple platforms.
- Model scan results with `Codable` types and document permission requirements plus fallback
  behaviours for unsupported targets.
- Emit structured diagnostics with `OSLog` and keep cross-thread access safe using Sendable
  constructs.
- When introducing new capture utilities, provide integration notes or tests that demonstrate
  researcher workflows on macOS/iOS simulators or devices.
- Avoid placeholder APIs—gracefully degrade with actionable error messages when capabilities
  are unavailable.
