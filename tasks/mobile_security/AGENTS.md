# Mobile Security Task Guidance

This scope governs `tasks/mobile_security/`, including Android VPN capture utilities.

- Implement Kotlin modules with coroutine-aware services, structured logging, and testability via
  `./gradlew test`. Provide README snippets covering sideloading or emulator validation steps.
- Prefer modern AndroidX APIs (VpnService, WorkManager, Notification channels) and document
  required permissions or entitlements.
- Persist packet captures using stable formats such as PCAP with deterministic metadata writers.
- Include integration tests or scripted validation that researchers can run locally without
  proprietary infrastructure.
- Mirror functionality with corresponding iOS tooling when applicable, or document planned parity.
