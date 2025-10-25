# Swift Sources Guidance

This scope applies to all Swift code under `Sources/`.

- Target Swift 5.9+ with compatibility for iOS 18 and macOS 15 as reflected in the Core ML
  packages produced by `dolphin2coreml_full.py`.
- Keep modules free of placeholder implementations—mirror the production utilities already in
  place (Core ML wrappers, benchmark harnesses) and ensure public APIs expose thorough
  documentation comments when behaviour is non-obvious.
- Maintain Core ML imports and avoid introducing platform-specific APIs without `#if canImport`
  guards.
- Prefer deterministic performance measurements and guard against unsafe pointer usage by
  validating tensor ranks and data types, as demonstrated in existing files.
- When introducing new Swift sources, accompany them with integration notes in the README or
  inline doc comments so app developers understand how to wire them into Xcode projects.
