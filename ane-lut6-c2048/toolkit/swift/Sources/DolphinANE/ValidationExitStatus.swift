public enum ValidationExitStatus {
    /// Stable CLI contract. Runtime failures take precedence, followed by
    /// numerical parity, then the optional strict-ANE policy.
    public static func resolve(runtimeFailed: Bool, numericalParityFailed: Bool,
                               strictANERequested: Bool, anePassed: Bool) -> Int32 {
        if runtimeFailed { return 1 }
        if numericalParityFailed { return 3 }
        if strictANERequested && !anePassed { return 2 }
        return 0
    }
}
