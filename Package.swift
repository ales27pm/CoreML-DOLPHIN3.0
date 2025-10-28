// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DolphinCoreMLRuntime",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
        .visionOS(.v2)
    ],
    products: [
        .library(
            name: "DolphinCoreMLRuntime",
            targets: ["DolphinCoreMLRuntime"]
        ),
        .library(
            name: "DolphinBenchmarkHarness",
            targets: ["DolphinBenchmarkHarness"]
        )
    ],
    targets: [
        .target(
            name: "DolphinCoreMLRuntime",
            path: "Sources/App/LLM",
            exclude: ["AGENTS.md"]
        ),
        .target(
            name: "DolphinBenchmarkHarness",
            dependencies: ["DolphinCoreMLRuntime"],
            path: "Sources/App/Bench",
            exclude: ["AGENTS.md"]
        ),
        .testTarget(
            name: "DolphinCoreMLRuntimeTests",
            dependencies: ["DolphinCoreMLRuntime"],
            path: "Tests/DolphinCoreMLRuntimeTests"
        )
    ]
)
