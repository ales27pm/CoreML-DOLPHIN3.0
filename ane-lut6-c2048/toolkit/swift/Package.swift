// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DolphinANE",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [.library(name: "DolphinANE", targets: ["DolphinANE"]),
               .executable(name: "dolphin-ane-validate", targets: ["Validate"])],
    targets: [.target(name: "DolphinANE"),
              .executableTarget(name: "Validate", dependencies: ["DolphinANE"]),
              .testTarget(name: "DolphinANETests", dependencies: ["DolphinANE"])],
    swiftLanguageModes: [.v5]
)
