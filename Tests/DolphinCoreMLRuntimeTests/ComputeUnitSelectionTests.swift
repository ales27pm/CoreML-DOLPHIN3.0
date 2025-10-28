import XCTest
import CoreML
@testable import DolphinCoreMLRuntime

final class ComputeUnitSelectionTests: XCTestCase {
    func testAllMapsToAllComputeUnits() {
        XCTAssertEqual(ComputeUnitSelection.all.coreML, .all)
    }

    func testCpuAndGpuMapsToCpuAndGpuComputeUnits() {
        XCTAssertEqual(ComputeUnitSelection.cpuAndGPU.coreML, .cpuAndGPU)
    }

    func testCpuOnlyMapsToCpuOnlyComputeUnits() {
        XCTAssertEqual(ComputeUnitSelection.cpuOnly.coreML, .cpuOnly)
    }
}
