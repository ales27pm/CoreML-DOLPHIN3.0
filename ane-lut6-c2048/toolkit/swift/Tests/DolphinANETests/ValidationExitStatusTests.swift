import XCTest
@testable import DolphinANE

final class ValidationExitStatusTests: XCTestCase {
    func testExitStatusTruthTableAndFailurePriority() {
        XCTAssertEqual(ValidationExitStatus.resolve(runtimeFailed: false, numericalParityFailed: false,
                                                     strictANERequested: false, anePassed: false), 0)
        XCTAssertEqual(ValidationExitStatus.resolve(runtimeFailed: false, numericalParityFailed: false,
                                                     strictANERequested: true, anePassed: false), 2)
        XCTAssertEqual(ValidationExitStatus.resolve(runtimeFailed: false, numericalParityFailed: true,
                                                     strictANERequested: true, anePassed: false), 3)
        XCTAssertEqual(ValidationExitStatus.resolve(runtimeFailed: true, numericalParityFailed: true,
                                                     strictANERequested: true, anePassed: false), 1)
        XCTAssertEqual(ValidationExitStatus.resolve(runtimeFailed: false, numericalParityFailed: false,
                                                     strictANERequested: true, anePassed: true), 0)
    }
}
