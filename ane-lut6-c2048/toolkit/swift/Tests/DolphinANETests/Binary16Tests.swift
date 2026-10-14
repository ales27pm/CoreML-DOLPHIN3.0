import CoreML
import XCTest
@testable import DolphinANE

final class Binary16Tests: XCTestCase {
    func testKnownFiniteValues() {
        XCTAssertEqual(Binary16.float(from: 0x0000).bitPattern, Float(0).bitPattern)
        XCTAssertEqual(Binary16.float(from: 0x8000).bitPattern, 0x8000_0000)
        XCTAssertEqual(Binary16.float(from: 0x3c00), 1)
        XCTAssertEqual(Binary16.float(from: 0xc000), -2)
        XCTAssertEqual(Binary16.float(from: 0x7bff), 65_504)
    }

    func testSubnormalValues() {
        XCTAssertEqual(Binary16.float(from: 0x0001), Float(sign: .plus, exponent: -24, significand: 1))
        XCTAssertEqual(Binary16.float(from: 0x0400), Float(sign: .plus, exponent: -14, significand: 1))
    }

    func testSpecialValuesAndFiniteCheck() {
        XCTAssertTrue(Binary16.isFinite(0x7bff))
        XCTAssertFalse(Binary16.isFinite(0x7c00))
        XCTAssertFalse(Binary16.isFinite(0x7e00))
        XCTAssertEqual(Binary16.float(from: 0x7c00), .infinity)
        XCTAssertTrue(Binary16.float(from: 0x7e00).isNaN)
    }

    func testMultiArrayWritePreservesBinary16Bits() throws {
        let expected: [UInt16] = [0x0000, 0x3c00, 0xc000, 0x7bff]
        let array = try Binary16.multiArray(expected, shape: [1, expected.count, 1, 1])
        array.withUnsafeBytes { bytes in
            let valueStride = array.strides[1].intValue
            for (index, bits) in expected.enumerated() {
                let elementOffset = index * valueStride
                let byteOffset = elementOffset * MemoryLayout<UInt16>.size
                let observed = UInt16(bytes[byteOffset]) | (UInt16(bytes[byteOffset + 1]) << 8)
                XCTAssertEqual(observed, bits)
            }
        }
    }

    func testStridedMultiArrayReadUsesElementOffsets() throws {
        let expected: [UInt16] = [0x3c00, 0xc000, 0x7bff]
        let array = MLMultiArray(shape: [1, expected.count, 1, 1], dataType: .float16,
                                 strides: [expected.count * 2, 2, 2, 1])
        array.withUnsafeMutableBytes { bytes, strides in
            for (index, bits) in expected.enumerated() {
                let byteOffset = index * strides[1] * MemoryLayout<UInt16>.size
                bytes[byteOffset] = UInt8(truncatingIfNeeded: bits)
                bytes[byteOffset + 1] = UInt8(truncatingIfNeeded: bits >> 8)
            }
        }
        let values = try array.withUnsafeBytes { bytes in
            try expected.indices.map {
                Binary16.float(from: try Binary16.bits(from: bytes, elementOffset: $0 * array.strides[1].intValue))
            }
        }
        XCTAssertEqual(values, [1, -2, 65_504])
    }
}
