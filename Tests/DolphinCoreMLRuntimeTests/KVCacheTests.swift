import CoreML
import XCTest
@testable import DolphinCoreMLRuntime

final class KVCacheTests: XCTestCase {
    private func makeCacheTensor() throws -> MLMultiArray {
        try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float32)
    }

    func testKVCacheInitialisesWhenLayerCountsMatch() throws {
        let keys = [try makeCacheTensor(), try makeCacheTensor()]
        let values = [try makeCacheTensor(), try makeCacheTensor()]
        let cache = try DolphinCoreML.KVCache(keys: keys, values: values, expectedLayers: 2)
        XCTAssertEqual(cache.keys.count, 2)
        XCTAssertEqual(cache.values.count, 2)
    }

    func testKVCacheThrowsWhenLayerCountsMismatch() throws {
        let keys = [try makeCacheTensor()]
        let values = [try makeCacheTensor(), try makeCacheTensor()]
        XCTAssertThrowsError(try DolphinCoreML.KVCache(keys: keys, values: values, expectedLayers: 2))
    }

    func testKVCacheUpdateReplacesStorage() throws {
        let initialK = try makeCacheTensor()
        let initialV = try makeCacheTensor()
        var cache = try DolphinCoreML.KVCache(keys: [initialK], values: [initialV], expectedLayers: 1)

        let logits = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        let lastHidden = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        let updatedK = try makeCacheTensor()
        let updatedV = try makeCacheTensor()
        let output = DolphinCoreML.DecodeOutput(logits: logits,
                                               lastHidden: lastHidden,
                                               outK: [updatedK],
                                               outV: [updatedV])

        cache.update(with: output)

        XCTAssertTrue(cache.keys.first === updatedK)
        XCTAssertTrue(cache.values.first === updatedV)
    }
}
