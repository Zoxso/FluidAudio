import CoreML
import XCTest

@testable import FluidAudio

final class MagpieKvCacheTests: XCTestCase {

    func testInitialShapeAndZeroPosition() throws {
        let cache = try MagpieKvCache(
            numLayers: MagpieConstants.numDecoderLayers,
            maxCacheLength: MagpieConstants.maxCacheLength,
            numHeads: MagpieConstants.numHeads,
            headDim: MagpieConstants.headDim)

        XCTAssertEqual(cache.caches.count, MagpieConstants.numDecoderLayers)
        XCTAssertEqual(cache.positions.count, MagpieConstants.numDecoderLayers)
        XCTAssertEqual(cache.position, 0)

        let expectedShape: [NSNumber] = [
            2, 1,
            NSNumber(value: MagpieConstants.maxCacheLength),
            NSNumber(value: MagpieConstants.numHeads),
            NSNumber(value: MagpieConstants.headDim),
        ]
        XCTAssertEqual(cache.caches[0].shape, expectedShape)
        XCTAssertEqual(cache.positions[0].shape, [1])
    }

    func testAddInputsProvidesAllLayerKeys() throws {
        let cache = try MagpieKvCache(
            numLayers: 3, maxCacheLength: 32, numHeads: 4, headDim: 8)
        var inputs: [String: MLMultiArray] = [:]
        cache.addInputs(to: &inputs)
        XCTAssertEqual(inputs.count, 6)
        for i in 0..<3 {
            XCTAssertNotNil(inputs["cache\(i)"])
            XCTAssertNotNil(inputs["position\(i)"])
        }
    }

    func testStaticOutputKeyCountMatchesLayers() {
        XCTAssertEqual(
            MagpieKvCache.cacheOutputKeys.count, MagpieConstants.numDecoderLayers,
            "cacheOutputKeys must match numDecoderLayers — regenerate list if the exporter changes.")
        XCTAssertEqual(
            MagpieKvCache.positionOutputKeys.count, MagpieConstants.numDecoderLayers)
    }
}
