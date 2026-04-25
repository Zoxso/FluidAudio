@preconcurrency import CoreML
import Foundation

/// Holds one path's KV cache state for the 12-layer decoder_step model.
///
/// Each layer has:
///   - `cache{i}`   : `MLMultiArray` shaped `[2, 1, 512, numHeads, headDim]` fp32
///   - `position{i}`: `MLMultiArray` shaped `[1]` fp32 (scalar index into the cache)
///
/// After each `decoder_step` forward pass the model returns new cache + position
/// buffers under output names that do not match the input names (scatter rewrite).
/// The exact output key names are hard-coded in
/// `mobius/.../generate_coreml.py` (`DECODER_CACHE_OUT_KEYS`, `DECODER_POSITION_KEYS`);
/// this Swift port mirrors that list and should be regenerated if the Python
/// compile pipeline changes.
public final class MagpieKvCache {

    public static let cacheOutputKeys: [String] = [
        "new_cache_1", "new_cache_3", "new_cache_5", "new_cache_7",
        "new_cache_9", "new_cache_11", "new_cache_13", "new_cache_15",
        "new_cache_17", "new_cache_19", "new_cache_21", "new_cache",
    ]

    public static let positionOutputKeys: [String] = [
        "var_169", "var_346", "var_523", "var_700",
        "var_877", "var_1054", "var_1231", "var_1408",
        "var_1585", "var_1762", "var_1939", "var_2116",
    ]

    public static let decoderHiddenKey = "input"
    public static let decoderLogitsKey = "var_2201"

    public private(set) var caches: [MLMultiArray]
    public private(set) var positions: [MLMultiArray]

    public let numLayers: Int
    public let maxCacheLength: Int
    public let numHeads: Int
    public let headDim: Int

    public init(numLayers: Int, maxCacheLength: Int, numHeads: Int, headDim: Int) throws {
        self.numLayers = numLayers
        self.maxCacheLength = maxCacheLength
        self.numHeads = numHeads
        self.headDim = headDim
        self.caches = try (0..<numLayers).map { _ -> MLMultiArray in
            let shape: [NSNumber] = [
                2, 1, NSNumber(value: maxCacheLength),
                NSNumber(value: numHeads),
                NSNumber(value: headDim),
            ]
            let arr = try MLMultiArray(shape: shape, dataType: .float32)
            arr.zeroFill()
            return arr
        }
        self.positions = try (0..<numLayers).map { _ -> MLMultiArray in
            let arr = try MLMultiArray(shape: [1], dataType: .float32)
            arr[0] = NSNumber(value: 0.0)
            return arr
        }
    }

    /// Populate `inputs` with `cache{i}` + `position{i}` keys.
    public func addInputs(to inputs: inout [String: MLMultiArray]) {
        for i in 0..<numLayers {
            inputs["cache\(i)"] = caches[i]
            inputs["position\(i)"] = positions[i]
        }
    }

    /// Consume the output dictionary of one `decoder_step.predict()` call and
    /// rotate the cache / position buffers in-place.
    public func absorbOutputs(_ output: MLFeatureProvider) throws {
        for i in 0..<numLayers {
            guard let newCache = output.featureValue(for: Self.cacheOutputKeys[i])?.multiArrayValue else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_step",
                    underlying: "missing cache output key \(Self.cacheOutputKeys[i])")
            }
            guard let newPos = output.featureValue(for: Self.positionOutputKeys[i])?.multiArrayValue else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_step",
                    underlying: "missing position output key \(Self.positionOutputKeys[i])")
            }
            caches[i] = newCache
            positions[i] = newPos
        }
    }

    /// Current decoder position as read from layer 0's position tensor.
    public var position: Int {
        guard numLayers > 0 else { return 0 }
        return Int(positions[0][0].floatValue)
    }
}

// MARK: - Helpers

extension MLMultiArray {
    /// Zero-fill an fp32 `MLMultiArray` fast (uses `memset` under the hood).
    fileprivate func zeroFill() {
        guard dataType == .float32 else {
            for i in 0..<count { self[i] = NSNumber(value: 0.0) }
            return
        }
        let bytes = count * MemoryLayout<Float>.size
        memset(dataPointer, 0, bytes)
    }
}
