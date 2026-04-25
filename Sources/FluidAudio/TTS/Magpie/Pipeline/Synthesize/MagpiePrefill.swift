@preconcurrency import CoreML
import Foundation

/// Prefills the decoder KV cache with the 110-token speaker context.
///
/// Currently always runs the step-by-step path (driving `decoder_step` 110 times
/// per path) so the Swift port works regardless of whether `decoder_prefill.mlmodelc`
/// shipped in the repo. Using the fast `decoder_prefill` model will come as a
/// follow-up optimization once its exact I/O signature is documented.
public struct MagpiePrefill {

    private let logger = AppLogger(category: "MagpiePrefill")
    private let decoderStep: MLModel

    public init(decoderStep: MLModel) {
        self.decoderStep = decoderStep
    }

    public func prefill(
        speakerEmbedding: [Float],
        speakerContextLength: Int,
        dModel: Int,
        encoderOutput: MLMultiArray,
        encoderMask: MLMultiArray,
        cache: MagpieKvCache
    ) throws {
        precondition(speakerEmbedding.count == speakerContextLength * dModel)

        for t in 0..<speakerContextLength {
            let tokenBuffer = try MLMultiArray(
                shape: [1, 1, NSNumber(value: dModel)], dataType: .float32)
            let srcStart = t * dModel
            tokenBuffer.withUnsafeMutableBytes { ptr, strides in
                let base = ptr.bindMemory(to: Float.self).baseAddress!
                for i in 0..<dModel {
                    base[i] = speakerEmbedding[srcStart + i]
                }
                _ = strides
            }

            var inputs: [String: MLMultiArray] = [
                "audio_embed": tokenBuffer,
                "encoder_output": encoderOutput,
                "encoder_mask": encoderMask,
            ]
            cache.addInputs(to: &inputs)

            let provider = try MLDictionaryFeatureProvider(
                dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
            let output = try decoderStep.prediction(from: provider)
            try cache.absorbOutputs(output)
        }
        logger.info("Prefill complete: position = \(cache.position)")
    }

    /// Build the unconditional (CFG) encoder output + mask pair: zero tensor +
    /// mask with only slot 0 unmasked (mirrors NeMo's `prepare_dummy_cond_for_cfg`).
    public static func makeUnconditional(
        encoderOutputShape shape: [NSNumber], maxTextLen: Int
    ) throws -> (encoderOutput: MLMultiArray, encoderMask: MLMultiArray) {
        let encOut = try MLMultiArray(shape: shape, dataType: .float32)
        encOut.zeroFillFloat()
        let mask = try MLMultiArray(
            shape: [1, NSNumber(value: maxTextLen)], dataType: .float32)
        mask.zeroFillFloat()
        mask[[0, 0] as [NSNumber]] = NSNumber(value: 1.0)
        return (encOut, mask)
    }
}

extension MLMultiArray {
    fileprivate func zeroFillFloat() {
        guard dataType == .float32 else {
            for i in 0..<count { self[i] = NSNumber(value: 0.0) }
            return
        }
        memset(dataPointer, 0, count * MemoryLayout<Float>.size)
    }
}
