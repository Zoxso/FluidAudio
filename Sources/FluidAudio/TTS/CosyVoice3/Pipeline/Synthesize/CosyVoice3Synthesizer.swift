@preconcurrency import CoreML
import Foundation

/// Top-level synthesizer orchestrating prefill → decode loop → Flow → HiFT.
///
/// Mirrors `verify/test_coreml_e2e_fp16.py::main()` in Python. Each stage is
/// implemented as a method on this type, keeping the state (KV cache, running
/// decoded list) local to a single synthesis call.
public final class CosyVoice3Synthesizer: @unchecked Sendable {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "CosyVoice3Synthesizer")

    private let models: CosyVoice3Models
    private let embeddings: CosyVoice3SpeechEmbeddings

    public init(models: CosyVoice3Models, embeddings: CosyVoice3SpeechEmbeddings) {
        self.models = models
        self.embeddings = embeddings
    }

    /// Entry point for the Phase 1 parity harness.
    public func synthesize(
        fixture: CosyVoice3FrontendFixture,
        options: CosyVoice3ParityOptions
    ) async throws -> CosyVoice3SynthesisResult {

        let nPrompt = fixture.promptSpeechIds.count
        let roomForNew = CosyVoice3Constants.flowTotalTokens - nPrompt
        guard roomForNew > 0 else {
            throw CosyVoice3Error.sequenceTooLong(nPrompt)
        }
        let maxNew: Int = {
            if let cap = options.maxNewTokens, cap > 0 { return min(cap, roomForNew) }
            return roomForNew
        }()

        // Sampler. Parity harness seeds the Python-recorded decode stream.
        let sampler = CosyVoice3RasSampler(seed: options.seed)
        if options.replayDecodedTokens {
            sampler.seedTokens(fixture.decodedTokens)
        }

        // 1) Prefill
        let (prefillLogits, initialKvK, initialKvV) = try await runPrefill(fixture: fixture)
        var kvK = initialKvK
        var kvV = initialKvV

        // First token from prefill tail logits.
        var decoded: [Int32] = []
        let firstLogits = sliceLastStepLogits(
            from: prefillLogits,
            tPre: fixture.tPre,
            vocab: CosyVoice3Constants.speechVocab)
        var topId = sampler.sample(logits: firstLogits, decodedSoFar: decoded)
        if CosyVoice3Constants.stopRange.contains(topId) {
            logger.info("First token \(topId) is a stop token; no speech generated")
        } else {
            decoded.append(topId)
        }

        // 2) Decode loop
        var curLen = fixture.tPre
        for step in 1..<maxNew {
            let nextEmb = try embeddings.embedding(tokenId: topId)
            let (logits, newKvK, newKvV) = try await runDecode(
                inputsEmbeds: nextEmb,
                kvK: kvK,
                kvV: kvV,
                curLen: Int32(curLen))
            kvK = newKvK
            kvV = newKvV
            topId = sampler.sample(logits: logits, decodedSoFar: decoded)
            curLen += 1
            if CosyVoice3Constants.stopRange.contains(topId) {
                logger.info("EOS at step \(step) (token=\(topId))")
                break
            }
            decoded.append(topId)
        }
        guard !decoded.isEmpty else {
            throw CosyVoice3Error.predictionFailed("LLM produced no speech tokens")
        }

        // 3) Flow
        let nNew = decoded.count
        let mel = try await runFlow(
            promptSpeechIds: fixture.promptSpeechIds,
            decodedTokens: decoded,
            promptMel: fixture.promptMel,
            promptMelFrames: fixture.promptMelFrames,
            spkEmbedding: fixture.spkEmbedding)

        // 4) Slice mel to new portion + HiFT
        let numPromptMel = mel.numPromptMel
        let newMelStart = numPromptMel
        let newMelFrames = nNew * CosyVoice3Constants.tokenMelRatio
        let audio = try await runHiFT(
            fullMel: mel.mel,
            newMelStart: newMelStart,
            newMelFrames: newMelFrames)

        return CosyVoice3SynthesisResult(
            samples: audio,
            sampleRate: CosyVoice3Constants.sampleRate,
            generatedTokenCount: nNew,
            decodedTokens: decoded)
    }

    // MARK: - Stages

    private func runPrefill(
        fixture: CosyVoice3FrontendFixture
    ) async throws -> (logits: MLMultiArray, kvK: MLMultiArray, kvV: MLMultiArray) {
        guard fixture.tPre <= CosyVoice3Constants.prefillLength else {
            throw CosyVoice3Error.prefillTooLong(fixture.tPre)
        }
        // Pad lm_input_embeds from [1, tPre, 896] to [1, 256, 896].
        // Strides may be non-compact (e.g. [T*D_padded, D_padded, 1]).
        let embeds = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: CosyVoice3Constants.prefillLength),
                NSNumber(value: CosyVoice3Constants.embedDim),
            ],
            dataType: .float32)
        let embedDim = CosyVoice3Constants.embedDim
        let embedsStrides = embeds.strides.map { $0.intValue }
        let dst = embeds.dataPointer.bindMemory(to: Float.self, capacity: embeds.count)
        let physicalCount = embedsStrides[0] * embeds.shape[0].intValue
        dst.initialize(repeating: 0, count: physicalCount)
        for t in 0..<fixture.tPre {
            for d in 0..<embedDim {
                let srcIdx = t * embedDim + d
                let dstOff = t * embedsStrides[1] + d * embedsStrides[2]
                dst[dstOff] = fixture.lmInputEmbeds[srcIdx]
            }
        }
        let inputLen = try MLMultiArray(shape: [1], dataType: .int32)
        inputLen[0] = NSNumber(value: Int32(fixture.tPre))

        let features: [String: Any] = [
            "inputs_embeds": embeds,
            "input_len": inputLen,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await models.prefill.compatPrediction(
            from: provider, options: MLPredictionOptions())

        guard
            let logits = output.featureValue(for: "speech_logits")?.multiArrayValue,
            let kvK = output.featureValue(for: "kv_k")?.multiArrayValue,
            let kvV = output.featureValue(for: "kv_v")?.multiArrayValue
        else {
            throw CosyVoice3Error.predictionFailed("prefill: missing outputs")
        }
        return (logits, kvK, kvV)
    }

    private func runDecode(
        inputsEmbeds: MLMultiArray,
        kvK: MLMultiArray,
        kvV: MLMultiArray,
        curLen: Int32
    ) async throws -> (logits: [Float], kvK: MLMultiArray, kvV: MLMultiArray) {
        let curLenArr = try MLMultiArray(shape: [1], dataType: .int32)
        curLenArr[0] = NSNumber(value: curLen)

        let features: [String: Any] = [
            "inputs_embeds": inputsEmbeds,
            "kv_k": kvK,
            "kv_v": kvV,
            "cur_len": curLenArr,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await models.decode.compatPrediction(
            from: provider, options: MLPredictionOptions())

        guard
            let logitsArr = output.featureValue(for: "speech_logits")?.multiArrayValue,
            let newKvK = output.featureValue(for: "kv_k_out")?.multiArrayValue,
            let newKvV = output.featureValue(for: "kv_v_out")?.multiArrayValue
        else {
            throw CosyVoice3Error.predictionFailed("decode: missing outputs")
        }
        // logits shape = [1, 1, 6761] fp32; strides may be non-compact.
        let count = CosyVoice3Constants.speechVocab
        var logits = [Float](repeating: 0, count: count)
        let strides = logitsArr.strides.map { $0.intValue }
        let vocabStride = strides.last ?? 1
        let base = logitsArr.dataPointer.bindMemory(to: Float.self, capacity: logitsArr.count)
        for i in 0..<count { logits[i] = base[i * vocabStride] }
        return (logits, newKvK, newKvV)
    }

    private func runFlow(
        promptSpeechIds: [Int32],
        decodedTokens: [Int32],
        promptMel: [Float],
        promptMelFrames: Int,
        spkEmbedding: [Float]
    ) async throws -> (mel: MLMultiArray, numPromptMel: Int) {
        let N = CosyVoice3Constants.flowTotalTokens
        let nPrompt = promptSpeechIds.count
        let nNew = decodedTokens.count
        let nTotal = nPrompt + nNew
        guard nTotal <= N else {
            throw CosyVoice3Error.sequenceTooLong(nTotal)
        }
        // token_total: [1, 250] int32, zero-padded. Respect strides.
        let tokenTotal = try MLMultiArray(
            shape: [1, NSNumber(value: N)],
            dataType: .int32)
        let ttStrides = tokenTotal.strides.map { $0.intValue }
        let ttPtr = tokenTotal.dataPointer.bindMemory(to: Int32.self, capacity: tokenTotal.count)
        let ttPhysical = ttStrides[0] * tokenTotal.shape[0].intValue
        ttPtr.initialize(repeating: 0, count: ttPhysical)
        for i in 0..<nPrompt { ttPtr[i * ttStrides[1]] = promptSpeechIds[i] }
        for i in 0..<nNew { ttPtr[(nPrompt + i) * ttStrides[1]] = decodedTokens[i] }

        // num_prompt_tokens: [1] int32
        let numPromptTokens = try MLMultiArray(shape: [1], dataType: .int32)
        numPromptTokens[0] = NSNumber(value: Int32(nPrompt))

        // prompt_feat: [1, 500, 80] fp32, zero-padded along axis 1. Respect strides.
        let hiftFrames = CosyVoice3Constants.hiftMaxFrames
        let melBins = CosyVoice3Constants.melBins
        let promptFeat = try MLMultiArray(
            shape: [
                1, NSNumber(value: hiftFrames), NSNumber(value: melBins),
            ],
            dataType: .float32)
        let pfStrides = promptFeat.strides.map { $0.intValue }
        let pfPtr = promptFeat.dataPointer.bindMemory(to: Float.self, capacity: promptFeat.count)
        let pfPhysical = pfStrides[0] * promptFeat.shape[0].intValue
        pfPtr.initialize(repeating: 0, count: pfPhysical)
        let copyFrames = min(promptMelFrames, hiftFrames)
        for f in 0..<copyFrames {
            for b in 0..<melBins {
                let srcIdx = f * melBins + b
                let dstOff = f * pfStrides[1] + b * pfStrides[2]
                pfPtr[dstOff] = promptMel[srcIdx]
            }
        }

        // embedding: [1, 192] fp32. Respect strides.
        let embedding = try MLMultiArray(
            shape: [1, NSNumber(value: CosyVoice3Constants.speakerEmbeddingDim)],
            dataType: .float32)
        let eStrides = embedding.strides.map { $0.intValue }
        let ePtr = embedding.dataPointer.bindMemory(to: Float.self, capacity: embedding.count)
        let ePhysical = eStrides[0] * embedding.shape[0].intValue
        ePtr.initialize(repeating: 0, count: ePhysical)
        for i in 0..<spkEmbedding.count { ePtr[i * eStrides[1]] = spkEmbedding[i] }

        let features: [String: Any] = [
            "token_total": tokenTotal,
            "num_prompt_tokens": numPromptTokens,
            "prompt_feat": promptFeat,
            "embedding": embedding,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await models.flow.compatPrediction(
            from: provider, options: MLPredictionOptions())

        guard
            let mel = output.featureValue(for: "mel")?.multiArrayValue,
            let nPromptMelArr = output.featureValue(for: "num_prompt_mel")?.multiArrayValue
        else {
            throw CosyVoice3Error.predictionFailed("flow: missing outputs")
        }
        let nPromptMel = nPromptMelArr[0].intValue
        return (mel, nPromptMel)
    }

    private func runHiFT(
        fullMel: MLMultiArray,
        newMelStart: Int,
        newMelFrames: Int
    ) async throws -> [Float] {
        // fullMel logical shape = [1, 80, 500] fp32. Physical strides may be
        // non-compact (e.g. [40960, 512, 1]) — use logical indexing.
        let hiftFrames = CosyVoice3Constants.hiftMaxFrames
        let melBins = CosyVoice3Constants.melBins
        let validFrames = min(newMelFrames, hiftFrames)

        let melInput = try MLMultiArray(
            shape: [1, NSNumber(value: melBins), NSNumber(value: hiftFrames)],
            dataType: .float32)
        // melInput strides may also be non-compact — use logical indexing.
        let melInputStrides = melInput.strides.map { $0.intValue }
        let dstBase = melInput.dataPointer.bindMemory(to: Float.self, capacity: melInput.count)
        // Zero-fill entire physical extent (handles padded strides).
        let totalPhysical = melInputStrides[0] * melInput.shape[0].intValue
        dstBase.initialize(repeating: 0, count: totalPhysical)

        let srcStrides = fullMel.strides.map { $0.intValue }
        let srcBase = fullMel.dataPointer.bindMemory(to: Float.self, capacity: fullMel.count)
        // fullMel logical: [1, 80, 500]; copy new slice → melInput [1, 80, 500].
        for b in 0..<melBins {
            for f in 0..<validFrames {
                let srcOff = b * srcStrides[1] + (newMelStart + f) * srcStrides[2]
                let dstOff = b * melInputStrides[1] + f * melInputStrides[2]
                dstBase[dstOff] = srcBase[srcOff]
            }
        }

        let numValid = try MLMultiArray(shape: [1], dataType: .int32)
        numValid[0] = NSNumber(value: Int32(validFrames))

        let features: [String: Any] = [
            "mel": melInput,
            "num_valid_frames": numValid,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await models.hift.compatPrediction(
            from: provider, options: MLPredictionOptions())

        guard
            let audioArr = output.featureValue(for: "audio")?.multiArrayValue,
            let audioLenArr = output.featureValue(for: "audio_length_samples")?.multiArrayValue
        else {
            throw CosyVoice3Error.predictionFailed("hift: missing outputs")
        }
        let audioLen = audioLenArr[0].intValue
        var out = [Float](repeating: 0, count: audioLen)
        // audio logical shape = [1, 240000]; honor strides.
        let audioStrides = audioArr.strides.map { $0.intValue }
        let aBase = audioArr.dataPointer.bindMemory(to: Float.self, capacity: audioArr.count)
        for i in 0..<audioLen {
            out[i] = aBase[i * audioStrides[1]]
        }
        return out
    }

    // MARK: - Helpers

    /// Extracts logits for the last real prefill position (`tPre - 1`).
    /// Prefill output logical shape is `[1, 256, 6761]` fp32; strides may be
    /// non-compact.
    private func sliceLastStepLogits(
        from logits: MLMultiArray,
        tPre: Int,
        vocab: Int
    ) -> [Float] {
        let strides = logits.strides.map { $0.intValue }
        // shape = [1, T, V]; row (time) stride is strides[1], vocab stride is strides[2].
        let rowStride = strides[1]
        let vocabStride = strides[2]
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        let base = (tPre - 1) * rowStride
        var out = [Float](repeating: 0, count: vocab)
        for i in 0..<vocab { out[i] = ptr[base + i * vocabStride] }
        return out
    }
}
