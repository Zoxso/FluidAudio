import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "CohereAsrManager")

// MARK: - Cohere Transcribe Manager

/// Manages Cohere Transcribe 03-2026 CoreML inference.
///
/// Architecture:
/// 1. Audio -> mel spectrogram -> encoder -> acoustic features
/// 2. Decoder (with KV cache) processes tokens autoregressively
/// 3. LM head converts hidden states to logits
///
/// Supports 14 languages with state-of-the-art multilingual ASR:
/// - Western: EN, FR, DE, IT, ES, PT, EL, NL, PL, AR (WER 3.8-9.3%)
/// - Asian: ZH, JA, KO, VI (CER 0-7.3%)
@available(macOS 15, iOS 18, *)
public actor CohereAsrManager {
    private var models: CohereAsrModels?
    private let melExtractor: AudioMelSpectrogram

    public init() {
        // Cohere uses 128 mel bins (not Whisper's 80)
        self.melExtractor = AudioMelSpectrogram(
            sampleRate: 16000,
            nMels: 128,
            nFFT: 512,
            hopLength: 160,
            winLength: 400
        )
    }

    /// Load all models from the specified directory.
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await CohereAsrModels.load(from: directory, computeUnits: computeUnits)
        logger.info("Cohere Transcribe models loaded successfully")
    }

    /// Transcribe raw audio samples.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono Float32 audio samples.
    ///   - language: Optional language hint (ISO code like "en", "zh", or English name like "English").
    ///               Pass nil for automatic language detection.
    ///   - maxNewTokens: Maximum number of tokens to generate (default: 512).
    /// - Returns: Transcribed text.
    public func transcribe(
        audioSamples: [Float],
        language: String? = nil,
        maxNewTokens: Int = 512
    ) async throws -> String {
        // Use computeFlatTransposed for NeMo preprocessing (preemph, center padding, log)
        let (melFlat, melLength, numFrames) = melExtractor.computeFlatTransposed(audio: audioSamples)

        guard melLength > 0 else {
            throw CohereAsrError.generationFailed("Audio too short to extract mel spectrogram")
        }

        // Reshape from flat [T * nMels] to [[nMels, T]] and apply per-feature normalization
        let nMels = CohereAsrConfig.numMelBins
        let eps: Float = 1e-5  // DITHER_CONSTANT

        var mel = [[Float]](repeating: [Float](repeating: 0, count: melLength), count: nMels)

        // Step 1: Reshape
        for t in 0..<melLength {
            for m in 0..<nMels {
                mel[m][t] = melFlat[t * nMels + m]
            }
        }

        // Step 2: Per-feature normalization (mean=0, std=1 per mel bin)
        for m in 0..<nMels {
            // Compute mean
            var sum: Float = 0
            for t in 0..<melLength {
                sum += mel[m][t]
            }
            let mean = sum / Float(melLength)

            // Compute std
            var sumSq: Float = 0
            for t in 0..<melLength {
                let diff = mel[m][t] - mean
                sumSq += diff * diff
            }
            let variance = sumSq / Float(melLength - 1)
            let std = sqrt(variance) + eps

            // Normalize
            for t in 0..<melLength {
                mel[m][t] = (mel[m][t] - mean) / std
            }
        }

        return try await transcribe(melSpectrogram: mel, language: language, maxNewTokens: maxNewTokens)
    }

    /// Transcribe raw audio samples with typed language.
    public func transcribe(
        audioSamples: [Float],
        language: CohereAsrConfig.Language?,
        maxNewTokens: Int = 512
    ) async throws -> String {
        try await transcribe(
            audioSamples: audioSamples,
            language: language?.rawValue,
            maxNewTokens: maxNewTokens
        )
    }

    /// Transcribe from a pre-computed mel spectrogram.
    public func transcribe(
        melSpectrogram: [[Float]],
        language: String? = nil,
        maxNewTokens: Int = 512
    ) async throws -> String {
        guard let models = models else {
            throw CohereAsrError.generationFailed("Models not loaded")
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Resolve language
        let resolvedLanguage: CohereAsrConfig.Language?
        if let lang = language {
            resolvedLanguage = CohereAsrConfig.Language(from: lang)
            if resolvedLanguage == nil {
                logger.warning("Unknown language '\(lang)', using automatic detection")
            }
        } else {
            resolvedLanguage = nil
        }

        // Step 1: Encode audio (mel -> encoder hidden states)
        let t1 = CFAbsoluteTimeGetCurrent()
        let encoderHiddenStates = try encodeAudio(melSpectrogram: melSpectrogram, models: models)
        let audioEncodeTime = CFAbsoluteTimeGetCurrent() - t1

        // Debug: Check encoder output
        if encoderHiddenStates.count > 0 {
            let ptr = encoderHiddenStates.dataPointer.bindMemory(to: Float.self, capacity: encoderHiddenStates.count)
            var minVal: Float = Float.greatestFiniteMagnitude
            var maxVal: Float = -Float.greatestFiniteMagnitude
            for i in 0..<min(10000, encoderHiddenStates.count) {
                let val = ptr[i]
                if val.isFinite {
                    minVal = min(minVal, val)
                    maxVal = max(maxVal, val)
                }
            }
            print("ENCODER OUTPUT: min=\(minVal) max=\(maxVal)")
        }

        logger.debug("Audio encoding: \(String(format: "%.2f", audioEncodeTime))s")

        // Step 2: Autoregressive generation with decoder + LM head
        let t2 = CFAbsoluteTimeGetCurrent()
        let generatedTokenIds = try generate(
            encoderHiddenStates: encoderHiddenStates,
            maxNewTokens: maxNewTokens,
            models: models
        )
        let generateTime = CFAbsoluteTimeGetCurrent() - t2

        // Step 3: Decode tokens to text
        let text = decodeTokens(generatedTokenIds, vocabulary: models.vocabulary)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.debug(
            "Timing: encode=\(String(format: "%.2f", audioEncodeTime))s gen=\(String(format: "%.2f", generateTime))s total=\(String(format: "%.2f", elapsed))s tokens=\(generatedTokenIds.count)"
        )

        return text
    }

    // MARK: - Audio Encoding

    private func encodeAudio(
        melSpectrogram: [[Float]],
        models: CohereAsrModels
    ) throws -> MLMultiArray {
        // Create mel input MLMultiArray (shape: [1, 128, 3000])
        let melInput = try createMelInput(melSpectrogram: melSpectrogram)

        // Run encoder
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: melInput)
        ])

        let prediction = try models.audioEncoder.prediction(from: input)
        guard let encoderOutput = prediction.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw CohereAsrError.encoderFailed("No encoder_output from audio encoder")
        }

        // Pad encoder output to fixed length (1500) for decoder compatibility
        return try padEncoderOutput(encoderOutput, targetLength: 1500)
    }

    private func padEncoderOutput(_ encoderOutput: MLMultiArray, targetLength: Int) throws -> MLMultiArray {
        // Encoder output shape: [batch, seq_len, hidden_dim]
        let batch = encoderOutput.shape[0].intValue
        let seqLen = encoderOutput.shape[1].intValue
        let hiddenDim = encoderOutput.shape[2].intValue

        if seqLen >= targetLength {
            // Already at or exceeds target length, return as is
            return encoderOutput
        }

        // Create padded array (already initialized to zeros by MLMultiArray)
        let paddedShape: [NSNumber] = [
            NSNumber(value: batch), NSNumber(value: targetLength), NSNumber(value: hiddenDim),
        ]
        let padded = try MLMultiArray(shape: paddedShape, dataType: encoderOutput.dataType)

        // Copy original data using MLMultiArray subscripts
        for b in 0..<batch {
            for t in 0..<seqLen {
                for h in 0..<hiddenDim {
                    let srcIdx: [NSNumber] = [NSNumber(value: b), NSNumber(value: t), NSNumber(value: h)]
                    let dstIdx: [NSNumber] = [NSNumber(value: b), NSNumber(value: t), NSNumber(value: h)]
                    padded[dstIdx] = encoderOutput[srcIdx]
                }
            }
        }

        return padded
    }

    private func createMelInput(melSpectrogram: [[Float]]) throws -> MLMultiArray {
        let numMels = min(melSpectrogram.count, CohereAsrConfig.numMelBins)
        let numFrames = melSpectrogram.first?.count ?? 0

        // Pad or truncate to fixed length
        let targetFrames = CohereAsrConfig.fixedAudioLength
        let actualFrames = min(numFrames, targetFrames)

        // Note: Mel spectrogram is already per-feature normalized
        // Fill MLMultiArray
        let shape: [NSNumber] = [1, NSNumber(value: CohereAsrConfig.numMelBins), NSNumber(value: targetFrames)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)

        // Initialize with zeros (padding)
        ptr.initialize(repeating: 0.0, count: array.count)

        // Fill with mel spectrogram data
        for mel in 0..<numMels {
            for t in 0..<actualFrames {
                let idx = mel * targetFrames + t
                ptr[idx] = melSpectrogram[mel][t]
            }
        }

        return array
    }

    // MARK: - Autoregressive Generation

    private func generate(
        encoderHiddenStates: MLMultiArray,
        maxNewTokens: Int,
        models: CohereAsrModels
    ) throws -> [Int] {
        var generatedTokens: [Int] = []

        // Initialize decoder input with decoder start token
        var currentTokenId = CohereAsrConfig.decoderStartTokenId
        var position = 0

        for step in 0..<maxNewTokens {
            // Prepare decoder inputs
            let inputIds = try createInputIds(tokenId: currentTokenId)
            let positions = try createPositions(position: position)

            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "positions": MLFeatureValue(multiArray: positions),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
            ])

            // Run decoder
            let decoderOutput = try models.decoder.prediction(from: decoderInput)
            guard let hiddenStates = decoderOutput.featureValue(for: "hidden_states")?.multiArrayValue else {
                throw CohereAsrError.decoderFailed("No hidden_states from decoder at step \(step)")
            }

            // Run LM head
            let lmInput = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": MLFeatureValue(multiArray: hiddenStates)
            ])
            let lmOutput = try models.lmHead.prediction(from: lmInput)
            guard let logits = lmOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereAsrError.generationFailed("No logits from LM head at step \(step)")
            }

            // Get next token
            let tokenId = argmaxFromLogits(logits)

            // Check for EOS
            if tokenId == CohereAsrConfig.eosTokenId {
                break
            }

            generatedTokens.append(tokenId)
            currentTokenId = tokenId
            position += 1
        }

        return generatedTokens
    }

    // MARK: - Helper Methods

    private func createInputIds(tokenId: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1]  // [batch, seq_len=1]
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        array[0] = NSNumber(value: tokenId)
        return array
    }

    private func createPositions(position: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1]  // [batch, seq_len=1]
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        array[0] = NSNumber(value: position)
        return array
    }

    private func argmaxFromLogits(_ logits: MLMultiArray) -> Int {
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: CohereAsrConfig.vocabSize)
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(CohereAsrConfig.vocabSize))
        return Int(maxIdx)
    }

    private func decodeTokens(_ tokenIds: [Int], vocabulary: [Int: String]) -> String {
        // SentencePiece-style decoding
        var pieces: [String] = []
        for id in tokenIds {
            guard let piece = vocabulary[id] else { continue }

            // Skip special tokens
            if piece.hasPrefix("<|") && piece.hasSuffix("|>") {
                continue
            }

            pieces.append(piece)
        }

        let raw = pieces.joined()

        // Replace SentencePiece space marker (▁ U+2581) with actual spaces
        let decoded = raw.replacingOccurrences(of: "\u{2581}", with: " ")
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
