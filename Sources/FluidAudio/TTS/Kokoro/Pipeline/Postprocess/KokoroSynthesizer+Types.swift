import Foundation

extension KokoroSynthesizer {
    public struct TokenCapacities {
        public let short: Int
        public let long: Int

        public func capacity(for variant: ModelNames.TTS.Variant) -> Int {
            switch variant {
            case .fiveSecond:
                return short
            case .fifteenSecond:
                return long
            }
        }
    }

    public struct SynthesisResult: Sendable {
        public let audio: Data
        public let chunks: [ChunkInfo]
        public let diagnostics: Diagnostics?

        public init(audio: Data, chunks: [ChunkInfo], diagnostics: Diagnostics? = nil) {
            self.audio = audio
            self.chunks = chunks
            self.diagnostics = diagnostics
        }
    }

    public struct Diagnostics: Sendable {
        public let variantFootprints: [ModelNames.TTS.Variant: Int]
        public let lexiconEntryCount: Int
        public let lexiconEstimatedBytes: Int
        public let audioSampleBytes: Int
        public let outputWavBytes: Int

        public func updating(audioSampleBytes: Int, outputWavBytes: Int) -> Diagnostics {
            Diagnostics(
                variantFootprints: variantFootprints,
                lexiconEntryCount: lexiconEntryCount,
                lexiconEstimatedBytes: lexiconEstimatedBytes,
                audioSampleBytes: audioSampleBytes,
                outputWavBytes: outputWavBytes
            )
        }
    }

    /// Predicted duration for a single phoneme/input token, derived from the model's `pred_dur` output.
    /// Times are in seconds, measured from the start of the chunk's audio (`ChunkInfo.samples`).
    public struct TokenTiming: Sendable {
        /// Phoneme glyph from the chunker's vocabulary lookup.
        public let phoneme: String
        /// Cumulative start time in seconds, measured from the start of this chunk's audio.
        public let startTime: TimeInterval
        /// End time in seconds, measured from the start of this chunk's audio.
        public let endTime: TimeInterval
        /// Raw frame count from the model's `pred_dur` output.
        /// One frame = `TtsConstants.kokoroFrameSamples` (600) samples = 25 ms at 24 kHz.
        public let frames: Float

        public init(phoneme: String, startTime: TimeInterval, endTime: TimeInterval, frames: Float) {
            self.phoneme = phoneme
            self.startTime = startTime
            self.endTime = endTime
            self.frames = frames
        }
    }

    /// Word-level timing aggregated from `TokenTiming`s using the chunker's phoneme→atom alignment.
    /// Times are in seconds, measured from the start of the chunk's audio (`ChunkInfo.samples`).
    public struct WordTiming: Sendable {
        /// Source word/atom text — matches an entry in `ChunkInfo.atoms`.
        public let word: String
        /// Index into `ChunkInfo.atoms` for traceability.
        public let atomIndex: Int
        /// Start time in seconds, measured from the start of this chunk's audio.
        public let startTime: TimeInterval
        /// End time in seconds, measured from the start of this chunk's audio.
        public let endTime: TimeInterval

        public init(word: String, atomIndex: Int, startTime: TimeInterval, endTime: TimeInterval) {
            self.word = word
            self.atomIndex = atomIndex
            self.startTime = startTime
            self.endTime = endTime
        }
    }

    public struct ChunkInfo: Sendable {
        public let index: Int
        public let text: String
        public let wordCount: Int
        public let words: [String]
        public let atoms: [String]
        public let pauseAfterMs: Int
        public let tokenCount: Int
        public let samples: [Float]
        public let variant: ModelNames.TTS.Variant
        /// Per-phoneme timing derived from the model's duration predictor.
        /// `nil` when the underlying Core ML model didn't expose `pred_dur` for this chunk.
        public let tokenTimings: [TokenTiming]?
        /// Per-word timing aggregated from `tokenTimings` via the chunker's phoneme→atom alignment.
        /// `nil` when `tokenTimings` is nil or alignment couldn't be built.
        public let wordTimings: [WordTiming]?

        public init(
            index: Int,
            text: String,
            wordCount: Int,
            words: [String],
            atoms: [String],
            pauseAfterMs: Int,
            tokenCount: Int,
            samples: [Float],
            variant: ModelNames.TTS.Variant,
            tokenTimings: [TokenTiming]? = nil,
            wordTimings: [WordTiming]? = nil
        ) {
            self.index = index
            self.text = text
            self.wordCount = wordCount
            self.words = words
            self.atoms = atoms
            self.pauseAfterMs = pauseAfterMs
            self.tokenCount = tokenCount
            self.samples = samples
            self.variant = variant
            self.tokenTimings = tokenTimings
            self.wordTimings = wordTimings
        }
    }

    struct ChunkInfoTemplate: Sendable {
        let index: Int
        let text: String
        let wordCount: Int
        let words: [String]
        let atoms: [String]
        let pauseAfterMs: Int
        let tokenCount: Int
        let variant: ModelNames.TTS.Variant
        let targetTokens: Int
    }

    struct ChunkEntry: Sendable {
        let chunk: TextChunk
        let inputIds: [Int32]
        let template: ChunkInfoTemplate
    }
}
