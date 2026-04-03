import Foundation

// MARK: - Cohere Transcribe 03-2026 Model Configuration

/// Configuration constants for the Cohere Transcribe 03-2026 CoreML model.
///
/// Architecture: Conformer encoder (2B params) -> Transformer decoder -> LM head
/// Supports 14 languages with state-of-the-art multilingual ASR performance.
public enum CohereAsrConfig {
    // MARK: Audio

    public static let sampleRate = 16000
    public static let numMelBins = 128  // Cohere uses 128 mel bins (not Whisper's 80)

    /// Fixed audio length in frames (30 seconds at 16kHz).
    public static let fixedAudioLength = 3000

    // MARK: Encoder (Conformer)

    /// Encoder hidden dimension.
    public static let encoderDModel = 2048

    /// Number of Conformer encoder layers.
    public static let encoderNumLayers = 24

    /// Number of attention heads in encoder.
    public static let encoderNumHeads = 16

    // MARK: Decoder (Transformer)

    public static let decoderHiddenSize = 512
    public static let decoderNumLayers = 6
    public static let decoderNumHeads = 8
    public static let vocabSize = 32000

    // MARK: Special Tokens

    /// Decoder start token ID (used to initialize decoder input).
    public static let decoderStartTokenId = 13764

    /// End-of-sequence token ID.
    public static let eosTokenId = 2

    /// Padding token ID.
    public static let padTokenId = 1

    // MARK: Generation

    /// Maximum number of tokens to generate.
    public static let maxNewTokens = 512

    /// Maximum audio duration in seconds.
    public static let maxAudioSeconds: Double = 30.0

    // MARK: - Supported Languages

    /// Supported languages for Cohere Transcribe 03-2026.
    /// Use ISO 639-1 codes or English names. Pass nil for automatic detection.
    ///
    /// Benchmark results (FLEURS dataset, 100 samples per language):
    /// - Spanish: 3.80% WER
    /// - Italian: 3.80% WER
    /// - German: 4.57% WER
    /// - Portuguese: 5.03% WER
    /// - English: 5.44% WER
    /// - French: 5.80% WER
    /// - Polish: 5.97% WER
    /// - Dutch: 6.57% WER
    /// - Arabic: 7.31% WER
    /// - Greek: 9.25% WER
    /// - Chinese: ~0% CER (perfect)
    /// - Vietnamese: 3.43% CER
    /// - Korean: 3.48% CER
    /// - Japanese: 7.25% CER
    public enum Language: String, CaseIterable, Sendable {
        case english = "en"
        case french = "fr"
        case german = "de"
        case italian = "it"
        case spanish = "es"
        case portuguese = "pt"
        case greek = "el"
        case dutch = "nl"
        case polish = "pl"
        case chinese = "zh"
        case japanese = "ja"
        case korean = "ko"
        case vietnamese = "vi"
        case arabic = "ar"

        /// English name for the language.
        public var englishName: String {
            switch self {
            case .english: return "English"
            case .french: return "French"
            case .german: return "German"
            case .italian: return "Italian"
            case .spanish: return "Spanish"
            case .portuguese: return "Portuguese"
            case .greek: return "Greek"
            case .dutch: return "Dutch"
            case .polish: return "Polish"
            case .chinese: return "Chinese"
            case .japanese: return "Japanese"
            case .korean: return "Korean"
            case .vietnamese: return "Vietnamese"
            case .arabic: return "Arabic"
            }
        }

        /// FLEURS dataset language code (used for benchmarking).
        public var fleursCode: String {
            switch self {
            case .english: return "en_us"
            case .french: return "fr_fr"
            case .german: return "de_de"
            case .italian: return "it_it"
            case .spanish: return "es_419"
            case .portuguese: return "pt_br"
            case .greek: return "el_gr"
            case .dutch: return "nl_nl"
            case .polish: return "pl_pl"
            case .chinese: return "cmn_hans_cn"
            case .japanese: return "ja_jp"
            case .korean: return "ko_kr"
            case .vietnamese: return "vi_vn"
            case .arabic: return "ar_eg"
            }
        }

        /// Whether this language should use CER (Character Error Rate) instead of WER.
        /// Asian languages without clear word boundaries use CER as the primary metric.
        public var usesCER: Bool {
            switch self {
            case .chinese, .japanese, .korean:
                return true
            default:
                return false
            }
        }

        /// Initialize from ISO code or English name.
        public init?(from string: String) {
            let lowercased = string.lowercased()
            // Try ISO code first
            if let lang = Language(rawValue: lowercased) {
                self = lang
                return
            }
            // Try English name
            if let lang = Language.allCases.first(where: { $0.englishName.lowercased() == lowercased }) {
                self = lang
                return
            }
            return nil
        }
    }
}
