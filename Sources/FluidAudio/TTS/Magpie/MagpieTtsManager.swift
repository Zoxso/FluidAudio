import CoreML
import Foundation

/// Manages text-to-speech synthesis with the NVIDIA Magpie TTS Multilingual 357M model.
///
/// Magpie is an encoder-decoder transformer that emits discrete NanoCodec tokens
/// autoregressively at 21.5 fps; NanoCodec then decodes them to 22 kHz audio. The
/// Swift port uses four CoreML models (text_encoder, decoder_prefill, decoder_step,
/// nanocodec_decoder) plus a small 1-layer "local transformer" implemented in Swift
/// to sample the 8 codebook tokens per step.
///
/// Usage:
/// ```swift
/// let manager = try await MagpieTtsManager.downloadAndCreate(
///     languages: [.english, .spanish])
/// let result = try await manager.synthesize(
///     text: "Hello from Magpie.", speaker: .john, language: .english)
/// ```
public actor MagpieTtsManager {

    private let logger = AppLogger(category: "MagpieTtsManager")

    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private let preferredLanguages: Set<MagpieLanguage>

    private var store: MagpieModelStore?
    private var tokenizer: MagpieTokenizer?
    private var synthesizer: MagpieSynthesizer?

    public init(
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        preferredLanguages: Set<MagpieLanguage> = [.english]
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.preferredLanguages = preferredLanguages
    }

    public var isAvailable: Bool {
        synthesizer != nil
    }

    /// Convenience factory: download assets and return a ready-to-use manager.
    public static func downloadAndCreate(
        languages: Set<MagpieLanguage> = [.english],
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> MagpieTtsManager {
        let manager = MagpieTtsManager(
            directory: cacheDirectory,
            computeUnits: computeUnits,
            preferredLanguages: languages)
        try await manager.initialize()
        return manager
    }

    /// Download models + constants from HuggingFace and load everything needed to synthesize.
    public func initialize() async throws {
        if synthesizer != nil { return }

        let store = MagpieModelStore(
            directory: directory,
            computeUnits: computeUnits,
            preferredLanguages: preferredLanguages)
        try await store.loadIfNeeded()
        self.store = store

        let bundle = try await store.constants()
        let repoDir = try await store.repoDir()
        let tokenizerDir = MagpieResourceDownloader.tokenizerDirectory(in: repoDir)
        let tokenizer = MagpieTokenizer(
            tokenizerDir: tokenizerDir, eosId: bundle.textEosId)
        self.tokenizer = tokenizer

        self.synthesizer = MagpieSynthesizer(store: store, tokenizer: tokenizer)
        logger.info("Magpie TTS ready (languages: \(preferredLanguages.map { $0.rawValue }.sorted()))")
    }

    /// Ensure tokenizer data for `language` exists on disk (downloads if missing).
    /// Useful when you want to synthesize in a language that wasn't in
    /// `preferredLanguages` at init time.
    public func prepareLanguage(_ language: MagpieLanguage) async throws {
        guard let store = store else {
            throw MagpieError.notInitialized
        }
        let repoDir = try await store.repoDir()
        try await MagpieResourceDownloader.ensureTokenizer(
            for: language, repoDirectory: repoDir)
    }

    /// Synthesize `text` into 22 kHz float PCM using the given speaker and language.
    ///
    /// Text flows through the normal language tokenizer / G2P. When
    /// `options.allowIpaOverride` is `true` (default), any `|…|` region in the text
    /// is treated as a space-separated IPA pronunciation override and tokenized
    /// directly against the language's `token2id` map — no G2P.
    public func synthesize(
        text: String,
        speaker: MagpieSpeaker = .john,
        language: MagpieLanguage = .english,
        options: MagpieSynthesisOptions = .default
    ) async throws -> MagpieSynthesisResult {
        guard let synthesizer = synthesizer else {
            throw MagpieError.notInitialized
        }
        return try await synthesizer.synthesize(
            text: text, speaker: speaker, language: language, options: options)
    }

    /// Synthesize from pre-tokenized phoneme/IPA tokens, bypassing the text frontend.
    public func synthesize(
        phonemes: MagpiePhonemeTokens,
        speaker: MagpieSpeaker = .john,
        options: MagpieSynthesisOptions = .default
    ) async throws -> MagpieSynthesisResult {
        guard let synthesizer = synthesizer else {
            throw MagpieError.notInitialized
        }
        return try await synthesizer.synthesize(
            phonemes: phonemes, speaker: speaker, options: options)
    }

    public func cleanup() async {
        if let store = store {
            await store.unload()
        }
        store = nil
        tokenizer = nil
        synthesizer = nil
    }
}
