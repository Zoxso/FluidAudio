import Foundation

/// Decoded shape / hyperparameter metadata from `constants/constants.json`.
///
/// The field names mirror the Python exporter
/// (`mobius/.../export_constants.py`). Unknown keys are ignored so the exporter
/// can add fields without breaking Swift. All fields have safe defaults matching
/// the published 357M checkpoint so the Swift port remains usable if a key is
/// dropped in a future rebuild.
public struct MagpieModelConfig: Sendable, Decodable {
    public let dModel: Int
    public let numDecoderLayers: Int
    public let numHeads: Int
    public let headDim: Int
    public let numCodebooks: Int
    public let numCodesPerCodebook: Int
    public let maxCacheLength: Int
    public let maxTextLength: Int
    public let audioBosId: Int32
    public let audioEosId: Int32
    public let speakerContextLength: Int

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case numDecoderLayers = "num_decoder_layers"
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case numCodebooks = "num_codebooks"
        case numCodesPerCodebook = "num_codes_per_codebook"
        case maxCacheLength = "max_cache_length"
        case maxTextLength = "max_text_length"
        case audioBosId = "audio_bos_id"
        case audioEosId = "audio_eos_id"
        case speakerContextLength = "speaker_context_length"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dModel = (try? c.decode(Int.self, forKey: .dModel)) ?? MagpieConstants.dModel
        numDecoderLayers =
            (try? c.decode(Int.self, forKey: .numDecoderLayers)) ?? MagpieConstants.numDecoderLayers
        numHeads = (try? c.decode(Int.self, forKey: .numHeads)) ?? MagpieConstants.numHeads
        headDim = (try? c.decode(Int.self, forKey: .headDim)) ?? MagpieConstants.headDim
        numCodebooks =
            (try? c.decode(Int.self, forKey: .numCodebooks)) ?? MagpieConstants.numCodebooks
        numCodesPerCodebook =
            (try? c.decode(Int.self, forKey: .numCodesPerCodebook))
            ?? MagpieConstants.numCodesPerCodebook
        maxCacheLength =
            (try? c.decode(Int.self, forKey: .maxCacheLength)) ?? MagpieConstants.maxCacheLength
        maxTextLength =
            (try? c.decode(Int.self, forKey: .maxTextLength)) ?? MagpieConstants.maxTextLength
        audioBosId = (try? c.decode(Int32.self, forKey: .audioBosId)) ?? MagpieConstants.audioBosId
        audioEosId = (try? c.decode(Int32.self, forKey: .audioEosId)) ?? MagpieConstants.audioEosId
        speakerContextLength =
            (try? c.decode(Int.self, forKey: .speakerContextLength))
            ?? MagpieConstants.speakerContextLength
    }

    public init(
        dModel: Int = MagpieConstants.dModel,
        numDecoderLayers: Int = MagpieConstants.numDecoderLayers,
        numHeads: Int = MagpieConstants.numHeads,
        headDim: Int = MagpieConstants.headDim,
        numCodebooks: Int = MagpieConstants.numCodebooks,
        numCodesPerCodebook: Int = MagpieConstants.numCodesPerCodebook,
        maxCacheLength: Int = MagpieConstants.maxCacheLength,
        maxTextLength: Int = MagpieConstants.maxTextLength,
        audioBosId: Int32 = MagpieConstants.audioBosId,
        audioEosId: Int32 = MagpieConstants.audioEosId,
        speakerContextLength: Int = MagpieConstants.speakerContextLength
    ) {
        self.dModel = dModel
        self.numDecoderLayers = numDecoderLayers
        self.numHeads = numHeads
        self.headDim = headDim
        self.numCodebooks = numCodebooks
        self.numCodesPerCodebook = numCodesPerCodebook
        self.maxCacheLength = maxCacheLength
        self.maxTextLength = maxTextLength
        self.audioBosId = audioBosId
        self.audioEosId = audioEosId
        self.speakerContextLength = speakerContextLength
    }
}

/// Decoded metadata from `constants/speaker_info.json`.
public struct MagpieSpeakerInfo: Sendable, Decodable {
    public let contextLength: Int
    public let dim: Int
    public let names: [String]

    enum CodingKeys: String, CodingKey {
        case contextLength = "context_length"
        case dim = "dim"
        case names = "names"
        case T = "T"
        case D = "D"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        contextLength =
            (try? c.decode(Int.self, forKey: .contextLength))
            ?? (try? c.decode(Int.self, forKey: .T))
            ?? MagpieConstants.speakerContextLength
        dim =
            (try? c.decode(Int.self, forKey: .dim))
            ?? (try? c.decode(Int.self, forKey: .D))
            ?? MagpieConstants.dModel
        let decodedNames = (try? c.decode([String].self, forKey: .names)) ?? []
        if decodedNames.isEmpty {
            names = MagpieSpeakerInfo.defaultNames
        } else {
            names = decodedNames
        }
    }

    /// Direct initializer used by the fallback path when `speaker_info.json`
    /// is missing. Keeps us out of synthesizing fake Decoder instances.
    public init(
        contextLength: Int = MagpieConstants.speakerContextLength,
        dim: Int = MagpieConstants.dModel,
        names: [String] = MagpieSpeakerInfo.defaultNames
    ) {
        self.contextLength = contextLength
        self.dim = dim
        self.names = names
    }

    public static let defaultNames: [String] = ["John", "Sofia", "Aria", "Jason", "Leo"]
}

/// Loaded constants: config, speaker info, per-speaker embeddings (fp32), per-codebook
/// audio embeddings (fp32). All arrays are stored row-major.
public struct MagpieConstantsBundle: Sendable {
    public let config: MagpieModelConfig
    public let speakers: MagpieSpeakerInfo
    /// Shape: [numSpeakers][contextLength × dModel]. Row-major.
    public let speakerEmbeddings: [[Float]]
    /// Shape: [numCodebooks][numCodesPerCodebook × dModel]. Row-major.
    public let audioEmbeddings: [[Float]]
    /// Text tokenizer EOS id (from `tokenizer_metadata.json`; 0 if absent).
    public let textEosId: Int32
}

/// Loads Magpie constants from a directory (typically `<repo>/constants/`).
public enum MagpieConstantsLoader {

    private static let logger = AppLogger(category: "MagpieConstantsLoader")

    public static func load(from constantsDir: URL) throws -> MagpieConstantsBundle {
        let config = try loadConfig(from: constantsDir)
        let speakers = try loadSpeakerInfo(from: constantsDir)

        var speakerEmbeddings: [[Float]] = []
        speakerEmbeddings.reserveCapacity(MagpieConstants.numSpeakers)
        for idx in 0..<MagpieConstants.numSpeakers {
            let url = constantsDir.appendingPathComponent(
                MagpieConstants.Files.speakerEmbedding(index: idx))
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MagpieError.modelFileNotFound(url.lastPathComponent)
            }
            let array = try NpyReader.read(from: url)
            try array.assertShape([config.speakerContextLength, config.dModel], label: url.lastPathComponent)
            speakerEmbeddings.append(array.data)
        }

        var audioEmbeddings: [[Float]] = []
        audioEmbeddings.reserveCapacity(config.numCodebooks)
        for cb in 0..<config.numCodebooks {
            let url = constantsDir.appendingPathComponent(
                MagpieConstants.Files.audioEmbedding(codebook: cb))
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MagpieError.modelFileNotFound(url.lastPathComponent)
            }
            let array = try NpyReader.read(from: url)
            try array.assertShape([config.numCodesPerCodebook, config.dModel], label: url.lastPathComponent)
            audioEmbeddings.append(array.data)
        }

        let textEosId = loadTextEosId(from: constantsDir)

        logger.info(
            "Loaded Magpie constants: \(speakerEmbeddings.count) speakers × \(config.speakerContextLength)×\(config.dModel), \(audioEmbeddings.count) codebooks × \(config.numCodesPerCodebook)×\(config.dModel), textEosId=\(textEosId)"
        )

        return MagpieConstantsBundle(
            config: config,
            speakers: speakers,
            speakerEmbeddings: speakerEmbeddings,
            audioEmbeddings: audioEmbeddings,
            textEosId: textEosId
        )
    }

    private static func loadTextEosId(from dir: URL) -> Int32 {
        let url = dir.appendingPathComponent(MagpieConstants.Files.tokenizerMetadataJson)
        guard FileManager.default.fileExists(atPath: url.path),
            let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return 0
        }
        if let eos = json["eos_token_id"] as? Int {
            return Int32(eos)
        }
        if let eos = json["text_eos_id"] as? Int {
            return Int32(eos)
        }
        return 0
    }

    private static func loadConfig(from dir: URL) throws -> MagpieModelConfig {
        let url = dir.appendingPathComponent(MagpieConstants.Files.constantsJson)
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.warning("constants.json missing; falling back to built-in defaults")
            return MagpieModelConfig()
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(MagpieModelConfig.self, from: data)
        } catch {
            throw MagpieError.invalidConstants("constants.json: \(error)")
        }
    }

    private static func loadSpeakerInfo(from dir: URL) throws -> MagpieSpeakerInfo {
        let url = dir.appendingPathComponent(MagpieConstants.Files.speakerInfoJson)
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.warning("speaker_info.json missing; falling back to built-in defaults")
            return MagpieSpeakerInfo()
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(MagpieSpeakerInfo.self, from: data)
        } catch {
            throw MagpieError.invalidConstants("speaker_info.json: \(error)")
        }
    }
}
