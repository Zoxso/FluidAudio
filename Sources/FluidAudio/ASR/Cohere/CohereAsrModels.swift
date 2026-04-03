@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "CohereAsrModels")

// MARK: - Cohere Transcribe CoreML Model Container

/// Holds CoreML model components for Cohere Transcribe 03-2026.
///
/// Components:
/// - `audioEncoder`: mel spectrogram -> encoder hidden states
/// - `decoder`: decoder with KV cache
/// - `lmHead`: hidden states -> logits
@available(macOS 15, iOS 18, *)
public struct CohereAsrModels: Sendable {
    public let audioEncoder: MLModel
    public let decoder: MLModel
    public let lmHead: MLModel
    public let vocabulary: [Int: String]

    /// Load Cohere Transcribe models from a directory.
    ///
    /// Expected directory structure:
    /// ```
    /// cohere-transcribe/
    ///   cohere_audio_encoder.mlmodelc
    ///   cohere_decoder.mlmodelc
    ///   cohere_lm_head.mlmodelc
    ///   vocab.json (optional, for text decoding)
    /// ```
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> CohereAsrModels {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits

        logger.info("Loading Cohere Transcribe models from \(directory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        // Load encoder (3.6GB)
        let audioEncoder = try await loadModel(
            named: "cohere_audio_encoder",
            from: directory,
            configuration: modelConfig
        )

        // Load decoder (293MB)
        let decoder = try await loadModel(
            named: "cohere_decoder",
            from: directory,
            configuration: modelConfig
        )

        // Load LM head (32MB)
        let lmHead = try await loadModel(
            named: "cohere_lm_head",
            from: directory,
            configuration: modelConfig
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Loaded Cohere Transcribe models in \(String(format: "%.2f", elapsed))s")

        // Load vocabulary if available
        let vocabulary = (try? loadVocabulary(from: directory)) ?? [:]

        return CohereAsrModels(
            audioEncoder: audioEncoder,
            decoder: decoder,
            lmHead: lmHead,
            vocabulary: vocabulary
        )
    }

    /// Download models from HuggingFace and load them.
    ///
    /// Downloads to the default cache directory if not already present,
    /// then loads all model components.
    public static func downloadAndLoad(
        to directory: URL? = nil,
        computeUnits: MLComputeUnits = .all,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CohereAsrModels {
        let targetDir = try await download(to: directory, progressHandler: progressHandler)
        return try await load(from: targetDir, computeUnits: computeUnits)
    }

    /// Download Cohere Transcribe models from HuggingFace.
    ///
    /// - Parameters:
    ///   - directory: Target directory. Uses default cache directory if nil.
    ///   - force: Force re-download even if models exist.
    ///   - progressHandler: Optional callback for download progress updates.
    /// - Returns: Path to the directory containing the downloaded models.
    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        let modelsRoot = modelsRootDirectory()

        if !force && modelsExist(at: targetDir) {
            logger.info("Cohere Transcribe models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            try? FileManager.default.removeItem(at: targetDir)
        }

        logger.info("Downloading Cohere Transcribe models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.cohereTranscribe, to: modelsRoot, progressHandler: progressHandler)
        logger.info("Successfully downloaded Cohere Transcribe models")
        return targetDir
    }

    /// Check if all required model files exist locally.
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let requiredFiles = [
            "cohere_audio_encoder.mlmodelc",
            "cohere_decoder.mlmodelc",
            "cohere_lm_head.mlmodelc",
        ]
        return requiredFiles.allSatisfy { file in
            fm.fileExists(atPath: directory.appendingPathComponent(file).path)
        }
    }

    /// Root directory for all FluidAudio model caches.
    private static func modelsRootDirectory() -> URL {
        guard
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            // Fallback to temporary directory if application support unavailable
            return FileManager.default.temporaryDirectory
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    /// Default cache directory for Cohere Transcribe models.
    public static func defaultCacheDirectory() -> URL {
        modelsRootDirectory()
            .appendingPathComponent(Repo.cohereTranscribe.folderName, isDirectory: true)
    }

    // MARK: Private

    private static func loadModel(
        named name: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        // Try .mlmodelc first (pre-compiled), then compile .mlpackage on the fly
        let compiledPath = directory.appendingPathComponent("\(name).mlmodelc")
        let packagePath = directory.appendingPathComponent("\(name).mlpackage")

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledPath.path) {
            modelURL = compiledPath
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            // .mlpackage must be compiled to .mlmodelc before loading
            logger.info("Compiling \(name).mlpackage -> .mlmodelc ...")
            let compileStart = CFAbsoluteTimeGetCurrent()
            let compiledURL = try await MLModel.compileModel(at: packagePath)
            let compileElapsed = CFAbsoluteTimeGetCurrent() - compileStart
            logger.info("  \(name): compiled in \(String(format: "%.2f", compileElapsed))s")

            // Move compiled model next to the package for caching
            try? FileManager.default.removeItem(at: compiledPath)
            try FileManager.default.copyItem(at: compiledURL, to: compiledPath)
            // Clean up the temp compiled model
            try? FileManager.default.removeItem(at: compiledURL)

            modelURL = compiledPath
        } else {
            throw CohereAsrError.modelNotFound(name)
        }

        let start = CFAbsoluteTimeGetCurrent()
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.debug("  \(name): loaded in \(String(format: "%.2f", elapsed))s")
        return model
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent("vocab.json")
        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            logger.warning("vocab.json not found, text decoding will not work")
            return [:]
        }

        let data = try Data(contentsOf: vocabPath)
        guard let stringToId = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw CohereAsrError.invalidVocabulary
        }

        // Invert: token string -> token ID becomes token ID -> token string
        var idToString: [Int: String] = [:]
        idToString.reserveCapacity(stringToId.count)
        for (token, id) in stringToId {
            idToString[id] = token
        }
        logger.info("Loaded vocabulary: \(idToString.count) tokens")
        return idToString
    }
}

// MARK: - Errors

public enum CohereAsrError: Error, LocalizedError {
    case modelNotFound(String)
    case invalidVocabulary
    case encoderFailed(String)
    case decoderFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Cohere Transcribe model not found: \(name)"
        case .invalidVocabulary:
            return "Invalid vocabulary file"
        case .encoderFailed(let detail):
            return "Audio encoder failed: \(detail)"
        case .decoderFailed(let detail):
            return "Decoder failed: \(detail)"
        case .generationFailed(let detail):
            return "Generation failed: \(detail)"
        }
    }
}
