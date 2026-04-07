@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "CohereAsrModels")

// MARK: - Cohere Transcribe CoreML Model Container

/// Holds CoreML model components for Cohere Transcribe ASR.
///
/// **Hybrid Quantization Only**: Uses INT8 encoder + FP16 decoder for optimal balance.
///
/// Specifications:
/// - Total size: 2.1 GB (1.8 GB encoder + 291 MB decoder)
/// - Quality: Same as full FP16
/// - Stability: 0% loops (FP16 decoder prevents repetition)
/// - Memory savings: 46% vs full FP16 (3.9 GB → 2.1 GB)
///
/// Model architecture:
/// - `encoder`: Mel spectrogram → encoder hidden states (1, 438, 1024)
/// - `decoder`: Stateful decoder with self-attention and cross-attention (CoreML State API)
@available(macOS 14, iOS 17, *)
public struct CohereAsrModels: Sendable {
    public let encoder: MLModel
    public let decoder: MLModel
    public let vocabulary: [Int: String]

    /// Download and load Cohere Transcribe models from HuggingFace (hybrid quantization).
    ///
    /// Downloads INT8 encoder and FP16 decoder if not already cached.
    ///
    /// - Parameters:
    ///   - computeUnits: Compute units for model execution
    ///   - progressHandler: Optional progress callback
    /// - Returns: CohereAsrModels with hybrid quantization
    public static func downloadAndLoad(
        computeUnits: MLComputeUnits = .all,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CohereAsrModels {
        // Download INT8 encoder
        logger.info("Downloading Cohere Transcribe (hybrid: INT8 encoder + FP16 decoder)...")
        let int8Dir = try await download(repo: .cohereTranscribeCoremlInt8, progressHandler: progressHandler)

        // Download FP16 decoder
        let fp16Dir = try await download(repo: .cohereTranscribeCoreml, progressHandler: progressHandler)

        // Load hybrid: INT8 encoder from int8Dir, FP16 decoder from fp16Dir
        return try await loadHybrid(
            encoderDirectory: int8Dir,
            decoderDirectory: fp16Dir,
            computeUnits: computeUnits
        )
    }

    /// Load Cohere Transcribe models with hybrid quantization (INT8 encoder + FP16 decoder).
    ///
    /// - Parameters:
    ///   - encoderDirectory: Directory containing INT8 encoder
    ///   - decoderDirectory: Directory containing FP16 decoder
    ///   - computeUnits: Compute units for model execution
    /// - Returns: CohereAsrModels with hybrid quantization
    public static func loadHybrid(
        encoderDirectory: URL,
        decoderDirectory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> CohereAsrModels {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits

        logger.info("Loading HYBRID Cohere Transcribe models")
        logger.info("  INT8 encoder from: \(encoderDirectory.path)")
        logger.info("  FP16 decoder from: \(decoderDirectory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        // Load INT8 encoder
        let encoder = try await loadModel(
            named: ModelNames.CohereTranscribe.encoder,
            from: encoderDirectory,
            configuration: modelConfig
        )

        // Load FP16 decoder
        let decoder = try await loadModel(
            named: ModelNames.CohereTranscribe.decoderStateful,
            from: decoderDirectory,
            configuration: modelConfig
        )

        // Load vocabulary (from either directory, they're identical)
        let vocabulary = try loadVocabulary(from: decoderDirectory)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Loaded HYBRID Cohere Transcribe models in \(String(format: "%.2f", elapsed))s")

        return CohereAsrModels(
            encoder: encoder,
            decoder: decoder,
            vocabulary: vocabulary
        )
    }

    /// Download Cohere Transcribe models from HuggingFace.
    ///
    /// - Parameters:
    ///   - repo: Model repository to download (INT8 or FP16 variant)
    ///   - progressHandler: Optional callback for download progress updates
    /// - Returns: Path to the directory containing the downloaded models
    @discardableResult
    private static func download(
        repo: Repo,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = defaultCacheDirectory(repo: repo)
        let modelsRoot = modelsRootDirectory()

        if modelsExist(at: targetDir, repo: repo) {
            logger.info("Cohere Transcribe models already present at: \(targetDir.path)")
            return targetDir
        }

        try await DownloadUtils.downloadRepo(repo, to: modelsRoot, progressHandler: progressHandler)
        logger.info("Successfully downloaded Cohere Transcribe models from \(repo.name)")
        return targetDir
    }

    /// Check if all required model files exist locally.
    private static func modelsExist(at directory: URL, repo: Repo) -> Bool {
        let fm = FileManager.default
        let requiredFiles = ModelNames.getRequiredModelNames(for: repo, variant: nil)
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
    private static func defaultCacheDirectory(repo: Repo) -> URL {
        modelsRootDirectory()
            .appendingPathComponent(repo.folderName, isDirectory: true)
    }
}

// MARK: - Helpers

@available(macOS 14, iOS 17, *)
extension CohereAsrModels {
    /// Load a CoreML model, compiling .mlpackage if necessary.
    internal static func loadModel(
        named name: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiledURL = directory.appendingPathComponent("\(name).mlmodelc")
        let packageURL = directory.appendingPathComponent("\(name).mlpackage")

        // Try .mlmodelc first (faster), fall back to compiling .mlpackage
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            logger.debug("Loading \(name) from compiled model")
            return try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        } else if FileManager.default.fileExists(atPath: packageURL.path) {
            logger.debug("Compiling \(name) from package...")
            let compileStart = CFAbsoluteTimeGetCurrent()

            // Compile .mlpackage to .mlmodelc
            let tempCompiledURL = try await MLModel.compileModel(at: packageURL)
            let compileElapsed = CFAbsoluteTimeGetCurrent() - compileStart
            logger.info("  \(name): compiled in \(String(format: "%.2f", compileElapsed))s")

            // Move compiled model next to the package for caching
            try? FileManager.default.removeItem(at: compiledURL)
            try FileManager.default.moveItem(at: tempCompiledURL, to: compiledURL)

            // Load the compiled model
            return try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        } else {
            logger.error("Model not found: \(name)")
            throw CohereAsrError.modelNotFound("Model not found: \(name)")
        }
    }

    /// Load vocabulary from JSON file.
    internal static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent("vocab.json")

        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            logger.error("Vocabulary file not found at \(vocabPath.path)")
            throw CohereAsrError.modelNotFound("vocab.json not found at \(vocabPath.path)")
        }

        do {
            let data = try Data(contentsOf: vocabPath)
            let json = try JSONSerialization.jsonObject(with: data)

            var vocabulary: [Int: String] = [:]

            if let jsonDict = json as? [String: String] {
                // Dictionary format: {"0": "<unk>", "1": "<|nospeech|>", ...}
                for (key, value) in jsonDict {
                    if let tokenId = Int(key) {
                        vocabulary[tokenId] = value
                    }
                }
            } else {
                throw CohereAsrError.modelNotFound("Invalid vocab.json format")
            }

            logger.info("Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch {
            logger.error("Failed to load vocabulary: \(error.localizedDescription)")
            throw CohereAsrError.modelNotFound("Failed to load vocab.json: \(error.localizedDescription)")
        }
    }
}

// MARK: - Error

public enum CohereAsrError: Error, LocalizedError {
    case modelNotFound(String)
    case encodingFailed(String)
    case decodingFailed(String)
    case invalidInput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return "Model not found: \(msg)"
        case .encodingFailed(let msg): return "Encoding failed: \(msg)"
        case .decodingFailed(let msg): return "Decoding failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        }
    }
}
