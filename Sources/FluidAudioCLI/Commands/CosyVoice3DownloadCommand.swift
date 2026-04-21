import FluidAudio
import Foundation

/// Eagerly downloads every CosyVoice3 asset from HuggingFace
/// (`FluidInference/CosyVoice3-0.5B-coreml`) into `~/.cache/fluidaudio` so
/// subsequent `--backend cosyvoice3-text` runs are cache hits.
///
/// Downloads (~5.8 GB total):
/// - 4 `.mlmodelc` bundles (LLM-Prefill, LLM-Decode, Flow, HiFT)
/// - `embeddings/speech_embedding-fp16.safetensors`
/// - `embeddings/embeddings-runtime-fp32.safetensors` (~542 MB)
/// - Tokenizer files (vocab.json, merges.txt, tokenizer_config.json, special_tokens.json)
/// - Default voice bundle (`voices/cosyvoice3-default-zh.safetensors` + `.json`)
///
/// Usage:
/// ```
/// fluidaudiocli tts --backend cosyvoice3-download
/// ```
enum CosyVoice3DownloadCLI {

    private static let logger = AppLogger(category: "CosyVoice3DownloadCLI")

    static func run() async {
        let tStart = Date()
        logger.info("Starting CosyVoice3 asset download from HuggingFace…")

        do {
            let repoDir = try await CosyVoice3ResourceDownloader.ensureCoreModels()
            logger.info("Core models + speech embedding cached at: \(repoDir.path)")

            let frontend = try await CosyVoice3ResourceDownloader.ensureTextFrontendAssets(
                repoDirectory: repoDir)
            logger.info("Tokenizer: \(frontend.tokenizerDirectory.path)")
            logger.info("Runtime embeddings: \(frontend.runtimeEmbeddingsFile.path)")
            logger.info("Special tokens: \(frontend.specialTokensFile.path)")

            let voiceURL = try await CosyVoice3ResourceDownloader.ensureVoice(
                repoDirectory: repoDir)
            logger.info("Default voice bundle: \(voiceURL.path)")

            let elapsed = Date().timeIntervalSince(tStart)
            logger.info("CosyVoice3 download complete in \(String(format: "%.1fs", elapsed))")
        } catch {
            logger.error("CosyVoice3 download failed: \(error)")
            exit(2)
        }
    }
}
