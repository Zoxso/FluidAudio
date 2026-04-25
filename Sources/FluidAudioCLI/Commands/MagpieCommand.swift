#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// CLI surface for the Magpie TTS Multilingual Swift port.
///
/// Subcommands:
///   - `download`             Fetch models + constants + tokenizer data from HuggingFace.
///   - `text`                 Synthesize text → WAV.
///   - `parity`               Compare Swift intermediates against a Python fixture (Phase 5).
///   - `tokenizer-parity`     Compare Swift tokenizer output against a language fixture.
public enum MagpieCommand {

    private static let logger = AppLogger(category: "MagpieCommand")

    public static func run(arguments: [String]) async {
        guard let sub = arguments.first else {
            printUsage()
            return
        }
        let rest = Array(arguments.dropFirst())
        switch sub {
        case "download":
            await runDownload(arguments: rest)
        case "text":
            await runText(arguments: rest)
        case "parity":
            await runParity(arguments: rest)
        case "tokenizer-parity":
            await runTokenizerParity(arguments: rest)
        case "help", "--help", "-h":
            printUsage()
        default:
            logger.error("Unknown magpie subcommand: \(sub)")
            printUsage()
            exit(1)
        }
    }

    // MARK: - download

    private static func runDownload(arguments: [String]) async {
        var languageCodes: [String] = ["en"]
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            if arg == "--languages" || arg == "-l", i + 1 < arguments.count {
                languageCodes = arguments[i + 1].split(separator: ",").map(String.init)
                i += 1
            }
            i += 1
        }
        let langs: Set<MagpieLanguage> = Set(languageCodes.compactMap { MagpieLanguage(rawValue: $0) })
        if langs.isEmpty {
            logger.error("No valid language codes provided")
            exit(1)
        }
        do {
            let repoDir = try await MagpieResourceDownloader.ensureAssets(languages: langs)
            logger.info("Magpie assets ready at: \(repoDir.path)")
        } catch {
            logger.error("Magpie download failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - text

    private static func runText(arguments: [String]) async {
        var text: String? = nil
        var output = "magpie.wav"
        var speakerIdx = MagpieSpeaker.john.rawValue
        var languageCode = "en"
        var cfg: Float = MagpieConstants.defaultCfgScale
        var topK = MagpieConstants.defaultTopK
        var temperature = MagpieConstants.defaultTemperature
        var seed: UInt64? = nil
        var allowIpa = true

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--speaker":
                if i + 1 < arguments.count, let idx = Int(arguments[i + 1]) {
                    speakerIdx = idx
                    i += 1
                }
            case "--language", "-L":
                if i + 1 < arguments.count {
                    languageCode = arguments[i + 1]
                    i += 1
                }
            case "--cfg":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    cfg = v
                    i += 1
                }
            case "--topk":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    topK = v
                    i += 1
                }
            case "--temperature":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    temperature = v
                    i += 1
                }
            case "--seed":
                if i + 1 < arguments.count, let v = UInt64(arguments[i + 1]) {
                    seed = v
                    i += 1
                }
            case "--no-ipa-override":
                allowIpa = false
            default:
                if text == nil { text = arg }
            }
            i += 1
        }

        guard let text = text, !text.isEmpty else {
            logger.error("Missing text argument")
            printUsage()
            exit(1)
        }
        guard let speaker = MagpieSpeaker(rawValue: speakerIdx) else {
            logger.error("Invalid speaker index \(speakerIdx); valid range 0..<\(MagpieConstants.numSpeakers)")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language code '\(languageCode)'")
            exit(1)
        }

        do {
            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            let opts = MagpieSynthesisOptions(
                temperature: temperature,
                topK: topK,
                maxSteps: MagpieConstants.maxSteps,
                minFrames: MagpieConstants.minFrames,
                cfgScale: cfg,
                seed: seed,
                peakNormalize: true,
                allowIpaOverride: allowIpa)
            let start = Date()
            let result = try await manager.synthesize(
                text: text, speaker: speaker, language: language, options: opts)
            let elapsed = Date().timeIntervalSince(start)

            let wav = try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(result.sampleRate))
            let outURL = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try wav.write(to: outURL)

            let audioSecs = result.durationSeconds
            let rtfx = elapsed > 0 ? audioSecs / elapsed : 0
            logger.info("Magpie synthesis complete")
            logger.info("  Speaker: \(speaker.displayName), Language: \(language.rawValue)")
            logger.info("  Codes: \(result.codeCount), EOS: \(result.finishedOnEos)")
            logger.info(
                "  Audio: \(String(format: "%.3f", audioSecs))s, Synthesis: \(String(format: "%.3f", elapsed))s, RTFx: \(String(format: "%.2f", rtfx))x"
            )
            logger.info("  Output: \(outURL.path)")
        } catch {
            logger.error("Magpie synthesis failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - parity (stub)

    private static func runParity(arguments: [String]) async {
        var fixturePath: String? = nil
        var i = 0
        while i < arguments.count {
            if arguments[i] == "--fixture", i + 1 < arguments.count {
                fixturePath = arguments[i + 1]
                i += 1
            }
            i += 1
        }
        guard let fixturePath = fixturePath else {
            logger.error("--fixture <path> is required for magpie parity")
            exit(1)
        }
        let url = URL(fileURLWithPath: fixturePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.error("Fixture not found at \(url.path)")
            logger.info(
                "Emit one from mobius using: uv run python generate_coreml.py --emit-fixture <out.json>")
            exit(1)
        }

        do {
            let fixture = try MagpieParityFixture.load(from: url)
            logger.info(
                "Loaded fixture: text=\"\(fixture.text)\" speaker=\(fixture.speakerIndex) language=\(fixture.languageCode)"
            )

            guard let language = MagpieLanguage(rawValue: fixture.languageCode) else {
                logger.error("Fixture language '\(fixture.languageCode)' not supported in Swift port")
                exit(1)
            }

            // Stage 1 — tokenize and compare token ids.
            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            _ = manager  // parity comparison will grow once mobius emits fixture intermediates.

            let expected = fixture.expectedTokenIds
            logger.info("Fixture contains \(expected.count) expected token ids (parity harness Phase 5 stub)")
            logger.info(
                "Full per-stage parity (encoder_output, caches, LT samples, audio) will light up once the mobius exporter emits NPZ intermediates; see plan Phase 5."
            )
        } catch {
            logger.error("Parity harness failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - tokenizer-parity (stub)

    private static func runTokenizerParity(arguments: [String]) async {
        var languageCode = "en"
        var fixturePath: String? = nil
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            if arg == "--language" || arg == "-L", i + 1 < arguments.count {
                languageCode = arguments[i + 1]
                i += 1
            } else if arg == "--fixture", i + 1 < arguments.count {
                fixturePath = arguments[i + 1]
                i += 1
            }
            i += 1
        }
        guard let fixturePath = fixturePath else {
            logger.error("--fixture <path> is required")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language '\(languageCode)'")
            exit(1)
        }

        do {
            let url = URL(fileURLWithPath: fixturePath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                logger.error("Fixture not found at \(url.path)")
                exit(1)
            }
            let data = try Data(contentsOf: url)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let text = json["text"] as? String,
                let expected = json["token_ids"] as? [Int]
            else {
                logger.error("Fixture must be a JSON object with keys {text, token_ids}")
                exit(1)
            }

            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            _ = manager
            // Tokenizer is actor-internal; we construct a second tokenizer view against the
            // same on-disk tokenizer directory for parity.
            let repoDir = try await MagpieResourceDownloader.ensureAssets(languages: [language])
            let tokenizerDir = MagpieResourceDownloader.tokenizerDirectory(in: repoDir)
            let tok = MagpieTokenizer(tokenizerDir: tokenizerDir, eosId: 0)
            let tokenized = try await tok.tokenize(
                text, language: language, options: MagpieSynthesisOptions())
            let actual = Swift.Array(tokenized.paddedIds.prefix(tokenized.realLength))
            let expectedInt32 = expected.map { Int32($0) }

            let match = actual == expectedInt32
            if match {
                logger.info("Tokenizer parity OK (\(actual.count) tokens)")
            } else {
                logger.error("Tokenizer parity MISMATCH")
                logger.error("  expected: \(expectedInt32.prefix(32))… (\(expectedInt32.count) tokens)")
                logger.error("  actual:   \(actual.prefix(32))… (\(actual.count) tokens)")
                exit(1)
            }
        } catch {
            logger.error("Tokenizer parity failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - usage

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio magpie <subcommand> [options]

            Subcommands:
              download                Download Magpie models + constants + tokenizers
                --languages en,es,de    Comma-separated language codes (default: en)

              text "<text>"           Synthesize text and write a WAV file
                --output, -o PATH       Output WAV path (default: magpie.wav)
                --speaker N             Speaker index 0-4 (default: 0 = John)
                --language CODE         Language code (en, es, de, fr, it, vi, zh, hi)
                --cfg FLOAT             CFG guidance scale (default: 1.0 = off)
                --topk N                Top-K sampling (default: 80)
                --temperature FLOAT     Sampling temperature (default: 0.6)
                --seed N                Deterministic RNG seed
                --no-ipa-override       Disable `|…|` IPA pass-through

              parity --fixture PATH   Run Swift-side parity against a mobius fixture
              tokenizer-parity --fixture PATH --language CODE
                                      Verify tokenizer matches a fixture {text, token_ids}

            IPA override example:
              fluidaudio magpie text "Hello | ˈ n ɛ m o ʊ | Text." --output demo.wav

            """
        )
    }
}

// MARK: - Fixture loader

/// Minimal fixture shape the mobius exporter is expected to emit. Only the stable
/// fields are declared; additional intermediate tensors will be added in Phase 5 once
/// the exporter lands on the Python side.
private struct MagpieParityFixture: Decodable {
    let text: String
    let speakerIndex: Int
    let languageCode: String
    let expectedTokenIds: [Int32]

    enum CodingKeys: String, CodingKey {
        case text
        case speakerIndex = "speaker_index"
        case languageCode = "language"
        case expectedTokenIds = "token_ids"
    }

    static func load(from url: URL) throws -> MagpieParityFixture {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MagpieParityFixture.self, from: data)
    }
}
#endif
