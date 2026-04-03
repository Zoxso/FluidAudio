#if os(macOS)
import CoreML
import FluidAudio
import Foundation

@available(macOS 15, *)
enum CohereEncoderTest {
    static let logger = AppLogger(category: "CohereEncoderTest")

    static func run(arguments: [String]) async {
        do {
            // Load reference mel from Cohere's official processor
            let melPath = URL(fileURLWithPath: "/tmp/cohere_reference_mel.bin")
            logger.info("Loading reference mel from: \(melPath.path)")

            let melData = try Data(contentsOf: melPath)
            logger.info("Loaded \(melData.count) bytes")

            // Convert to MLMultiArray [1, 128, 3000]
            let melArray = try MLMultiArray(shape: [1, 128, 3000], dataType: .float32)
            let ptr = melArray.dataPointer.bindMemory(to: Float.self, capacity: melArray.count)

            // Initialize with zeros first (MLMultiArray doesn't auto-zero)
            ptr.initialize(repeating: 0.0, count: melArray.count)

            // Load reference data (1001 frames from Cohere)
            melData.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
                let floatPtr = buffer.bindMemory(to: Float.self)
                // Copy 128 * 1001 values, rest stays zero-padded
                for mel in 0..<128 {
                    for t in 0..<1001 {
                        let srcIdx = mel * 1001 + t
                        let dstIdx = mel * 3000 + t
                        ptr[dstIdx] = floatPtr[srcIdx]
                    }
                }
            }

            // Verify input mel is valid
            var melMin: Float = Float.greatestFiniteMagnitude
            var melMax: Float = -Float.greatestFiniteMagnitude
            var nanCount = 0
            var infCount = 0
            for i in 0..<melArray.count {
                let val = ptr[i]
                if val.isNaN {
                    nanCount += 1
                } else if val.isInfinite {
                    infCount += 1
                } else {
                    melMin = min(melMin, val)
                    melMax = max(melMax, val)
                }
            }

            logger.info("Loaded reference mel spectrogram from Cohere processor")
            logger.info("  Shape: [1, 128, 3000]")
            logger.info("  Input min: \(String(format: "%.3f", melMin)), max: \(String(format: "%.3f", melMax))")
            logger.info("  NaN count: \(nanCount), Inf count: \(infCount)")
            logger.info(
                "  First 10 values: \(Array(UnsafeBufferPointer(start: ptr, count: 10)).map { String(format: "%.3f", $0) }.joined(separator: ", "))"
            )

            // Load encoder models
            let cacheDir = CohereAsrModels.defaultCacheDirectory()
            logger.info("Loading encoder from: \(cacheDir.path)")

            // Check which encoder is available
            let encoderV4Path = cacheDir.appendingPathComponent("cohere_audio_encoder_v4.mlpackage")
            let encoderV3Path = cacheDir.appendingPathComponent("cohere_audio_encoder_v3.mlpackage")
            let encoderV2Path = cacheDir.appendingPathComponent("cohere_audio_encoder_v2.mlpackage")
            let encoderV1Path = cacheDir.appendingPathComponent("cohere_audio_encoder.mlpackage")

            let encoderName: String
            if FileManager.default.fileExists(atPath: encoderV4Path.path) {
                encoderName = "encoder_v4 (coremltools 8.2)"
            } else if FileManager.default.fileExists(atPath: encoderV3Path.path) {
                encoderName = "encoder_v3 (coremltools 8.1)"
            } else if FileManager.default.fileExists(atPath: encoderV2Path.path) {
                encoderName = "encoder_v2 (coremltools 9.0b1)"
            } else if FileManager.default.fileExists(atPath: encoderV1Path.path) {
                encoderName = "encoder_v1 (coremltools 9.0b1)"
            } else {
                logger.error("No encoder found")
                exit(1)
            }

            logger.info("Testing with: \(encoderName)")

            let models = try await CohereAsrModels.load(from: cacheDir, computeUnits: .cpuOnly)
            logger.info("Encoder loaded successfully")

            // Run encoder
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_features": MLFeatureValue(multiArray: melArray)
            ])

            logger.info("Running encoder...")
            let output = try await models.audioEncoder.prediction(from: input)
            guard let encoderOutput = output.featureValue(for: "encoder_output")?.multiArrayValue else {
                logger.error("No encoder_output")
                exit(1)
            }

            // Check output
            let outPtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
            var minVal: Float = Float.greatestFiniteMagnitude
            var maxVal: Float = -Float.greatestFiniteMagnitude
            var sum: Double = 0
            for i in 0..<encoderOutput.count {
                let val = outPtr[i]
                if val.isFinite {
                    minVal = min(minVal, val)
                    maxVal = max(maxVal, val)
                    sum += Double(val)
                }
            }
            let mean = sum / Double(encoderOutput.count)

            print("\n=== ENCODER OUTPUT ===")
            print("Shape: \(encoderOutput.shape)")
            print("Min:   \(String(format: "%.6f", minVal))")
            print("Max:   \(String(format: "%.6f", maxVal))")
            print("Mean:  \(String(format: "%.6f", mean))")
            print("\n=== EXPECTED (Python) ===")
            print("Min:   -1.185547")
            print("Max:   1.593750")
            print("Mean:  -0.007178")

            if abs(maxVal - 1.59) < 0.1 {
                print("\n✓ SUCCESS: Encoder produces correct outputs!")
                exit(0)
            } else {
                print("\n❌ FAILURE: Encoder outputs are wrong!")
                print("   Difference of ~\(Int((1.59 / maxVal)))x")
                print("   This confirms a CoreML Runtime bug.")
                exit(1)
            }
        } catch {
            logger.error("Error: \(error)")
            exit(1)
        }
    }
}
#endif
