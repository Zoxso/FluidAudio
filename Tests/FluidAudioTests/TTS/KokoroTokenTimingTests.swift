import XCTest

@testable import FluidAudio

/// Unit coverage for the timing-derivation helpers introduced alongside `pred_dur` exposure.
/// These are pure functions on synthesized inputs — no Core ML model required.
final class KokoroTokenTimingTests: XCTestCase {

    // MARK: - buildTokenTimings

    /// One frame is `kokoroFrameSamples / audioSampleRate` seconds. Verify the helper
    /// uses both constants correctly so future tweaks to either don't silently drift timings.
    func testTokenTimingsUseFrameAndSampleRateConstants() throws {
        let predDur: [Float] = [2.0, 1.0, 3.0]
        let phonemes = ["a", "b", "c"]

        let timings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: predDur, phonemes: phonemes)
        )
        XCTAssertEqual(timings.count, 3)

        let frameSeconds =
            TimeInterval(TtsConstants.kokoroFrameSamples) / TimeInterval(TtsConstants.audioSampleRate)
        XCTAssertEqual(timings[0].startTime, 0.0, accuracy: 1e-9)
        XCTAssertEqual(timings[0].endTime, 2.0 * frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(timings[1].startTime, 2.0 * frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(timings[1].endTime, 3.0 * frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(timings[2].startTime, 3.0 * frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(timings[2].endTime, 6.0 * frameSeconds, accuracy: 1e-9)
    }

    func testTokenTimingsCarryPhonemeAndFrames() throws {
        let predDur: [Float] = [1.5, 2.5]
        let phonemes = ["h", "ɛ"]

        let timings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: predDur, phonemes: phonemes)
        )
        XCTAssertEqual(timings.map(\.phoneme), phonemes)
        XCTAssertEqual(timings.map(\.frames), predDur)
    }

    func testTokenTimingsMonotonic() throws {
        let predDur: [Float] = [4, 1, 2, 3, 1]
        let phonemes = ["a", "b", "c", "d", "e"]

        let timings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: predDur, phonemes: phonemes)
        )
        for i in 1..<timings.count {
            XCTAssertGreaterThanOrEqual(timings[i].startTime, timings[i - 1].endTime)
        }
    }

    func testTokenTimingsLastEndMatchesTotalFrames() throws {
        let predDur: [Float] = [3, 7, 2, 5]
        let phonemes = ["a", "b", "c", "d"]

        let timings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: predDur, phonemes: phonemes)
        )
        let frameSeconds =
            TimeInterval(TtsConstants.kokoroFrameSamples) / TimeInterval(TtsConstants.audioSampleRate)
        let totalFrames = predDur.reduce(0, +)
        XCTAssertEqual(timings.last?.endTime ?? 0, TimeInterval(totalFrames) * frameSeconds, accuracy: 1e-9)
    }

    func testTokenTimingsReturnsNilOnLengthMismatch() {
        XCTAssertNil(
            KokoroSynthesizer.buildTokenTimings(predDur: [1.0, 2.0], phonemes: ["a"])
        )
        XCTAssertNil(
            KokoroSynthesizer.buildTokenTimings(predDur: [1.0], phonemes: ["a", "b"])
        )
    }

    func testTokenTimingsReturnsNilOnEmpty() {
        XCTAssertNil(
            KokoroSynthesizer.buildTokenTimings(predDur: [], phonemes: [])
        )
    }

    // MARK: - buildWordTimings

    func testWordTimingsAggregateAcrossPhonemeRanges() throws {
        let frameSeconds =
            TimeInterval(TtsConstants.kokoroFrameSamples) / TimeInterval(TtsConstants.audioSampleRate)

        // 5 phonemes split as: word "hi" = [0..<2], punct "," = [2..<3], word "bye" = [3..<5]
        let predDur: [Float] = [2, 3, 1, 4, 2]
        let phonemes = ["h", "ɪ", ",", "b", "aɪ"]
        let atoms = ["hi", ",", "bye"]
        let phonemeRanges: [Range<Int>] = [0..<2, 2..<3, 3..<5]

        let tokenTimings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: predDur, phonemes: phonemes)
        )
        let wordTimings = try XCTUnwrap(
            KokoroSynthesizer.buildWordTimings(
                tokenTimings: tokenTimings,
                atoms: atoms,
                phonemeRanges: phonemeRanges
            )
        )

        XCTAssertEqual(wordTimings.count, 3)

        XCTAssertEqual(wordTimings[0].word, "hi")
        XCTAssertEqual(wordTimings[0].atomIndex, 0)
        XCTAssertEqual(wordTimings[0].startTime, 0.0, accuracy: 1e-9)
        XCTAssertEqual(wordTimings[0].endTime, 5.0 * frameSeconds, accuracy: 1e-9)

        XCTAssertEqual(wordTimings[1].word, ",")
        XCTAssertEqual(wordTimings[1].atomIndex, 1)
        XCTAssertEqual(wordTimings[1].startTime, 5.0 * frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(wordTimings[1].endTime, 6.0 * frameSeconds, accuracy: 1e-9)

        XCTAssertEqual(wordTimings[2].word, "bye")
        XCTAssertEqual(wordTimings[2].atomIndex, 2)
        XCTAssertEqual(wordTimings[2].startTime, 6.0 * frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(wordTimings[2].endTime, 12.0 * frameSeconds, accuracy: 1e-9)
    }

    func testWordTimingsSkipAtomsBeyondAvailableTokens() throws {
        // Truncation case: tokenTimings shorter than the chunker's full atom set.
        let predDur: [Float] = [1, 1]
        let phonemes = ["a", "b"]
        let tokenTimings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: predDur, phonemes: phonemes)
        )
        let atoms = ["one", "two", "three"]
        let phonemeRanges: [Range<Int>] = [0..<1, 1..<2, 2..<3]  // third atom out of range

        let wordTimings = try XCTUnwrap(
            KokoroSynthesizer.buildWordTimings(
                tokenTimings: tokenTimings,
                atoms: atoms,
                phonemeRanges: phonemeRanges
            )
        )
        XCTAssertEqual(wordTimings.count, 2)
        XCTAssertEqual(wordTimings.map(\.word), ["one", "two"])
    }

    func testWordTimingsClipPartialRange() throws {
        // Atom range partially overlaps available tokens — should clip end to last available token.
        let predDur: [Float] = [1, 2, 3]
        let phonemes = ["a", "b", "c"]
        let tokenTimings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: predDur, phonemes: phonemes)
        )
        let atoms = ["partial"]
        let phonemeRanges: [Range<Int>] = [1..<5]  // upperBound past available tokens

        let wordTimings = try XCTUnwrap(
            KokoroSynthesizer.buildWordTimings(
                tokenTimings: tokenTimings,
                atoms: atoms,
                phonemeRanges: phonemeRanges
            )
        )
        XCTAssertEqual(wordTimings.count, 1)
        XCTAssertEqual(wordTimings[0].startTime, tokenTimings[1].startTime, accuracy: 1e-9)
        XCTAssertEqual(wordTimings[0].endTime, tokenTimings[2].endTime, accuracy: 1e-9)
    }

    func testWordTimingsReturnsNilOnLengthMismatch() throws {
        let tokenTimings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: [1, 1], phonemes: ["a", "b"])
        )
        XCTAssertNil(
            KokoroSynthesizer.buildWordTimings(
                tokenTimings: tokenTimings,
                atoms: ["x"],
                phonemeRanges: [0..<1, 1..<2]   // atoms.count != phonemeRanges.count
            )
        )
    }

    func testWordTimingsReturnsNilWhenEveryRangeOutOfBounds() throws {
        let tokenTimings = try XCTUnwrap(
            KokoroSynthesizer.buildTokenTimings(predDur: [1, 1], phonemes: ["a", "b"])
        )
        XCTAssertNil(
            KokoroSynthesizer.buildWordTimings(
                tokenTimings: tokenTimings,
                atoms: ["x", "y"],
                phonemeRanges: [5..<6, 7..<8]   // both ranges past tokenTimings.count
            )
        )
    }
}
