import XCTest

@testable import FluidAudio

/// Verifies the `phonemeRanges` field added to `TextChunk` so per-word timing aggregation
/// downstream can attribute synthesizer output back to source atoms.
final class KokoroChunkerPhonemeRangesTests: XCTestCase {

    private let allowed: Set<String> = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "n", "o", "r", "t", " ", ".", ","]

    func testWordRangesAlignWithAtoms() async throws {
        let lexicon: [String: [String]] = [
            "hi": ["h", "i"],
            "bye": ["b", "i"],
        ]

        let chunks = try await KokoroChunker.chunk(
            text: "hi bye",
            wordToPhonemes: lexicon,
            caseSensitiveLexicon: [:],
            customLexicon: nil,
            targetTokens: 120,
            hasLanguageToken: false,
            allowedPhonemes: allowed,
            phoneticOverrides: []
        )

        XCTAssertEqual(chunks.count, 1)
        guard let chunk = chunks.first else {
            XCTFail("missing chunk output")
            return
        }

        // atoms.count and phonemeRanges.count must always match.
        XCTAssertEqual(chunk.atoms.count, chunk.phonemeRanges.count)

        // For "hi bye": phonemes laid out as [h, i, " ", b, i] with separator at index 2.
        // atoms[0]="hi" → [0..<2], atoms[1]="bye" → [3..<5]; separator excluded.
        XCTAssertEqual(chunk.atoms, ["hi", "bye"])
        XCTAssertEqual(chunk.phonemeRanges[0], 0..<2)
        XCTAssertEqual(chunk.phonemeRanges[1], 3..<5)

        // Each range must reference real phoneme indices.
        for range in chunk.phonemeRanges {
            XCTAssertGreaterThanOrEqual(range.lowerBound, 0)
            XCTAssertLessThanOrEqual(range.upperBound, chunk.phonemes.count)
            XCTAssertLessThan(range.lowerBound, range.upperBound)
        }
    }

    func testPunctuationGetsItsOwnRange() async throws {
        let lexicon: [String: [String]] = [
            "hi": ["h", "i"],
            "bye": ["b", "i"],
        ]

        let chunks = try await KokoroChunker.chunk(
            text: "hi, bye",
            wordToPhonemes: lexicon,
            caseSensitiveLexicon: [:],
            customLexicon: nil,
            targetTokens: 120,
            hasLanguageToken: false,
            allowedPhonemes: allowed,
            phoneticOverrides: []
        )

        XCTAssertEqual(chunks.count, 1)
        guard let chunk = chunks.first else {
            XCTFail("missing chunk output")
            return
        }
        XCTAssertEqual(chunk.atoms.count, chunk.phonemeRanges.count)

        // Expect atoms ["hi", ",", "bye"]; punctuation atom is its own one-phoneme range.
        guard let commaIndex = chunk.atoms.firstIndex(of: ",") else {
            XCTFail("comma atom missing")
            return
        }
        let commaRange = chunk.phonemeRanges[commaIndex]
        XCTAssertEqual(commaRange.count, 1)
        XCTAssertEqual(chunk.phonemes[commaRange.lowerBound], ",")
    }

    func testRangesMonotonicAndNonOverlapping() async throws {
        let lexicon: [String: [String]] = [
            "one": ["o", "n"],
            "two": ["t"],
            "three": ["t", "h", "r"],
        ]

        let chunks = try await KokoroChunker.chunk(
            text: "one two three",
            wordToPhonemes: lexicon,
            caseSensitiveLexicon: [:],
            customLexicon: nil,
            targetTokens: 120,
            hasLanguageToken: false,
            allowedPhonemes: allowed,
            phoneticOverrides: []
        )

        XCTAssertEqual(chunks.count, 1)
        guard let chunk = chunks.first else {
            XCTFail("missing chunk output")
            return
        }
        XCTAssertEqual(chunk.atoms.count, chunk.phonemeRanges.count)

        for i in 1..<chunk.phonemeRanges.count {
            XCTAssertGreaterThanOrEqual(
                chunk.phonemeRanges[i].lowerBound,
                chunk.phonemeRanges[i - 1].upperBound,
                "ranges must not overlap"
            )
        }
    }
}
