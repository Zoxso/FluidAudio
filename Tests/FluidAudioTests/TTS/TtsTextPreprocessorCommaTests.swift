import XCTest

@testable import FluidAudio

/// Regression tests for comma handling in `removeCommasFromNumbers`.
///
/// The original implementation matched number groups with a regex but then ran
/// `.replacingOccurrences(of: ",", with: "")` on the WHOLE string, stripping
/// every clause comma — not just thousands separators. Downstream, Kokoro
/// needs commas to survive into the phoneme/token stream to produce clause
/// pauses, so the over-broad strip flattened prosody (the voice "read straight
/// through" commas). The fix removes a comma only when it sits between digits.
final class TtsTextPreprocessorCommaTests: XCTestCase {

    // MARK: - Clause commas must survive (the reported bug)

    func testClauseCommaIsPreserved() {
        let output = TtsTextPreprocessor.preprocess("He paused, then spoke.")
        XCTAssertTrue(output.contains(","), "Clause comma was stripped, got: \(output)")
    }

    func testMultipleClauseCommasPreserved() {
        let output = TtsTextPreprocessor.preprocess("One, two, three.")
        XCTAssertEqual(
            output.filter { $0 == "," }.count, 2,
            "Expected both clause commas to survive, got: \(output)"
        )
    }

    // MARK: - Thousands separators must still be removed

    func testThousandsSeparatorRemoved() {
        // No clause commas → no comma should remain after the separator is stripped.
        let output = TtsTextPreprocessor.preprocess("It cost 1,000 dollars")
        XCTAssertFalse(output.contains(","), "Thousands separator not removed, got: \(output)")
    }

    func testMultiGroupThousandsSeparatorRemoved() {
        let output = TtsTextPreprocessor.preprocess("Population 1,000,000 today")
        XCTAssertFalse(output.contains(","), "Multi-group separators not removed, got: \(output)")
    }

    // MARK: - Mixed: in-number comma stripped, clause comma kept

    func testNumberSeparatorStrippedButClauseCommaKept() {
        // "1,500" loses its separator; the clause comma that follows survives.
        let output = TtsTextPreprocessor.preprocess("Total was 1,500, then it doubled.")
        XCTAssertEqual(
            output.filter { $0 == "," }.count, 1,
            "Only the clause comma should remain after the separator is removed, got: \(output)"
        )
    }
}
