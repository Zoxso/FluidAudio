import Foundation

/// GPT-2 style reversible byte-to-unicode mapping used by Qwen2 BPE.
///
/// Mirrors `transformers.models.qwen2.tokenization_qwen2.bytes_to_unicode`:
/// - Printable ASCII, Latin-1 supplement (¡..¬), and (®..ÿ) map to themselves.
/// - The 68 "unprintable" bytes are remapped to code points 256..323.
///
/// After mapping, every byte of a UTF-8 string becomes a single-code-point
/// unicode character that vocab/merges.txt expect.
public enum Qwen2ByteEncoder {

    /// byte (0..255) → single Unicode scalar.
    public static let byteToUnicode: [Character] = {
        var map = [Character](repeating: Character(" "), count: 256)
        var printable = [Int]()
        printable.reserveCapacity(188)
        printable.append(contentsOf: Int(Character("!").asciiValue!)...Int(Character("~").asciiValue!))
        printable.append(contentsOf: 0xA1...0xAC)
        printable.append(contentsOf: 0xAE...0xFF)

        for b in printable {
            map[b] = Character(UnicodeScalar(b)!)
        }

        var extra = 0
        for b in 0..<256 {
            if !printable.contains(b) {
                let scalar = UnicodeScalar(256 + extra)!
                map[b] = Character(scalar)
                extra += 1
            }
        }
        return map
    }()

    /// Inverse table: Unicode scalar value → byte (0..255). Built lazily.
    public static let unicodeToByte: [UInt32: UInt8] = {
        var dict: [UInt32: UInt8] = [:]
        dict.reserveCapacity(256)
        for (b, ch) in byteToUnicode.enumerated() {
            let scalar = ch.unicodeScalars.first!.value
            dict[scalar] = UInt8(b)
        }
        return dict
    }()

    /// Encode a UTF-8 byte sequence as a string of mapped characters.
    public static func encode(_ bytes: some Sequence<UInt8>) -> String {
        var out = ""
        for b in bytes {
            out.append(byteToUnicode[Int(b)])
        }
        return out
    }
}
