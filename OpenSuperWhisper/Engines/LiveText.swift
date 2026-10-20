import Foundation

/// A word of transcribed text, or one character of a script written without
/// spaces. `text` keeps the whitespace before it, so words join back into
/// the text they came from.
struct LiveWord: Equatable {
    let text: String

    /// Compared without case or punctuation, so "Hello," matches "hello".
    var key: String {
        let letters = text.lowercased().filter { $0.isLetter || $0.isNumber }
        return letters.isEmpty ? text.trimmingCharacters(in: .whitespaces) : letters
    }
}

/// Text handling for live transcription.
enum LiveText {
    /// Joins chunk texts the way WhisperEngine.assembleSegmentTexts joins
    /// segments: each text exactly as decoded, in order, since Whisper's text
    /// already carries its own leading space. A chunk that starts without one
    /// would glue two words together, so a space is added there, except next
    /// to Chinese or Japanese, which use no spaces. Nothing else is changed:
    /// a phrase Whisper repeats at a chunk's start stays, as the batch decode
    /// of that chunk has it.
    static func join(_ parts: [String]) -> String {
        var result = ""
        for part in parts where !part.isEmpty {
            if let last = result.last, let first = part.first,
               needsSpace(between: last, and: first) {
                result.append(" ")
            }
            result.append(part)
        }
        return result
    }

    static func words(in text: String) -> [LiveWord] {
        var words: [LiveWord] = []
        var leading = ""
        var word = ""
        func flush() {
            guard !word.isEmpty else { return }
            words.append(LiveWord(text: leading + word))
            leading = ""
            word = ""
        }
        for character in text {
            if character.isWhitespace {
                flush()
                leading.append(character)
            } else if isUnspacedScript(character) {
                flush()
                word = String(character)
                flush()
            } else {
                word.append(character)
            }
        }
        flush()
        return words
    }

    private static func needsSpace(between last: Character, and first: Character) -> Bool {
        guard !last.isWhitespace, !first.isWhitespace,
              !isUnspacedScript(last), !isUnspacedScript(first) else { return false }
        let endsWord = last.isLetter || last.isNumber || ".,!?;:)]}…%\"'»”’".contains(last)
        let startsWord = first.isLetter || first.isNumber || "([{«“‘\"'".contains(first)
        return endsWord && startsWord
    }

    /// Han, kana and their punctuation: scripts written without spaces.
    /// Hangul is written with spaces, so it is not included.
    static func isUnspacedScript(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x3000...0x303F, // CJK symbols and punctuation
                 0x3040...0x30FF, // Hiragana, Katakana
                 0x31F0...0x31FF, // Katakana phonetic extensions
                 0x3400...0x4DBF, // CJK Extension A
                 0x4E00...0x9FFF, // CJK Unified Ideographs
                 0xF900...0xFAFF, // CJK Compatibility Ideographs
                 0xFF00...0xFFEF, // Halfwidth and fullwidth forms
                 0x20000...0x2FA1F: // CJK Extensions B and later
                return true
            default:
                return false
            }
        }
    }
}

/// LocalAgreement-2 over preview passes of the open chunk: a word is settled
/// once two passes in a row agree on it and on everything before it. Settled
/// words never change within the chunk, so the indicator does not flicker.
struct LiveAgreement {
    private(set) var settled: [LiveWord] = []
    private(set) var latest: [LiveWord] = []
    private var previous: [LiveWord]?

    /// Words of the latest pass after the settled ones.
    var unsettled: [LiveWord] { Array(latest.dropFirst(settled.count)) }

    var settledText: String { settled.map(\.text).joined() }
    var unsettledText: String { unsettled.map(\.text).joined() }

    mutating func update(with words: [LiveWord]) {
        defer {
            previous = words
            latest = words
        }
        guard let previous,
              Self.agree(previous.prefix(settled.count), settled[...]),
              Self.agree(words.prefix(settled.count), settled[...]) else { return }
        var agreed = settled.count
        while agreed < min(previous.count, words.count), previous[agreed].key == words[agreed].key {
            agreed += 1
        }
        settled += words[settled.count..<agreed]
    }

    private static func agree(_ lhs: ArraySlice<LiveWord>, _ rhs: ArraySlice<LiveWord>) -> Bool {
        lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { $0.key == $1.key }
    }
}
