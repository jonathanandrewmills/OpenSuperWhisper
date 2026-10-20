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
    /// segments: Whisper's text already carries its own leading space. A chunk
    /// that starts without one would glue two words together, so a space is
    /// added there, except next to Chinese or Japanese, which use no spaces.
    static func join(_ parts: [String]) -> String {
        var result = ""
        for part in parts where !part.isEmpty {
            let part = droppingRepeatedLead(of: part, after: result)
            if let last = result.last, let first = part.first,
               needsSpace(between: last, and: first) {
                result.append(" ")
            }
            result.append(part)
        }
        return result
    }

    /// Whisper, prompted with the text so far, sometimes opens a chunk that
    /// starts mid-sentence by repeating a phrase: its own first words, or the
    /// last words of the chunk before. Chunks never share audio, so such a
    /// repeat right at the join is dropped. Short repeats are kept, since
    /// speech has those ("that that", "two words, two words", "谢谢").
    static func droppingRepeatedLead(of text: String, after previous: String) -> String {
        let words = words(in: text)
        let earlier = Array(Self.words(in: previous).suffix(8))
        func isPhrase(_ phrase: ArraySlice<LiveWord>) -> Bool {
            phrase.count >= 2 && phrase.reduce(0) { $0 + $1.key.count } >= 4
        }
        func same(_ lhs: ArraySlice<LiveWord>, _ rhs: ArraySlice<LiveWord>) -> Bool {
            lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { $0.key == $1.key }
        }
        var repeated = 0
        for length in stride(from: min(8, words.count / 2), through: 3, by: -1)
        where isPhrase(words[0..<length]) && same(words[0..<length], words[length..<2 * length]) {
            repeated = length
            break
        }
        if repeated == 0 {
            for length in stride(from: min(8, words.count, earlier.count), through: 2, by: -1)
            where isPhrase(words[0..<length]) && same(words[0..<length], earlier.suffix(length)[...]) {
                repeated = length
                break
            }
        }
        guard repeated > 0, repeated < words.count else { return text }
        let leading = words[0].text.prefix(while: \.isWhitespace)
        let kept = words[repeated...]
        return String(leading) + String(kept.first!.text.drop(while: \.isWhitespace)) + kept.dropFirst().map(\.text).joined()
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
