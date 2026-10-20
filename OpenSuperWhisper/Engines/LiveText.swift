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
            if let last = result.last, let first = part.first,
               needsSpace(between: last, and: first) {
                result.append(" ")
            }
            result.append(part)
        }
        return result
    }

    /// Joins decoded chunks into the dictation's text: each chunk's batch
    /// decode, in order, as decoded. The one exception is a chunk that
    /// follows a forced cut, which can split a sentence (see
    /// droppingRepeatedLead). After a pause, a chunk starts a new phrase,
    /// and repeating words is ordinary speech ("I think we should, I think
    /// we should go"; "see you on the call. On the call we will review"),
    /// so it is kept as the batch decode keeps it. The first text is never
    /// trimmed, so a dictation with no cut reads exactly like today's.
    static func joinChunks(_ chunks: [(text: String, followsForcedCut: Bool)]) -> String {
        var result = ""
        for chunk in chunks where !chunk.text.isEmpty {
            let text = !result.isEmpty && chunk.followsForcedCut
                ? droppingRepeatedLead(of: chunk.text, after: result)
                : chunk.text
            result = join([result, text])
        }
        return result
    }

    /// Whisper, prompted with the text so far, sometimes opens a chunk that
    /// starts mid-sentence by repeating a phrase: its own first three or more
    /// words, or two or more of the last words before the cut. Chunks never
    /// share audio, so such a repeat is dropped. Short repeats are kept,
    /// since speech has those ("that that", "two words, two words", "谢谢").
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

extension LiveText {
    /// Keeps a preview pass up to the first copy of a phrase of up to eight
    /// words that repeats three or more times in a row: Whisper's repetition
    /// loops would otherwise fill the indicator.
    static func cutRepetitionLoop(_ words: [LiveWord]) -> [LiveWord] {
        let keys = words.map(\.key)
        guard let (start, length) = firstLoop(in: keys, minimumCopies: 3) else { return words }
        return Array(words.prefix(start + length))
    }

    /// Keeps at most two copies of any looping phrase, so one looping chunk
    /// cannot prompt the next pass into the same loop.
    static func capRepetitions<T: Equatable>(_ items: [T], maximumCopies: Int = 2) -> [T] {
        var result: [T] = []
        var index = 0
        while index < items.count {
            var collapsed = false
            for length in 1...8 where index + length <= items.count {
                let copies = copiesInARow(of: index..<index + length, in: items)
                if copies > maximumCopies {
                    for _ in 0..<maximumCopies { result.append(contentsOf: items[index..<index + length]) }
                    index += copies * length
                    collapsed = true
                    break
                }
            }
            if !collapsed {
                result.append(items[index])
                index += 1
            }
        }
        return result
    }

    private static func firstLoop<T: Equatable>(in items: [T], minimumCopies: Int) -> (Int, Int)? {
        for start in items.indices {
            for length in 1...8 where start + length * minimumCopies <= items.count {
                if copiesInARow(of: start..<start + length, in: items) >= minimumCopies {
                    return (start, length)
                }
            }
        }
        return nil
    }

    private static func copiesInARow<T: Equatable>(of phrase: Range<Int>, in items: [T]) -> Int {
        var copies = 1
        var next = phrase.upperBound
        while next + phrase.count <= items.count,
              items[next..<next + phrase.count].elementsEqual(items[phrase]) {
            copies += 1
            next += phrase.count
        }
        return copies
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
