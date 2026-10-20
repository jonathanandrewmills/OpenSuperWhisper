import Foundation
import XCTest
@testable import OpenSuperWhisper

final class ChunkCutterTests: XCTestCase {
    private let speech: Float = 0.9
    private let silence: Float = 0.05

    /// Feeds frames and returns every cut, as frame indices.
    private func cuts(_ frames: [Float], cutter: inout ChunkCutter) -> [Int] {
        frames.compactMap { cutter.append($0) }
    }

    private func frames(_ value: Float, _ count: Int) -> [Float] {
        Array(repeating: value, count: count)
    }

    func testFrameCountsFollowTheDesign() {
        let parameters = LiveTranscriptionParameters.standard
        XCTAssertEqual(parameters.frames(parameters.minimumChunkDuration), 250)
        XCTAssertEqual(parameters.frames(parameters.maximumChunkDuration), 750)
        XCTAssertEqual(parameters.frames(parameters.cutPauseDuration), 19)
        XCTAssertEqual(parameters.frames(parameters.forcedCutSearchDuration), 125)
        XCTAssertEqual(parameters.frames(parameters.forcedCutStretchDuration), 3)
    }

    func testNoCutBeforeTheMinimumChunkLength() {
        var cutter = ChunkCutter()
        let timeline = frames(speech, 90) + frames(silence, 40) + frames(speech, 100) + frames(silence, 15)
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [])
    }

    func testPauseAfterMinimumCutsInTheMiddleOfThePause() {
        var cutter = ChunkCutter()
        let timeline = frames(speech, 281) + frames(silence, 31) + frames(speech, 100)
        // The cut happens as soon as the pause reaches 19 frames, in its middle.
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [281 + 9])
        XCTAssertEqual(cutter.chunkStart, 290)
    }

    func testPauseStartingBeforeTheMinimumCutsOnceItsMiddlePassesIt() {
        var cutter = ChunkCutter()
        // Silence from frame 240: its middle reaches frame 250 at length 20.
        let timeline = frames(speech, 240) + frames(silence, 30) + frames(speech, 10)
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [250])
    }

    func testShortPauseAfterMinimumDoesNotCut() {
        var cutter = ChunkCutter()
        let timeline = frames(speech, 300) + frames(silence, 18) + frames(speech, 100)
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [])
    }

    func testFramesBetweenThresholdsDoNotStartASilence() {
        var cutter = ChunkCutter()
        let timeline = frames(speech, 300) + frames(0.4, 40) + frames(speech, 10) + frames(0.3, 19)
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [350 + 9])
    }

    func testForcedCutAtTheMaximumUsesTheQuietestStretch() {
        var cutter = ChunkCutter()
        var timeline = frames(speech, 750)
        for index in 700..<703 { timeline[index] = 0.1 }
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [701])
        XCTAssertEqual(cutter.chunkStart, 701)
    }

    func testForcedCutWithoutAQuietStretchKeepsTheChunkLong() {
        var cutter = ChunkCutter()
        XCTAssertEqual(cuts(frames(speech, 750), cutter: &cutter), [748])
    }

    func testEveryChunkAfterACutNeedsTheMinimumAgain() {
        var cutter = ChunkCutter()
        let timeline = frames(speech, 281) + frames(silence, 31) // cut at 290
            + frames(speech, 100) + frames(silence, 25)            // too soon
            + frames(speech, 150) + frames(silence, 25)            // 8 s reached
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [290, 587 + 9])
    }

    func testOngoingSilenceAfterAPauseCutDoesNotCutAgainRightAway() {
        var cutter = ChunkCutter()
        let timeline = frames(speech, 281) + frames(silence, 200)
        XCTAssertEqual(cuts(timeline, cutter: &cutter), [290])
    }
}

final class LivePromptTests: XCTestCase {
    func testCarriedInitialPromptStaysFirstAndHistoryFillsTheRest() {
        let initial: [WhisperToken] = [1, 2, 3]
        let history = (100..<600).map { WhisperToken($0) }
        let tokens = LivePrompt.tokens(initialPrompt: initial, carriesInitialPrompt: true, history: history, capacity: 224)
        XCTAssertEqual(tokens.count, 223)
        XCTAssertEqual(Array(tokens.prefix(3)), initial)
        XCTAssertEqual(Array(tokens.suffix(220)), Array(history.suffix(220)))
    }

    func testUncarriedInitialPromptIsTheOldestTextAndDropsFirst() {
        let initial = (1...150).map { WhisperToken($0) }
        let history = (1000..<1100).map { WhisperToken($0) }
        let tokens = LivePrompt.tokens(initialPrompt: initial, carriesInitialPrompt: false, history: history, capacity: 224)
        XCTAssertEqual(tokens, Array((initial + history).suffix(223)))
    }

    func testShortHistoryIsKeptWhole() {
        let tokens = LivePrompt.tokens(initialPrompt: [], carriesInitialPrompt: false, history: [7, 8, 9], capacity: 224)
        XCTAssertEqual(tokens, [7, 8, 9])
    }
}

final class LiveTextJoinTests: XCTestCase {
    func testWhisperLeadingSpacesAreKept() {
        XCTAssertEqual(LiveText.join([" Hello world.", " Next part."]), " Hello world. Next part.")
    }

    func testMissingSpaceBetweenWordsIsAdded() {
        XCTAssertEqual(LiveText.join([" Hello", "world"]), " Hello world")
        XCTAssertEqual(LiveText.join([" Done.", "Next"]), " Done. Next")
    }

    func testPunctuationAttachesWithoutASpace() {
        XCTAssertEqual(LiveText.join([" end", ", and more"]), " end, and more")
        XCTAssertEqual(LiveText.join([" Really", "?"]), " Really?")
    }

    func testChineseAndJapaneseJoinWithoutSpaces() {
        XCTAssertEqual(LiveText.join(["你好。", "世界"]), "你好。世界")
        XCTAssertEqual(LiveText.join(["こんにちは", "世界"]), "こんにちは世界")
    }

    func testKoreanIsSpaced() {
        XCTAssertEqual(LiveText.join(["안녕하세요", "감사합니다"]), "안녕하세요 감사합니다")
    }

    func testEmptyChunksAreSkipped() {
        XCTAssertEqual(LiveText.join(["", " One.", "", " Two."]), " One. Two.")
        XCTAssertEqual(LiveText.join(["", ""]), "")
    }

    func testAChunkRepeatingItsOwnOpeningPhraseKeepsOneCopy() {
        // Turbo, prompted with the text so far, decoded the chunk that began
        // mid-sentence in the long English fixture this way.
        XCTAssertEqual(
            LiveText.join([" If this phrase is present,", " the last part of the last part of the audio was not clipped."]),
            " If this phrase is present, the last part of the audio was not clipped."
        )
    }

    func testAChunkRepeatingTheEndOfThePreviousOneDropsTheRepeat() {
        XCTAssertEqual(
            LiveText.join([" The final checkpoint phrase is", " checkpoint phrase is silver compass."]),
            " The final checkpoint phrase is silver compass."
        )
        XCTAssertEqual(LiveText.join([" Remember the Alamo,", " Remember the Alamo!"]), " Remember the Alamo, Remember the Alamo!",
                       "A chunk that is nothing but the repeat is kept whole")
    }

    func testShortRepeatsAndRepeatsAwayFromTheJoinAreKept() {
        XCTAssertEqual(LiveText.join([" I said that", " that is fine."]), " I said that that is fine.")
        XCTAssertEqual(LiveText.join(["谢谢", "谢谢你"]), "谢谢谢谢你")
        XCTAssertEqual(LiveText.join([" One.", " Two words, two words again."]), " One. Two words, two words again.")
    }
}

final class LiveTextWordsTests: XCTestCase {
    func testWordsKeepTheirLeadingSpace() {
        XCTAssertEqual(LiveText.words(in: " Hello, world.").map(\.text), [" Hello,", " world."])
        XCTAssertEqual(LiveText.words(in: " Hello, world.").map(\.text).joined(), " Hello, world.")
    }

    func testChineseAndJapaneseSplitIntoCharacters() {
        XCTAssertEqual(LiveText.words(in: "你好 world").map(\.text), ["你", "好", " world"])
        XCTAssertEqual(LiveText.words(in: "こんにちは").count, 5)
    }

    func testKeysIgnoreCaseAndPunctuation() {
        XCTAssertEqual(LiveWord(text: " Hello,").key, LiveWord(text: "hello").key)
        XCTAssertEqual(LiveWord(text: " —").key, "—")
    }
}

final class LiveTranscriptionSettingsTests: XCTestCase {
    private let keys = ["liveTranscription", "selectedEngine", "showTimestamps"]
    private var saved: [String: Any] = [:]

    override func setUp() {
        super.setUp()
        for key in keys {
            saved[key] = AppPreferences.store.object(forKey: key)
            AppPreferences.store.removeObject(forKey: key)
        }
    }

    override func tearDown() {
        for key in keys {
            AppPreferences.store.set(saved[key], forKey: key)
        }
        super.tearDown()
    }

    func testLiveTranscriptionIsOffByDefault() {
        XCTAssertFalse(AppPreferences.shared.liveTranscription)
        XCTAssertFalse(AppPreferences.shared.isLiveTranscriptionActive)
    }

    func testLiveTranscriptionIsWhisperOnlyAndStandsDownForTimestamps() {
        let prefs = AppPreferences.shared
        prefs.liveTranscription = true
        XCTAssertTrue(prefs.isLiveTranscriptionActive)
        prefs.selectedEngine = "fluidaudio"
        XCTAssertFalse(prefs.isLiveTranscriptionActive)
        prefs.selectedEngine = "whisper"
        prefs.showTimestamps = true
        XCTAssertFalse(prefs.isLiveTranscriptionActive)
    }
}

/// Speech is any frame with signal; tests build audio from flat blocks.
final class AmplitudeSpeechProbe: SpeechProbing {
    func probabilities(for frames: ArraySlice<Float>) -> [Float]? {
        let size = MyWhisperVadContext.frameSampleCount
        guard frames.count % size == 0 else { return nil }
        return stride(from: frames.startIndex, to: frames.endIndex, by: size).map { start in
            frames[start..<start + size].contains { abs($0) > 0.01 } ? 0.9 : 0.05
        }
    }
}

/// Records every chunk pass. A pass can be held open with `hold(pass:)`,
/// and `cancel()` releases a held pass with CancellationError, as the native
/// abort callback does.
final class FakeLiveDecoder: LiveDecoding, @unchecked Sendable {
    struct Call {
        let sampleCount: Int
        let history: [WhisperToken]
        let language: String?
    }

    private let lock = NSLock()
    private var callsStorage: [Call] = []
    private var heldPasses: Set<Int> = []
    private var startedSignals: [Int: DispatchSemaphore] = [:]
    private var releaseSignals: [Int: DispatchSemaphore] = [:]
    private(set) var isCancelled = false
    private(set) var isClosed = false
    var failingPass: Int?
    var detectedLanguage: String?
    /// Preview passes return these in turn, then repeat the last one.
    var previewTexts: [String] = [" preview"]
    /// A held preview pass blocks until it is aborted or cancelled.
    var holdsPreviews = false
    var onPreviewStart: ((Int) -> Void)?
    private var previewCallsStorage = 0
    private var abortCountStorage = 0
    private var previewAbort: DispatchSemaphore?

    var previewCalls: Int { lock.withLock { previewCallsStorage } }
    var abortCount: Int { lock.withLock { abortCountStorage } }

    var calls: [Call] {
        lock.lock()
        defer { lock.unlock() }
        return callsStorage
    }

    func hold(pass: Int) {
        lock.lock()
        heldPasses.insert(pass)
        startedSignals[pass] = DispatchSemaphore(value: 0)
        releaseSignals[pass] = DispatchSemaphore(value: 0)
        lock.unlock()
    }

    /// Waits off the main thread until a held pass has started.
    func waitUntilStarted(pass: Int) async {
        let signal = lock.withLock { startedSignals[pass] }
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                signal?.wait()
                continuation.resume()
            }
        }
    }

    func release(pass: Int) {
        lock.withLock { releaseSignals[pass] }?.signal()
    }

    func decodeChunk(_ samples: [Float], history: [WhisperToken], language: String?) throws -> LiveChunkResult {
        lock.lock()
        let index = callsStorage.count
        callsStorage.append(Call(sampleCount: samples.count, history: history, language: language))
        let held = heldPasses.contains(index)
        let started = startedSignals[index]
        let release = releaseSignals[index]
        lock.unlock()

        if held {
            started?.signal()
            release?.wait()
        }
        if lock.withLock({ isCancelled }) { throw CancellationError() }
        if failingPass == index { throw TranscriptionError.processingFailed }
        guard samples.contains(where: { abs($0) > 0.01 }) else {
            return LiveChunkResult(text: "", tokens: [], detectedLanguage: nil)
        }
        return LiveChunkResult(
            text: " chunk\(index).",
            tokens: [WhisperToken(index * 10), WhisperToken(index * 10 + 1)],
            detectedLanguage: index == 0 ? detectedLanguage : nil
        )
    }

    func decodePreview(_ samples: [Float], history: [WhisperToken], language: String?) throws -> String {
        lock.lock()
        let index = previewCallsStorage
        previewCallsStorage += 1
        let text = previewTexts[min(index, previewTexts.count - 1)]
        let abort = holdsPreviews ? DispatchSemaphore(value: 0) : nil
        previewAbort = abort
        let onStart = onPreviewStart
        lock.unlock()

        onStart?(index)
        if let abort {
            abort.wait()
            throw CancellationError()
        }
        if lock.withLock({ isCancelled }) { throw CancellationError() }
        return text
    }

    func abortCurrentPass() {
        lock.lock()
        abortCountStorage += 1
        let abort = previewAbort
        previewAbort = nil
        lock.unlock()
        abort?.signal()
    }

    func cancel() {
        lock.lock()
        isCancelled = true
        let held = Array(releaseSignals.values) + [previewAbort].compactMap { $0 }
        lock.unlock()
        held.forEach { $0.signal() }
    }

    func close() {
        lock.withLock { isClosed = true }
    }
}

final class LiveTranscriberTests: XCTestCase {
    private let sampleRate = 16_000

    private func audio(_ parts: [(seconds: Double, speech: Bool)]) -> [Float] {
        parts.flatMap { part in
            Array(repeating: part.speech ? Float(0.5) : 0, count: Int(part.seconds * Double(sampleRate)))
        }
    }

    /// 10 s of speech, a 1 s pause, 10 s of speech, a 1 s pause, 5 s of speech:
    /// two cuts while recording and a last chunk at release.
    private var threeChunkDictation: [Float] {
        audio([(10, true), (1, false), (10, true), (1, false), (5, true)])
    }

    private func makeTranscriber(_ decoder: FakeLiveDecoder) -> (LiveTranscriber, LiveAudioBuffer) {
        let buffer = LiveAudioBuffer(recordingURL: URL(fileURLWithPath: "/tmp/osw-live-\(UUID().uuidString).wav"))
        let transcriber = LiveTranscriber(
            audio: buffer,
            makeDecoder: { decoder },
            makeProbe: { AmplitudeSpeechProbe() }
        )
        transcriber.start()
        return (transcriber, buffer)
    }

    /// Appends audio in 20 ms blocks, as the capture queue does.
    private func feed(_ samples: [Float], to buffer: LiveAudioBuffer) {
        for start in stride(from: 0, to: samples.count, by: 320) {
            buffer.append(Array(samples[start..<min(start + 320, samples.count)]))
        }
    }

    func testFinishedChunksAreDecodedWhileRecordingAndOnlyTheLastAtRelease() async throws {
        let decoder = FakeLiveDecoder()
        let (transcriber, buffer) = makeTranscriber(decoder)
        let samples = threeChunkDictation
        feed(samples, to: buffer)
        await transcriber.waitUntilIdle()

        XCTAssertEqual(decoder.calls.count, 2, "Both cut chunks are decoded before release")

        buffer.finish()
        let transcript = try await transcriber.finish()

        XCTAssertEqual(decoder.calls.count, 3, "Release decodes only the last chunk")
        XCTAssertEqual(transcript.chunks.map(\.text), [" chunk0.", " chunk1.", " chunk2."])
        XCTAssertEqual(transcript.text, " chunk0. chunk1. chunk2.")
        XCTAssertEqual(transcript.chunks.first?.samples.lowerBound, 0)
        XCTAssertEqual(transcript.chunks.last?.samples.upperBound, samples.count)
        for (earlier, later) in zip(transcript.chunks, transcript.chunks.dropFirst()) {
            XCTAssertEqual(earlier.samples.upperBound, later.samples.lowerBound, "Chunks must cover the audio without gaps")
        }
        // Cuts land inside the pauses, about 0.3 s after each pause begins.
        let cuts = transcript.chunks.dropLast().map { Double($0.samples.upperBound) / Double(sampleRate) }
        XCTAssertEqual(cuts[0], 10.3, accuracy: 0.05)
        XCTAssertEqual(cuts[1], 21.3, accuracy: 0.05)
        XCTAssertTrue(decoder.isClosed)
    }

    func testEachChunkIsPromptedWithTheEarlierChunksText() async throws {
        let decoder = FakeLiveDecoder()
        let (transcriber, buffer) = makeTranscriber(decoder)
        feed(threeChunkDictation, to: buffer)
        buffer.finish()
        _ = try await transcriber.finish()

        XCTAssertEqual(decoder.calls.map(\.history), [[], [0, 1], [0, 1, 10, 11]])
    }

    func testDetectedLanguageIsPinnedForLaterChunks() async throws {
        let decoder = FakeLiveDecoder()
        decoder.detectedLanguage = "de"
        let (transcriber, buffer) = makeTranscriber(decoder)
        feed(threeChunkDictation, to: buffer)
        buffer.finish()
        _ = try await transcriber.finish()

        XCTAssertEqual(decoder.calls.map(\.language), [nil, "de", "de"])
    }

    func testReleaseDuringAChunkPassWaitsForItThenDecodesTheLastChunk() async throws {
        let decoder = FakeLiveDecoder()
        decoder.hold(pass: 0)
        let (transcriber, buffer) = makeTranscriber(decoder)
        feed(audio([(10, true), (1, false), (4, true)]), to: buffer)
        await decoder.waitUntilStarted(pass: 0)
        buffer.finish()

        let finishing = Task { try await transcriber.finish() }
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(decoder.calls.count, 1, "The last chunk waits for the running pass")

        decoder.release(pass: 0)
        let transcript = try await finishing.value
        XCTAssertEqual(transcript.chunks.map(\.text), [" chunk0.", " chunk1."])
    }

    func testCancelDuringAChunkPassAbortsItAndReturnsNoText() async throws {
        let decoder = FakeLiveDecoder()
        decoder.hold(pass: 0)
        let (transcriber, buffer) = makeTranscriber(decoder)
        feed(threeChunkDictation, to: buffer)
        await decoder.waitUntilStarted(pass: 0)
        buffer.finish()

        let finishing = Task { try await transcriber.finish() }
        finishing.cancel()

        do {
            _ = try await finishing.value
            XCTFail("A cancelled dictation must not produce text")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertTrue(decoder.isCancelled)
        await transcriber.waitUntilIdle()
        XCTAssertEqual(decoder.calls.count, 1, "No pass may start after a cancel")
        XCTAssertTrue(decoder.isClosed)
    }

    func testDiscardedRecordingCancelsTheSession() async throws {
        let decoder = FakeLiveDecoder()
        let (transcriber, buffer) = makeTranscriber(decoder)
        feed(audio([(3, true)]), to: buffer)
        buffer.cancel()
        await transcriber.waitUntilIdle()

        XCTAssertTrue(decoder.isCancelled)
        XCTAssertTrue(decoder.isClosed)
        do {
            _ = try await transcriber.finish()
            XCTFail("A discarded recording has no text")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
    }

    func testAFailedPassMakesFinishThrowSoTheCallerDecodesTheWholeRecording() async throws {
        let decoder = FakeLiveDecoder()
        decoder.failingPass = 0
        let (transcriber, buffer) = makeTranscriber(decoder)
        feed(threeChunkDictation, to: buffer)
        await transcriber.waitUntilIdle()
        XCTAssertEqual(decoder.calls.count, 1, "The session stops after a failed pass")
        XCTAssertTrue(decoder.isClosed, "A stopped session frees its state right away")

        buffer.finish()
        do {
            _ = try await transcriber.finish()
            XCTFail("Expected the failed pass to surface")
        } catch {
            XCTAssertEqual(error as? TranscriptionError, .processingFailed)
        }
    }

    func testFinishBeforeTheAudioEndsThrows() async throws {
        let decoder = FakeLiveDecoder()
        let (transcriber, buffer) = makeTranscriber(decoder)
        feed(audio([(2, true)]), to: buffer)
        do {
            _ = try await transcriber.finish()
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual(error as? LiveTranscriptionError, .audioNotFinished)
        }
    }

    func testSilentChunksAddNoText() async throws {
        let decoder = FakeLiveDecoder()
        let (transcriber, buffer) = makeTranscriber(decoder)
        // A long pause is cut again once its middle is 8 s past the last
        // cut, which leaves one chunk with no speech at all.
        feed(audio([(10, true), (18, false), (3, true)]), to: buffer)
        buffer.finish()
        let transcript = try await transcriber.finish()

        XCTAssertTrue(transcript.chunks.contains { $0.text.isEmpty })
        XCTAssertEqual(transcript.text, " chunk0. chunk2.")
    }

    func testShortDictationIsOneChunkDecodedAtRelease() async throws {
        let decoder = FakeLiveDecoder()
        let (transcriber, buffer) = makeTranscriber(decoder)
        let samples = audio([(6, true)])
        feed(samples, to: buffer)
        await transcriber.waitUntilIdle()
        XCTAssertEqual(decoder.calls.count, 0)

        buffer.finish()
        let transcript = try await transcriber.finish()
        XCTAssertEqual(transcript.chunks, [LiveTranscript.Chunk(samples: 0..<samples.count, text: " chunk0.")])
        XCTAssertEqual(decoder.calls.first?.history, [])
    }
}
