import AppKit
import Combine
import Foundation
import XCTest
@testable import OpenSuperWhisper

final class LiveAgreementTests: XCTestCase {
    private func update(_ agreement: inout LiveAgreement, _ text: String) {
        agreement.update(with: LiveText.words(in: text))
    }

    func testWordsSettleOnceTwoPassesAgree() {
        var agreement = LiveAgreement()
        update(&agreement, " The quick brown")
        XCTAssertEqual(agreement.settledText, "")
        XCTAssertEqual(agreement.unsettledText, " The quick brown")

        update(&agreement, " The quick brown fox")
        XCTAssertEqual(agreement.settledText, " The quick brown")
        XCTAssertEqual(agreement.unsettledText, " fox")
    }

    func testCaseAndPunctuationDoNotBreakAgreement() {
        var agreement = LiveAgreement()
        update(&agreement, " the quick, brown")
        update(&agreement, " The quick brown. Fox")
        XCTAssertEqual(agreement.settled.count, 3)
    }

    func testSettledWordsNeverShrinkOrChange() {
        var agreement = LiveAgreement()
        update(&agreement, " a b c")
        update(&agreement, " a b c d")
        XCTAssertEqual(agreement.settledText, " a b c")

        update(&agreement, " a x c d e")
        XCTAssertEqual(agreement.settledText, " a b c")
        update(&agreement, " a x c d e")
        XCTAssertEqual(agreement.settledText, " a b c", "Passes that contradict settled words cannot extend them")
    }

    func testChineseAgreesCharacterByCharacter() {
        var agreement = LiveAgreement()
        update(&agreement, "你好世界")
        update(&agreement, "你好世")
        XCTAssertEqual(agreement.settledText, "你好世")
    }
}

final class LiveTextLayoutTests: XCTestCase {
    private let width = IndicatorWindow.liveTextWidth
    private let lines = IndicatorWindow.liveTextLines

    private func height(_ text: String) -> CGFloat {
        NSAttributedString(string: text, attributes: [.font: NSFont.systemFont(ofSize: IndicatorWindow.liveTextFontSize)])
            .boundingRect(with: CGSize(width: width, height: .greatestFiniteMagnitude),
                          options: [.usesLineFragmentOrigin, .usesFontLeading]).height
    }

    func testShortTextIsKeptWithoutItsLeadingSpace() {
        let fitted = LiveTextLayout.fitting(LivePreview(settled: " Hello there.", unsettled: " How"), width: width, lines: lines)
        XCTAssertEqual(fitted, LivePreview(settled: "Hello there.", unsettled: " How"))
    }

    func testLongTextKeepsTheNewestWords() {
        let settled = (1...80).map { " word\($0)" }.joined()
        let fitted = LiveTextLayout.fitting(LivePreview(settled: settled, unsettled: " still changing"), width: width, lines: lines)

        XCTAssertTrue(fitted.settled.hasPrefix("…word"), fitted.settled)
        XCTAssertTrue(fitted.settled.hasSuffix(" word80"))
        XCTAssertEqual(fitted.unsettled, " still changing")
        let lineHeight = NSLayoutManager().defaultLineHeight(for: NSFont.systemFont(ofSize: IndicatorWindow.liveTextFontSize))
        XCTAssertLessThanOrEqual(height(fitted.settled + fitted.unsettled), CGFloat(lines) * lineHeight + 0.5)
        XCTAssertGreaterThan(height("x " + fitted.settled + fitted.unsettled), CGFloat(lines - 1) * lineHeight,
                             "Only as many words are dropped as needed")
    }

    func testCutInsideChangingWordsKeepsThemDim() {
        let unsettled = (1...80).map { " word\($0)" }.joined()
        let fitted = LiveTextLayout.fitting(LivePreview(settled: " Settled.", unsettled: unsettled), width: width, lines: lines)
        XCTAssertEqual(fitted.settled, "…")
        XCTAssertTrue(fitted.unsettled.hasSuffix(" word80"))
    }
}

/// Time as the live session sees it, moved by the test.
final class TestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var value: TimeInterval = 0

    var now: TimeInterval { lock.withLock { value } }

    func advance(_ seconds: TimeInterval) {
        lock.withLock { value += seconds }
    }
}

final class PreviewRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [LivePreview] = []

    var previews: [LivePreview] { lock.withLock { storage } }

    func record(_ preview: LivePreview) {
        lock.withLock { storage.append(preview) }
    }
}

final class LivePreviewSchedulingTests: XCTestCase {
    private let sampleRate = 16_000
    private let clock = TestClock()
    private let recorder = PreviewRecorder()

    private func audio(_ parts: [(seconds: Double, speech: Bool)]) -> [Float] {
        parts.flatMap { part in
            Array(repeating: part.speech ? Float(0.5) : 0, count: Int(part.seconds * Double(sampleRate)))
        }
    }

    private func makeTranscriber(
        _ decoder: FakeLiveDecoder,
        parameters: LiveTranscriptionParameters = .standard
    ) -> (LiveTranscriber, LiveAudioBuffer) {
        let buffer = LiveAudioBuffer(recordingURL: URL(fileURLWithPath: "/tmp/osw-preview-\(UUID().uuidString).wav"))
        let transcriber = LiveTranscriber(
            audio: buffer,
            parameters: parameters,
            makeDecoder: { decoder },
            makeProbe: { AmplitudeSpeechProbe() },
            onPreview: { [recorder] in recorder.record($0) },
            now: { [clock] in clock.now }
        )
        transcriber.start()
        return (transcriber, buffer)
    }

    /// Delivers 20 ms blocks in step with the clock, letting the session run
    /// after each one, as if passes took no time.
    private func play(_ samples: [Float], to buffer: LiveAudioBuffer, on transcriber: LiveTranscriber) async {
        for start in stride(from: 0, to: samples.count, by: 320) {
            buffer.append(Array(samples[start..<min(start + 320, samples.count)]))
            clock.advance(0.02)
            await transcriber.waitUntilIdle()
        }
    }

    func testAFastDecoderKeepsAOneSecondCadence() async throws {
        let decoder = FakeLiveDecoder()
        var starts: [TimeInterval] = []
        let lock = NSLock()
        decoder.onPreviewStart = { [clock] _ in lock.withLock { starts.append(clock.now) } }
        let (transcriber, buffer) = makeTranscriber(decoder)
        await play(audio([(20, true)]), to: buffer, on: transcriber)

        let times = lock.withLock { starts }
        XCTAssertGreaterThanOrEqual(times.count, 18)
        XCTAssertLessThanOrEqual(times.count, 20)
        for (earlier, later) in zip(times, times.dropFirst()) {
            XCTAssertGreaterThanOrEqual(later - earlier, 1.0 - 1e-9)
            XCTAssertLessThanOrEqual(later - earlier, 1.3)
        }
        buffer.finish()
        _ = try await transcriber.finish()
    }

    func testPreviewsSettleWordsTwoPassesAgreeOn() async throws {
        let decoder = FakeLiveDecoder()
        decoder.previewTexts = [" The quick", " The quick brown", " The quick brown fox"]
        let (transcriber, buffer) = makeTranscriber(decoder)
        await play(audio([(3.3, true)]), to: buffer, on: transcriber)

        XCTAssertEqual(recorder.previews, [
            LivePreview(settled: "", unsettled: " The quick"),
            LivePreview(settled: " The quick", unsettled: " brown"),
            LivePreview(settled: " The quick brown", unsettled: " fox"),
        ])
        buffer.finish()
        _ = try await transcriber.finish()
    }

    func testAChunksOwnTextReplacesItsPreviewWords() async throws {
        let decoder = FakeLiveDecoder()
        decoder.previewTexts = [" hello world"]
        let (transcriber, buffer) = makeTranscriber(decoder)
        await play(audio([(10, true), (1, false), (3, true)]), to: buffer, on: transcriber)

        let previews = recorder.previews
        XCTAssertTrue(previews.contains(LivePreview(settled: " hello world", unsettled: "")),
                      "At the cut the chunk keeps its settled preview words")
        XCTAssertTrue(previews.contains(LivePreview(settled: " chunk0.", unsettled: "")),
                      "Its pass then replaces them with its own text")
        XCTAssertEqual(previews.last?.settled.hasPrefix(" chunk0."), true)
        buffer.finish()
        _ = try await transcriber.finish()
    }

    func testSlowChunkPassesShedPreviewsButEveryChunkIsDecoded() async throws {
        let decoder = FakeLiveDecoder()
        decoder.hold(pass: 0)
        let (transcriber, buffer) = makeTranscriber(decoder)
        let dictation = audio([(10, true), (1, false), (10, true), (1, false), (10, true), (1, false), (10, true), (1, false), (3, true)])
        // The held pass blocks the session, so audio is delivered without
        // waiting for it.
        func deliver(_ range: Range<Int>) {
            for start in stride(from: range.lowerBound, to: range.upperBound, by: 320) {
                buffer.append(Array(dictation[start..<min(start + 320, range.upperBound)]))
                clock.advance(0.02)
            }
        }
        let firstCut = 11 * sampleRate
        deliver(0..<firstCut)
        await decoder.waitUntilStarted(pass: 0)

        // The rest of the dictation arrives while the first pass is stuck.
        deliver(firstCut..<dictation.count)
        let previewsBefore = decoder.previewCalls
        decoder.release(pass: 0)
        await transcriber.waitUntilIdle()
        XCTAssertEqual(decoder.previewCalls, previewsBefore, "Previews stop once chunks queue up")

        buffer.finish()
        let transcript = try await transcriber.finish()
        XCTAssertEqual(transcript.chunks.count, 5)
        XCTAssertEqual(transcript.text, " chunk0. chunk1. chunk2. chunk3. chunk4.")
    }

    func testAStuckPreviewIsAbortedByItsBudget() async throws {
        let decoder = FakeLiveDecoder()
        decoder.holdsPreviews = true
        var parameters = LiveTranscriptionParameters.standard
        parameters.previewBudgetMinimum = 0.2
        let (transcriber, buffer) = makeTranscriber(decoder, parameters: parameters)
        await play(audio([(1.5, true)]), to: buffer, on: transcriber)

        XCTAssertGreaterThanOrEqual(decoder.abortCount, 1)
        buffer.finish()
        let transcript = try await transcriber.finish()
        XCTAssertEqual(transcript.text, " chunk0.", "An abandoned preview does not stop the session")
    }

    func testReleaseAbortsARunningPreviewInsteadOfWaitingForIt() async throws {
        let decoder = FakeLiveDecoder()
        decoder.holdsPreviews = true
        var parameters = LiveTranscriptionParameters.standard
        parameters.previewBudgetMinimum = 60
        let started = expectation(description: "preview started")
        decoder.onPreviewStart = { _ in started.fulfill() }
        let (transcriber, buffer) = makeTranscriber(decoder, parameters: parameters)
        let samples = audio([(2, true)])
        for start in stride(from: 0, to: samples.count, by: 320) {
            buffer.append(Array(samples[start..<min(start + 320, samples.count)]))
            clock.advance(0.02)
        }
        await fulfillment(of: [started], timeout: 5)
        buffer.finish()

        let release = ContinuousClock.now
        let transcript = try await transcriber.finish()
        XCTAssertLessThan(release.duration(to: .now), .seconds(10), "Release does not wait out the 60 s preview budget")
        XCTAssertEqual(decoder.abortCount, 1)
        XCTAssertEqual(transcript.text, " chunk0.")
    }

    func testNoPreviewWithoutSpeech() async throws {
        let decoder = FakeLiveDecoder()
        let (transcriber, buffer) = makeTranscriber(decoder)
        await play(audio([(5, false)]), to: buffer, on: transcriber)
        XCTAssertEqual(decoder.previewCalls, 0)
        buffer.finish()
        _ = try await transcriber.finish()
    }
}

@MainActor
final class LivePreviewServiceTests: XCTestCase {
    private var savedLive: Any?

    override func setUp() async throws {
        savedLive = AppPreferences.store.object(forKey: "liveTranscription")
        AppPreferences.store.set(true, forKey: "liveTranscription")
    }

    override func tearDown() async throws {
        AppPreferences.store.set(savedLive, forKey: "liveTranscription")
    }

    private func makeService(_ decoder: FakeLiveDecoder) -> (TranscriptionService, () -> ((LivePreview) -> Void)?) {
        let service = TranscriptionService(engine: WhisperEngine())
        var captured: ((LivePreview) -> Void)?
        service.makeLiveTranscriber = { audio, _, _, onPreview in
            captured = onPreview
            return LiveTranscriber(audio: audio, makeDecoder: { decoder }, makeProbe: { AmplitudeSpeechProbe() }, onPreview: onPreview)
        }
        return (service, { captured })
    }

    func testPreviewsArePublishedAtMostFiveTimesASecondAndLateOnesAreIgnored() async throws {
        let decoder = FakeLiveDecoder()
        let (service, onPreview) = makeService(decoder)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("preview-\(UUID()).wav")
        let buffer = LiveAudioBuffer(recordingURL: url)
        service.prepareForRecording(liveAudio: buffer)
        let send = try XCTUnwrap(onPreview())

        var published: [(preview: LivePreview?, at: ContinuousClock.Instant)] = []
        let subscription = service.$livePreview.dropFirst().sink { published.append(($0, .now)) }
        defer { subscription.cancel() }
        send(LivePreview(settled: " word1", unsettled: ""))
        try await waitUntil("the first preview is published") { published.count == 1 }

        for index in 2...10 {
            send(LivePreview(settled: " word\(index)", unsettled: ""))
        }
        try await waitUntil("the burst is published") { published.count == 2 }
        XCTAssertEqual(published.map(\.preview), [LivePreview(settled: " word1", unsettled: ""), LivePreview(settled: " word10", unsettled: "")],
                       "A burst of previews is coalesced into its newest one")
        XCTAssertGreaterThanOrEqual(published[0].at.duration(to: published[1].at), .milliseconds(200),
                                    "The indicator changes at most five times a second")

        buffer.append(Array(repeating: 0.5, count: 2 * 16_000))
        buffer.finish()
        let text = try await service.transcribeAudio(url: url, settings: Settings(), pcmSamples: nil)
        XCTAssertEqual(text, "chunk0.")
        XCTAssertNil(service.livePreview, "The preview ends with its dictation")

        send(LivePreview(settled: " late", unsettled: ""))
        // A late preview would be published within the 200 ms throttle; on a
        // slow machine this check can only pass early, never fail falsely.
        try await Task.sleep(nanoseconds: 400_000_000)
        XCTAssertNil(service.livePreview, "A finished session's preview is ignored")
    }

    /// Polls until `condition` holds, failing after a generous deadline:
    /// fixed sleeps would race a loaded machine.
    private func waitUntil(_ what: String, timeout: Duration = .seconds(5), _ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + timeout
        while !condition() {
            guard ContinuousClock.now < deadline else {
                XCTFail("Timed out waiting until \(what)")
                return
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
    }

    /// Starts a live session for a recording at `url` and shows one preview.
    private func startDictation(_ service: TranscriptionService, url: URL, onPreview: () -> ((LivePreview) -> Void)?) async throws {
        service.prepareForRecording(liveAudio: LiveAudioBuffer(recordingURL: url))
        try XCTUnwrap(onPreview())(LivePreview(settled: " discarded words", unsettled: ""))
        try await waitUntil("the preview is shown") { service.livePreview != nil }
    }

    private func waitUntilDropped(_ decoder: FakeLiveDecoder, _ service: TranscriptionService) async throws {
        try await waitUntil("the live session is dropped") { decoder.isClosed && service.livePreview == nil }
    }

    func testARecordingTooShortToKeepDropsItsLiveSession() async throws {
        let decoder = FakeLiveDecoder()
        let (service, onPreview) = makeService(decoder)
        try await startDictation(service, url: FileManager.default.temporaryDirectory.appendingPathComponent("short-\(UUID()).wav"), onPreview: onPreview)
        let viewModel = IndicatorViewModel(transcriptionService: service, stopRecording: { nil }, cancelAudioRecording: {})
        defer { viewModel.cleanup() }
        viewModel.state = .recording

        viewModel.startDecoding()
        try await waitUntilDropped(decoder, service)

        XCTAssertTrue(decoder.isCancelled, "The live session's pass is aborted")
        XCTAssertTrue(decoder.isClosed, "Its whisper state is freed")
        XCTAssertNil(service.livePreview, "The next indicator must not show the discarded text")
    }

    func testEscDuringTheStopTailDropsTheLiveSession() async throws {
        let decoder = FakeLiveDecoder()
        let (service, onPreview) = makeService(decoder)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("tail-\(UUID()).wav")
        try await startDictation(service, url: url, onPreview: onPreview)
        let viewModel = IndicatorViewModel(
            transcriptionService: service,
            stopRecording: {
                // The 250 ms stop tail, during which Esc arrives.
                try? await Task.sleep(nanoseconds: 250_000_000)
                return RecordedAudio(url: url, samples: [])
            },
            cancelAudioRecording: {}
        )
        defer { viewModel.cleanup() }
        viewModel.state = .recording

        viewModel.startDecoding()
        viewModel.cancelRecording()
        try await waitUntilDropped(decoder, service)

        XCTAssertTrue(decoder.isClosed)
        XCTAssertNil(service.livePreview)
        XCTAssertFalse(service.isTranscribing)
    }

    func testEscWhileRecordingClearsTheLivePreview() async throws {
        let decoder = FakeLiveDecoder()
        let (service, onPreview) = makeService(decoder)
        try await startDictation(service, url: FileManager.default.temporaryDirectory.appendingPathComponent("esc-\(UUID()).wav"), onPreview: onPreview)
        let viewModel = IndicatorViewModel(transcriptionService: service, stopRecording: { nil }, cancelAudioRecording: {})
        defer { viewModel.cleanup() }
        viewModel.state = .recording

        viewModel.cancelRecording()
        try await waitUntilDropped(decoder, service)

        XCTAssertTrue(decoder.isClosed)
        XCTAssertNil(service.livePreview)
    }

    func testTheLiveIndicatorShowsTheFittedPreview() async throws {
        let decoder = FakeLiveDecoder()
        let (service, onPreview) = makeService(decoder)
        let buffer = LiveAudioBuffer(recordingURL: FileManager.default.temporaryDirectory.appendingPathComponent("preview-\(UUID()).wav"))
        service.prepareForRecording(liveAudio: buffer)
        let viewModel = IndicatorViewModel(transcriptionService: service, stopRecording: { nil }, cancelAudioRecording: {}, usesLiveLayout: true)
        defer {
            viewModel.cleanup()
            service.discardLiveTranscription()
        }

        try XCTUnwrap(onPreview())(LivePreview(settled: " Hello", unsettled: " there"))
        try await waitUntil("the indicator shows the preview") { viewModel.liveText != nil }
        XCTAssertEqual(viewModel.liveText, LivePreview(settled: "Hello", unsettled: " there"))

        let compact = IndicatorViewModel(transcriptionService: service, stopRecording: { nil }, cancelAudioRecording: {}, usesLiveLayout: false)
        defer { compact.cleanup() }
        XCTAssertNil(compact.liveText, "The compact indicator never shows live text")
    }
}

@MainActor
final class LiveIndicatorLayoutTests: XCTestCase {
    func testTheLivePanelLeavesTheCompactMargins() {
        XCTAssertEqual(IndicatorWindow.liveCardBottomInset, (IndicatorWindow.windowSize.height - IndicatorWindow.cardSize.height) / 2)
        XCTAssertEqual(IndicatorWindow.liveWindowSize.width - IndicatorWindow.liveCardWidth,
                       IndicatorWindow.windowSize.width - IndicatorWindow.cardSize.width)
        XCTAssertEqual(IndicatorWindow.liveWindowSize.height - IndicatorWindow.liveCardMaxHeight,
                       IndicatorWindow.windowSize.height - IndicatorWindow.cardSize.height)
        let lineHeight = NSLayoutManager().defaultLineHeight(for: NSFont.systemFont(ofSize: IndicatorWindow.liveTextFontSize))
        XCTAssertGreaterThanOrEqual(IndicatorWindow.liveCardMaxHeight,
                                    IndicatorWindow.cardSize.height + 4 + 2 + CGFloat(IndicatorWindow.liveTextLines) * lineHeight,
                                    "Three lines of text fit in the tallest card")
    }

    func testTheLiveCardRisesAroundItsOwnCenter() {
        let center = CGPoint(x: IndicatorWindow.liveWindowSize.width / 2,
                             y: IndicatorWindow.liveCardBottomInset + IndicatorWindow.cardSize.height / 2)
        let transform = IndicatorWindowManager.hiddenTransform(scalingAround: center)
        let mappedX = transform.m11 * center.x + transform.m21 * center.y + transform.m41
        let mappedY = transform.m12 * center.x + transform.m22 * center.y + transform.m42
        XCTAssertEqual(mappedX, center.x, accuracy: 0.001)
        XCTAssertEqual(mappedY, center.y - IndicatorWindow.appearOffset, accuracy: 0.001)
    }

    func testThePanelTakesTheLiveSizeOnlyInLiveMode() async throws {
        let saved = AppPreferences.store.object(forKey: "liveTranscription")
        defer { AppPreferences.store.set(saved, forKey: "liveTranscription") }
        let manager = IndicatorWindowManager.shared

        AppPreferences.store.set(true, forKey: "liveTranscription")
        let live = manager.prepare()
        XCTAssertTrue(live.usesLiveLayout)
        XCTAssertEqual(manager.window?.frame.size, IndicatorWindow.liveWindowSize)
        manager.stopForce()
        try await Task.sleep(nanoseconds: 700_000_000)

        AppPreferences.store.set(false, forKey: "liveTranscription")
        let compact = manager.prepare()
        XCTAssertFalse(compact.usesLiveLayout)
        XCTAssertEqual(manager.window?.frame.size, IndicatorWindow.windowSize)
        manager.stopForce()
        try await Task.sleep(nanoseconds: 700_000_000)
    }
}
