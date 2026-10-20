import Foundation
import XCTest
@testable import OpenSuperWhisper

/// Live transcription through the real whisper.cpp decode. The short tests
/// use the tiny.en model at the repository root when it is present; the long
/// ones need OSW_TEST_TURBO_MODEL (or a multilingual model, as the long-form
/// tests find it), and the real-time bench also needs OSW_TEST_LIVE_BENCH=1.
final class LiveTranscriptionModelTests: XCTestCase {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
    private static let sampleRate = 16_000

    private struct Fixture {
        let name: String
        let language: String
        let checkpoints: [String]
    }

    private static let fixtures = [
        Fixture(name: "long_en", language: "en", checkpoints: ["amber lighthouse", "quiet river", "silver compass"]),
        Fixture(name: "long_ru", language: "ru", checkpoints: ["янтарный маяк", "тихая река", "серебряный компас"]),
    ]

    private func loadEngine(_ path: String) async throws -> WhisperEngine {
        let engine = WhisperEngine(modelPath: path)
        try await engine.initialize()
        return engine
    }

    private func tinyEnglishModel() throws -> String {
        let model = Self.repoRoot.appendingPathComponent("ggml-tiny.en.bin")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: model.path), "ggml-tiny.en.bin is not in the repository root")
        return model.path
    }

    private func longFormModel() throws -> String {
        let environment = ProcessInfo.processInfo.environment
        if let turbo = environment["OSW_TEST_TURBO_MODEL"] { return turbo }
        let candidates = [
            environment["OSW_TEST_MULTILINGUAL_MODEL"],
            Self.repoRoot.appendingPathComponent(".build/test-models/ggml-tiny.bin").path,
        ].compactMap { $0 }
        guard let model = candidates.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
            throw XCTSkip("Set OSW_TEST_TURBO_MODEL or OSW_TEST_MULTILINGUAL_MODEL")
        }
        return model
    }

    private func samples(of fixture: String) throws -> [Float] {
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("live-\(UUID()).wav")
        defer { try? FileManager.default.removeItem(at: output) }
        let input = Self.repoRoot.appendingPathComponent("OpenSuperWhisperTests/Fixtures/\(fixture).m4a")
        return try PCMRecordingTests.prepare(file: input, output: output).samples
    }

    private func settings(language: String, beamSearch: Bool) -> Settings {
        var settings = Settings()
        settings.selectedLanguage = language
        settings.showTimestamps = false
        settings.initialPrompt = ""
        settings.useBeamSearch = beamSearch
        settings.beamSize = 5
        settings.temperature = 0
        settings.noSpeechThreshold = 0.6
        settings.suppressBlankAudio = true
        return settings
    }

    private func makeTranscriber(engine: WhisperEngine, settings: Settings) -> (LiveTranscriber, LiveAudioBuffer) {
        let buffer = LiveAudioBuffer(recordingURL: URL(fileURLWithPath: "/tmp/osw-live-model-\(UUID().uuidString).wav"))
        let transcriber = LiveTranscriber(
            audio: buffer,
            makeDecoder: { try engine.makeLiveDecoder(settings: settings) },
            makeProbe: { try WhisperSpeechProbe() }
        )
        transcriber.start()
        return (transcriber, buffer)
    }

    /// Feeds 20 ms blocks as fast as possible. The cut points depend only on
    /// the audio, so the final text is the same as with real-time capture.
    private func feed(_ samples: [Float], to buffer: LiveAudioBuffer) {
        for start in stride(from: 0, to: samples.count, by: 320) {
            buffer.append(Array(samples[start..<min(start + 320, samples.count)]))
        }
    }

    private func words(_ text: String) -> [String] {
        text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
    }

    private func containsPhrase(_ phrase: String, in text: String) -> Bool {
        let haystack = words(text)
        let needle = words(phrase)
        guard haystack.count >= needle.count else { return false }
        return (0...(haystack.count - needle.count)).contains { Array(haystack[$0..<$0 + needle.count]) == needle }
    }

    /// Words repeated at a chunk join show up in the final text as a
    /// three-word sequence that occurs twice near the join.
    private func repeatedTrigramsAtJoins(_ transcript: LiveTranscript) -> [String] {
        var repeated: [String] = []
        var joined = ""
        for (index, chunk) in transcript.chunks.enumerated() {
            let next = LiveText.join([joined, chunk.text])
            if index > 0 {
                let window = Array(words(joined).suffix(8) + words(String(next.dropFirst(joined.count))).prefix(8))
                var seen = Set<String>()
                for start in 0..<max(0, window.count - 2) {
                    let trigram = window[start..<start + 3].joined(separator: " ")
                    if !seen.insert(trigram).inserted { repeated.append(trigram) }
                }
            }
            joined = next
        }
        return repeated
    }

    func testShortDictationsWithoutACutMatchTheBatchTextExactly() async throws {
        let engine = try await loadEngine(try tinyEnglishModel())
        let fixture = try samples(of: "long_en")
        let settings = settings(language: "en", beamSearch: false)
        for seconds in [3, 6, 12] {
            let clip = Array(fixture.prefix(seconds * Self.sampleRate))
            let batch = try await engine.transcribeSamples(clip, settings: settings)

            let (transcriber, buffer) = makeTranscriber(engine: engine, settings: settings)
            feed(clip, to: buffer)
            buffer.finish()
            let transcript = try await transcriber.finish()

            XCTAssertEqual(transcript.chunks.count, 1, "\(seconds) s")
            XCTAssertEqual(WhisperEngine.cleanedTranscript(transcript.text, settings: settings), batch, "\(seconds) s")
            XCTAssertFalse(batch.isEmpty)
        }
    }

    @MainActor
    func testServiceTakesTheTextOfTheLiveSessionForItsRecording() async throws {
        let engine = try await loadEngine(try tinyEnglishModel())
        let clip = Array(try samples(of: "long_en").prefix(6 * Self.sampleRate))
        let settings = Settings()
        let expected = try await engine.transcribeSamples(clip, settings: settings)

        let service = TranscriptionService(engine: engine)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("live-service-\(UUID()).wav")
        let buffer = LiveAudioBuffer(recordingURL: url)
        service.prepareForRecording(liveAudio: buffer)
        feed(clip, to: buffer)
        buffer.finish()

        let text = try await service.transcribeAudio(url: url, settings: settings, pcmSamples: clip)
        XCTAssertEqual(text, expected)
        XCTAssertFalse(engine.hasPreparedState, "Live mode decodes on its own state, not a prepared one")
        XCTAssertFalse(service.isTranscribing)
    }

    @MainActor
    func testServiceFallsBackToTheWholeRecordingWhenTheSessionCannotFinish() async throws {
        let engine = try await loadEngine(try tinyEnglishModel())
        let clip = Array(try samples(of: "long_en").prefix(6 * Self.sampleRate))
        let settings = Settings()
        let expected = try await engine.transcribeSamples(clip, settings: settings)

        let service = TranscriptionService(engine: engine)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("live-fallback-\(UUID()).wav")
        let buffer = LiveAudioBuffer(recordingURL: url)
        service.prepareForRecording(liveAudio: buffer)
        feed(clip, to: buffer)
        // The audio is never marked finished, so the session cannot finish.

        let text = try await service.transcribeAudio(url: url, settings: settings, pcmSamples: clip)
        XCTAssertEqual(text, expected)
    }

    func testLongFixturesMatchOrBeatTheWholeRecordingPass() async throws {
        let modelPath = try longFormModel()
        let engine = try await loadEngine(modelPath)
        for fixture in Self.fixtures {
            let audio = try samples(of: fixture.name)
            let reference = try String(
                contentsOf: Self.repoRoot.appendingPathComponent("OpenSuperWhisperTests/Fixtures/\(fixture.name).txt"),
                encoding: .utf8
            )
            let settings = settings(language: fixture.language, beamSearch: true)

            let batchStart = ContinuousClock.now
            let batch = try await engine.transcribeSamples(audio, settings: settings)
            let batchTime = batchStart.duration(to: .now)

            let (transcriber, buffer) = makeTranscriber(engine: engine, settings: settings)
            feed(audio, to: buffer)
            buffer.finish()
            let transcript = try await transcriber.finish()
            let live = WhisperEngine.cleanedTranscript(transcript.text, settings: settings)

            let batchWER = WhisperTurboRegressionTests.wordErrorRate(reference: reference, transcription: batch)
            let liveWER = WhisperTurboRegressionTests.wordErrorRate(reference: reference, transcription: live)
            let cuts = transcript.chunks.dropLast().map { String(format: "%.1f", Double($0.samples.upperBound) / Double(Self.sampleRate)) }
            print("[LIVE-PARITY] model=\(URL(fileURLWithPath: modelPath).lastPathComponent) fixture=\(fixture.name) batchWER=\(batchWER) liveWER=\(liveWER) batchTime=\(batchTime) cuts=\(cuts)")
            print("[LIVE-PARITY-TEXT] \(fixture.name) batch=\(batch)")
            print("[LIVE-PARITY-TEXT] \(fixture.name) live=\(live)")
            for chunk in transcript.chunks {
                print(String(format: "[LIVE-PARITY-CHUNK] %@ %.2f-%.2f s:%@", fixture.name,
                             Double(chunk.samples.lowerBound) / Double(Self.sampleRate),
                             Double(chunk.samples.upperBound) / Double(Self.sampleRate), chunk.text))
            }

            XCTAssertLessThanOrEqual(liveWER, batchWER + 0.005, fixture.name)
            // Small models misspell some checkpoints in both paths; live mode
            // must keep every one the whole-recording pass kept.
            for checkpoint in fixture.checkpoints where containsPhrase(checkpoint, in: batch) {
                XCTAssertTrue(containsPhrase(checkpoint, in: live), "\(fixture.name) lost \(checkpoint)")
            }
            XCTAssertEqual(repeatedTrigramsAtJoins(transcript), [], fixture.name)
        }
    }

    /// Plays dictations through the live session in real time and compares
    /// the decode left after release with the whole-recording decode of the
    /// same audio. Both paths share the 250 ms stop tail, so it is left out.
    func testRealTimeReleaseLatency() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["OSW_TEST_LIVE_BENCH"] == "1", "Set OSW_TEST_LIVE_BENCH=1")
        let modelPath = try longFormModel()
        let engine = try await loadEngine(modelPath)
        let english = try samples(of: "long_en")
        let runs: [(name: String, language: String, samples: [Float])] = [
            ("en-10s", "en", Array(english.prefix(10 * Self.sampleRate))),
            ("en-30s", "en", Array(english.prefix(30 * Self.sampleRate))),
            ("en-60s", "en", Array(english.prefix(60 * Self.sampleRate))),
            ("long_en", "en", english),
            ("long_ru", "ru", try samples(of: "long_ru")),
        ]
        for run in runs {
            let settings = settings(language: run.language, beamSearch: true)
            let batchStart = ContinuousClock.now
            _ = try await engine.transcribeSamples(run.samples, settings: settings)
            let batchTime = batchStart.duration(to: .now)

            let (transcriber, buffer) = makeTranscriber(engine: engine, settings: settings)
            let clock = ContinuousClock()
            let start = clock.now
            for (index, blockStart) in stride(from: 0, to: run.samples.count, by: 320).enumerated() {
                try await clock.sleep(until: start + .milliseconds(20 * (index + 1)))
                buffer.append(Array(run.samples[blockStart..<min(blockStart + 320, run.samples.count)]))
            }
            buffer.finish()
            let releaseStart = clock.now
            let transcript = try await transcriber.finish()
            let releaseTime = releaseStart.duration(to: clock.now)

            print("[LIVE-BENCH] model=\(URL(fileURLWithPath: modelPath).lastPathComponent) run=\(run.name) seconds=\(run.samples.count / Self.sampleRate) chunks=\(transcript.chunks.count) batchDecode=\(batchTime) liveAfterRelease=\(releaseTime)")
        }
    }

    /// The same real-time playback with preview passes on, as the indicator
    /// sees it: when the first words appear, how often they change, and the
    /// wait after release while previews share the GPU with chunk passes.
    func testRealTimePreviews() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["OSW_TEST_LIVE_BENCH"] == "1", "Set OSW_TEST_LIVE_BENCH=1")
        let modelPath = try longFormModel()
        let engine = try await loadEngine(modelPath)
        let english = try samples(of: "long_en")
        let runs: [(name: String, language: String, samples: [Float])] = [
            ("en-30s", "en", Array(english.prefix(30 * Self.sampleRate))),
            ("long_en", "en", english),
            ("long_ru", "ru", try samples(of: "long_ru")),
        ]
        for run in runs {
            let settings = settings(language: run.language, beamSearch: true)
            let clock = ContinuousClock()
            let start = clock.now
            let lock = NSLock()
            var updates: [Duration] = []
            let buffer = LiveAudioBuffer(recordingURL: URL(fileURLWithPath: "/tmp/osw-live-bench-\(UUID().uuidString).wav"))
            let transcriber = LiveTranscriber(
                audio: buffer,
                makeDecoder: { try engine.makeLiveDecoder(settings: settings) },
                makeProbe: { try WhisperSpeechProbe() },
                onPreview: { preview in
                    guard !preview.isEmpty else { return }
                    lock.withLock { updates.append(start.duration(to: clock.now)) }
                }
            )
            transcriber.start()
            for (index, blockStart) in stride(from: 0, to: run.samples.count, by: 320).enumerated() {
                try await clock.sleep(until: start + .milliseconds(20 * (index + 1)))
                buffer.append(Array(run.samples[blockStart..<min(blockStart + 320, run.samples.count)]))
            }
            buffer.finish()
            let releaseStart = clock.now
            let transcript = try await transcriber.finish()
            let releaseTime = releaseStart.duration(to: clock.now)

            let releaseOffset = start.duration(to: releaseStart)
            let times = lock.withLock { updates }.filter { $0 <= releaseOffset }
            let gaps = zip(times, times.dropFirst()).map { $1 - $0 }.sorted()
            let speechStart = (try? firstSpeech(in: run.samples)) ?? .zero
            print("[LIVE-PREVIEW-BENCH] model=\(URL(fileURLWithPath: modelPath).lastPathComponent) run=\(run.name) chunks=\(transcript.chunks.count) speechStart=\(speechStart) firstText=\(times.first ?? .zero) updates=\(times.count) medianGap=\(gaps.isEmpty ? .zero : gaps[gaps.count / 2]) maxGap=\(gaps.last ?? .zero) liveAfterRelease=\(releaseTime)")
        }
    }

    /// When Silero first hears speech, for measuring first text from it.
    private func firstSpeech(in samples: [Float]) throws -> Duration {
        let probe = try WhisperSpeechProbe()
        let frame = MyWhisperVadContext.frameSampleCount
        let frames = min(samples.count / frame, 10 * Self.sampleRate / frame)
        let probabilities = probe.probabilities(for: samples[0..<frames * frame]) ?? []
        let index = probabilities.firstIndex { $0 >= 0.5 } ?? 0
        return .milliseconds(index * frame * 1000 / Self.sampleRate)
    }
}
