import AppKit
import SwiftUI
import XCTest
@testable import OpenSuperWhisper

/// Renders the recording indicator in its states to PNG files for review.
/// Opt-in: set OSW_TEST_INDICATOR_SNAPSHOTS to an output directory.
@MainActor
final class IndicatorSnapshotTests: XCTestCase {
    private func outputDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["OSW_TEST_INDICATOR_SNAPSHOTS"] else {
            throw XCTSkip("Set OSW_TEST_INDICATOR_SNAPSHOTS to a directory")
        }
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Draws the panel content at 2x in an off-screen window.
    private func render(_ viewModel: IndicatorViewModel, size: CGSize, dark: Bool, to url: URL) throws {
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        let host = NSHostingView(rootView: IndicatorWindow(viewModel: viewModel))
        host.sizingOptions = []
        host.frame = NSRect(origin: .zero, size: size)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2), pixelsHigh: Int(size.height * 2),
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ))
        bitmap.size = size
        host.cacheDisplay(in: host.bounds, to: bitmap)
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: url)
    }

    func testCompactCard() throws {
        let directory = try outputDirectory()
        let service = TranscriptionService(engine: NamedTestEngine("idle"))
        for state in [RecordingState.recording, .decoding, .busy] {
            for dark in [false, true] {
                let viewModel = IndicatorViewModel(transcriptionService: service, stopRecording: { nil }, cancelAudioRecording: {})
                viewModel.state = state
                try render(viewModel, size: IndicatorWindow.windowSize, dark: dark,
                           to: directory.appendingPathComponent("compact-\(state)-\(dark ? "dark" : "light").png"))
                viewModel.cleanup()
            }
        }
    }

    func testLiveCard() async throws {
        let directory = try outputDirectory()
        let service = TranscriptionService(engine: WhisperEngine())
        var send: ((LivePreview) -> Void)?
        service.makeLiveTranscriber = { audio, _, _, onPreview in
            send = onPreview
            return LiveTranscriber(audio: audio, makeDecoder: { FakeLiveDecoder() }, makeProbe: { AmplitudeSpeechProbe() })
        }
        service.prepareForRecording(liveAudio: LiveAudioBuffer(recordingURL: URL(fileURLWithPath: "/tmp/osw-snapshot.wav")))
        defer { service.discardLiveTranscription() }
        let long = (1...14).map { " Sentence number \($0) of a long dictation keeps going." }.joined()
        let cases: [(String, RecordingState, LivePreview?, Bool)] = [
            ("recording-empty", .recording, nil, false),
            ("recording-short", .recording, LivePreview(settled: " The first checkpoint phrase is", unsettled: " amber lighthouse"), false),
            ("recording-long", .recording, LivePreview(settled: long, unsettled: " and these words are still changing"), false),
            ("confirm-cancel", .recording, LivePreview(settled: " I was saying that the northern path", unsettled: " needs better lighting"), true),
            ("decoding", .decoding, LivePreview(settled: " The report should be clear, calm, and easy to read.", unsettled: ""), false),
        ]
        for (name, state, preview, confirming) in cases {
            for dark in [false, true] {
                let viewModel = IndicatorViewModel(transcriptionService: service, stopRecording: { nil },
                                                   cancelAudioRecording: {}, usesLiveLayout: true)
                viewModel.state = state
                viewModel.isConfirmingCancel = confirming
                if let preview {
                    try XCTUnwrap(send)(preview)
                    try await Task.sleep(nanoseconds: 250_000_000)
                }
                try render(viewModel, size: IndicatorWindow.liveWindowSize, dark: dark,
                           to: directory.appendingPathComponent("live-\(name)-\(dark ? "dark" : "light").png"))
                viewModel.cleanup()
            }
        }
    }
}
