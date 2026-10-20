import Foundation

/// Values that shape live transcription. They live in one place on purpose:
/// none of them is a user setting.
struct LiveTranscriptionParameters {
    var sampleRate = 16_000
    /// Every chunk pass pays the full encoder cost (Whisper always encodes
    /// 30 s), so shorter chunks spend more GPU per minute of speech and give
    /// the decoder less context.
    var minimumChunkDuration: TimeInterval = 8
    /// Keeps each chunk inside one 30 s decoder window, so a chunk pass never
    /// seeks across a window seam, where the whole-recording pass loses words.
    var maximumChunkDuration: TimeInterval = 24
    /// The pause that ends a chunk.
    var cutPauseDuration: TimeInterval = 0.6
    /// A chunk that reaches the maximum is cut at the quietest stretch of
    /// this length within the last few seconds.
    var forcedCutSearchDuration: TimeInterval = 4
    var forcedCutStretchDuration: TimeInterval = 0.1
    /// Silero's own speech threshold, and a lower one for silence so a frame
    /// near the threshold does not flip the state.
    var speechThreshold: Float = 0.5
    var silenceThreshold: Float = 0.35
    /// Audio is probed in batches: whisper.cpp logs every probe call.
    var probeBatchDuration: TimeInterval = 0.25
    /// Only the newest tokens can fit in a prompt.
    var historyTokenLimit = 448

    static let standard = LiveTranscriptionParameters()

    func frames(_ duration: TimeInterval) -> Int {
        max(1, Int((duration * Double(sampleRate) / Double(MyWhisperVadContext.frameSampleCount)).rounded()))
    }
}

/// Mono 16 kHz audio of one recording, appended by the capture queue as it is
/// converted. The live session reads it at its own pace and never makes the
/// capture wait, so falling behind costs passes and never audio.
final class LiveAudioBuffer: @unchecked Sendable {
    enum Event {
        case appended
        case finished
        case cancelled
    }

    let recordingURL: URL
    private let lock = NSLock()
    private var storage: [Float] = []
    private var end: Event?
    private var observer: ((Event) -> Void)?

    init(recordingURL: URL) {
        self.recordingURL = recordingURL
    }

    func append(_ block: [Float]) {
        lock.lock()
        guard end == nil else {
            lock.unlock()
            return
        }
        storage.append(contentsOf: block)
        let observer = self.observer
        lock.unlock()
        observer?(.appended)
    }

    /// All audio of the recording has been appended.
    func finish() { close(with: .finished) }

    /// The recording was discarded; nobody will ask for its text.
    func cancel() { close(with: .cancelled) }

    private func close(with event: Event) {
        lock.lock()
        guard end == nil else {
            lock.unlock()
            return
        }
        end = event
        let observer = self.observer
        lock.unlock()
        observer?(event)
    }

    func observe(_ observer: @escaping (Event) -> Void) {
        lock.lock()
        self.observer = observer
        lock.unlock()
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage.count
    }

    var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return end == .finished
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return end == .cancelled
    }

    func samples(in range: Range<Int>) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        return Array(storage[range.clamped(to: 0..<storage.count)])
    }
}

/// Decides where a recording is cut into chunks from Silero's per-frame
/// speech probabilities. Positions are frame indices from the recording start.
struct ChunkCutter {
    private let minimumFrames: Int
    private let maximumFrames: Int
    private let pauseFrames: Int
    private let searchFrames: Int
    private let stretchFrames: Int
    private let speechThreshold: Float
    private let silenceThreshold: Float

    private(set) var chunkStart = 0
    /// Probabilities of the open chunk's frames.
    private var probabilities: [Float] = []
    /// First frame of the silence in progress, if any.
    private var silenceStart: Int?

    init(parameters: LiveTranscriptionParameters = .standard) {
        minimumFrames = parameters.frames(parameters.minimumChunkDuration)
        maximumFrames = parameters.frames(parameters.maximumChunkDuration)
        pauseFrames = parameters.frames(parameters.cutPauseDuration)
        searchFrames = parameters.frames(parameters.forcedCutSearchDuration)
        stretchFrames = parameters.frames(parameters.forcedCutStretchDuration)
        speechThreshold = parameters.speechThreshold
        silenceThreshold = parameters.silenceThreshold
    }

    var frameCount: Int { chunkStart + probabilities.count }

    /// Adds the next frame and returns the frame the open chunk ends at, if
    /// this frame completes a chunk.
    mutating func append(_ probability: Float) -> Int? {
        let frame = frameCount
        probabilities.append(probability)
        if probability >= speechThreshold {
            silenceStart = nil
        } else if probability < silenceThreshold && silenceStart == nil {
            silenceStart = frame
        }

        // Cut in the middle of a pause so both chunks keep their padding, and
        // only where the chunk before it is long enough to be worth a pass.
        if let start = silenceStart {
            let length = frame + 1 - start
            let middle = start + length / 2
            if length >= pauseFrames && middle - chunkStart >= minimumFrames {
                return cut(at: middle)
            }
        }
        if probabilities.count >= maximumFrames {
            return cut(at: quietestFrame())
        }
        return nil
    }

    /// The middle of the lowest-probability stretch near the chunk's end;
    /// the latest one wins a tie, keeping chunks long.
    private func quietestFrame() -> Int {
        let searchStart = probabilities.count - min(searchFrames, probabilities.count)
        let lastStart = max(searchStart, probabilities.count - stretchFrames)
        var best = lastStart
        var bestSum = Float.infinity
        for start in searchStart...lastStart {
            let sum = probabilities[start..<min(start + stretchFrames, probabilities.count)].reduce(0, +)
            if sum <= bestSum {
                best = start
                bestSum = sum
            }
        }
        return chunkStart + best + stretchFrames / 2
    }

    private mutating func cut(at frame: Int) -> Int {
        probabilities.removeFirst(frame - chunkStart)
        chunkStart = frame
        if let start = silenceStart, start < frame {
            silenceStart = frame
        }
        return frame
    }
}

/// Prompt tokens for a chunk pass: the user's initial prompt under the batch
/// path's budget rule, followed by the newest text of the finished chunks,
/// the way whisper.cpp carries text between its own 30 s windows.
enum LivePrompt {
    static func tokens(
        initialPrompt: [WhisperToken],
        carriesInitialPrompt: Bool,
        history: [WhisperToken],
        capacity: Int
    ) -> [WhisperToken] {
        // whisper.cpp prepends one marker token within the same capacity.
        let budget = max(0, capacity - 1)
        guard carriesInitialPrompt else {
            return Array((initialPrompt + history).suffix(budget))
        }
        let kept = initialPrompt.suffix(budget)
        return Array(kept) + history.suffix(budget - kept.count)
    }
}

struct LiveChunkResult {
    var text: String
    var tokens: [WhisperToken]
    /// Set by the first pass that detected the language in "auto" mode.
    var detectedLanguage: String?
}

/// Decodes chunks with the user's batch settings. Passes are serial.
protocol LiveDecoding: AnyObject {
    func decodeChunk(_ samples: [Float], history: [WhisperToken], language: String?) throws -> LiveChunkResult
    /// Aborts the running pass and every later one. Safe from any thread.
    func cancel()
    /// Frees native resources; called once no pass can run any more.
    func close()
}

/// Scores consecutive Silero frames, keeping state between calls.
protocol SpeechProbing: AnyObject {
    func probabilities(for frames: ArraySlice<Float>) -> [Float]?
}

final class WhisperSpeechProbe: SpeechProbing {
    private let vad: MyWhisperVadContext

    init() throws {
        guard let path = WhisperEngine.vadModelPath,
              let vad = MyWhisperVadContext(modelPath: path) else {
            throw TranscriptionError.contextInitializationFailed
        }
        vad.resetStreamingState()
        self.vad = vad
    }

    func probabilities(for frames: ArraySlice<Float>) -> [Float]? {
        vad.streamingSpeechProbabilities(for: frames)
    }
}

struct LiveTranscript {
    struct Chunk: Equatable {
        let samples: Range<Int>
        let text: String
    }

    let chunks: [Chunk]

    /// Raw joined text; the caller runs the usual cleanup once over it.
    var text: String { LiveText.join(chunks.map(\.text)) }
}

enum LiveTranscriptionError: Error {
    case audioNotFinished
    case probeFailed
}

/// Transcribes a recording while it is still being made. The audio is cut
/// into chunks at natural pauses; each finished chunk is decoded right away
/// with the earlier chunks' text as prompt, so at release only the last chunk
/// is left to decode. All work runs on one serial queue, off the capture path.
final class LiveTranscriber: @unchecked Sendable {
    let recordingURL: URL

    private let audio: LiveAudioBuffer
    private let parameters: LiveTranscriptionParameters
    private let makeDecoder: () throws -> LiveDecoding
    private let makeProbe: () throws -> SpeechProbing
    private let queue = DispatchQueue(label: "com.opensuperwhisper.live-transcription", qos: .userInitiated)
    private let frameSampleCount = MyWhisperVadContext.frameSampleCount

    // Shared with other threads, under the lock.
    private let lock = NSLock()
    private var cancelled = false
    private var processingScheduled = false
    private var activeDecoder: LiveDecoding?

    // Session queue only.
    private var decoder: LiveDecoding?
    private var probe: SpeechProbing?
    private var cutter: ChunkCutter
    private var probedSamples = 0
    private var openChunkStart = 0
    private var cutChunks: [Range<Int>] = []
    private var chunks: [LiveTranscript.Chunk] = []
    private var history: [WhisperToken] = []
    private var language: String?
    private var failure: Error?
    private var isFinishing = false

    init(
        audio: LiveAudioBuffer,
        parameters: LiveTranscriptionParameters = .standard,
        makeDecoder: @escaping () throws -> LiveDecoding,
        makeProbe: @escaping () throws -> SpeechProbing
    ) {
        self.audio = audio
        recordingURL = audio.recordingURL
        self.parameters = parameters
        self.makeDecoder = makeDecoder
        self.makeProbe = makeProbe
        cutter = ChunkCutter(parameters: parameters)
    }

    func start() {
        audio.observe { [weak self] event in
            switch event {
            case .appended: self?.scheduleProcessing()
            case .finished: break
            case .cancelled: self?.cancel()
            }
        }
        queue.async { [self] in
            do {
                // Allocating the state takes a moment for large models, so it
                // happens here, while the first seconds are being recorded.
                let decoder = try makeDecoder()
                self.decoder = decoder
                probe = try makeProbe()
                lock.lock()
                activeDecoder = decoder
                let wasCancelled = cancelled
                lock.unlock()
                if wasCancelled { decoder.cancel() }
            } catch {
                fail(error)
                return
            }
            if audio.isCancelled { cancel() }
            process()
        }
    }

    /// Decodes whatever the recording still holds and returns every chunk's
    /// text. The audio must be finished. Throws when a pass failed, so the
    /// caller can decode the whole recording the usual way instead.
    func finish() async throws -> LiveTranscript {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    continuation.resume(with: Result { try self.finishOnQueue() })
                }
            }
        } onCancel: {
            cancel()
        }
    }

    /// Stops all work. A running pass is aborted, and finish() then throws
    /// CancellationError. Safe from any thread, any number of times.
    func cancel() {
        lock.lock()
        guard !cancelled else {
            lock.unlock()
            return
        }
        cancelled = true
        let decoder = activeDecoder
        lock.unlock()
        decoder?.cancel()
        queue.async { self.closeResources() }
    }

    /// Resolves once the work queued so far has run. After cancel() that
    /// includes the aborted pass and freeing the decoder.
    func waitUntilIdle() async {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume() }
        }
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    private func scheduleProcessing() {
        lock.lock()
        guard !processingScheduled, !cancelled else {
            lock.unlock()
            return
        }
        processingScheduled = true
        lock.unlock()
        queue.async {
            self.lock.lock()
            self.processingScheduled = false
            self.lock.unlock()
            self.process()
        }
    }

    private func process() {
        guard !isCancelled, failure == nil, !isFinishing, decoder != nil else { return }
        do {
            try probeAudio(final: false)
            try decodeCutChunks()
        } catch {
            fail(error)
        }
    }

    private func probeAudio(final: Bool) throws {
        guard let probe else { return }
        let frames = (audio.count - probedSamples) / frameSampleCount
        let batch = final ? 1 : parameters.frames(parameters.probeBatchDuration)
        guard frames >= batch else { return }
        let range = probedSamples..<(probedSamples + frames * frameSampleCount)
        guard let probabilities = probe.probabilities(for: audio.samples(in: range)[...]),
              probabilities.count == frames else {
            throw LiveTranscriptionError.probeFailed
        }
        probedSamples = range.upperBound
        for probability in probabilities {
            if let cutFrame = cutter.append(probability) {
                let cutSample = cutFrame * frameSampleCount
                cutChunks.append(openChunkStart..<cutSample)
                openChunkStart = cutSample
            }
        }
    }

    private func decodeCutChunks() throws {
        while !cutChunks.isEmpty {
            try decode(cutChunks.removeFirst())
        }
    }

    private func decode(_ range: Range<Int>) throws {
        guard !isCancelled else { throw CancellationError() }
        guard let decoder else { throw TranscriptionError.contextInitializationFailed }
        let started = Date()
        let result = try decoder.decodeChunk(audio.samples(in: range), history: history, language: language)
        chunks.append(LiveTranscript.Chunk(samples: range, text: result.text))
        history = Array((history + result.tokens).suffix(parameters.historyTokenLimit))
        if language == nil {
            language = result.detectedLanguage
        }
        print(String(
            format: "[live] chunk %d: %.1f s of audio decoded in %.2f s",
            chunks.count,
            Double(range.count) / Double(parameters.sampleRate),
            Date().timeIntervalSince(started)
        ))
    }

    private func finishOnQueue() throws -> LiveTranscript {
        isFinishing = true
        defer { closeResources() }
        if isCancelled { throw CancellationError() }
        if let failure { throw failure }
        guard audio.isFinished else { throw LiveTranscriptionError.audioNotFinished }
        do {
            try probeAudio(final: true)
            try decodeCutChunks()
            // After the stop tail, the open chunk is the last chunk.
            let end = audio.count
            if end > openChunkStart {
                try decode(openChunkStart..<end)
            }
        } catch {
            if isCancelled { throw CancellationError() }
            throw error
        }
        if isCancelled { throw CancellationError() }
        return LiveTranscript(chunks: chunks)
    }

    private func fail(_ error: Error) {
        guard failure == nil else { return }
        failure = error
        if !isCancelled {
            print("[live] stopped, the recording will be decoded whole at release: \(error)")
        }
        closeResources()
    }

    private func closeResources() {
        lock.lock()
        activeDecoder = nil
        lock.unlock()
        decoder?.close()
        decoder = nil
        probe = nil
    }
}
