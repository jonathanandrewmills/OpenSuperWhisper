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
    /// Preview passes start about once a second, and no sooner than this
    /// factor times the last pass's duration after the previous one started,
    /// which leaves the GPU idle at least a fifth of the time.
    var previewInterval: TimeInterval = 1
    var previewSpacingFactor = 1.25
    /// A preview pass running past the larger of these is aborted.
    var previewBudgetMinimum: TimeInterval = 2
    var previewBudgetFactor = 4.0
    /// Less open audio than this has nothing worth showing.
    var minimumPreviewDuration: TimeInterval = 0.5

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
    /// Index of storage[0] in the recording: samples before it were freed.
    private var offset = 0
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

    /// Samples appended so far, including any freed.
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return offset + storage.count
    }

    /// Samples still held in memory.
    var heldCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage.count
    }

    /// Frees the samples before `index`, which the reader will not ask for
    /// again. The recording itself is saved separately, so a long dictation
    /// need not keep a second full copy here. Small amounts are left in
    /// place, since freeing moves the rest of the buffer.
    func discard(before index: Int) {
        lock.lock()
        defer { lock.unlock() }
        let freed = min(index, offset + storage.count) - offset
        guard freed >= Self.minimumDiscard else { return }
        storage.removeFirst(freed)
        offset += freed
    }

    private static let minimumDiscard = 10 * 16_000

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
        let held = range.clamped(to: offset..<(offset + storage.count))
        return Array(storage[(held.lowerBound - offset)..<(held.upperBound - offset)])
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
    /// A quick pass over the open chunk, for display only: greedy, at
    /// temperature 0 and without fallback. An abort sent before it starts
    /// stops it; callers clear stale aborts with resetAbort() first.
    func decodePreview(_ samples: [Float], history: [WhisperToken], language: String?) throws -> String
    /// Aborts the running pass only. Safe from any thread.
    func abortCurrentPass()
    /// Clears an abort meant for an earlier pass; a cancel stays.
    func resetAbort()
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

/// What the indicator shows while recording: text that will not change
/// any more within its chunk, then the words still changing.
struct LivePreview: Equatable {
    var settled: String
    var unsettled: String

    var isEmpty: Bool { settled.isEmpty && unsettled.isEmpty }
}

enum LiveTranscriptionError: Error {
    case audioNotFinished
    case probeFailed
}

/// Transcribes a recording while it is still being made. The audio is cut
/// into chunks at natural pauses; each finished chunk is decoded right away
/// with the earlier chunks' text as prompt, so at release only the last chunk
/// is left to decode. In between, preview passes over the open chunk feed
/// `onPreview`. All work runs on one serial queue, off the capture path.
final class LiveTranscriber: @unchecked Sendable {
    let recordingURL: URL

    private let audio: LiveAudioBuffer
    private let parameters: LiveTranscriptionParameters
    private let makeDecoder: () throws -> LiveDecoding
    private let makeProbe: () throws -> SpeechProbing
    private let onPreview: ((LivePreview) -> Void)?
    private let now: () -> TimeInterval
    private let queue = DispatchQueue(label: "com.opensuperwhisper.live-transcription", qos: .userInitiated)
    private let frameSampleCount = MyWhisperVadContext.frameSampleCount

    // Shared with other threads, under the lock.
    private let lock = NSLock()
    private var cancelled = false
    private var processingScheduled = false
    private var activeDecoder: LiveDecoding?
    private var finishRequested = false
    private var runningPreview: Int?
    private var previewCount = 0

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
    /// The saved recording, once finish() hands it over.
    private var finalSamples: [Float]?
    private var agreement = LiveAgreement()
    /// Settled preview words of chunks that are cut but not decoded yet.
    private var pendingPreviewTexts: [String] = []
    private var lastPreviewStart: TimeInterval?
    private var lastPreviewDuration: TimeInterval = 0
    private var previewDurations: [TimeInterval] = []
    private var heardSpeechSincePreview = false
    private var previewsStopped = false
    private var lastPreview: LivePreview?

    init(
        audio: LiveAudioBuffer,
        parameters: LiveTranscriptionParameters = .standard,
        makeDecoder: @escaping () throws -> LiveDecoding,
        makeProbe: @escaping () throws -> SpeechProbing,
        onPreview: ((LivePreview) -> Void)? = nil,
        now: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
    ) {
        self.audio = audio
        recordingURL = audio.recordingURL
        self.parameters = parameters
        self.makeDecoder = makeDecoder
        self.makeProbe = makeProbe
        self.onPreview = onPreview
        self.now = now
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
    ///
    /// `recordedSamples` are the recording's saved samples. What is left to
    /// decode at release is taken from them, so a dictation with no cut is
    /// decoded from exactly the samples the batch path would decode, even
    /// where the live mix of a multi-channel input differs from the saved one.
    func finish(recordedSamples: [Float]? = nil) async throws -> LiveTranscript {
        requestFinish()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    continuation.resume(with: Result { try self.finishOnQueue(recordedSamples: recordedSamples) })
                }
            }
        } onCancel: {
            cancel()
        }
    }

    /// No preview starts after this, and a running one is aborted so the
    /// release does not wait for it. The abort happens under the lock, so it
    /// cannot reach a chunk pass that starts after the preview ends.
    private func requestFinish() {
        lock.lock()
        defer { lock.unlock() }
        finishRequested = true
        if runningPreview != nil {
            activeDecoder?.abortCurrentPass()
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
            // A long chunk pass leaves a backlog: probe it, and decode what it
            // cuts, before spending the GPU on a preview of stale audio.
            var probed: Int
            repeat {
                probed = probedSamples
                try probeAudio(final: false)
                try decodeCutChunks()
            } while probedSamples != probed
            // Only the open chunk is read from here on.
            audio.discard(before: openChunkStart)
            try previewIfDue()
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
        var didCut = false
        for probability in probabilities {
            heardSpeechSincePreview = heardSpeechSincePreview || probability >= parameters.speechThreshold
            if let cutFrame = cutter.append(probability) {
                let cutSample = cutFrame * frameSampleCount
                cutChunks.append(openChunkStart..<cutSample)
                openChunkStart = cutSample
                // The chunk keeps its settled words on screen until its own
                // pass replaces them; what was still changing goes.
                pendingPreviewTexts.append(agreement.settledText)
                agreement = LiveAgreement()
                didCut = true
            }
        }
        // More than one chunk waiting means passes are slower than speech:
        // preview work is shed first, so every chunk is still decoded.
        if cutChunks.count > 1 && !previewsStopped {
            previewsStopped = true
            print("[live] preview stopped for this recording: \(cutChunks.count) chunks waiting")
        }
        if didCut { publishPreview() }
    }

    private func decodeCutChunks() throws {
        while !cutChunks.isEmpty {
            try decode(cutChunks.removeFirst())
            // Its own text now replaces its preview words.
            if !pendingPreviewTexts.isEmpty { pendingPreviewTexts.removeFirst() }
            publishPreview()
        }
    }

    private func previewIfDue() throws {
        guard onPreview != nil, !previewsStopped, cutChunks.isEmpty, heardSpeechSincePreview,
              let decoder else { return }
        let start = now()
        if let lastPreviewStart,
           start - lastPreviewStart < max(parameters.previewInterval, parameters.previewSpacingFactor * lastPreviewDuration) {
            return
        }
        // Passes always take the newest audio.
        let range = openChunkStart..<audio.count
        guard Double(range.count) >= parameters.minimumPreviewDuration * Double(parameters.sampleRate),
              let generation = beginPreview() else { return }
        defer { endPreview() }
        heardSpeechSincePreview = false
        lastPreviewStart = start

        let mean = previewDurations.isEmpty ? 0 : previewDurations.reduce(0, +) / Double(previewDurations.count)
        let budget = max(parameters.previewBudgetMinimum, parameters.previewBudgetFactor * mean)
        DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + budget) { [weak self] in
            self?.abortPreview(generation, budget: budget)
        }

        let text: String
        do {
            text = try decoder.decodePreview(
                audio.samples(in: range),
                history: history,
                language: language
            )
        } catch {
            if isCancelled { throw CancellationError() }
            // An aborted or failed preview costs only its display update.
            lastPreviewDuration = now() - start
            print(String(format: "[live] preview pass abandoned after %.2f s: %@", lastPreviewDuration, "\(error)"))
            return
        }
        lastPreviewDuration = now() - start
        previewDurations.append(lastPreviewDuration)
        agreement.update(with: LiveText.words(in: text))
        publishPreview()
    }

    private func beginPreview() -> Int? {
        lock.lock()
        defer { lock.unlock() }
        guard !finishRequested, !cancelled else { return nil }
        // Cleared before the preview can be aborted, so an abort that
        // arrives as the pass starts reaches it.
        activeDecoder?.resetAbort()
        previewCount += 1
        runningPreview = previewCount
        return previewCount
    }

    /// Ends the preview under the lock, so a budget timer firing late can
    /// never abort the pass that follows it.
    private func endPreview() {
        lock.lock()
        runningPreview = nil
        lock.unlock()
    }

    private func abortPreview(_ generation: Int, budget: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        guard runningPreview == generation else { return }
        print(String(format: "[live] preview pass over its %.1f s budget, aborting", budget))
        activeDecoder?.abortCurrentPass()
    }

    private func publishPreview() {
        guard let onPreview else { return }
        let settled = LiveText.join([LiveTranscript(chunks: chunks).text] + pendingPreviewTexts + [agreement.settledText])
        let joined = LiveText.join([settled, agreement.unsettledText])
        let preview = LivePreview(settled: settled, unsettled: String(joined.dropFirst(settled.count)))
        guard preview != lastPreview else { return }
        lastPreview = preview
        onPreview(preview)
    }

    private func decode(_ range: Range<Int>) throws {
        guard !isCancelled else { throw CancellationError() }
        guard let decoder else { throw TranscriptionError.contextInitializationFailed }
        let started = Date()
        let samples = finalSamples.map { Array($0[range]) } ?? audio.samples(in: range)
        let result = try decoder.decodeChunk(samples, history: history, language: language)
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

    private func finishOnQueue(recordedSamples: [Float]?) throws -> LiveTranscript {
        isFinishing = true
        defer { closeResources() }
        if isCancelled { throw CancellationError() }
        if let failure { throw failure }
        guard audio.isFinished else { throw LiveTranscriptionError.audioNotFinished }
        if let recordedSamples, recordedSamples.count == audio.count {
            finalSamples = recordedSamples
        }
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
