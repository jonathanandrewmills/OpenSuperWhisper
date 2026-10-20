import Foundation

@MainActor
class TranscriptionService: ObservableObject {
    /// Unit tests run inside the app and build the engines they need, so there
    /// the shared service never loads the user's model: a load still running
    /// when the test process exits aborts in ggml's Metal teardown.
    static let shared = OpenSuperWhisperApp.isRunningTests
        ? TranscriptionService(engineLoader: { _ in throw TranscriptionError.contextInitializationFailed })
        : TranscriptionService()
    
    @Published private(set) var isTranscribing = false
    @Published private(set) var transcribedText = ""
    @Published private(set) var currentSegment = ""
    @Published private(set) var isLoading = false
    @Published private(set) var progress: Float = 0.0
    @Published private(set) var isConverting = false
    @Published private(set) var conversionProgress: Float = 0.0
    /// Live text of the recording being made, for the recording indicator.
    @Published private(set) var livePreview: LivePreview?
    
    private final class TranscriptionTaskBox {
        let id: UUID
        let engine: TranscriptionEngine
        let task: Task<String, Error>

        init(id: UUID, engine: TranscriptionEngine, task: Task<String, Error>) {
            self.id = id
            self.engine = engine
            self.task = task
        }
    }
    
    private var currentEngine: TranscriptionEngine?
    private var transcriptionTask: TranscriptionTaskBox? = nil
    var activeOperationID: UUID? { transcriptionTask?.id }
    private var cancellationRequestedFor: UUID?

    private struct RecordingPreparation {
        let engine: WhisperEngine
        let task: Task<Void, Error>
    }

    private var recordingPreparation: RecordingPreparation?
    /// Transcribes the current recording while it is made. Its text is used
    /// when that recording's URL is transcribed.
    private var liveTranscriber: LiveTranscriber?
    /// The recording whose previews are shown; previews of any other are late.
    private var livePreviewSource: URL?
    private var pendingLivePreview: LivePreview?
    private var livePreviewPublishedAt: ContinuousClock.Instant?
    private var livePreviewFlush: Task<Void, Never>?
    /// Re-rendering the indicator's material card costs main-thread time, so
    /// its text changes at most five times a second.
    static let livePreviewInterval: Duration = .milliseconds(200)
    var makeLiveTranscriber: (LiveAudioBuffer, WhisperEngine, Settings, @escaping (LivePreview) -> Void) -> LiveTranscriber = { audio, engine, settings, onPreview in
        LiveTranscriber(
            audio: audio,
            makeDecoder: { try engine.makeLiveDecoder(settings: settings) },
            makeProbe: { try WhisperSpeechProbe() },
            onPreview: onPreview
        )
    }
    private var backgroundOperations: [UUID: Task<Void, Never>] = [:]
    private var shutdownTask: Task<Void, Never>?
    private var shutdownEngines: [WhisperEngine] = []
    private(set) var isShuttingDown = false

    func shutdown() async {
        if let shutdownTask {
            await shutdownTask.value
            return
        }
        isShuttingDown = true
        engineLoadTask?.cancel()
        cancelTranscription()
        let engine = currentEngine
        let preparation = recordingPreparation
        let activeTask = transcriptionTask
        let live = liveTranscriber
        liveTranscriber = nil
        live?.cancel()
        endLivePreview()
        let operations = Array(backgroundOperations.values)
        let task = Task {
            _ = await activeTask?.task.result
            _ = await preparation?.task.result
            // A live pass may still be running natively; its state must be
            // freed before the model it decodes with.
            await live?.waitUntilIdle()
            for operation in operations {
                await operation.value
            }
            for engine in shutdownEngines {
                engine.unload()
            }
            shutdownEngines.removeAll()
            (activeTask?.engine as? WhisperEngine)?.unload()
            preparation?.engine.unload()
            (engine as? WhisperEngine)?.unload()
            currentEngine = nil
            recordingPreparation = nil
            transcriptionTask = nil
            engineLoadTask = nil
            engineLoadID = nil
            isLoading = false
            isTranscribing = false
        }
        shutdownTask = task
        await task.value
    }

    /// Called when recording starts. With live audio, the recording is
    /// transcribed while it is made; otherwise a decoding state is prepared so
    /// the decode after release does not pay for it.
    func prepareForRecording(liveAudio: LiveAudioBuffer? = nil) {
        guard !isShuttingDown, !isLoading, transcriptionTask == nil,
              let engine = currentEngine as? WhisperEngine else {
            liveAudio?.cancel()
            return
        }
        if let liveAudio, !liveAudio.isCancelled {
            if startLiveTranscription(audio: liveAudio, engine: engine) { return }
            liveAudio.cancel()
        }
        guard recordingPreparation == nil else { return }
        let task = Task.detached(priority: .userInitiated) {
            try engine.prepareForRecording()
        }
        recordingPreparation = RecordingPreparation(engine: engine, task: task)
        let id = UUID()
        backgroundOperations[id] = Task { [weak self] in
            _ = await task.result
            self?.backgroundOperations[id] = nil
        }
    }

    private func startLiveTranscription(audio: LiveAudioBuffer, engine: WhisperEngine) -> Bool {
        let settings = Settings()
        guard !settings.showTimestamps else { return false }
        // Only one recording is made at a time; a session nobody collected
        // (its recording was queued instead) must not keep its state.
        discardLiveTranscription()
        let source = audio.recordingURL
        let transcriber = makeLiveTranscriber(audio, engine, settings) { [weak self] preview in
            guard let self else { return }
            Task { @MainActor in self.receiveLivePreview(preview, from: source) }
        }
        livePreviewSource = source
        liveTranscriber = transcriber
        transcriber.start()
        return true
    }

    private func receiveLivePreview(_ preview: LivePreview, from source: URL) {
        guard livePreviewSource == source else { return }
        pendingLivePreview = preview
        guard livePreviewFlush == nil else { return }
        let wait = livePreviewPublishedAt.map { Self.livePreviewInterval - $0.duration(to: .now) } ?? .zero
        livePreviewFlush = Task { @MainActor [weak self] in
            if wait > .zero { try? await Task.sleep(for: wait) }
            guard let self, !Task.isCancelled else { return }
            self.livePreviewFlush = nil
            guard self.livePreviewSource == source, let pending = self.pendingLivePreview else { return }
            self.pendingLivePreview = nil
            self.livePreview = pending
            self.livePreviewPublishedAt = .now
        }
    }

    private func endLivePreview() {
        livePreviewSource = nil
        pendingLivePreview = nil
        livePreviewFlush?.cancel()
        livePreviewFlush = nil
        livePreviewPublishedAt = nil
        livePreview = nil
    }

    /// Drops the live session (only the one for `url`, when given); its
    /// recording, if kept, is decoded whole.
    func discardLiveTranscription(for url: URL? = nil) {
        guard let liveTranscriber, url == nil || liveTranscriber.recordingURL == url else { return }
        liveTranscriber.cancel()
        self.liveTranscriber = nil
        endLivePreview()
    }
    
    struct EngineSelection: Equatable {
        let engine: String
        let modelPath: String?
        let modelVersion: String

        static var current: Self {
            let prefs = AppPreferences.shared
            return Self(engine: prefs.selectedEngine,
                        modelPath: prefs.selectedWhisperModelPath ?? prefs.selectedModelPath,
                        modelVersion: prefs.fluidAudioModelVersion)
        }
    }

    @Published private(set) var loadingError: String?
    private let engineLoader: (EngineSelection) async throws -> TranscriptionEngine
    private var engineLoadTask: Task<TranscriptionEngine, Error>?
    private var engineLoadID: UUID?
    private var engineSelection: EngineSelection?

    init(selection: EngineSelection = .current, engineLoader: @escaping (EngineSelection) async throws -> TranscriptionEngine = TranscriptionService.makeEngine) {
        self.engineLoader = engineLoader
        loadEngine(selection: selection)
    }

    init(engine: TranscriptionEngine) {
        engineLoader = Self.makeEngine
        currentEngine = engine
    }

    private static func makeEngine(_ selection: EngineSelection) async throws -> TranscriptionEngine {
        let engine: TranscriptionEngine
        if selection.engine == "fluidaudio" {
            engine = FluidAudioEngine(modelVersion: selection.modelVersion)
        } else {
            guard let path = selection.modelPath else { throw TranscriptionError.contextInitializationFailed }
            engine = WhisperEngine(modelPath: path)
        }
        try await engine.initialize()
        return engine
    }

    func cancelTranscription() {
        guard let activeTask = transcriptionTask else { return }

        cancel(activeTask)
    }

    func cancelTranscription(operationID: UUID) {
        guard let activeTask = transcriptionTask,
              activeTask.id == operationID else { return }

        cancel(activeTask)
    }

    private func cancel(_ activeTask: TranscriptionTaskBox) {

        // Keep the task registered, and keep isTranscribing true, until the
        // engine's native call has actually returned. Clearing either here lets
        // a new recording enter the same engine while whisper.cpp is aborting.
        cancellationRequestedFor = activeTask.id
        activeTask.engine.cancelTranscription()
        activeTask.task.cancel()

        currentSegment = ""
        transcribedText = ""
        progress = 0.0
    }
    
    func loadEngine(selection: EngineSelection) {
        guard !isShuttingDown else { return }
        guard selection != engineSelection || (!isLoading && currentEngine == nil) else { return }
        engineLoadTask?.cancel()
        engineSelection = selection
        let id = UUID()
        engineLoadID = id
        currentEngine = nil
        recordingPreparation = nil
        discardLiveTranscription()
        loadingError = nil
        isLoading = true
        let loader = engineLoader
        let task = Task.detached(priority: .userInitiated) {
            let engine = try await loader(selection)
            try Task.checkCancellation()
            return engine
        }
        engineLoadTask = task
        backgroundOperations[id] = Task { [weak self] in
            let result = await task.result
            self?.finishEngineLoad(id: id, result: result)
            self?.backgroundOperations[id] = nil
        }
    }

    private func finishEngineLoad(id: UUID, result: Result<TranscriptionEngine, Error>) {
        guard !isShuttingDown else {
            if case .success(let engine as WhisperEngine) = result {
                shutdownEngines.append(engine)
            }
            return
        }
        guard engineLoadID == id else { return }
        engineLoadTask = nil
        engineLoadID = nil
        isLoading = false
        switch result {
        case .success(let engine): currentEngine = engine
        case .failure(let error): loadingError = error.localizedDescription
        }
    }

    func waitUntilReady() async throws {
        guard !isShuttingDown else { throw CancellationError() }
        while let task = engineLoadTask, let id = engineLoadID {
            let result = await task.result
            try Task.checkCancellation()
            guard !isShuttingDown else { throw CancellationError() }
            finishEngineLoad(id: id, result: result)
        }
        try Task.checkCancellation()
        guard !isShuttingDown else { throw CancellationError() }
        guard currentEngine != nil else { throw TranscriptionError.contextInitializationFailed }
    }

    func reloadEngine() {
        loadEngine(selection: .current)
    }

    func reloadModel(with path: String) {
        if AppPreferences.shared.selectedEngine == "whisper" {
            AppPreferences.shared.selectedWhisperModelPath = path
            reloadEngine()
        }
    }
    
    func transcribeAudio(url: URL, settings: Settings, pcmSamples: [Float]? = nil) async throws -> String {
        try await transcribeAudio(
            url: url,
            settings: settings,
            operationID: UUID(),
            pcmSamples: pcmSamples
        )
    }

    func transcribeAudio(
        url: URL,
        settings: Settings,
        operationID: UUID,
        pcmSamples: [Float]? = nil
    ) async throws -> String {
        try Task.checkCancellation()

        // Serialize access to the engine: a whisper context must not process
        // two transcriptions concurrently (indicator flow and queue flow can
        // both reach this point due to async busy checks).
        while true {
            try await waitUntilReady()
            guard let existing = transcriptionTask else { break }
            _ = try? await existing.task.value
            try Task.checkCancellation()
            if transcriptionTask === existing {
                transcriptionTask = nil
                if cancellationRequestedFor == existing.id {
                    cancellationRequestedFor = nil
                }
            }
        }
        
        progress = 0.0
        conversionProgress = 0.0
        isConverting = true
        isTranscribing = true
        transcribedText = ""
        currentSegment = ""
        cancellationRequestedFor = nil
        
        guard let engine = currentEngine else {
            isTranscribing = false
            isConverting = false
            throw TranscriptionError.contextInitializationFailed
        }

        let preparation = recordingPreparation
        recordingPreparation = nil
        let live = liveTranscriber?.recordingURL == url ? liveTranscriber : nil
        if live != nil {
            liveTranscriber = nil
        }

        // Setup progress callback for engines
        if let whisperEngine = engine as? WhisperEngine {
            whisperEngine.onProgressUpdate = { [weak self] newProgress in
                Task { @MainActor in
                    guard let self,
                          self.transcriptionTask?.id == operationID,
                          self.cancellationRequestedFor != operationID else { return }
                    self.progress = newProgress
                }
            }
        } else if let fluidEngine = engine as? FluidAudioEngine {
            fluidEngine.onProgressUpdate = { [weak self] newProgress in
                Task { @MainActor in
                    guard let self,
                          self.transcriptionTask?.id == operationID,
                          self.cancellationRequestedFor != operationID else { return }
                    self.progress = newProgress
                }
            }
        }
        
        let task = Task.detached(priority: .userInitiated) { [weak self] in
            if let preparation, preparation.engine === engine {
                try await preparation.task.value
            }
            try Task.checkCancellation()
            
            let cancelled = await MainActor.run {
                guard let self = self else { return true }
                return self.cancellationRequestedFor == operationID
            }
            
            guard !cancelled else {
                throw CancellationError()
            }
            
            let result: String
            do {
                if let live {
                    result = try await Self.finishLiveTranscription(
                        live, engine: engine, url: url, settings: settings, pcmSamples: pcmSamples
                    )
                } else if let pcmSamples, let whisper = engine as? WhisperEngine {
                    result = try await whisper.transcribeSamples(pcmSamples, settings: settings)
                } else {
                    result = try await engine.transcribeAudio(url: url, settings: settings)
                }
            } catch {
                // Native engines may surface their own generic error after an
                // abort callback. Preserve cancellation as cancellation for the
                // indicator and queue instead of treating it as a failed decode.
                try Task.checkCancellation()
                let cancellationRequested = await MainActor.run {
                    guard let self = self else { return true }
                    return self.cancellationRequestedFor == operationID
                }
                if cancellationRequested {
                    throw CancellationError()
                }
                throw error
            }
            
            try Task.checkCancellation()
            
            let finalCancelled = await MainActor.run {
                guard let self = self else { return true }
                return self.cancellationRequestedFor == operationID
                    || self.transcriptionTask?.id != operationID
            }

            guard !finalCancelled else {
                throw CancellationError()
            }

            let didPublish = await MainActor.run {
                guard let self,
                      self.transcriptionTask?.id == operationID,
                      self.cancellationRequestedFor != operationID else { return false }
                self.transcribedText = result
                self.progress = 1.0
                return true
            }

            guard didPublish else { throw CancellationError() }
            try Task.checkCancellation()
            
            return result
        }
        
        let taskBox = TranscriptionTaskBox(
            id: operationID,
            engine: engine,
            task: task
        )
        transcriptionTask = taskBox

        defer {
            // The indicator shows the live text until the dictation is done.
            if let live, livePreviewSource == live.recordingURL {
                endLivePreview()
            }
            if transcriptionTask === taskBox {
                let wasCancelled = cancellationRequestedFor == taskBox.id
                transcriptionTask = nil
                if wasCancelled {
                    cancellationRequestedFor = nil
                    transcribedText = ""
                }
                isTranscribing = false
                isConverting = false
                currentSegment = ""
                progress = wasCancelled ? 0.0 : 1.0
            }
        }
        
        do {
            return try await task.value
        } catch is CancellationError {
            throw CancellationError()
        }
    }

    /// The live session has decoded every chunk but the last; this decodes
    /// the last one and cleans the joined text once, as the batch path does.
    /// If any pass failed, the whole recording is decoded the usual way.
    private nonisolated static func finishLiveTranscription(
        _ live: LiveTranscriber,
        engine: TranscriptionEngine,
        url: URL,
        settings: Settings,
        pcmSamples: [Float]?
    ) async throws -> String {
        do {
            let transcript = try await live.finish(recordedSamples: pcmSamples)
            return WhisperEngine.cleanedTranscript(transcript.text, settings: settings)
        } catch {
            try Task.checkCancellation()
            print("Live transcription unavailable, decoding the whole recording: \(error)")
            if let pcmSamples, let whisper = engine as? WhisperEngine {
                return try await whisper.transcribeSamples(pcmSamples, settings: settings)
            }
            return try await engine.transcribeAudio(url: url, settings: settings)
        }
    }
}

enum TranscriptionError: Error {
    case contextInitializationFailed
    case audioConversionFailed
    case processingFailed
}
