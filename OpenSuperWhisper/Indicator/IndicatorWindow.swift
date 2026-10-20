import Cocoa
import Combine
import SwiftUI

enum RecordingState {
    case idle
    case connecting
    case recording
    case decoding
    case busy
    case noMicrophone
}

@MainActor
protocol IndicatorViewDelegate: AnyObject {
    
    @discardableResult
    func didFinishDecoding(from viewModel: IndicatorViewModel) -> Bool
}

@MainActor
class IndicatorViewModel: ObservableObject {
    static let cancelConfirmationThreshold: TimeInterval = 10.0
    static let cancelConfirmationWindow: TimeInterval = 5.0
    
    @Published var state: RecordingState = .idle
    @Published var isBlinking = false
    @Published var isConfirmingCancel = false
    @Published var recorder: AudioRecorder = .shared
    /// The text of a live dictation, fitted to the card.
    @Published private(set) var liveText: LivePreview?
    
    /// Live transcription shows its text in a wider card that grows with it;
    /// without it the indicator is the compact card.
    let usesLiveLayout: Bool
    /// The live card's height as laid out, for the hide animation.
    var liveCardHeight = IndicatorWindow.cardSize.height
    
    var recordingStartedAt: Date?
    
    var delegate: IndicatorViewDelegate?
    private var blinkTimer: Timer?
    private var hideTimer: Timer?
    private var confirmCancelTimer: Timer?
    private var decodingTask: Task<Void, Never>?
    private var decodingSessionID: UUID?
    private var recordingSessionID: UUID?
    private var cancellables = Set<AnyCancellable>()
    
    private let recordingStore: RecordingStore
    private let transcriptionService: TranscriptionService
    private let transcriptionQueue: TranscriptionQueue
    private let stopRecordingOperation: () async -> RecordedAudio?
    private let cancelAudioRecordingOperation: () -> Void
    
    init(
        transcriptionService: TranscriptionService = .shared,
        recordingStore: RecordingStore = .shared,
        stopRecording: @escaping () async -> RecordedAudio? = {
            await AudioRecorder.shared.stopRecording()
        },
        cancelAudioRecording: @escaping () -> Void = {
            AudioRecorder.shared.cancelRecording()
        },
        usesLiveLayout: Bool = AppPreferences.shared.isLiveTranscriptionActive
    ) {
        self.usesLiveLayout = usesLiveLayout
        self.recordingStore = recordingStore
        self.transcriptionService = transcriptionService
        self.transcriptionQueue = TranscriptionQueue.shared
        self.stopRecordingOperation = stopRecording
        self.cancelAudioRecordingOperation = cancelAudioRecording
        
        recorder.$startFailure
            .compactMap { $0 }
            .sink { [weak self] failure in
                guard let self, self.recordingSessionID == failure.sessionID else { return }
                self.resetAfterRecordingFailure()
                AppErrorCenter.shared.report("Recording failed", message: failure.message)
            }
            .store(in: &cancellables)

        recorder.$isConnecting
            .receive(on: RunLoop.main)
            .sink { [weak self] isConnecting in
                guard let self, let id = self.recordingSessionID,
                      RecordingSessionController.shared.currentID == id,
                      RecordingSessionController.shared.isCapturing else { return }
                if isConnecting && self.recorder.isConnecting {
                    self.state = .connecting
                    self.stopBlinking()
                }
            }
            .store(in: &cancellables)
        
        recorder.$isRecording
            .receive(on: RunLoop.main)
            .sink { [weak self] isRecording in
                guard let self, let id = self.recordingSessionID,
                      RecordingSessionController.shared.currentID == id,
                      RecordingSessionController.shared.isCapturing else { return }
                if isRecording && self.recorder.isRecording {
                    self.state = .recording
                    self.startBlinking()
                }
            }
            .store(in: &cancellables)

        if usesLiveLayout {
            transcriptionService.$livePreview
                .sink { [weak self] preview in
                    self?.liveText = preview.map {
                        LiveTextLayout.fitting($0, width: IndicatorWindow.liveTextWidth, lines: IndicatorWindow.liveTextLines)
                    }
                }
                .store(in: &cancellables)
        }
    }
    
    var isTranscriptionBusy: Bool {
        transcriptionService.isLoading || transcriptionService.isTranscribing || transcriptionQueue.isProcessing
    }
    
    func showBusyMessage() {
        showAutoDismissingMessage(.busy)
    }

    private func showAutoDismissingMessage(_ message: RecordingState) {
        state = message

        hideTimer?.invalidate()
        hideTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                _ = self.delegate?.didFinishDecoding(from: self)
            }
        }
    }

    func resetAfterRecordingFailure() {
        RecordingSessionController.shared.finish(recordingSessionID)
        recordingSessionID = nil
        state = .idle
        stopBlinking()
        recordingStartedAt = nil
        resetCancelConfirmation()
        _ = delegate?.didFinishDecoding(from: self)
    }

    func startRecording() {
        if isTranscriptionBusy {
            showBusyMessage()
            return
        }

        // getActiveMicrophone() only reads the cached currentMicrophone, so
        // this guard costs no CoreAudio HAL round-trip on the main thread.
        guard MicrophoneService.shared.getActiveMicrophone() != nil else {
            showAutoDismissingMessage(.noMicrophone)
            return
        }
        
        // Optimistically assume recording: querying the microphone here costs
        // CoreAudio HAL round-trips on the main thread right before the appear
        // animation. The recorder resolves the real state on its own queue and
        // publishes isConnecting/isRecording, which the sinks above translate
        // into .connecting/.recording.
        guard let id = RecordingSessionController.shared.begin(stop: { self.decodeRecording() }) else { return }
        recordingSessionID = id
        state = .recording
        startBlinking()
        recordingStartedAt = Date()
        
        recorder.startRecording(sessionID: id)
    }
    
    func handleCancelRequest() -> Bool {
        guard state == .recording,
              !AppPreferences.shared.escCancelWithoutConfirmation,
              !isConfirmingCancel,
              let startedAt = recordingStartedAt,
              Date().timeIntervalSince(startedAt) >= Self.cancelConfirmationThreshold
        else {
            return true
        }
        
        isConfirmingCancel = true
        confirmCancelTimer?.invalidate()
        confirmCancelTimer = Timer.scheduledTimer(withTimeInterval: Self.cancelConfirmationWindow, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.resetCancelConfirmation()
            }
        }
        return false
    }
    
    private func resetCancelConfirmation() {
        confirmCancelTimer?.invalidate()
        confirmCancelTimer = nil
        isConfirmingCancel = false
    }
    
    func startDecoding() {
        if RecordingSessionController.shared.hasSession {
            RecordingSessionController.shared.requestStop()
        } else {
            decodeRecording()
        }
    }

    private func decodeRecording() {
        // A second stop request (double hotkey press, hold-mode key-up) must not
        // restart decoding or hide the window while transcription is in flight.
        guard state == .recording || state == .connecting else { return }
        
        resetCancelConfirmation()
        stopBlinking()
        
        if isTranscriptionBusy {
            // The engine is busy with another transcription: keep the user's audio
            // and put it into the queue instead of deleting it.
            Task { [weak self] in
                guard let self = self else { return }
                defer {
                    RecordingSessionController.shared.finish(self.recordingSessionID)
                    self.recordingSessionID = nil
                }
                if let audio = await self.stopRecordingOperation() {
                    self.transcriptionService.discardLiveTranscription(for: audio.url)
                    await self.transcriptionQueue.addFileToQueue(url: audio.url)
                }
            }
            showBusyMessage()
            return
        }
        
        state = .decoding

        let sessionID = UUID()
        decodingSessionID = sessionID
        decodingTask = Task { [weak self] in
            guard let self = self else { return }

            guard let audio = await self.stopRecordingOperation() else {
                print("!!! Not found record url !!!")
                // Too short or failed: nobody will ask for its live text.
                self.transcriptionService.discardLiveTranscription()
                self.finishDecoding(sessionID: sessionID)
                return
            }
            let tempURL = audio.url
            var savedRecording: Recording?
            // An Esc before transcribeAudio takes the live session (during
            // the stop tail) would otherwise leave it running.
            defer { self.transcriptionService.discardLiveTranscription(for: tempURL) }

            do {
                try Task.checkCancellation()
                guard self.decodingSessionID == sessionID else {
                    throw CancellationError()
                }

                print("start decoding...")
                let duration = audio.duration
                try Task.checkCancellation()
                guard self.decodingSessionID == sessionID else {
                    throw CancellationError()
                }

                let text = try await transcriptionService.transcribeAudio(
                    url: tempURL,
                    settings: Settings(),
                    operationID: sessionID,
                    pcmSamples: audio.samples
                )
                try Task.checkCancellation()
                guard self.decodingSessionID == sessionID else {
                    throw CancellationError()
                }

                if text.isEmpty {
                    try? FileManager.default.removeItem(at: tempURL)
                    print("No speech detected, dictation discarded")
                } else {
                    let timestamp = Date()
                    let recordingId = UUID()
                    let fileName = Recording.fileName(for: recordingId)
                    let newRecording = Recording(
                        id: recordingId,
                        timestamp: timestamp,
                        fileName: fileName,
                        transcription: text,
                        duration: duration,
                        status: .completed,
                        progress: 1.0,
                        sourceFileURL: nil
                    )

                    try recorder.moveTemporaryRecording(from: tempURL, to: newRecording.url)
                    savedRecording = newRecording
                    do { try await self.recordingStore.addRecordingSync(newRecording) }
                    catch { throw PreservedAudioError(url: newRecording.url, underlying: error) }

                    try Task.checkCancellation()
                    guard self.decodingSessionID == sessionID else { throw CancellationError() }
                    insertText(text)
                    print("Transcription result: \(text)")
                }
            } catch is CancellationError {
                if let savedRecording {
                    do { try await Task { try await recordingStore.deleteRecordingSync(savedRecording, cancelTranscription: false) }.value }
                    catch { AppErrorCenter.shared.report("Cancelled recording could not be removed", error: error) }
                } else {
                    try? FileManager.default.removeItem(at: tempURL)
                }
                print("Transcription cancelled")
            } catch {
                if Task.isCancelled || self.decodingSessionID != sessionID {
                    if let savedRecording {
                        do { try await Task { try await recordingStore.deleteRecordingSync(savedRecording, cancelTranscription: false) }.value }
                        catch { AppErrorCenter.shared.report("Cancelled recording could not be removed", error: error) }
                    } else {
                        try? FileManager.default.removeItem(at: tempURL)
                    }
                    print("Transcription cancelled")
                } else {
                    let source = (error as? PreservedAudioError)?.url ?? audio.url
                    await self.recordingStore.preserveFailedDictation(RecordedAudio(url: source, samples: audio.samples), error: error)
                }
            }

            self.finishDecoding(sessionID: sessionID)
        }
    }

    private func finishDecoding(sessionID: UUID) {
        guard decodingSessionID == sessionID else { return }
        decodingSessionID = nil
        decodingTask = nil
        RecordingSessionController.shared.finish(recordingSessionID)
        recordingSessionID = nil
        _ = delegate?.didFinishDecoding(from: self)
    }
    
    func insertText(_ text: String) {
        guard !text.isEmpty else { return }
        let finalText = Self.applyPostProcessing(text)
        let prefs = AppPreferences.shared

        if prefs.autoPasteTranscription {
            if prefs.autoCopyToClipboard {
                // Paste and keep in clipboard
                ClipboardUtil.insertTextAndKeepInClipboard(finalText)
            } else {
                // Paste but restore original clipboard (legacy behavior)
                ClipboardUtil.insertText(finalText)
            }
        } else if prefs.autoCopyToClipboard {
            // Only copy to clipboard, don't paste
            ClipboardUtil.copyToClipboard(finalText)
        }
        // If both are false, do nothing

    }
    
    static func applyPostProcessing(_ text: String) -> String {
        guard AppPreferences.shared.addSpaceAfterSentence,
              let lastChar = text.last,
              lastChar.isPunctuation else {
            return text
        }
        return text + " "
    }
    
    private func startBlinking() {
        blinkTimer?.invalidate()
        blinkTimer = Timer.scheduledTimer(withTimeInterval: 0.8, repeats: true) { [weak self] _ in
            // Update UI on the main thread
            Task { @MainActor in
                guard let self = self else { return }
                self.isBlinking.toggle()
            }
        }
    }
    
    private func stopBlinking() {
        blinkTimer?.invalidate()
        blinkTimer = nil
        isBlinking = false
    }

    func cleanup() {
        stopBlinking()
        resetCancelConfirmation()
        recordingStartedAt = nil
        hideTimer?.invalidate()
        hideTimer = nil
        cancellables.removeAll()
    }

    func cancelRecording() {
        hideTimer?.invalidate()
        hideTimer = nil

        if state == .decoding {
            // In decoding the recorder is already stopped. Cancel both the
            // Swift task and the native engine operation, and invalidate the
            // session before either can save or paste a late result.
            let cancelledSessionID = decodingSessionID
            decodingSessionID = nil
            decodingTask?.cancel()
            decodingTask = nil
            if let cancelledSessionID {
                transcriptionService.cancelTranscription(
                    operationID: cancelledSessionID
                )
            }
        }

        if state != .decoding {
            cancelAudioRecordingOperation()
            // The recording is gone, and so is any text shown for it.
            transcriptionService.discardLiveTranscription()
        }
        RecordingSessionController.shared.finish(recordingSessionID)
        recordingSessionID = nil
    }
}

struct RecordingIndicator: View {
    let isBlinking: Bool
    
    var body: some View {
        Circle()
            .fill(
                LinearGradient(
                    colors: [
                        Color.red.opacity(0.8),
                        Color.red
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .frame(width: 8, height: 8)
            .shadow(color: .red.opacity(0.5), radius: 4)
            .opacity(isBlinking ? 0.3 : 1.0)
            .animation(.easeInOut(duration: 0.4), value: isBlinking)
    }
}

struct CancelConfirmationBar: View {
    @State private var progress: CGFloat = 1
    
    var body: some View {
        GeometryReader { geo in
            Capsule()
                .fill(Color.orange)
                .frame(width: geo.size.width * progress, height: 2)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(height: 2)
        .padding(.horizontal, 12)
        .padding(.bottom, 3)
        .onAppear {
            withAnimation(.linear(duration: IndicatorViewModel.cancelConfirmationWindow)) {
                progress = 0
            }
        }
    }
}

struct IndicatorWindow: View {
    /// Geometry shared with IndicatorWindowManager. The panel must be larger
    /// than the card: everything drawn outside the window bounds is cut off,
    /// so the appear offset (moves the card down) and the spring overshoot
    /// need margins, otherwise the card edges are visibly clipped mid-animation.
    static let cardSize = CGSize(width: 200, height: 36)
    static let windowSize = CGSize(width: 256, height: 96)
    static let appearOffset: CGFloat = 20
    static let appearInitialScale: CGFloat = 0.5
    
    /// The live card is wider and grows upward from the compact card's
    /// height to three lines of text; the panel leaves the same margins.
    static let liveCardWidth: CGFloat = 360
    /// The text lines up with the status label, after the 24 pt icon slot
    /// and its 8 pt spacing, like the body of a notification.
    static let liveTextIndent: CGFloat = 32
    static let liveTextWidth = liveCardWidth - 48 - liveTextIndent
    static let liveTextLines = 3
    static let liveTextFontSize: CGFloat = 12
    static let liveCardMaxHeight: CGFloat = 88
    static let liveCardBottomInset = (windowSize.height - cardSize.height) / 2
    static let liveWindowSize = CGSize(
        width: liveCardWidth + (windowSize.width - cardSize.width),
        height: liveCardMaxHeight + (windowSize.height - cardSize.height)
    )
    
    @ObservedObject var viewModel: IndicatorViewModel
    @Environment(\.colorScheme) private var colorScheme
    
    private var backgroundColor: Color {
        colorScheme == .dark
            ? Color.black.opacity(0.24)
            : Color.white.opacity(0.24)
    }
    
    var body: some View {
        if viewModel.usesLiveLayout {
            liveCard
        } else {
            compactCard
        }
    }

    private var compactCard: some View {

        let rect = RoundedRectangle(cornerRadius: 24)
        
        return VStack(spacing: 12) {
            statusRow
        }
        .padding(.horizontal, 24)
        .frame(height: Self.cardSize.height)
        .background {
            rect
                .fill(backgroundColor)
                .background {
                    rect
                        .fill(Material.thinMaterial)
                }
        }
        .overlay(alignment: .bottom) {
            if viewModel.isConfirmingCancel {
                CancelConfirmationBar()
            }
        }
        .clipShape(rect)
        .frame(width: Self.cardSize.width)
        // The ideal size of the root view must match the panel: NSHostingView
        // resizes the window down to SwiftUI's ideal size, and a window sized
        // to the bare card clips the appear offset, bounce overshoot and shadow.
        .frame(width: Self.windowSize.width, height: Self.windowSize.height)
        // The appear/hide animation is NOT done in SwiftUI on purpose:
        // animating scaleEffect/offset/opacity re-rasterizes the card (material
        // + gradients + shadow) on the CPU every frame and stalls the main
        // thread in CABackingStoreUpdate/wait_for_synchronize (20-60 ms per
        // frame in traces). IndicatorWindowManager animates the hosting view's
        // layer with CASpringAnimation instead: content is drawn once and the
        // spring runs entirely in the render server on the GPU.
    }

    /// The compact card's row on top, the live text below it. The card sits
    /// at the bottom of the panel, so it grows upward, away from the caret.
    /// Its text changes without animation: re-rendering the material card
    /// is expensive, and the service limits changes to five a second.
    private var liveCard: some View {
        // At the compact height the corners make the same capsule ends.
        let rect = RoundedRectangle(cornerRadius: Self.cardSize.height / 2)

        return VStack(spacing: 0) {
            Spacer(minLength: 0)
            VStack(alignment: .leading, spacing: 4) {
                statusRow
                    .frame(height: Self.cardSize.height - 16)
                if let text = viewModel.liveText, !text.isEmpty {
                    (Text(text.settled).foregroundColor(.primary)
                        + Text(text.unsettled).foregroundColor(.secondary))
                        .font(.system(size: Self.liveTextFontSize))
                        .lineLimit(Self.liveTextLines)
                        .truncationMode(.head)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.leading, Self.liveTextIndent)
                        // Clears the cancel confirmation bar at the bottom.
                        .padding(.bottom, 2)
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 8)
            .frame(width: Self.liveCardWidth)
            .background {
                rect
                    .fill(backgroundColor)
                    .background {
                        rect
                            .fill(Material.thinMaterial)
                    }
            }
            .overlay(alignment: .bottom) {
                if viewModel.isConfirmingCancel {
                    CancelConfirmationBar()
                }
            }
            .clipShape(rect)
            .background {
                GeometryReader { proxy in
                    Color.clear.preference(key: LiveCardHeightKey.self, value: proxy.size.height)
                }
            }
            .padding(.bottom, Self.liveCardBottomInset)
        }
        .frame(width: Self.liveWindowSize.width, height: Self.liveWindowSize.height)
        .onPreferenceChange(LiveCardHeightKey.self) { height in
            viewModel.liveCardHeight = height
        }
    }

    @ViewBuilder
    private var statusRow: some View {
        switch viewModel.state {
        case .connecting:
            HStack(spacing: 8) {
                ProgressView()
                    .scaleEffect(0.7)
                    .frame(width: 24)
                
                Text("Connecting...")
                    .font(.system(size: 13, weight: .semibold))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            
        case .recording:
            HStack(spacing: 8) {
                RecordingIndicator(isBlinking: viewModel.isBlinking)
                    .frame(width: 24)
                
                if viewModel.isConfirmingCancel {
                    Text("Press Esc to cancel")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.orange)
                        .transition(.opacity)
                } else {
                    Text("Recording...")
                        .font(.system(size: 13, weight: .semibold))
                        .transition(.opacity)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .animation(.easeInOut(duration: 0.2), value: viewModel.isConfirmingCancel)
            
        case .decoding:
            HStack(spacing: 8) {
                ProgressView()
                    .scaleEffect(0.7)
                    .frame(width: 24)
                
                Text("Transcribing...")
                    .font(.system(size: 13, weight: .semibold))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            
        case .busy:
            HStack(spacing: 8) {
                Image(systemName: "hourglass")
                    .foregroundColor(.orange)
                    .frame(width: 24)

                Text("Processing...")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.orange)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case .noMicrophone:
            HStack(spacing: 8) {
                Image(systemName: "mic.slash")
                    .foregroundColor(.orange)
                    .frame(width: 24)

                Text("No microphone")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.orange)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case .idle:
            EmptyView()
        }
    }
}

private struct LiveCardHeightKey: PreferenceKey {
    static let defaultValue = IndicatorWindow.cardSize.height

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

/// Fits live text into the card's lines by dropping its oldest words, so
/// the newest words stay visible.
enum LiveTextLayout {
    static func fitting(_ preview: LivePreview, width: CGFloat, lines: Int) -> LivePreview {
        let font = NSFont.systemFont(ofSize: IndicatorWindow.liveTextFontSize)
        var settled = preview.settled
        var unsettled = preview.unsettled
        if settled.isEmpty {
            unsettled = String(unsettled.drop(while: \.isWhitespace))
        } else {
            settled = String(settled.drop(while: \.isWhitespace))
        }
        let text = Array(settled + unsettled)
        let maxHeight = CGFloat(lines) * NSLayoutManager().defaultLineHeight(for: font) + 0.5
        func fits(_ candidate: String) -> Bool {
            NSAttributedString(string: candidate, attributes: [.font: font])
                .boundingRect(with: CGSize(width: width, height: .greatestFiniteMagnitude),
                              options: [.usesLineFragmentOrigin, .usesFontLeading])
                .height <= maxHeight
        }
        guard !fits(String(text)) else { return LivePreview(settled: settled, unsettled: unsettled) }

        // Candidate starts: the first character of each word (or of each
        // character in scripts written without spaces).
        let starts = text.indices.filter { index in
            !text[index].isWhitespace && (index == 0 || text[index - 1].isWhitespace
                || LiveText.isUnspacedScript(text[index]) || LiveText.isUnspacedScript(text[index - 1]))
        }
        var low = 0
        var high = starts.count - 1
        while low < high {
            let middle = (low + high) / 2
            if fits("…" + String(text[starts[middle]...])) {
                high = middle
            } else {
                low = middle + 1
            }
        }
        let cut = starts.isEmpty ? 0 : starts[low]
        if cut < settled.count {
            return LivePreview(settled: "…" + String(text[cut..<settled.count]), unsettled: unsettled)
        }
        return LivePreview(settled: "…", unsettled: String(text[cut...]))
    }
}

struct IndicatorWindowPreview: View {
    @StateObject private var recordingVM = {
        let vm = IndicatorViewModel()
//        vm.startRecording()
        return vm
    }()
    
    @StateObject private var decodingVM = {
        let vm = IndicatorViewModel()
        vm.state = .decoding
        return vm
    }()
    
    var body: some View {
        VStack(spacing: 20) {
            IndicatorWindow(viewModel: recordingVM)
            IndicatorWindow(viewModel: decodingVM)
        }
        .padding()
        .frame(height: 200)
        .background(Color(.windowBackgroundColor))
    }
}

#Preview {
    IndicatorWindowPreview()
}
