import Foundation

/// The subset of UserDefaults that preferences use.
protocol PreferenceStore: AnyObject {
    func object(forKey defaultName: String) -> Any?
    func set(_ value: Any?, forKey defaultName: String)
    func removeObject(forKey defaultName: String)
}

extension UserDefaults: PreferenceStore {}

/// Preferences that live only as long as the process.
final class InMemoryPreferenceStore: PreferenceStore {
    private let lock = NSLock()
    private var values: [String: Any] = [:]

    func object(forKey defaultName: String) -> Any? {
        lock.lock()
        defer { lock.unlock() }
        return values[defaultName]
    }

    func set(_ value: Any?, forKey defaultName: String) {
        lock.lock()
        defer { lock.unlock() }
        values[defaultName] = value
    }

    func removeObject(forKey defaultName: String) {
        set(nil, forKey: defaultName)
    }
}

@propertyWrapper
struct UserDefault<T> {
    let key: String
    let defaultValue: T
    
    var wrappedValue: T {
        get { AppPreferences.store.object(forKey: key) as? T ?? defaultValue }
        set { AppPreferences.store.set(newValue, forKey: key) }
    }
}

@propertyWrapper
struct OptionalUserDefault<T> {
    let key: String
    
    var wrappedValue: T? {
        get { AppPreferences.store.object(forKey: key) as? T }
        set { AppPreferences.store.set(newValue, forKey: key) }
    }
}

final class AppPreferences {
    static let shared = AppPreferences()

    /// Unit tests run inside the app, so with UserDefaults.standard they would
    /// read and overwrite the user's real settings, and parallel test
    /// processes would race on them. Each test process gets its own store.
    static let store: PreferenceStore = OpenSuperWhisperApp.isRunningTests
        ? InMemoryPreferenceStore()
        : UserDefaults.standard

    private init() {
        migrateOldPreferences()
    }
    
    private func migrateOldPreferences() {
        let store = Self.store
        if let oldPath = store.object(forKey: "selectedModelPath") as? String,
           store.object(forKey: "selectedWhisperModelPath") as? String == nil {
            store.set(oldPath, forKey: "selectedWhisperModelPath")
        }
    }
    
    // Engine settings
    @UserDefault(key: "selectedEngine", defaultValue: "whisper")
    var selectedEngine: String
    
    // Model settings
    var selectedModelPath: String? {
        get {
            if selectedEngine == "whisper" {
                return selectedWhisperModelPath
            }
            return nil
        }
        set {
            if selectedEngine == "whisper" {
                selectedWhisperModelPath = newValue
            }
        }
    }
    
    @OptionalUserDefault(key: "selectedWhisperModelPath")
    var selectedWhisperModelPath: String?
    
    @UserDefault(key: "fluidAudioModelVersion", defaultValue: "v3")
    var fluidAudioModelVersion: String
    
    @UserDefault(key: "qwen3Variant", defaultValue: "f32")
    var qwen3Variant: String
    
    @UserDefault(key: "whisperLanguage", defaultValue: "en")
    var whisperLanguage: String
    
    // Transcription settings
    @UserDefault(key: "suppressBlankAudio", defaultValue: true)
    var suppressBlankAudio: Bool
    
    @UserDefault(key: "showTimestamps", defaultValue: false)
    var showTimestamps: Bool

    @UserDefault(key: "liveTranscription", defaultValue: false)
    var liveTranscription: Bool

    /// Live transcription is Whisper only, and stands down while timestamps
    /// are shown: they refer to positions in the whole recording.
    var isLiveTranscriptionActive: Bool {
        liveTranscription && selectedEngine == "whisper" && !showTimestamps
    }
    
    @UserDefault(key: "temperature", defaultValue: 0.0)
    var temperature: Double
    
    @UserDefault(key: "noSpeechThreshold", defaultValue: 0.6)
    var noSpeechThreshold: Double
    
    @UserDefault(key: "initialPrompt", defaultValue: "")
    var initialPrompt: String
    
    @UserDefault(key: "useBeamSearch", defaultValue: false)
    var useBeamSearch: Bool
    
    @UserDefault(key: "beamSize", defaultValue: 5)
    var beamSize: Int
    
    @UserDefault(key: "debugMode", defaultValue: false)
    var debugMode: Bool
    
    @UserDefault(key: "playSoundOnRecordStart", defaultValue: false)
    var playSoundOnRecordStart: Bool
    
    @UserDefault(key: "hasCompletedOnboarding", defaultValue: false)
    var hasCompletedOnboarding: Bool
    
    @UserDefault(key: "useAsianAutocorrect", defaultValue: true)
    var useAsianAutocorrect: Bool
    
    @OptionalUserDefault(key: "selectedMicrophoneData")
    var selectedMicrophoneData: Data?
    
    @UserDefault(key: "modifierOnlyHotkey", defaultValue: "none")
    var modifierOnlyHotkey: String
    
    /// Last non-none modifier key, used to restore the user's choice
    /// when switching back to Single Modifier Key mode.
    @UserDefault(key: "lastModifierOnlyHotkey", defaultValue: "leftCommand")
    var lastModifierOnlyHotkey: String
    
    @UserDefault(key: "mouseButtonHotkey", defaultValue: "none")
    var mouseButtonHotkey: String


    @UserDefault(key: "holdToRecord", defaultValue: true)
    var holdToRecord: Bool
    
    @UserDefault(key: "addSpaceAfterSentence", defaultValue: true)
    var addSpaceAfterSentence: Bool

    // Clipboard settings
    @UserDefault(key: "autoCopyToClipboard", defaultValue: false)
    var autoCopyToClipboard: Bool

    @UserDefault(key: "autoPasteTranscription", defaultValue: true)
    var autoPasteTranscription: Bool

    @UserDefault(key: "escCancelWithoutConfirmation", defaultValue: false)
    var escCancelWithoutConfirmation: Bool

    @UserDefault(key: "startHiddenInMenuBar", defaultValue: false)
    var startHiddenInMenuBar: Bool

    @UserDefault(key: "autoDeleteRecordingsEnabled", defaultValue: false)
    var autoDeleteRecordingsEnabled: Bool

    @UserDefault(key: "autoDeleteRecordingsAfterDays", defaultValue: 30)
    var autoDeleteRecordingsAfterDays: Int
}
