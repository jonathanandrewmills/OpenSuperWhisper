# Dictation pipeline

How a dictation travels from the microphone to the focused app.
It reflects the code on `fix/osw-streaming-build` (live transcription, on top of `c8e6fe7`).
Render it with `Scripts/render_architecture.py`; the HTML it writes is not committed.

```mermaid
flowchart LR
  subgraph Capture["Capture: AudioRecorder and PCMRecordingSession"]
    Tap["AVAudioEngine input tap"] --> Convert["PCM queue<br/>convert to 16 kHz"]
    Convert --> Wav[("Temporary WAV<br/>and in-memory samples")]
  end

  Convert -->|"mono blocks,<br/>Live Transcription on"| Buffer["LiveAudioBuffer"]

  subgraph Live["LiveTranscriber: one serial queue per recording"]
    Buffer --> Probe["Silero probe<br/>streaming, 32 ms frames"]
    Probe --> Cutter{"ChunkCutter<br/>600 ms pause after 8 s,<br/>or quietest 100 ms by 24 s"}
    Cutter -->|"finished chunk"| ChunkPass["Chunk pass<br/>user's settings,<br/>earlier text as prompt"]
    Buffer -->|"open chunk,<br/>about once a second"| PreviewPass["Preview pass<br/>greedy, temperature 0"]
    PreviewPass --> Agreement["Repetition guard and<br/>LocalAgreement-2"]
  end

  subgraph Engine["WhisperEngine: one loaded model"]
    BatchState["Prepared whisper_state"]
    LiveState["Live whisper_state"]
  end

  ChunkPass -.-> LiveState
  PreviewPass -.-> LiveState

  Release["Hotkey release<br/>after a 250 ms tail"] --> Service["TranscriptionService<br/>transcribeAudio"]
  Wav --> Service
  Service -->|"live session"| LastChunk["Last chunk pass,<br/>join, cleanup"]
  Service -->|"no session, or a pass failed"| Whole["Whole-recording pass"]
  LastChunk -.-> LiveState
  Whole -.-> BatchState

  LastChunk --> Deliver["Save recording<br/>and paste once"]
  Whole --> Deliver
  Deliver --> Store[("Recordings folder<br/>and recordings.sqlite")]
  Deliver --> App["Focused app<br/>through the clipboard"]

  Agreement -->|"at most 5 updates a second"| Indicator["Recording indicator<br/>live card"]
  ChunkPass --> Indicator
```

Dotted arrows show which decoding state a pass runs on.
With Live Transcription off, which is the default, there is no live session: the recorder hands no blocks to a `LiveAudioBuffer`, the prepared state decodes the whole recording at release, and the indicator shows the compact card.

Live mode also stands down for one recording, which then takes the whole-recording path, when Show Timestamps is on, when the engine is Parakeet, or when the transcription service is busy as recording starts.
If any pass fails, the session frees its state and the recording is decoded whole at release.
Esc, quitting, an engine reload and a recording sent to the queue cancel the session.
