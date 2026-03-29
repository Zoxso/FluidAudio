# Choosing an ASR API

FluidAudio provides three ASR approaches, each suited to different use cases. All are based on NVIDIA's Parakeet model family and run locally on Apple Silicon via CoreML.

## At a Glance

| | `AsrManager` | `SlidingWindowAsrManager` | `StreamingAsrManager` |
|---|---|---|---|
| **Use case** | Transcribe a complete file | Live mic input with high accuracy | Live mic input with low latency |
| **Latency** | N/A (batch) | ~15s chunks, seconds of delay | 80ms–1280ms per chunk |
| **Accuracy** | Best (full-context encoder) | Same as batch (same encoder) | Lower (streaming encoder trades context for speed) |
| **Partial results** | No | Yes (volatile + confirmed) | Yes (via callback) |
| **Model size** | 0.6B params | 0.6B params (same model) | 120M (EOU) or 0.6B (Nemotron) |
| **Encoder** | Offline (sees entire input) | Offline (sees each window) | Cache-aware (incremental state) |
| **Custom vocabulary** | Yes | Yes | No |

## `AsrManager` — Batch Transcription

Use this when you have a complete audio file or buffer and want the most accurate transcription.

```swift
let models = try await AsrModels.downloadAndLoad(version: .v3)
let asrManager = AsrManager(config: .default)
try await asrManager.loadModels(models)

// From a file
let result = try await asrManager.transcribe(URL(fileURLWithPath: "audio.wav"))

// From samples (16 kHz Float array)
let result = try await asrManager.transcribe(samples)

// From an AVAudioPCMBuffer
let result = try await asrManager.transcribe(audioBuffer)
```

**When to use:**
- Transcribing recorded audio files
- Batch processing (benchmarks, datasets)
- Any scenario where audio is available upfront
- When accuracy matters more than responsiveness

**CLI:**
```bash
swift run fluidaudiocli transcribe audio.wav
```

## `SlidingWindowAsrManager` — Sliding Window (Pseudo-Streaming)

Use this when you need live transcription from a microphone with the same accuracy as batch mode. Audio is processed in large overlapping windows using the same offline encoder as `AsrManager`.

```swift
let slidingWindow = SlidingWindowAsrManager(config: .default)
try await slidingWindow.loadModels(version: .v3)

// Feed audio as it arrives
let updates = try await slidingWindow.start()
slidingWindow.appendAudio(buffer)

// Consume transcription updates
for await update in updates {
    switch update {
    case .volatile(let text):
        // Quick hypothesis — may change
        print("Volatile: \(text)")
    case .confirmed(let text):
        // Stable text — won't change
        print("Confirmed: \(text)")
    case .finished(let text):
        print("Final: \(text)")
    }
}
```

**When to use:**
- Dictation apps where accuracy is the priority
- When a few seconds of latency is acceptable
- When you need vocabulary boosting (custom word lists)
- When you want the two-tier volatile/confirmed transcript pattern (like Apple's Speech API)

**How it works:** Audio accumulates in a buffer. Every ~15 seconds, a window is assembled with ~10s of left context and ~2s of right context, then transcribed with the offline TDT encoder. Overlapping windows let the decoder see surrounding context, producing accurate results at the cost of higher latency.

**CLI:**
```bash
swift run fluidaudiocli transcribe audio.wav --low-latency
```

## `StreamingAsrManager` — True Streaming

Use this when you need the lowest possible latency. These engines use cache-aware encoders that maintain state across chunks, producing results every 80ms–1280ms.

```swift
// Create a streaming manager from the variant
let engine = StreamingModelVariant.parakeetEou160ms.createManager()
try await engine.loadModels()

// Set up partial transcript callback
await engine.setPartialTranscriptCallback { text in
    print("Partial: \(text)")
}

// Feed audio as it arrives
try engine.appendAudio(buffer)
try await engine.processBufferedAudio()

// When done
let finalText = try await engine.finish()
```

**When to use:**
- Real-time captioning or subtitles
- Voice commands where response time matters
- Any scenario where sub-second latency is required

### Available Variants

**Parakeet EOU (120M params)** — Lightweight streaming with end-of-utterance detection:

| Variant | Latency | Notes |
|---|---|---|
| `.parakeetEou160ms` | 160ms | Lowest latency, ~8-9% WER |
| `.parakeetEou320ms` | 320ms | Balanced, ~5.7% WER |
| `.parakeetEou1280ms` | 1280ms | Best streaming accuracy |

**Nemotron (0.6B params)** — Full-size streaming encoder:

| Variant | Latency | Notes |
|---|---|---|
| `.nemotron560ms` | 560ms | Balanced |
| `.nemotron1120ms` | 1120ms | Best accuracy |

### Choosing a Variant

- For voice commands or live captions: `.parakeetEou160ms` or `.parakeetEou320ms`
- For longer-form streaming where accuracy matters: `.nemotron1120ms`
- For balanced latency and quality: `.parakeetEou320ms` or `.nemotron560ms`

## Decision Flowchart

1. **Is the audio already recorded?** → Use `AsrManager`
2. **Is sub-second latency required?** → Use `StreamingAsrManager`
3. **Is accuracy more important than latency?** → Use `SlidingWindowAsrManager`
4. **Do you need custom vocabulary boosting?** → Use `SlidingWindowAsrManager`
5. **Otherwise** → Start with `SlidingWindowAsrManager` (most forgiving) and switch to streaming if latency is a problem
