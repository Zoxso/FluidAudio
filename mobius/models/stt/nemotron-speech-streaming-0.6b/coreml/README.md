# Nemotron Speech Streaming 0.6B - CoreML Conversion

CoreML conversion of NVIDIA's `nvidia/nemotron-speech-streaming-en-0.6b` for real-time streaming ASR on Apple devices.

## Model Overview

| Property | Value |
|----------|-------|
| Source Model | `nvidia/nemotron-speech-streaming-en-0.6b` |
| Architecture | FastConformer RNNT (Streaming) |
| Parameters | 0.6B |
| Chunk Size | 1.12 seconds (112 mel frames) |
| Sample Rate | 16kHz |
| Mel Features | 128 bins |

## CoreML Models

4 mlpackage files for the streaming RNNT pipeline:

| Model | Size | Function |
|-------|------|----------|
| `preprocessor.mlpackage` | 1.2M | audio → 128-dim mel spectrogram |
| `encoder.mlpackage` | 2.2G | mel + cache → encoded + new_cache |
| `decoder.mlpackage` | 28M | token + LSTM state → decoder_out + new_state |
| `joint.mlpackage` | 6.6M | encoder + decoder → logits |

Plus:
- `metadata.json` - Model configuration
- `tokenizer.json` - Vocabulary (1024 tokens)

## Streaming Configuration

```json
{
  "sample_rate": 16000,
  "mel_features": 128,
  "chunk_mel_frames": 112,
  "pre_encode_cache": 9,
  "total_mel_frames": 121,
  "vocab_size": 1024,
  "blank_idx": 1024,
  "encoder_dim": 1024,
  "decoder_hidden": 640,
  "decoder_layers": 2
}
```

### Chunk Timing

| Parameter | Value |
|-----------|-------|
| window_stride | 10ms |
| chunk_mel_frames | 112 |
| **chunk duration** | 112 × 10ms = **1.120s** |
| samples per chunk | 17,920 |

### Cache Shapes

| Cache | Shape | Description |
|-------|-------|-------------|
| cache_channel | [1, 24, 70, 1024] | Attention context cache |
| cache_time | [1, 24, 1024, 8] | Convolution time cache |
| cache_len | [1] | Cache fill level |

## Benchmark Results

### WER on LibriSpeech test-clean

| Mode | Files | WER | Notes |
|------|-------|-----|-------|
| PyTorch `pad_and_drop=False` | 100 | 1.88% | Non-streaming (full context) |
| PyTorch `pad_and_drop=True` | 10 | 3.57% | True streaming |
| CoreML Non-streaming | 100 | 1.83% | Full audio preprocessed |
| CoreML Streaming | 100 | 1.79% | Audio chunked at 1.12s |
| NVIDIA Claimed | 2620 | 2.31% | Full test-clean |

### Streaming Modes Explained

```
NON-STREAMING (test_coreml_inference.py):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Full audio → preprocessor → FULL mel (one continuous spectrogram)
2. Slice mel into chunks for encoder
3. Each slice has natural continuity (no chunk boundaries)

CHEAT: The mel was computed with full audio context
WER: ~1.83%
```

```
TRUE STREAMING (test_coreml_streaming.py):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Audio chunk 1 → preprocessor → mel_1
2. Audio chunk 2 → preprocessor → mel_2 (computed separately!)
3. Prepend last 9 frames of mel_1 to mel_2 (mel_cache)

mel_cache = bridge between separately-computed mels (NOT cheating)
WER: ~1.79%
```

### What is mel_cache?

The encoder's subsampling layer needs 9 frames (~90ms) of look-back context:

```
ENCODER INPUT (needs 121 frames = 9 cache + 112 new)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│9│      112 frames      │
 ↑
 mel_cache = last 9 frames from PREVIOUS chunk's mel

Chunk 1: [000000000][mel_chunk_1]  ← pad with zeros (no previous)
Chunk 2: [mel_1_end][mel_chunk_2]  ← 9 frames from chunk 1
Chunk 3: [mel_2_end][mel_chunk_3]  ← 9 frames from chunk 2
```

This is **NOT cheating** - in real-time streaming you DO have the previous 90ms of audio.

## Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMING RNNT PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

1. PREPROCESSOR (per 1.12s audio chunk)
   audio [1, 17920] → mel [1, 128, 112]

2. ENCODER (with cache)
   mel [1, 128, 121] + cache → encoded [1, 1024, 14] + new_cache
   (121 = 9 mel_cache + 112 new frames)
   (14 output frames after 8x subsampling)

3. DECODER + JOINT (greedy loop per encoder frame)
   For each of 14 encoder frames:
     ┌──────────────────────────────────────────┐
     │  token → DECODER → decoder_out           │
     │  encoder_step + decoder_out → JOINT      │
     │  → logits → argmax → predicted token     │
     │  if token == BLANK: next encoder frame   │
     │  else: emit token, update decoder state  │
     └──────────────────────────────────────────┘
```

## Usage

### Convert to CoreML

```bash
cd conversion_scripts
uv sync
uv run python convert_nemotron_streaming.py --output-dir ../nemotron_coreml
```

Options:
- `--encoder-cu`: Encoder compute units (default: CPU_AND_NE)
- `--precision`: FLOAT32 or FLOAT16

### Run WER Benchmark (PyTorch)

```bash
cd conversion_scripts
uv run python ../benchmark_wer.py --num-files 100
```

### Test CoreML Inference

Non-streaming (full audio preprocessing):
```bash
uv run python ../test_coreml_inference.py --model-dir ../nemotron_coreml --num-files 10
```

True streaming (audio chunked at 1.12s):
```bash
uv run python ../test_coreml_streaming.py --model-dir ../nemotron_coreml --num-files 10
```

## Files

```
nemotron-speech-streaming-0.6b/coreml/
├── README.md                    # This file
├── BENCHMARK_RESULTS.md         # WER benchmark results
├── benchmark_wer.py             # PyTorch streaming WER benchmark
├── nemo_streaming_reference.py  # NeMo streaming reference implementation
├── test_coreml_inference.py     # CoreML non-streaming test
├── test_coreml_streaming.py     # CoreML true streaming test
├── conversion_scripts/
│   ├── pyproject.toml           # Python dependencies (uv)
│   ├── convert_nemotron_streaming.py  # Main conversion script
│   └── individual_components.py       # Wrapper classes for export
├── nemotron_coreml/             # Exported CoreML models
│   ├── preprocessor.mlpackage
│   ├── encoder.mlpackage
│   ├── decoder.mlpackage
│   ├── joint.mlpackage
│   ├── metadata.json
│   └── tokenizer.json
└── datasets/
    └── LibriSpeech/test-clean/  # 2620 test files
```

## Dependencies

- Python 3.10
- PyTorch 2.x
- NeMo Toolkit 2.x
- CoreMLTools 7.x
- soundfile, numpy, typer

## Notes

- The encoder is the largest model (2.2GB) with 24 Conformer layers
- Model uses 128 mel bins (not the typical 80)
- RNNT blank token index is 1024 (vocab_size)
- Decoder uses 2-layer LSTM with 640 hidden units
