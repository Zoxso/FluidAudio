# Nemotron Speech Streaming EN 0.6B - CoreML Conversion

Convert NVIDIA's [nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) to CoreML for on-device streaming ASR on Apple Silicon.

## Model Overview

| Feature | Value |
|---------|-------|
| Architecture | FastConformer-CacheAware-RNNT |
| Parameters | 600M |
| Sample Rate | 16kHz |
| Mel Bins | **128** |
| Encoder Layers | 24 |
| Encoder Dim | 1024 |
| Subsampling | 8x |
| Chunk Duration | 1.12s |

## Benchmark Results (LibriSpeech test-clean)

| Mode | WER | Notes |
|------|-----|-------|
| CoreML Streaming | **1.79%** | Audio chunked at 1.12s |
| PyTorch Streaming | 1.88% | `pad_and_drop=False` |
| NVIDIA Claimed | 2.31% | Full test-clean (2620 files) |

## Quick Start

```bash
cd coreml/conversion_scripts
uv sync
uv run python convert_nemotron_streaming.py --output-dir ../nemotron_coreml
```

Exports 4 CoreML models:
- `preprocessor.mlpackage` (1.2M) - audio → 128-dim mel
- `encoder.mlpackage` (2.2G) - mel + cache → encoded + new_cache
- `decoder.mlpackage` (28M) - token + LSTM state → decoder_out
- `joint.mlpackage` (6.6M) - encoder + decoder → logits

## Streaming Configuration

```
Chunk: 1.12s audio → 112 mel frames → 14 encoder frames
       (17,920 samples @ 16kHz)

Encoder input: 121 mel frames = 9 cache + 112 new
               (9 frames ≈ 90ms look-back context)
```

See [coreml/README.md](coreml/README.md) for detailed documentation.

## References

- [Model on HuggingFace](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [Scaling Voice Agents with Cache-Aware ASR](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
