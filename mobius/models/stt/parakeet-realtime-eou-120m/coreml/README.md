# Parakeet Realtime EOU 120M CoreML Conversion

This directory contains scripts for converting the [NVIDIA Parakeet Realtime EOU 120M](https://huggingface.co/alexwengg/parakeet-realtime-eou-120m-coreml/tree/main) model to CoreML format for use on Apple platforms.

## Model Overview

The Parakeet Realtime EOU model is a **cache-aware streaming FastConformer-RNNT** designed for low-latency end-of-utterance detection. Key characteristics:

- **120M parameters** (smaller than the 0.6B TDT models)
- **Streaming support** via cache-aware attention with attention context `[70, 1]`
- **EOU detection** - outputs a special `<EOU>` token at utterance boundaries
- **Low latency** - 50th percentile EOU detection at 160ms, 90th at 280ms
- **English only** - no punctuation or capitalization

## Model Components

The model is split into the following CoreML packages:

| Component | Description | Cache Support |
|-----------|-------------|---------------|
| `parakeet_eou_preprocessor.mlpackage` | Mel spectrogram extraction | No |
| `parakeet_eou_encoder_initial.mlpackage` | Encoder for first chunk (no cache) | No |
| `parakeet_eou_encoder_streaming.mlpackage` | Encoder for subsequent chunks | Yes (I/O) |
| `parakeet_eou_decoder.mlpackage` | RNNT prediction network | State (h, c) |
| `parakeet_eou_joint.mlpackage` | RNNT joint network | No |
| `parakeet_eou_joint_decision.mlpackage` | Joint + argmax | No |
| `parakeet_eou_joint_decision_single_step.mlpackage` | Single-step for streaming | No |

## Cache States

The streaming encoder requires two cache tensors:

1. **cache_last_channel**: `[batch, num_layers, cache_size, d_model]`
   - Stores attention context across layers

2. **cache_last_time**: `[batch, num_layers, d_model, conv_context_size]`
   - Stores convolution temporal context

3. **cache_last_channel_len**: `[batch]` (int64)
   - Tracks how much of the cache is filled

## Setup

```bash
# Create virtual environment with uv
uv venv --python 3.10.12

# Install dependencies
uv pip install -e .
```

Or use the existing parakeet-tdt-v3 environment which has compatible dependencies.

## Usage

```bash
# Auto-download model from HuggingFace and convert
python convert-parakeet-eou.py

# Or use a local .nemo checkpoint
python convert-parakeet-eou.py --nemo-path /path/to/model.nemo

# Custom output directory
python convert-parakeet-eou.py --output-dir ./my_models

# Adjust chunk size for streaming (default 160ms)
python convert-parakeet-eou.py --chunk-seconds 0.32
```

## Streaming Inference Pattern

```python
# 1. Initialize caches (zeros)
cache_channel = np.zeros([1, num_layers, cache_size, d_model], dtype=np.float32)
cache_time = np.zeros([1, num_layers, d_model, conv_ctx], dtype=np.float32)
cache_len = np.array([0], dtype=np.int64)

# 2. Process first chunk (use initial encoder, no cache)
mel = preprocessor.predict({"audio_signal": chunk1, "audio_length": len1})
encoded = encoder_initial.predict({"mel": mel["mel"], "mel_length": mel["mel_length"]})

# 3. Process subsequent chunks (use streaming encoder with cache)
for chunk in chunks[1:]:
    mel = preprocessor.predict({"audio_signal": chunk, "audio_length": chunk_len})
    result = encoder_streaming.predict({
        "mel": mel["mel"],
        "mel_length": mel["mel_length"],
        "cache_last_channel": cache_channel,
        "cache_last_time": cache_time,
        "cache_last_channel_len": cache_len,
    })
    encoded = result["encoder"]
    cache_channel = result["cache_last_channel_next"]
    cache_time = result["cache_last_time_next"]
    cache_len = result["cache_last_channel_len_next"]

    # 4. Run RNNT decoding loop
    # ... (similar to TDT greedy decoding)
```

## Differences from Parakeet TDT

| Feature | TDT | EOU |
|---------|-----|-----|
| Parameters | 0.6B | 120M |
| Duration outputs | Yes (bins 0-4) | No |
| EOU token | No | Yes |
| Streaming cache | No | Yes |
| Languages | 25 European | English only |
| Use case | Transcription | Voice agents |

## References

- [Model Card](https://huggingface.co/alexwengg/parakeet-realtime-eou-120m-coreml/tree/main)
- [FastConformer Paper](https://arxiv.org/abs/2305.05084)
- [Cache-Aware Streaming](https://arxiv.org/abs/2312.17279)
