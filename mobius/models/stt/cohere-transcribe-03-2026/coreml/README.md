# Cohere Transcribe CoreML Export

Export Cohere Transcribe (`CohereLabs/cohere-transcribe-03-2026`) to CoreML for on-device inference on Apple platforms.

## Quick Start

```bash
# Install dependencies
uv sync

# Export models
uv run python export-encoder.py --output-dir build --precision float16
uv run python export-decoder-cached.py --output-dir build --precision float16

# Test pipeline
uv run python test-full-pipeline.py
```

## Status

✅ **Complete and Working** - Full pipeline processes real audio end-to-end

### Encoder
- **Status:** ✅ Perfect match with reference (max diff: 0.041)
- **Size:** 3.6 GB (FP16)
- **Input:** `(1, 128, 3001)` mel spectrogram + length
- **Output:** `(1, 376, 1024)` hidden states

### Decoder
- **Status:** ✅ Working (generates tokens, reaches EOS)
- **Size:** 289 MB (FP16)
- **Architecture:** 8 layers, 8 heads, KV cache
- **Cache:** `(8, 8, 108, 128)` per K/V

### Preprocessing
- **Status:** ✅ Working Python implementation
- **Parameters:** n_fft=1024, hop_length=160, n_mels=128

## Files

### Export Scripts
- `export-encoder.py` - Encoder + projection layer export
- `export-decoder-cached.py` - Decoder with KV cache export
- `cohere_mel_spectrogram.py` - Python mel spectrogram preprocessing

### Test Scripts
- `test-full-pipeline.py` - Full pipeline test
- `compare-models.py` - Comparison with reference models
- `test-hybrid-our-encoder-ref-decoder.py` - Hybrid test (encoder validation)
- `test-hybrid-ref-encoder-our-decoder.py` - Hybrid test (decoder validation)
- `test-with-librispeech.py` - Ground truth WER testing

### Utilities
- `export-cross-kv-projector.py` - Cross-attention KV projector
- `benchmark-models.py` - Performance benchmarking
- `quantize-models.py` - Model quantization
- `measure-memory.py` - Memory profiling
- `download-librispeech-samples.py` - Dataset download utility
- `create-test-audio.py` - Test audio generation

### Documentation
- `STATUS.md` - Current status and test results
- `REVERSE_ENGINEERING.md` - Technical reverse engineering details
- `SUMMARY.md` - Executive summary

## Requirements

- Python 3.10+
- PyTorch
- CoreMLTools
- Transformers
- NumPy
- SentencePiece
- SoundFile

Managed via `uv` (see `pyproject.toml`)

## Model Specs

### Encoder (Conformer)
- Hidden size: 1280 (encoder) → 1024 (after projection)
- Input: 128 mel bins, up to 3001 frames (~30s audio)
- Output: 376 frames × 1024 dimensions

### Decoder (Transformer)
- Layers: 8
- Attention heads: 8
- Head dimension: 128
- Hidden size: 1024
- Max sequence length: 108 tokens
- Vocabulary: 51,865 tokens

## Architecture

The model is split into three components:

1. **Mel Preprocessing** (Python) - Converts raw audio to mel spectrogram
2. **Encoder** (CoreML) - Conformer blocks + projection layer
3. **Decoder** (CoreML) - Autoregressive transformer with KV caching

This separation allows for efficient on-device inference with proper cache management.

## Notes

- Models auto-download from HuggingFace on first use
- Requires ~4GB disk space for encoder + decoder
- FP16 precision recommended for size/speed balance
- Targets iOS 17+ / macOS 14+
