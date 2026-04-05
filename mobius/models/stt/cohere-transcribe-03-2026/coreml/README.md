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

# Test with ground truth
uv run python test-with-librispeech.py

# Compare with reference
uv run python compare-models.py
```

## Models

### Encoder (✅ Working Perfectly)
- **Size:** 3.6 GB (FP16)
- **Input:** `(1, 128, 3001)` mel spectrogram + length
- **Output:** `(1, 376, 1024)` hidden states
- **Status:** Perfect match with reference (max diff: 0.041)

### Decoder (⚠️ Has Known Issue)
- **Size:** 289 MB (FP16)
- **Architecture:** 8 layers, 8 heads, KV cache
- **Status:** Functional but diverges after token 3

## Files

**Export Scripts:**
- `export-encoder.py` - Encoder + projection layer export
- `export-decoder-cached.py` - Decoder with KV cache export
- `cohere_mel_spectrogram.py` - Python mel spectrogram preprocessing

**Test Scripts:**
- `test-hybrid-our-encoder-ref-decoder.py` - Proves encoder is correct
- `test-hybrid-ref-encoder-our-decoder.py` - Proves decoder has issue
- `test-with-librispeech.py` - Ground truth WER testing
- `test-full-pipeline.py` - Full pipeline test
- `compare-models.py` - Comparison with reference

**Documentation:**
- `STATUS.md` - Current status summary
- `REVERSE_ENGINEERING.md` - Technical details
- `HYBRID_TEST_RESULTS.md` - Hybrid test analysis
- `SUMMARY.md` - Executive summary

## Known Issue

The decoder gets stuck repeating token 16 (`<|emo:undefined|>`) after the first 3 tokens, likely due to KV cache handling. See `HYBRID_TEST_RESULTS.md` for detailed analysis.

**Hybrid Test Results:**

| Configuration | WER | Tokens | Status |
|--------------|-----|--------|--------|
| Our Encoder + Reference Decoder | 0.00% | 22 | ✅ Perfect |
| Reference Encoder + Our Decoder | N/A | 200 | ❌ Failed |

**Conclusion:** Encoder export is 100% correct. Decoder needs cache handling fixes.

## Requirements

- Python 3.10+
- PyTorch
- CoreMLTools
- Transformers
- NumPy
- SentencePiece

Managed via `uv` (see `pyproject.toml`)
