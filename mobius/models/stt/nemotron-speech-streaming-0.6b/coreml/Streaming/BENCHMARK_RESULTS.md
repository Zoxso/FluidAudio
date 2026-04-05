# Nemotron Streaming 0.6B - WER Benchmark Results

Model: `nvidia/nemotron-speech-streaming-en-0.6b`
Dataset: LibriSpeech test-clean
Chunk size: 1.12s

## Results

### 10 Files

| Mode | WER | Errors | Words |
|------|-----|--------|-------|
| `pad_and_drop_preencoded=False` | 1.79% | 3 | 168 |
| `pad_and_drop_preencoded=True` | 3.57% | 6 | 168 |

### 100 Files

| Mode | WER | Errors | Words |
|------|-----|--------|-------|
| `pad_and_drop_preencoded=False` | 1.88% | - | - |

### NVIDIA Claimed

| Dataset | WER |
|---------|-----|
| LibriSpeech test-clean (1.12s chunks) | 2.31% |

## Notes

- `pad_and_drop_preencoded=False`: Better WER, but cannot be exported to ONNX/CoreML
- `pad_and_drop_preencoded=True`: Worse WER (~3%), but required for ONNX/CoreML export
- NVIDIA's 2.31% likely uses `pad_and_drop_preencoded=True` on full 2620 files
- Our implementation uses `conformer_stream_step` API with `CacheAwareStreamingAudioBuffer`

## Run Benchmark

```bash
cd nemotron-speech-streaming-0.6b/coreml
uv sync
uv run python benchmark_wer.py --num-files 100
```
