# Sortformer CoreML Model Specifications

## Model Variants

| Model | Weights | Description |
|-------|---------|-------------|
| **Sortformer** | 229M (8.5M + 220M) | Gradient Descent version, lowest latency |
| **SortformerNvidiaHigh** | 242M (8.5M + 232M) | NVIDIA high resolution, larger chunks |
| **SortformerNvidiaLow** | 234M (8.5M + 225M) | NVIDIA low latency with extra outputs |

## Inputs

| Input | Sortformer | NvidiaHigh | NvidiaLow |
|-------|------------|------------|-----------|
| **chunk** | [1, 112, 128] | [1, 3048, 128] | [1, 112, 128] |
| **chunk_lengths** | [1] Int32 | [1] Int32 | [1] Int32 |
| **spkcache** | [1, 188, 512] | [1, 188, 512] | [1, 188, 512] |
| **spkcache_lengths** | [1] Int32 | [1] Int32 | [1] Int32 |
| **fifo** | [1, 40, 512] | [1, 40, 512] | [1, 188, 512] |
| **fifo_lengths** | [1] Int32 | [1] Int32 | [1] Int32 |

## Outputs

| Output | Sortformer | NvidiaHigh | NvidiaLow |
|--------|------------|------------|-----------|
| **speaker_preds** | [1, 242, 4] | [1, 609, 4] | [1, 390, 4] |
| **chunk_pre_encoder_embs** | [1, 14, 512] | [1, 381, 512] | [1, 14, 512] |
| **chunk_pre_encoder_lengths** | [1] Int32 | [1] Int32 | [1] Int32 |
| **nest_encoder_embs** | - | - | [1, 390, 192] |
| **nest_encoder_lengths** | - | - | [1] Int32 |

## Streaming Parameters (userDefinedMetadata)

| Parameter | Sortformer | NvidiaHigh | NvidiaLow |
|-----------|------------|------------|-----------|
| **chunk_len** | 6 | 340 | 6 |
| **mel_feature_frames** | 48 | 2720 | 48 |
| **fifo_len** | 40 | 40 | 188 |
| **spkcache_len** | 188 | 188 | 188 |
| **spkcache_update_period** | 31 | 300 | 144 |
| **chunk_left_context** | 1 | 1 | 1 |
| **chunk_right_context** | 7 | 40 | 7 |
| **subsampling_factor** | 8 | 8 | 8 |
| **frame_duration** | 0.08s | 0.08s | 0.08s |

## Common Specifications

- **Speakers**: 4 max
- **Embedding dimension**: 512
- **Mel features**: 128-dim
- **Subsampling factor**: 8x
- **Frame duration**: 80ms
- **Storage precision**: Mixed (Float16, Float32)
- **Compute precision**: Mixed (Float16, Float32, Int32)
- **Model type**: Pipeline (2x mlProgram)

## Platform Availability

- macOS 13.0+
- iOS 16.0+
- tvOS 16.0+
- watchOS 9.0+
- visionOS 1.0+

## Key Differences

### Sortformer (Gradient Descent)
- Smallest chunk size (112 mel frames = ~0.9s audio)
- Lowest latency streaming
- FIFO queue: 40 frames

### SortformerNvidiaHigh
- Largest chunk size (3048 mel frames = ~24s audio)
- Higher latency but potentially better accuracy
- Best for offline/batch processing

### SortformerNvidiaLow
- Same chunk size as Gradient Descent (112 mel frames)
- Larger FIFO queue (188 vs 40 frames)
- Extra outputs: `nest_encoder_embs` [1, 390, 192] for speaker embeddings
- Good balance of latency and accuracy

## Sources

- GitHub: https://github.com/Audivize-AI/Streaming-Sortformer-Conversion
- HuggingFace: https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml
