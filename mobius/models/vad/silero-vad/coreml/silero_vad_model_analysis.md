# Silero VAD Model Architecture Analysis

Detailed analysis of the Silero Voice Activity Detection ONNX model structure and operations.

## Model Information

- **IR Version**: 8
- **Producer**: spox v
- **Model Version**: 0
- **Opset Imports**:
  - default: v16

- **File Size**: 2.22 MB

## Model Inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| input | 1 | [dynamic, dynamic] | Audio chunk (batch_size, sequence_length) |
| state | 1 | [2, dynamic, 128] | Hidden state from previous chunk (2, batch_size, 128) |
| sr | 7 | [] | Sampling rate (scalar) |

## Model Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| output | 1 | [dynamic, 1] | Speech probability (batch_size, 1) |
| stateN | 1 | [dynamic, dynamic, dynamic] | Updated hidden state for next chunk |

## Main Graph Structure

The main graph contains minimal operations and uses conditional execution via If nodes:

### Constant Operations

**Constant_1**: 
- Inputs: []
- Outputs: ['Constant_0_output']
- Attributes: ['value']

### Equal Operations

**Equal_1**: 
- Inputs: ['sr', 'Constant_0_output']
- Outputs: ['Equal_0_C']

### If Operations

**If_1**: 
- Inputs: ['Equal_0_C']
- Outputs: ['If_0_outputs_0', 'If_0_outputs_1']

### Identity Operations

**Identity_1**: 
- Inputs: ['If_0_outputs_0']
- Outputs: ['output']

**Identity_2**: 
- Inputs: ['If_0_outputs_1']
- Outputs: ['stateN']

## Conditional Branches Analysis

The model uses If nodes with two branches that handle different sampling rates:

### Else Branch

**Total Operations**: 113

**Operation Breakdown**:

- Add: 1
- Cast: 5
- Concat: 2
- Constant: 59
- ConstantOfShape: 1
- Conv: 6
- Equal: 2
- Gather: 3
- Identity: 2
- If: 3
- Pad: 1
- Pow: 2
- ReduceMean: 1
- Relu: 5
- Reshape: 2
- Shape: 3
- Sigmoid: 1
- Slice: 6
- Sqrt: 1
- Squeeze: 1
- Transpose: 1
- Unsqueeze: 5

**Compute-Intensive Operations**:
- Conv: 6 instances

### Then Branch

**Total Operations**: 113

**Operation Breakdown**:

- Add: 1
- Cast: 5
- Concat: 2
- Constant: 59
- ConstantOfShape: 1
- Conv: 6
- Equal: 2
- Gather: 3
- Identity: 2
- If: 3
- Pad: 1
- Pow: 2
- ReduceMean: 1
- Relu: 5
- Reshape: 2
- Shape: 3
- Sigmoid: 1
- Slice: 6
- Sqrt: 1
- Squeeze: 1
- Transpose: 1
- Unsqueeze: 5

**Compute-Intensive Operations**:
- Conv: 6 instances

## Performance Characteristics

### Why ANE Doesn't Provide Speedup

1. **Lightweight Architecture**: Only ~113 operations per branch
2. **Small Input Size**: 512 samples per chunk (very small tensor)
3. **Optimized for Latency**: Designed for real-time streaming, not throughput
4. **CPU Efficiency**: Modern Apple Silicon CPUs handle this workload extremely well

### Benchmark Results

| Provider | Avg Time | Throughput | Notes |
|----------|----------|------------|-------|
| CPU | ~0.079ms | ~12,700 fps | Baseline performance |
| CoreML-ANE | ~0.080ms | ~12,500 fps | No meaningful speedup |
| CoreML-GPU | ~0.078ms | ~12,900 fps | Slight improvement |

## Recommendations

### For Production Use

- **Recommended**: Use CPU provider for maximum compatibility
- **Alternative**: CoreML with CPUOnly for consistent CoreML pipeline
- **Not Recommended**: ANE configuration (no performance benefit)

### When ANE Would Be Beneficial

ANE typically provides speedups for:
- Large models (>50MB, millions of parameters)
- Complex computer vision models
- Batch processing of many samples
- Models with many matrix multiplications

## Technical Implementation Details

### Model Architecture

The Silero VAD model uses a conditional architecture:

1. **Input Validation**: Checks sampling rate via Equal operation
2. **Conditional Processing**: If node selects appropriate branch
3. **Branch Processing**: Each branch contains identical operations for different sample rates
4. **Output Generation**: Produces speech probability and updated state

### Key Operations per Branch

- **Convolution Layers**: 6 (main computation)
- **Activation Functions**: 5 ReLU activations
- **Signal Processing**: Reshaping, padding, normalization
- **State Management**: LSTM-like state updates


## Detailed Layer Analysis

### Convolution Architecture

The model contains **6 convolution layers per branch** that form the core of the neural network:


#### Conv Layer 1: 
- **Inputs**: ['If_0_else_branch__Inline_0__/stft/Unsqueeze_output_0', 'If_0_else_branch__Inline_0__stft.forward_basis_buffer']
- **Outputs**: ['If_0_else_branch__Inline_0__/stft/Conv_output_0']
- **Type**: 1D Convolution (audio processing)

#### Conv Layer 2: 
- **Inputs**: ['If_0_else_branch__Inline_0__/stft/Sqrt_output_0', 'If_0_else_branch__Inline_0__encoder.0.reparam_conv.weight', 'If_0_else_branch__Inline_0__encoder.0.reparam_conv.bias']
- **Outputs**: ['If_0_else_branch__Inline_0__/encoder/0/reparam_conv/Conv_output_0']
- **Type**: 1D Convolution (audio processing)

#### Conv Layer 3: 
- **Inputs**: ['If_0_else_branch__Inline_0__/encoder/0/activation/Relu_output_0', 'If_0_else_branch__Inline_0__encoder.1.reparam_conv.weight', 'If_0_else_branch__Inline_0__encoder.1.reparam_conv.bias']
- **Outputs**: ['If_0_else_branch__Inline_0__/encoder/1/reparam_conv/Conv_output_0']
- **Type**: 1D Convolution (audio processing)

#### Conv Layer 4: 
- **Inputs**: ['If_0_else_branch__Inline_0__/encoder/1/activation/Relu_output_0', 'If_0_else_branch__Inline_0__encoder.2.reparam_conv.weight', 'If_0_else_branch__Inline_0__encoder.2.reparam_conv.bias']
- **Outputs**: ['If_0_else_branch__Inline_0__/encoder/2/reparam_conv/Conv_output_0']
- **Type**: 1D Convolution (audio processing)

#### Conv Layer 5: 
- **Inputs**: ['If_0_else_branch__Inline_0__/encoder/2/activation/Relu_output_0', 'If_0_else_branch__Inline_0__encoder.3.reparam_conv.weight', 'If_0_else_branch__Inline_0__encoder.3.reparam_conv.bias']
- **Outputs**: ['If_0_else_branch__Inline_0__/encoder/3/reparam_conv/Conv_output_0']
- **Type**: 1D Convolution (audio processing)

#### Conv Layer 6: 
- **Inputs**: ['If_0_else_branch__Inline_0__/decoder/decoder/1/Relu_output_0', 'If_0_else_branch__Inline_0__decoder.decoder.2.weight', 'If_0_else_branch__Inline_0__decoder.decoder.2.bias']
- **Outputs**: ['If_0_else_branch__Inline_0__/decoder/decoder/2/Conv_output_0']
- **Type**: 1D Convolution (audio processing)


### Model Execution Flow

```
Input Audio (512 samples)
    ↓
Sampling Rate Check (Equal operation)
    ↓
Conditional Branch Selection (If node)
    ↓
┌─────────────────────┬─────────────────────┐
│    Else Branch      │    Then Branch      │
│   (e.g., 16kHz)     │   (e.g., 8kHz)      │
├─────────────────────┼─────────────────────┤
│ 6 Conv Layers       │ 6 Conv Layers       │
│ 5 ReLU Activations  │ 5 ReLU Activations  │
│ Normalization       │ Normalization       │
│ State Updates       │ State Updates       │
└─────────────────────┴─────────────────────┘
    ↓
Speech Probability + Updated State
```

### Operation Complexity Analysis

| Operation Type | Count per Branch | Computational Cost | ANE Suitability |
|----------------|------------------|-------------------|-----------------|
| **Conv** | 6 | Medium | ✓ Good |
| **ReLU** | 5 | Low | ✓ Good |
| **Reshape** | 2 | Very Low | ✗ Poor |
| **Slice** | 6 | Very Low | ✗ Poor |
| **Constant** | 59 | None | N/A |
| **Cast** | 5 | Very Low | ✗ Poor |

**Analysis**: While the Conv operations are ANE-suitable, they're outnumbered by lightweight operations that don't benefit from ANE acceleration.

### Memory and Compute Footprint

- **Input Tensor**: 512 float32 values (2KB)
- **State Tensor**: 2×1×128 float32 values (1KB) 
- **Model Weights**: ~2.2MB embedded in constants
- **Output**: Single float32 probability + updated state
- **Peak Memory**: < 5MB total
- **Compute**: ~113 operations per inference

**Conclusion**: The model's extremely small memory footprint and low compute requirements make it ideal for CPU execution but unsuitable for ANE acceleration benefits.

## Implementation Notes

### CoreML Execution Provider Configuration

```python
# Optimal configuration (no ANE benefit, but technically correct)
model = load_silero_vad(
    onnx=True,
    use_coreml=True,
    coreml_compute_units='CPUOnly'  # Best performance
)

# ANE configuration (works but no speedup)
model = load_silero_vad(
    onnx=True,
    use_coreml=True,
    coreml_compute_units='CPUAndNeuralEngine'  # No benefit
)
```

### Real-World Performance

- **Streaming Audio**: Processes 512-sample chunks in ~0.08ms
- **Real-Time Factor**: 0.003x (300× faster than real-time)
- **Latency**: Suitable for live audio processing
- **Throughput**: >12,000 chunks/second
- **Memory**: Minimal RAM usage, cache-friendly

This analysis confirms that the Silero VAD model is expertly optimized for its intended use case: real-time voice activity detection with minimal computational overhead.

## Recent Debugging Summary (2025-09-15)

- **Problem**: After switching comparisons to the TorchScript (JIT) reference, unified CoreML exports drifted badly (correlation ≈0.02). Root cause was that our hand-built PyTorch modules diverged from the TorchScript implementation—STFT used symmetric padding and a log stage, encoder strides defaulted to 1, and the decoder averaged its temporal axis.
- **Fixes**: Tweaked `coreml/convert_model_components.py` so STFT uses right-side `ReflectionPad1d` with hop 128 and returns raw magnitudes, encoder layers read stride/padding metadata, and decoder mirrors the dropout→ReLU→1×1 Conv→Sigmoid sequence. Propagated these parameters through `coreml/convert-coreml.py` when building unified exports. Regenerated the CoreML packages and reran `compare-models.py`, restoring parity with correlation ≈0.99998 on `yc_first_minute.wav`.

### 256 ms Unified Variant Update (2025-09-15)

- **Problem**: The 256 ms CoreML package was still built from the older hand-written PyTorch modules, so its STFT windowing and encoder strides disagreed with the JIT baseline, leading to the large probability drift we observed when switching comparisons.
- **Change**: Re-exported the CoreML bundle with `uv run python convert-coreml.py --output-dir ./silero-vad-coreml --include-256ms`, letting the refreshed conversion pipeline (right-pad STFT, hop=128, decoder layout) drive the 8×512 noisy-OR aggregator used for long chunks.
- **Validation**: `uv run python compare-models.py --audio-file ../../../../../FluidAudio/yc_first_minute.wav --output-dir ./plots --coreml-dir ./silero-vad-coreml --include-256ms` now reports PyTorch ↔ CoreML agreement for the 256 ms variant (MAE ≈ 2.1×10⁻⁴, MSE ≈ 2.0×10⁻⁶, r ≈ 0.999988, max |Δ| ≈ 0.018). The 256 ms CoreML model also processes chunks ≈3.6× faster than the TorchScript reference (mean RTF ≈ 1485).

### Plot Label Clarification (2025-09-15)

- **Problem**: `coreml/compare-models.py` plotted "PyTorch" and "Unified" as generic labels, which made it hard to confirm which exported package a chart referenced once multiple CoreML bundles existed side by side.
- **Change**: Threaded explicit `model_name` metadata through the TorchScript and CoreML wrappers and taught the plotting helpers to reuse those identifiers (e.g., `silero-vad-jit`, `silero-vad-unified-v6.0.0`, `silero-vad-unified-256ms-v6.0.0`) in legends, axis labels, and performance summaries for both the base and 256 ms variants.
