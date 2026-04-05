#!/usr/bin/env python3
"""
Quantize Silero VAD Unified CoreML Models
This script applies various quantization techniques to both standard and 256ms unified models and analyzes the results.
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import coremltools as ct
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle


TECHNIQUE_DISPLAY_NAMES = {
    "original": "Original",
    "int8": "INT8",
    "int4": "INT4",
    "mixed": "Mixed Precision",
    "palette_2bit": "Palette 2-bit",
    "palette_4bit": "Palette 4-bit",
    "palette_6bit": "Palette 6-bit",
    "palette_8bit": "Palette 8-bit",
    "pruned_int8": "Pruned INT8",
}

# Maintain a consistent order when plotting technique comparisons so the
# unified and 256ms variants align even if some experiments fail.
TECHNIQUE_ORDER = [
    "original",
    "int8",
    "pruned_int8",
    "mixed",
    "int4",
    "palette_8bit",
    "palette_6bit",
    "palette_4bit",
    "palette_2bit",
]


def load_unified_model(model_path: str) -> ct.models.MLModel:
    """Load the unified CoreML model"""
    print(f"Loading unified model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = ct.models.MLModel(model_path)
    print(f"Model loaded successfully")
    return model


def get_model_size(model_path: str) -> int:
    """Get model size in bytes"""
    if os.path.isdir(model_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    else:
        return os.path.getsize(model_path)


def generate_test_inputs(is_256ms: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test inputs for the unified model"""
    if is_256ms:
        # Audio input: 4160 samples (260ms at 16kHz - 64 context + 4096 current)
        audio_input = np.random.randn(1, 4160).astype(np.float32)
    else:
        # Audio input: 576 samples (36ms at 16kHz)
        audio_input = np.random.randn(1, 576).astype(np.float32)

    # Hidden and cell states for LSTM
    hidden_state = np.zeros((1, 128), dtype=np.float32)
    cell_state = np.zeros((1, 128), dtype=np.float32)

    return audio_input, hidden_state, cell_state


def measure_inference_time_1s(model: ct.models.MLModel, is_256ms: bool = False, num_runs: int = 20) -> Tuple[float, float]:
    """Measure inference time for processing 1 second of audio"""
    # Calculate chunks needed for 1 second (16kHz)
    if is_256ms:
        # 256ms model: ~4 chunks to cover 1 second
        chunks_per_second = 4
        audio_duration_ms = 1000.0
    else:
        # Standard model: ~31 chunks to cover 1 second (32ms each with some overlap)
        chunks_per_second = 31
        audio_duration_ms = 1000.0

    # Generate inputs
    audio_input, hidden_state, cell_state = generate_test_inputs(is_256ms)

    # Warm up with a few single chunks
    for _ in range(3):
        try:
            _ = model.predict({
                'audio_input': audio_input,
                'hidden_state': hidden_state,
                'cell_state': cell_state
            })
        except Exception as e:
            print(f"Warning: Inference failed during warmup: {e}")
            return float('inf'), float('inf')

    # Measure processing 1 second of audio
    total_latencies = []
    single_chunk_times = []

    for _ in range(num_runs):
        # Reset states for each run
        curr_hidden = hidden_state.copy()
        curr_cell = cell_state.copy()

        start_time = time.time()

        # Process chunks needed for 1 second
        chunk_times = []
        for chunk_idx in range(chunks_per_second):
            chunk_start = time.time()
            try:
                result = model.predict({
                    'audio_input': audio_input,
                    'hidden_state': curr_hidden,
                    'cell_state': curr_cell
                })
                # Update states for next chunk
                curr_hidden = result.get('hidden_state_output', curr_hidden)
                curr_cell = result.get('cell_state_output', curr_cell)

                chunk_end = time.time()
                chunk_times.append(chunk_end - chunk_start)
            except Exception as e:
                print(f"Warning: Chunk {chunk_idx} failed: {e}")
                chunk_times.append(float('inf'))
                break

        end_time = time.time()
        total_latency = end_time - start_time
        avg_chunk_time = np.mean(chunk_times) if chunk_times else float('inf')

        total_latencies.append(total_latency)
        single_chunk_times.append(avg_chunk_time)

    if not total_latencies or all(t == float('inf') for t in total_latencies):
        return float('inf'), float('inf')

    # Return average total latency for 1s and average single chunk time
    avg_total_latency = np.mean([t for t in total_latencies if t != float('inf')])
    avg_chunk_time = np.mean([t for t in single_chunk_times if t != float('inf')])

    return avg_total_latency, avg_chunk_time


def test_model_accuracy(original_model: ct.models.MLModel,
                       quantized_model: ct.models.MLModel,
                       is_256ms: bool = False,
                       num_tests: int = 20) -> float:
    """Test accuracy preservation between original and quantized models"""
    mse_errors = []

    for _ in range(num_tests):
        audio_input, hidden_state, cell_state = generate_test_inputs(is_256ms)

        try:
            # Get original output
            orig_output = original_model.predict({
                'audio_input': audio_input,
                'hidden_state': hidden_state,
                'cell_state': cell_state
            })

            # Get quantized output
            quant_output = quantized_model.predict({
                'audio_input': audio_input,
                'hidden_state': hidden_state,
                'cell_state': cell_state
            })

            # Compare VAD outputs
            orig_vad = orig_output['vad_output']
            quant_vad = quant_output['vad_output']

            mse = np.mean((orig_vad - quant_vad) ** 2)
            mse_errors.append(mse)

        except Exception as e:
            print(f"Warning: Accuracy test failed: {e}")
            mse_errors.append(float('inf'))

    return np.mean(mse_errors)


def quantize_model_int8(model: ct.models.MLModel, output_path: str) -> ct.models.MLModel:
    """Apply INT8 quantization"""
    print("Applying INT8 quantization...")
    try:
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear", dtype="int8")
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        quantized_model = ct.optimize.coreml.linear_quantize_weights(model, config)
        quantized_model.save(output_path)
        return ct.models.MLModel(output_path)
    except Exception as e:
        print(f"INT8 quantization failed: {e}")
        return None


def quantize_model_int4(model: ct.models.MLModel, output_path: str) -> ct.models.MLModel:
    """Apply INT4 quantization"""
    print("Applying INT4 quantization...")
    try:
        # First convert model to iOS18 target for int4 support
        temp_path = output_path.replace('.mlpackage', '_temp_ios18.mlpackage')
        model_spec = model.get_spec()
        model_ios18 = ct.models.MLModel(model_spec, compute_units=ct.ComputeUnit.CPU_ONLY)

        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear", dtype="int4")
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        quantized_model = ct.optimize.coreml.linear_quantize_weights(model_ios18, config)
        quantized_model.save(output_path)
        return ct.models.MLModel(output_path)
    except Exception as e:
        print(f"INT4 quantization failed: {e}")
        return None


def quantize_model_mixed_precision(model: ct.models.MLModel, output_path: str) -> ct.models.MLModel:
    """Apply mixed precision quantization"""
    print("Applying mixed precision quantization...")
    try:
        # Use different precision for different layers
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        quantized_model = ct.optimize.coreml.linear_quantize_weights(model, config)
        quantized_model.save(output_path)
        return ct.models.MLModel(output_path)
    except Exception as e:
        print(f"Mixed precision quantization failed: {e}")
        return None


def quantize_model_palettization(model: ct.models.MLModel, output_path: str, nbits: int) -> ct.models.MLModel:
    """Apply palettization (weight clustering)"""
    print(f"Applying {nbits}-bit palettization...")
    try:
        op_config = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=nbits)
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        quantized_model = ct.optimize.coreml.palettize_weights(model, config)
        quantized_model.save(output_path)
        return ct.models.MLModel(output_path)
    except Exception as e:
        print(f"{nbits}-bit palettization failed: {e}")
        return None


def quantize_model_pruned_int8(model: ct.models.MLModel, output_path: str) -> ct.models.MLModel:
    """Apply pruning followed by INT8 quantization"""
    print("Applying pruning + INT8 quantization...")
    try:
        # First prune
        prune_config = ct.optimize.coreml.OpThresholdPrunerConfig(threshold=0.01)
        config = ct.optimize.coreml.OptimizationConfig(global_config=prune_config)
        pruned_model = ct.optimize.coreml.prune_weights(model, config)

        # Then quantize
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear", dtype="int8")
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        quantized_model = ct.optimize.coreml.linear_quantize_weights(pruned_model, config)
        quantized_model.save(output_path)
        return ct.models.MLModel(output_path)
    except Exception as e:
        print(f"Pruned INT8 quantization failed: {e}")
        return None


def run_quantization_experiments(unified_model_path: str, output_dir: str, is_256ms: bool = False) -> Dict[str, Any]:
    """Run all quantization experiments and collect results"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load original model
    original_model = load_unified_model(unified_model_path)
    original_size = get_model_size(unified_model_path)
    original_total_latency, original_chunk_time = measure_inference_time_1s(original_model, is_256ms)

    print(f"Original model size: {original_size / 1024:.1f} KB")
    print(f"Original 1s latency: {original_total_latency * 1000:.2f} ms")
    print(f"Original chunk time: {original_chunk_time * 1000:.2f} ms")

    # Calculate RTF (Real-Time Factor): audio_duration / processing_time
    # RTF > 1 means faster than real-time
    original_rtf = 1000.0 / (original_total_latency * 1000) if original_total_latency != float('inf') else 0.0

    results = {
        'original': {
            'model': original_model,
            'size_bytes': original_size,
            'total_latency_ms': original_total_latency * 1000,
            'chunk_time_ms': original_chunk_time * 1000,
            'mse_error': 0.0,  # Reference
            'compression_ratio': 1.0,
            'rtf': original_rtf
        }
    }

    # Define quantization experiments
    experiments = [
        ('int8', lambda m, p: quantize_model_int8(m, p)),
        ('int4', lambda m, p: quantize_model_int4(m, p)),
        ('mixed', lambda m, p: quantize_model_mixed_precision(m, p)),
        ('palette_2bit', lambda m, p: quantize_model_palettization(m, p, 2)),
        ('palette_4bit', lambda m, p: quantize_model_palettization(m, p, 4)),
        ('palette_6bit', lambda m, p: quantize_model_palettization(m, p, 6)),
        ('palette_8bit', lambda m, p: quantize_model_palettization(m, p, 8)),
        ('pruned_int8', lambda m, p: quantize_model_pruned_int8(m, p))
    ]

    for exp_name, quantize_func in experiments:
        print(f"\n--- {exp_name.upper()} ---")

        model_prefix = "unified_256ms" if is_256ms else "unified"
        output_path = os.path.join(output_dir, f"{model_prefix}_{exp_name}.mlpackage")

        # Apply quantization
        quantized_model = quantize_func(original_model, output_path)

        if quantized_model is None:
            print(f"Skipping {exp_name} due to failure")
            continue

        # Measure metrics
        size_bytes = get_model_size(output_path)
        total_latency, chunk_time = measure_inference_time_1s(quantized_model, is_256ms)
        mse_error = test_model_accuracy(original_model, quantized_model, is_256ms)

        compression_ratio = original_size / size_bytes
        rtf = 1000.0 / (total_latency * 1000) if total_latency != float('inf') else 0.0

        results[exp_name] = {
            'model': quantized_model,
            'size_bytes': size_bytes,
            'total_latency_ms': total_latency * 1000,
            'chunk_time_ms': chunk_time * 1000,
            'mse_error': mse_error,
            'compression_ratio': compression_ratio,
            'rtf': rtf
        }

        print(f"Size: {size_bytes / 1024:.1f} KB ({compression_ratio:.1f}x compression)")
        print(f"1s latency: {total_latency * 1000:.2f} ms (RTF: {rtf:.1f}x)")
        print(f"MSE error: {mse_error:.6f}")

    return results


def create_combined_visualizations(standard_results: Dict[str, Any],
                                  ms256_results: Dict[str, Any] = None,
                                  plots_dir: str = "./plots",
                                  standard_model_name: str = "unified",
                                  ms256_model_name: str = "unified-256ms"):
    """Create combined visualization plots for both model variants"""

    os.makedirs(plots_dir, exist_ok=True)

    def results_to_dataframe(results: Dict[str, Any], variant_label: str) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for technique, data in (results or {}).items():
            if not data or 'size_bytes' not in data:
                continue
            rows.append({
                'variant': variant_label,
                'technique': technique,
                'size_kb': float(data['size_bytes']) / 1024.0,
                'total_latency_ms': float(data['total_latency_ms']),
                'chunk_time_ms': float(data['chunk_time_ms']),
                'mse_error': float(data['mse_error']),
                'compression_ratio': float(data['compression_ratio']),
                'rtf': float(data['rtf']),
            })
        return pd.DataFrame(rows)

    data_frames: List[pd.DataFrame] = []
    if standard_results:
        data_frames.append(results_to_dataframe(standard_results, standard_model_name))
    if ms256_results:
        data_frames.append(results_to_dataframe(ms256_results, ms256_model_name))

    if not data_frames:
        print("No quantization results available for visualization.")
        return

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df['display_name'] = combined_df['technique'].map(
        lambda t: TECHNIQUE_DISPLAY_NAMES.get(t, t.replace('_', ' ').title())
    )

    available_techniques = list(combined_df['technique'].unique())
    technique_order = [tech for tech in TECHNIQUE_ORDER if tech in available_techniques]
    technique_order.extend(sorted(t for t in available_techniques if t not in technique_order))
    display_order = [TECHNIQUE_DISPLAY_NAMES.get(t, t.replace('_', ' ').title()) for t in technique_order]

    variant_order = list(combined_df['variant'].unique())
    default_palette = {
        standard_model_name: '#1f77b4',
        ms256_model_name: '#d62728'
    }
    fallback_colors = sns.color_palette('husl', len(variant_order))
    palette = {
        variant: default_palette.get(variant, fallback_colors[idx])
        for idx, variant in enumerate(variant_order)
    }

    # 1. Accuracy vs Compression
    scatter_df = combined_df.dropna(subset=['compression_ratio', 'mse_error'])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=scatter_df,
        x='compression_ratio',
        y='mse_error',
        hue='variant',
        style='variant',
        s=120,
        palette=palette,
        ax=ax,
    )

    for _, row in scatter_df.iterrows():
        ax.annotate(
            row['display_name'],
            (row['compression_ratio'], row['mse_error']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8,
        )

    ax.set_xlabel('Compression Ratio (higher is better)', fontsize=12)
    ax.set_ylabel('MSE Error (lower is better)', fontsize=12)
    ax.set_title('Accuracy vs Compression Trade-off', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Model Variant')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_vs_compression.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Combined Performance Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quantization Performance Dashboard', fontsize=18, fontweight='bold')

    size_df = combined_df.dropna(subset=['size_kb'])
    sns.barplot(
        data=size_df,
        x='technique',
        y='size_kb',
        hue='variant',
        order=technique_order,
        palette=palette,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title('Model Size Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Model Size (KB)')
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_xticklabels(display_order, rotation=45, ha='right')
    axes[0, 0].legend(title='Model Variant')

    latency_df = combined_df.dropna(subset=['size_kb', 'total_latency_ms'])
    sns.scatterplot(
        data=latency_df,
        x='size_kb',
        y='total_latency_ms',
        hue='variant',
        style='variant',
        s=120,
        palette=palette,
        ax=axes[0, 1],
    )
    for _, row in latency_df.iterrows():
        axes[0, 1].annotate(
            row['display_name'],
            (row['size_kb'], row['total_latency_ms']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8,
        )
    axes[0, 1].set_title('Inference Time vs Model Size', fontweight='bold')
    axes[0, 1].set_xlabel('Model Size (KB)')
    axes[0, 1].set_ylabel('Inference Time (ms)')
    axes[0, 1].grid(True, alpha=0.3)

    compression_df = combined_df.dropna(subset=['compression_ratio'])
    sns.barplot(
        data=compression_df,
        x='technique',
        y='compression_ratio',
        hue='variant',
        order=technique_order,
        palette=palette,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title('Compression Ratios', fontweight='bold')
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_xticklabels(display_order, rotation=45, ha='right')
    axes[1, 0].legend(title='Model Variant')

    rtf_df = combined_df.dropna(subset=['rtf'])
    sns.barplot(
        data=rtf_df,
        x='technique',
        y='rtf',
        hue='variant',
        order=technique_order,
        palette=palette,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title('Real-Time Factor (RTF)', fontweight='bold')
    axes[1, 1].set_ylabel('RTF (higher is better)')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_xticklabels(display_order, rotation=45, ha='right')
    axes[1, 1].legend(title='Model Variant')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'quantization_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Model Comparison Summary emphasising precision first
    if standard_results and ms256_results:
        comp_df = combined_df[combined_df['technique'].isin(technique_order)].copy()
        comp_df['mse_plot'] = comp_df['mse_error'].clip(lower=1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(f'{standard_model_name} vs {ms256_model_name} Model Comparison', fontsize=16, fontweight='bold')

        sns.barplot(
            data=comp_df,
            x='technique',
            y='mse_plot',
            hue='variant',
            order=technique_order,
            palette=palette,
            ax=axes[0],
        )
        axes[0].set_title('MSE Error (Lower is Better)', fontweight='bold')
        axes[0].set_ylabel('MSE (log scale)')
        axes[0].set_xlabel('')
        axes[0].set_xticklabels(display_order, rotation=45, ha='right')
        axes[0].set_yscale('log')
        axes[0].legend(title='Model Variant')

        sns.barplot(
            data=comp_df,
            x='technique',
            y='rtf',
            hue='variant',
            order=technique_order,
            palette=palette,
            ax=axes[1],
        )
        axes[1].set_title('Real-Time Factor (Higher is Better)', fontweight='bold')
        axes[1].set_ylabel('RTF')
        axes[1].set_xlabel('')
        axes[1].set_xticklabels(display_order, rotation=45, ha='right')
        axes[1].legend(title='Model Variant')

        sns.barplot(
            data=comp_df,
            x='technique',
            y='total_latency_ms',
            hue='variant',
            order=technique_order,
            palette=palette,
            ax=axes[2],
        )
        axes[2].set_title('1s Processing Latency (Lower is Better)', fontweight='bold')
        axes[2].set_ylabel('Latency (ms)')
        axes[2].set_xlabel('')
        axes[2].set_xticklabels(display_order, rotation=45, ha='right')
        axes[2].legend(title='Model Variant')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Combined visualizations saved to: {plots_dir}")



def save_combined_results_summary(standard_results: Dict[str, Any],
                                  ms256_results: Dict[str, Any] = None,
                                  output_file: str = "comparison_results.json"):
    """Save combined results summary to JSON"""

    def process_results(results, variant_name):
        variant_summary = {}
        for technique, data in results.items():
            if data is None or 'size_bytes' not in data:
                continue
            variant_summary[technique] = {
                'size_kb': float(data['size_bytes'] / 1024),
                'total_latency_ms': float(data['total_latency_ms']),
                'chunk_time_ms': float(data['chunk_time_ms']),
                'mse_error': float(data['mse_error']),
                'compression_ratio': float(data['compression_ratio']),
                'rtf': float(data['rtf'])
            }
        return variant_summary

    summary = {
        'standard': process_results(standard_results, 'standard')
    }

    if ms256_results:
        summary['256ms'] = process_results(ms256_results, '256ms')

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Combined results summary saved to: {output_file}")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Quantize Silero VAD Unified CoreML Models")
    parser.add_argument("--model-variant", choices=["standard", "256ms", "both"], default="both",
                       help="Model variant to quantize (default: both)")
    parser.add_argument("--plots-dir", default="plots/quantization", help="Output directory for plots")
    parser.add_argument("--models-dir", default="quantized_models", help="Output directory for quantized models")

    args = parser.parse_args()

    # Model paths
    standard_model_path = "silero-vad-coreml/silero-vad-unified-v6.0.0.mlpackage"
    ms256_model_path = "silero-vad-coreml/silero-vad-unified-256ms-v6.0.0.mlpackage"

    output_dir_std = os.path.join(args.models_dir, "unified_standard")
    output_dir_256 = os.path.join(args.models_dir, "unified_256ms")

    print("🚀 Starting Silero VAD Unified Model Quantization Analysis")
    print("=" * 60)
    print(f"Model variant: {args.model_variant}")

    standard_results = None
    ms256_results = None

    try:
        # Process standard model
        if args.model_variant in ["standard", "both"]:
            if not os.path.exists(standard_model_path):
                print(f"❌ Standard model not found at: {standard_model_path}")
                if args.model_variant == "standard":
                    sys.exit(1)
            else:
                print(f"\n📊 Running quantization experiments for {os.path.basename(standard_model_path).replace('.mlpackage', '')}...")
                standard_results = run_quantization_experiments(standard_model_path, output_dir_std, is_256ms=False)

        # Process 256ms model
        if args.model_variant in ["256ms", "both"]:
            if not os.path.exists(ms256_model_path):
                print(f"❌ 256ms model not found at: {ms256_model_path}")
                if args.model_variant == "256ms":
                    sys.exit(1)
            else:
                print(f"\n📊 Running quantization experiments for {os.path.basename(ms256_model_path).replace('.mlpackage', '')}...")
                ms256_results = run_quantization_experiments(ms256_model_path, output_dir_256, is_256ms=True)

        # Create combined visualizations
        if standard_results or ms256_results:
            print("\n📈 Creating combined visualizations...")

            # Extract model names from file paths (remove .mlpackage extension)
            standard_name = os.path.basename(standard_model_path).replace('.mlpackage', '') if standard_results else "unified"
            ms256_name = os.path.basename(ms256_model_path).replace('.mlpackage', '') if ms256_results else "unified-256ms"

            create_combined_visualizations(standard_results or {}, ms256_results, args.plots_dir,
                                         standard_name, ms256_name)

            # Save combined results summary
            summary_file = os.path.join(args.plots_dir, "comparison_results.json")
            save_combined_results_summary(standard_results or {}, ms256_results, summary_file)

            print("\n✅ Quantization analysis completed successfully!")
            if standard_results:
                print(f"📁 {standard_name} quantized models saved to: {output_dir_std}")
            if ms256_results:
                print(f"📁 {ms256_name} quantized models saved to: {output_dir_256}")
            print(f"📊 Combined plots saved to: {args.plots_dir}")
            print(f"📋 Summary saved to: {summary_file}")

            # Print best techniques
            print("\n🏆 Best Techniques:")
            if standard_results:
                print(f"\n📊 {standard_name}:")
                valid_std = {k: v for k, v in standard_results.items() if v and 'compression_ratio' in v}
                if valid_std:
                    best_accuracy = min(valid_std.items(), key=lambda x: x[1]['mse_error'])
                    best_rtf = max(valid_std.items(), key=lambda x: x[1]['rtf'])
                    best_compression = max(valid_std.items(), key=lambda x: x[1]['compression_ratio'])
                    print(f"  • Best accuracy: {best_accuracy[0]} (MSE: {best_accuracy[1]['mse_error']:.6f})")
                    print(f"  • Highest RTF: {best_rtf[0]} ({best_rtf[1]['rtf']:.1f}x real-time)")
                    print(f"  • Best compression: {best_compression[0]} ({best_compression[1]['compression_ratio']:.1f}x)")

            if ms256_results:
                print(f"\n📊 {ms256_name}:")
                valid_256 = {k: v for k, v in ms256_results.items() if v and 'compression_ratio' in v}
                if valid_256:
                    best_accuracy = min(valid_256.items(), key=lambda x: x[1]['mse_error'])
                    best_rtf = max(valid_256.items(), key=lambda x: x[1]['rtf'])
                    best_compression = max(valid_256.items(), key=lambda x: x[1]['compression_ratio'])
                    print(f"  • Best accuracy: {best_accuracy[0]} (MSE: {best_accuracy[1]['mse_error']:.6f})")
                    print(f"  • Highest RTF: {best_rtf[0]} ({best_rtf[1]['rtf']:.1f}x real-time)")
                    print(f"  • Best compression: {best_compression[0]} ({best_compression[1]['compression_ratio']:.1f}x)")
        else:
            print("❌ No valid models found for quantization.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during quantization analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
