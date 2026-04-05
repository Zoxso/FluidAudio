#!/usr/bin/env python3
"""Quantize Cohere Transcribe models for reduced size and faster inference."""

import argparse
from pathlib import Path
import coremltools as ct
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpPalettizerConfig,
    OpLinearQuantizerConfig,
)

def quantize_model(input_path: Path, output_path: Path, quantization: str):
    """
    Quantize a CoreML model.

    Args:
        input_path: Path to input .mlpackage
        output_path: Path to output .mlpackage
        quantization: Type of quantization ('int8', 'int4', 'int6_palettize')
    """
    print(f"Loading model from {input_path}...")
    model = ct.models.MLModel(str(input_path))

    # Get original size
    orig_size = sum(f.stat().st_size for f in input_path.rglob('*') if f.is_file()) / (1024**2)
    print(f"  Original size: {orig_size:.1f} MB")

    if quantization == 'int8':
        print("Applying int8 linear quantization...")
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
            )
        )
        quantized_model = ct.optimize.coreml.linear_quantize_weights(model, config=config)
    elif quantization == 'int4':
        print("Applying int4 linear quantization...")
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int4",
            )
        )
        quantized_model = ct.optimize.coreml.linear_quantize_weights(model, config=config)
    elif quantization == 'int6_palettize':
        print("Applying 6-bit palettization (nearest equivalent to Q6)...")
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                mode="kmeans",
                nbits=6,  # 6-bit palettization
            )
        )
        quantized_model = ct.optimize.coreml.palettize_weights(model, config=config)
    else:
        raise ValueError(f"Unknown quantization: {quantization}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized_model.save(str(output_path))

    # Get quantized size
    quant_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024**2)
    print(f"  Quantized size: {quant_size:.1f} MB")
    print(f"  Reduction: {(1 - quant_size/orig_size)*100:.1f}%")
    print(f"  ✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize Cohere Transcribe models")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("build"),
        help="Directory containing FP16 models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build-quantized"),
        help="Output directory for quantized models"
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "int4", "int6_palettize"],
        default="int6_palettize",
        help="Quantization type (int6_palettize is nearest to Q6)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["encoder", "decoder"],
        choices=["encoder", "decoder", "all"],
        help="Which models to quantize"
    )

    args = parser.parse_args()

    print("="*70)
    print(f"Cohere Transcribe Model Quantization ({args.quantization})")
    print("="*70)

    models_to_quantize = []
    if "all" in args.models or "encoder" in args.models:
        models_to_quantize.append(("encoder", "cohere_encoder.mlpackage"))
    if "all" in args.models or "decoder" in args.models:
        models_to_quantize.append(("decoder", "cohere_decoder_cached.mlpackage"))

    for name, filename in models_to_quantize:
        print(f"\n{'='*70}")
        print(f"Quantizing {name}")
        print(f"{'='*70}")

        input_path = args.input_dir / filename
        output_path = args.output_dir / filename

        if not input_path.exists():
            print(f"  ⚠️  Skipping - not found: {input_path}")
            continue

        try:
            quantize_model(input_path, output_path, args.quantization)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("QUANTIZATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
