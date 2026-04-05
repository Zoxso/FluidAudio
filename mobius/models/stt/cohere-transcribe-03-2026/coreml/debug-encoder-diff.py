#!/usr/bin/env python3
"""Debug encoder by comparing layer outputs with BarathwajAnandan's encoder."""

import torch
import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("=== Debugging Encoder Differences ===\n")

# Load PyTorch model
print("1. Loading PyTorch model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
model.eval()
print("   ✓ PyTorch model loaded")

# Load audio and create mel spectrogram
print("\n2. Loading test audio...")
audio, sr = sf.read("test-librispeech-real.wav")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
mel = inputs["input_features"]

# Pad to 3501
mel_padded = torch.nn.functional.pad(mel, (0, 3501 - mel.shape[2]))
print(f"   Mel shape: {mel_padded.shape}")

# Get PyTorch encoder output as ground truth
print("\n3. Running PyTorch encoder...")
with torch.no_grad():
    encoder_output = model.encoder(mel_padded)
    # Unpack tuple if encoder returns (hidden_states, attention_mask)
    if isinstance(encoder_output, tuple):
        pytorch_encoder_output = encoder_output[0]
    else:
        pytorch_encoder_output = encoder_output

    if model.encoder_decoder_proj is not None:
        pytorch_encoder_output = model.encoder_decoder_proj(pytorch_encoder_output)
print(f"   PyTorch output: {pytorch_encoder_output.shape}")
print(f"   Value range: [{pytorch_encoder_output.min():.4f}, {pytorch_encoder_output.max():.4f}]")

# Load BarathwajAnandan's CoreML encoder
print("\n4. Loading BarathwajAnandan's CoreML encoder...")
barathwaj_encoder = ct.models.MLModel(
    "build/barathwaj-models/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
barathwaj_output = barathwaj_encoder.predict({
    "input_features": mel_padded.numpy().astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})["var_8638"]
print(f"   BarathwajAnandan output: {barathwaj_output.shape}")
print(f"   Value range: [{barathwaj_output.min():.4f}, {barathwaj_output.max():.4f}]")

# Compare with PyTorch
diff_barathwaj = np.abs(pytorch_encoder_output.numpy() - barathwaj_output).max()
print(f"   Diff vs PyTorch: {diff_barathwaj:.6f}")

# Load OUR CoreML encoder
print("\n5. Loading OUR CoreML encoder...")
try:
    our_encoder = ct.models.MLModel(
        "build/ultra_static_encoder.mlpackage",
        compute_units=ct.ComputeUnit.ALL
    )
    our_output = our_encoder.predict({
        "input_features": mel_padded.numpy().astype(np.float32),
    })["encoder_output"]
    print(f"   Our output: {our_output.shape}")
    print(f"   Value range: [{our_output.min():.4f}, {our_output.max():.4f}]")

    # Compare
    diff_ours = np.abs(pytorch_encoder_output.numpy() - our_output).max()
    print(f"   Diff vs PyTorch: {diff_ours:.6f}")

    diff_vs_barathwaj = np.abs(barathwaj_output - our_output).max()
    print(f"   Diff vs BarathwajAnandan: {diff_vs_barathwaj:.6f}")

except FileNotFoundError:
    print("   ❌ ultra_static_encoder.mlpackage not found")
    print("   Run: python export-ultra-static-encoder.py")
    our_output = None

# Load OUR encoder with feature_length input (correct version)
print("\n6. Loading OUR encoder (with feature_length)...")
try:
    our_encoder_correct = ct.models.MLModel(
        "build/encoder_correct_static.mlpackage",
        compute_units=ct.ComputeUnit.ALL
    )
    our_correct_output = our_encoder_correct.predict({
        "input_features": mel_padded.numpy().astype(np.float32),
        "feature_length": np.array([3501], dtype=np.int32)
    })

    # Find output key
    output_key = None
    for key, value in our_correct_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 438:
            output_key = key
            our_correct_output = value
            break

    if output_key:
        print(f"   Our output ({output_key}): {our_correct_output.shape}")
        print(f"   Value range: [{our_correct_output.min():.4f}, {our_correct_output.max():.4f}]")

        # Compare
        diff_ours_correct = np.abs(pytorch_encoder_output.numpy() - our_correct_output).max()
        print(f"   Diff vs PyTorch: {diff_ours_correct:.6f}")

        diff_vs_barathwaj_correct = np.abs(barathwaj_output - our_correct_output).max()
        print(f"   Diff vs BarathwajAnandan: {diff_vs_barathwaj_correct:.6f}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    our_correct_output = None

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"PyTorch encoder (ground truth):")
print(f"  Shape: {pytorch_encoder_output.shape}")
print(f"  Range: [{pytorch_encoder_output.min():.4f}, {pytorch_encoder_output.max():.4f}]")
print()
print(f"BarathwajAnandan encoder:")
print(f"  Shape: {barathwaj_output.shape}")
print(f"  Range: [{barathwaj_output.min():.4f}, {barathwaj_output.max():.4f}]")
print(f"  Diff vs PyTorch: {diff_barathwaj:.6f} {'✅ GOOD' if diff_barathwaj < 0.1 else '⚠️ HIGH'}")
print()

if our_correct_output is not None:
    print(f"OUR encoder:")
    print(f"  Shape: {our_correct_output.shape}")
    print(f"  Range: [{our_correct_output.min():.4f}, {our_correct_output.max():.4f}]")
    print(f"  Diff vs PyTorch: {diff_ours_correct:.6f} {'✅ GOOD' if diff_ours_correct < 0.1 else '❌ BAD'}")
    print(f"  Diff vs BarathwajAnandan: {diff_vs_barathwaj_correct:.6f}")

print("="*70)

# Statistical analysis
if our_correct_output is not None:
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)

    # Check if values are systematically different
    our_flat = our_correct_output.flatten()
    barathwaj_flat = barathwaj_output.flatten()
    pytorch_flat = pytorch_encoder_output.numpy().flatten()

    print(f"\nMean values:")
    print(f"  PyTorch: {pytorch_flat.mean():.6f}")
    print(f"  BarathwajAnandan: {barathwaj_flat.mean():.6f}")
    print(f"  Ours: {our_flat.mean():.6f}")

    print(f"\nStd deviation:")
    print(f"  PyTorch: {pytorch_flat.std():.6f}")
    print(f"  BarathwajAnandan: {barathwaj_flat.std():.6f}")
    print(f"  Ours: {our_flat.std():.6f}")

    print(f"\nPearson correlation with PyTorch:")
    corr_barathwaj = np.corrcoef(pytorch_flat, barathwaj_flat)[0, 1]
    corr_ours = np.corrcoef(pytorch_flat, our_flat)[0, 1]
    print(f"  BarathwajAnandan: {corr_barathwaj:.6f}")
    print(f"  Ours: {corr_ours:.6f}")

    print("="*70)
