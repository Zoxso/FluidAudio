#!/usr/bin/env python3
"""Quick test to verify palettization hypothesis."""

import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("=== Testing Palettization Hypothesis ===\n")

# Load processor
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
mel = inputs["input_features"].numpy()
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3501 - mel.shape[2])), mode='constant')

print("1. Loading BarathwajAnandan encoder...")
barathwaj = ct.models.MLModel(
    "build/barathwaj-models/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

print("2. Running BarathwajAnandan encoder...")
barathwaj_out = barathwaj.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})["var_8638"]

print("3. Loading our encoder...")
ours = ct.models.MLModel(
    "build/encoder_correct_static.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

print("4. Running our encoder...")
ours_out = ours.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})["encoder_output"]

# Count unique values
barathwaj_unique = np.unique(barathwaj_out.flatten())
ours_unique = np.unique(ours_out.flatten())

print("\n" + "="*70)
print("PALETTIZATION TEST")
print("="*70)

print(f"\nBarathwajAnandan encoder:")
print(f"  Total elements: {barathwaj_out.size}")
print(f"  Unique values:  {len(barathwaj_unique)}")
print(f"  First 10 unique values: {barathwaj_unique[:10]}")

print(f"\nOur encoder:")
print(f"  Total elements: {ours_out.size}")
print(f"  Unique values:  {len(ours_unique)}")
print(f"  First 10 unique values: {ours_unique[:10]}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if len(barathwaj_unique) < 100:
    print(f"\n✅ CONFIRMED: BarathwajAnandan uses palettization!")
    print(f"   Only {len(barathwaj_unique)} unique values (expected ~64 for 6-bit palettization)")
    print(f"   This is a heavily quantized model")
elif len(barathwaj_unique) < 1000:
    print(f"\n⚠️  BarathwajAnandan uses some quantization")
    print(f"   {len(barathwaj_unique)} unique values")
else:
    print(f"\n❌ BarathwajAnandan does NOT use palettization")
    print(f"   {len(barathwaj_unique)} unique values (too many for palettization)")

if len(ours_unique) > 10000:
    print(f"\n✅ Our encoder uses standard FP16")
    print(f"   {len(ours_unique)} unique values (expected ~100k+ for FP16)")
else:
    print(f"\n⚠️  Our encoder might be quantized too")
    print(f"   Only {len(ours_unique)} unique values")

print("\n" + "="*70)

if len(barathwaj_unique) < 100 and len(ours_unique) > 10000:
    print("\nTHIS EXPLAINS THE INCOMPATIBILITY:")
    print("• BarathwajAnandan: Palettized (~64 values)")
    print("• Ours: Full FP16 (100k+ values)")
    print("• Decoder expects palettized inputs!")
    print("• Our FP16 outputs have wrong value distribution")
    print("• Result: Decoder ignores our encoder, produces garbage")
    print("="*70)
