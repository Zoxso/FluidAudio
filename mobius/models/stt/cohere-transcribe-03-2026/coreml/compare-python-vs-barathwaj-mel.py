#!/usr/bin/env python3
"""Compare Python mel vs BarathwajAnandan frontend mel."""

import numpy as np
import soundfile as sf
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram

print("=== Comparing Python Mel vs Barathwaj Frontend Mel ===\n")

# Load audio
print("1. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
print(f"   Audio: {len(audio)} samples, {sr}Hz")

# Prepare audio for frontend (560000 samples = 35 seconds)
audio_padded = np.zeros(560000, dtype=np.float32)
audio_len = min(len(audio), 560000)
audio_padded[:audio_len] = audio[:audio_len]

# Compute Python mel
print("\n2. Computing Python mel...")
mel_processor = CohereMelSpectrogram()
python_mel = mel_processor(audio)
python_mel_padded = np.pad(
    python_mel,
    ((0, 0), (0, 0), (0, 3501 - python_mel.shape[2])),
    mode='constant',
    constant_values=0
)
print(f"   Python mel: {python_mel_padded.shape}")
print(f"   Stats: mean={python_mel_padded.mean():.6f}, std={python_mel_padded.std():.6f}")
print(f"   Range: [{python_mel_padded.min():.4f}, {python_mel_padded.max():.4f}]")

# Compute BarathwajAnandan frontend mel
print("\n3. Loading BarathwajAnandan frontend...")
frontend = ct.models.MLModel(
    "build/barathwaj-models/cohere_frontend.mlpackage",
    compute_units=ct.ComputeUnit.CPU_ONLY
)

print("4. Running BarathwajAnandan frontend...")
frontend_output = frontend.predict({
    "audio_samples": audio_padded.reshape(1, -1),
    "audio_length": np.array([audio_len], dtype=np.int32)
})

# Find mel output
barathwaj_mel = None
for key, value in frontend_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 128:
        barathwaj_mel = value
        print(f"   Barathwaj mel ({key}): {barathwaj_mel.shape}")
        break

if barathwaj_mel is None:
    print("   ❌ Could not find mel output")
    exit(1)

print(f"   Stats: mean={barathwaj_mel.mean():.6f}, std={barathwaj_mel.std():.6f}")
print(f"   Range: [{barathwaj_mel.min():.4f}, {barathwaj_mel.max():.4f}]")

# Compare
print("\n5. Comparing mels...")
diff = np.abs(python_mel_padded - barathwaj_mel)
print(f"   Max diff: {diff.max():.6f}")
print(f"   Mean diff: {diff.mean():.6f}")

# Check shapes match
if python_mel_padded.shape == barathwaj_mel.shape:
    print(f"   ✓ Shapes match: {python_mel_padded.shape}")
else:
    print(f"   ❌ Shape mismatch: {python_mel_padded.shape} vs {barathwaj_mel.shape}")

# Check if they're close enough
if diff.max() < 0.1:
    print("\n✅ Excellent match! Python mel should work with encoder")
elif diff.max() < 1.0:
    print("\n✅ Good match! Python mel should work with encoder")
else:
    print(f"\n⚠️  Large differences detected")
    print(f"   This might cause issues with the encoder")

print("="*70)
