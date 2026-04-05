#!/usr/bin/env python3
"""Test if scaling our encoder output to match BarathwajAnandan's range fixes the decoder."""

import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("=== Testing Scaled Encoder Output ===\n")

# Load processor and tokenizer
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
tokenizer = processor.tokenizer

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
mel = inputs["input_features"].numpy()
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3501 - mel.shape[2])), mode='constant')

# Get both encoder outputs
print("1. Running encoders...")
barathwaj = ct.models.MLModel(
    "build/barathwaj-models/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
ours = ct.models.MLModel(
    "build/encoder_correct_static.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

barathwaj_out = barathwaj.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})["var_8638"]

ours_out = ours.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})["encoder_output"]

# Calculate scaling factor
barathwaj_std = barathwaj_out.std()
ours_std = ours_out.std()
scale_factor = barathwaj_std / ours_std

print(f"\n2. Value statistics:")
print(f"   BarathwajAnandan: mean={barathwaj_out.mean():.6f}, std={barathwaj_std:.6f}")
print(f"   Ours:             mean={ours_out.mean():.6f}, std={ours_std:.6f}")
print(f"   Scale factor:     {scale_factor:.6f}")

# Scale our output
ours_scaled = ours_out * scale_factor
print(f"\n3. After scaling our output by {scale_factor:.6f}:")
print(f"   Ours (scaled):    mean={ours_scaled.mean():.6f}, std={ours_scaled.std():.6f}")
print(f"   Range: [{ours_scaled.min():.4f}, {ours_scaled.max():.4f}]")

# Load decoder
print("\n4. Testing with decoder...")
decoder = ct.models.MLModel(
    "build/barathwaj-models/cohere_decoder_cached.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

# Test with SCALED encoder output
print("\n5. Autoregressive decoding with SCALED output...")
generated_tokens = [13764]
past_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
past_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

for step in range(100):
    decoder_output = decoder.predict({
        "input_id": np.array([[generated_tokens[-1]]], dtype=np.int32),
        "encoder_hidden_states": ours_scaled.astype(np.float16),  # Use SCALED output
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float16),
        "cache_k": past_cache_k,
        "cache_v": past_cache_v,
    })

    next_token = int(np.argmax(decoder_output["var_2891"][0]))
    generated_tokens.append(next_token)

    past_cache_k = decoder_output["var_2894"]
    past_cache_v = decoder_output["var_2897"]

    if next_token == 3:
        print(f"   ✓ EOS reached at step {step}")
        break

# Decode
transcription = tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\nOur encoder + Barathwaj decoder (SCALED):")
print(f'"{transcription}"')
print(f"Tokens generated: {len(generated_tokens)-1}")

print("\n" + "="*70)

# Compare with expected output
expected = "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce"

if "it's not a big deal" in transcription.lower():
    print("❌ FAILED: Still producing repetitive garbage")
    print("   Scaling did NOT fix the issue")
elif transcription.lower() == expected.lower():
    print("✅ SUCCESS: Scaling fixed the issue!")
    print("   Our encoder works with proper scaling!")
else:
    print("⚠️  PARTIAL: Different output (not garbage, not perfect)")
    print(f"   Expected: {expected}")

print("="*70)
