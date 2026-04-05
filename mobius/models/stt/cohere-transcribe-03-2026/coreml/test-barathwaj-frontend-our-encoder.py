#!/usr/bin/env python3
"""Test BarathwajAnandan frontend → Our encoder → BarathwajAnandan decoder."""

import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("=== Testing: Barathwaj Frontend + Our Encoder + Barathwaj Decoder ===\n")

# Load processor for tokenizer
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
tokenizer = processor.tokenizer

# Load audio
print("1. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
print(f"   Audio: {len(audio)} samples, {sr}Hz")

# Prepare audio for frontend (560000 samples = 35 seconds)
audio_padded = np.zeros(560000, dtype=np.float32)
audio_len = min(len(audio), 560000)
audio_padded[:audio_len] = audio[:audio_len]

# Load BarathwajAnandan's frontend
print("\n2. Loading BarathwajAnandan's frontend...")
frontend = ct.models.MLModel(
    "build/barathwaj-models/cohere_frontend.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

print("3. Running BarathwajAnandan's frontend...")
frontend_output = frontend.predict({
    "audio_samples": audio_padded.reshape(1, -1),
    "audio_length": np.array([audio_len], dtype=np.int32)
})

# Find mel output
mel = None
for key, value in frontend_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 128:
        mel = value
        print(f"   Frontend output ({key}): {mel.shape}")
        break

if mel is None:
    print("   ❌ Could not find mel output from frontend")
    exit(1)

# Load OUR encoder
print("\n4. Loading OUR encoder...")
our_encoder = ct.models.MLModel(
    "build/encoder_correct_static.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

print("5. Running OUR encoder...")
encoder_output = our_encoder.predict({
    "input_features": mel.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})

hidden_states = encoder_output["encoder_output"]
print(f"   Our encoder output: {hidden_states.shape}")
print(f"   Value range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

# Load BarathwajAnandan's decoder
print("\n6. Loading BarathwajAnandan's decoder...")
decoder = ct.models.MLModel(
    "build/barathwaj-models/cohere_decoder_cached.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

# Autoregressive decoding
print("7. Running autoregressive decoding...")
generated_tokens = [13764]  # decoder_start_token_id
past_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
past_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

for step in range(100):
    decoder_output = decoder.predict({
        "input_id": np.array([[generated_tokens[-1]]], dtype=np.int32),
        "encoder_hidden_states": hidden_states.astype(np.float16),
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float16),
        "cache_k": past_cache_k,
        "cache_v": past_cache_v,
    })

    next_token = int(np.argmax(decoder_output["var_2891"][0]))
    generated_tokens.append(next_token)

    past_cache_k = decoder_output["var_2894"]
    past_cache_v = decoder_output["var_2897"]

    if next_token == 3:  # eos_token_id
        print(f"   ✓ EOS reached at step {step}")
        break

# Decode
transcription = tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)

print("\n" + "="*70)
print("RESULT")
print("="*70)

print(f"\nBarathwaj Frontend + Our Encoder + Barathwaj Decoder:")
print(f'"{transcription}"')
print(f"\nTokens: {len(generated_tokens)-1}")

# Compare with expected
expected = "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce"

if "it's not a big deal" in transcription.lower():
    print("\n❌ FAILED: Still producing 'it's not a big deal' garbage")
elif "de la peine" in transcription.lower():
    print("\n❌ FAILED: Still producing 'de la peine' garbage")
elif transcription.lower().strip() == expected.lower().strip():
    print("\n✅ SUCCESS: Perfect transcription!")
    print("   BarathwajAnandan's frontend FIXES the issue!")
else:
    print(f"\n⚠️  PARTIAL: Different output")
    print(f"   Expected: {expected}")

print("="*70)
