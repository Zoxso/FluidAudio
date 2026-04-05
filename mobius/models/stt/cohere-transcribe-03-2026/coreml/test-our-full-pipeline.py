#!/usr/bin/env python3
"""Test our complete custom pipeline: frontend → encoder → decoder."""

import torch
import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("=== Testing OUR Full Pipeline ===\n")

# Load processor and tokenizer
print("1. Loading processor...")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
tokenizer = processor.tokenizer
print(f"   ✓ Vocab size: {tokenizer.vocab_size}")

# Load audio
print("\n2. Loading test audio...")
audio, sr = sf.read("test-librispeech-real.wav")
print(f"   Audio: {len(audio)} samples, {sr}Hz")

# ============================================================================
# OPTION 1: Try with OUR frontend (if available)
# ============================================================================

print("\n3. Testing with OUR frontend...")
try:
    frontend = ct.models.MLModel(
        "build/ultra_static_frontend.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    # Prepare audio input (must be exactly 560000 samples = 35 seconds)
    audio_padded = np.zeros(560000, dtype=np.float32)
    audio_len = min(len(audio), 560000)
    audio_padded[:audio_len] = audio[:audio_len]

    frontend_output = frontend.predict({
        "audio_samples": audio_padded.reshape(1, -1),
        "audio_length": np.array([audio_len], dtype=np.int32)
    })

    # Find mel output key
    mel_key = None
    for key, value in frontend_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 128:
            mel_key = key
            mel = value
            break

    if mel_key:
        print(f"   ✓ Frontend output ({mel_key}): {mel.shape}")
    else:
        print("   ❌ Could not find mel output")
        mel = None

except Exception as e:
    print(f"   ⚠️  Frontend not available: {e}")
    print("   Using processor for mel spectrogram")
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    mel = inputs["input_features"].numpy()
    # Pad to 3501
    mel = np.pad(mel, ((0, 0), (0, 0), (0, 3501 - mel.shape[2])), mode='constant')
    print(f"   Processor mel: {mel.shape}")

# ============================================================================
# OPTION 2: Use OUR encoder
# ============================================================================

print("\n4. Testing with OUR encoder...")
try:
    encoder = ct.models.MLModel(
        "build/encoder_correct_static.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    encoder_output = encoder.predict({
        "input_features": mel.astype(np.float32),
        "feature_length": np.array([3501], dtype=np.int32)
    })

    # Find encoder output key
    hidden_key = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 438:
            hidden_key = key
            hidden_states = value
            break

    if hidden_key:
        print(f"   ✓ Encoder output ({hidden_key}): {hidden_states.shape}")
        print(f"   Value range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
    else:
        print("   ❌ Could not find encoder output")
        hidden_states = None

except Exception as e:
    print(f"   ❌ Encoder failed: {e}")
    import traceback
    traceback.print_exc()
    hidden_states = None

if hidden_states is None:
    print("\n❌ Cannot continue without encoder output")
    exit(1)

# ============================================================================
# OPTION 3: Try with OUR decoder (if available)
# ============================================================================

print("\n5. Testing with OUR decoder...")
try:
    decoder = ct.models.MLModel(
        "build/ultra_static_decoder_cached.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print("   ✓ OUR decoder loaded")

    # Initialize autoregressive decoding
    generated_tokens = [13764]  # decoder_start_token_id
    past_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
    past_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

    print("   Starting autoregressive decoding...")
    for step in range(100):
        decoder_output = decoder.predict({
            "input_id": np.array([[generated_tokens[-1]]], dtype=np.int32),
            "encoder_hidden_states": hidden_states.astype(np.float16),
            "step": np.array([step], dtype=np.int32),
            "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float16),
            "cache_k": past_cache_k,
            "cache_v": past_cache_v,
        })

        # Find logits and cache outputs
        logits_key = None
        cache_k_key = None
        cache_v_key = None

        for key, value in decoder_output.items():
            if hasattr(value, 'shape'):
                if len(value.shape) == 3 and value.shape[-1] > 1000:  # Vocab size
                    logits_key = key
                elif 'cache' in key.lower() and 'k' in key.lower():
                    cache_k_key = key
                elif 'cache' in key.lower() and 'v' in key.lower():
                    cache_v_key = key

        if not logits_key:
            print(f"   ❌ Could not find logits in decoder output")
            break

        next_token = int(np.argmax(decoder_output[logits_key][0]))
        generated_tokens.append(next_token)

        if cache_k_key and cache_v_key:
            past_cache_k = decoder_output[cache_k_key]
            past_cache_v = decoder_output[cache_v_key]

        if next_token == 3:  # eos_token_id
            print(f"   ✓ EOS reached at step {step}")
            break

    # Decode tokens
    transcription = tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)
    print(f"\n✅ OUR PIPELINE TRANSCRIPTION ({len(generated_tokens)-1} tokens):")
    print(f'   "{transcription}"')

except FileNotFoundError:
    print("   ⚠️  OUR decoder not found, trying BarathwajAnandan's decoder...")

    # Fallback to BarathwajAnandan's decoder
    decoder = ct.models.MLModel(
        "build/barathwaj-models/cohere_decoder_cached.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print("   ✓ BarathwajAnandan decoder loaded")

    # Initialize autoregressive decoding
    generated_tokens = [13764]  # decoder_start_token_id
    past_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
    past_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

    print("   Starting autoregressive decoding...")
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

    # Decode tokens
    transcription = tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)
    print(f"\n⚠️  MIXED PIPELINE TRANSCRIPTION (our encoder + Barathwaj decoder):")
    print(f'   "{transcription}"')
    print(f"   Tokens: {len(generated_tokens)-1}")

except Exception as e:
    print(f"   ❌ Decoder failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
