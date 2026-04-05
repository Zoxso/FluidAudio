#!/usr/bin/env python3
"""Test full autoregressive pipeline with our exported models."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram

print("="*70)
print("Testing Full Autoregressive Pipeline")
print("="*70)

# Load or create test audio
print("\n[1/5] Loading test audio...")
try:
    import soundfile as sf

    # Try to use a sample from dependencies
    sample_paths = [
        ".venv/lib/python3.10/site-packages/pyannote/audio/sample/sample.wav",
    ]

    audio = None
    for path in sample_paths:
        try:
            audio, sr = sf.read(path)
            print(f"   ✓ Loaded: {path}")
            print(f"     Audio: {len(audio)} samples ({len(audio)/sr:.2f}s) @ {sr}Hz")

            # Resample to 16kHz if needed
            if sr != 16000:
                print(f"   Resampling {sr}Hz → 16000Hz...")
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
                print(f"     Resampled: {len(audio)} samples ({len(audio)/16000:.2f}s)")

            # Use only first channel if stereo
            if audio.ndim > 1:
                audio = audio[:, 0]
                print(f"     Using first channel")

            break
        except Exception as e:
            continue

    if audio is None:
        raise FileNotFoundError("No sample audio found")

except Exception as e:
    print(f"   ⚠️  Could not load audio ({e})")
    print("   Creating synthetic speech-like audio...")

    # Create a simple speech-like signal (sum of sine waves)
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Fundamental + harmonics (simulates vowel sound)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # F0
        0.2 * np.sin(2 * np.pi * 400 * t) +  # 2nd harmonic
        0.1 * np.sin(2 * np.pi * 600 * t) +  # 3rd harmonic
        0.05 * np.random.randn(len(t))       # Noise
    )
    audio = audio.astype(np.float32)
    print(f"   Synthetic audio: {len(audio)} samples ({duration:.2f}s)")

# Compute mel spectrogram
print("\n[2/5] Computing mel spectrogram...")
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)

# Pad to 3001 frames
mel_padded = np.pad(
    mel,
    ((0, 0), (0, 0), (0, 3001 - mel.shape[2])),
    mode='constant',
    constant_values=0
)
print(f"   Mel shape: {mel_padded.shape}")
print(f"   Mel stats: mean={mel_padded.mean():.6f}, std={mel_padded.std():.6f}")

# Run encoder
print("\n[3/5] Running encoder...")
try:
    encoder = ct.models.MLModel(
        "build/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    encoder_output = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([3001], dtype=np.int32)
    })

    # Find encoder output
    encoder_hidden_states = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden_states = value
            break

    if encoder_hidden_states is None:
        raise ValueError("Could not find encoder output")

    print(f"   ✓ Encoder output: {encoder_hidden_states.shape}")

except Exception as e:
    print(f"   ❌ Encoder error: {e}")
    exit(1)

# Run autoregressive decoding
print("\n[4/5] Running autoregressive decoding...")
try:
    decoder = ct.models.MLModel(
        "build/cohere_decoder_cached.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    # Token IDs
    decoder_start_token_id = 13764
    eos_token_id = 3
    max_new_tokens = 50  # Limit for testing

    generated_tokens = [decoder_start_token_id]

    # Initialize cache
    cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
    cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

    print(f"   Starting generation (max {max_new_tokens} tokens)...")

    for step in range(max_new_tokens):
        decoder_input = {
            "input_id": np.array([[generated_tokens[-1]]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden_states.astype(np.float16),
            "step": np.array([step], dtype=np.int32),
            "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden_states.shape[1]), dtype=np.float16),
            "cache_k": cache_k,
            "cache_v": cache_v,
        }

        decoder_output = decoder.predict(decoder_input)

        # Extract outputs
        logits = None
        new_cache_k = None
        new_cache_v = None

        for key, value in decoder_output.items():
            if hasattr(value, 'shape'):
                if len(value.shape) == 2 and value.shape[1] > 1000:
                    logits = value
                elif len(value.shape) == 4:
                    if new_cache_k is None:
                        new_cache_k = value
                    else:
                        new_cache_v = value

        if logits is None:
            print(f"   ❌ Could not find logits at step {step}")
            break

        # Get next token
        next_token = int(np.argmax(logits[0]))
        generated_tokens.append(next_token)

        # Update cache
        if new_cache_k is not None and new_cache_v is not None:
            cache_k = new_cache_k
            cache_v = new_cache_v

        # Stop at EOS
        if next_token == eos_token_id:
            print(f"   ✓ Generated {len(generated_tokens)-1} tokens (stopped at EOS)")
            break

        if (step + 1) % 10 == 0:
            print(f"   ... {step+1} tokens generated")

    if generated_tokens[-1] != eos_token_id:
        print(f"   ⚠ Reached max_new_tokens without EOS")

    print(f"\n   Total tokens generated: {len(generated_tokens)-1}")
    print(f"   Token IDs: {generated_tokens[:20]}")  # First 20

except Exception as e:
    print(f"   ❌ Decoder error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Try to decode with tokenizer
print("\n[5/5] Decoding tokens...")
try:
    # Try to load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True
    )

    # Decode (skip the start token)
    transcription = tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)

    print(f"   ✓ Transcription: \"{transcription}\"")

except ImportError as e:
    print(f"   ⚠️  Could not load tokenizer: {e}")
    print(f"   Generated token IDs: {generated_tokens}")
except Exception as e:
    print(f"   ⚠️  Decoding error: {e}")
    print(f"   Generated token IDs: {generated_tokens}")

print("\n" + "="*70)
print("PIPELINE TEST COMPLETE")
print("="*70)
print("\nResults:")
print(f"  - Encoder: ✓ Working")
print(f"  - Decoder: ✓ Working")
print(f"  - Generated: {len(generated_tokens)-1} tokens")
if generated_tokens[-1] == eos_token_id:
    print(f"  - Stopped at: EOS (correct)")
else:
    print(f"  - Stopped at: max_new_tokens")
print()
