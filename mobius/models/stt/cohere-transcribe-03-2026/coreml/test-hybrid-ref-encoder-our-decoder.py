#!/usr/bin/env python3
"""Test hybrid: BarathwajAnandan's encoder + our decoder."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram

print("="*70)
print("Testing Hybrid: Reference Encoder + Our Decoder")
print("="*70)

# Load LibriSpeech sample
print("\n[1/5] Loading LibriSpeech test-clean sample...")
try:
    from datasets import load_dataset

    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    sample = next(iter(dataset))

    audio = sample['audio']['array'].astype(np.float32)
    sr = sample['audio']['sampling_rate']
    ground_truth = sample['text'].lower()

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    print(f"   ✓ Loaded sample")
    print(f"     Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")
    print(f"     Ground truth: \"{ground_truth}\"")

except Exception as e:
    print(f"   ❌ Error loading LibriSpeech: {e}")
    exit(1)

# Load models
print("\n[2/5] Loading models...")
try:
    # Reference encoder
    ref_encoder = ct.models.MLModel(
        "barathwaj-models/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print(f"   ✓ Reference encoder loaded")

    # Our decoder
    our_decoder = ct.models.MLModel(
        "build/cohere_decoder_cached.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print(f"   ✓ Our decoder loaded")

except FileNotFoundError as e:
    print(f"   ❌ Model not found: {e}")
    exit(1)
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    exit(1)

# Load tokenizer
print("\n[3/5] Loading tokenizer...")
try:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("../tokenizer.model")
    print(f"   ✓ Tokenizer loaded")
    has_tokenizer = True
except Exception as e:
    print(f"   ⚠️  Could not load tokenizer: {e}")
    has_tokenizer = False

# Compute mel spectrogram
print("\n[4/5] Running inference...")
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
mel_padded = np.pad(
    mel,
    ((0, 0), (0, 0), (0, 3001 - mel.shape[2])),
    mode='constant',
    constant_values=0
)
print(f"   Mel shape: {mel_padded.shape}")

# Run REFERENCE encoder
print(f"   Running REFERENCE encoder...")
ref_encoder_output = ref_encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([mel.shape[2]], dtype=np.int32)
})

encoder_hidden = None
for key, value in ref_encoder_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3:
        encoder_hidden = value
        break

print(f"   ✓ Reference encoder output: {encoder_hidden.shape}")

# Run OUR decoder with reference encoder output
print(f"   Running OUR decoder...")
decoder_start_token_id = 13764
eos_token_id = 3
max_new_tokens = 200

tokens = [decoder_start_token_id]
cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

for step in range(max_new_tokens):
    decoder_input = {
        "input_id": np.array([[tokens[-1]]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
        "cache_k": cache_k,
        "cache_v": cache_v,
    }

    decoder_output = our_decoder.predict(decoder_input)

    # Extract outputs
    logits = None
    for key, value in decoder_output.items():
        if hasattr(value, 'shape'):
            if len(value.shape) == 2 and value.shape[1] > 1000:
                logits = value
            elif len(value.shape) == 4:
                if 'k' in key.lower() or key == 'new_cache_k':
                    cache_k = value
                else:
                    cache_v = value

    next_token = int(np.argmax(logits[0]))
    tokens.append(next_token)

    if next_token == eos_token_id:
        print(f"   ✓ Generated {len(tokens)-1} tokens (stopped at EOS)")
        break

    if (step + 1) % 50 == 0:
        print(f"   ... {step+1} tokens")

if tokens[-1] != eos_token_id:
    print(f"   ⚠️  Reached max tokens without EOS")

print(f"   Total tokens: {len(tokens)-1}")
print(f"   First 30 tokens: {tokens[:30]}")

# Decode
print("\n[5/5] Results:")
print("\n" + "="*70)

hypothesis = ""
if has_tokenizer:
    hypothesis = sp.DecodeIds(tokens[1:])
    # Remove special tokens
    for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                   '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                   '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                   '<|endoftext|>', '<|en|>', '<|ar|>', '<|eo|>', '<|tt|>', '<|ay|>',
                   '<|af|>', '<|am|>', '<|audioseparator|>', '<|emo:happy|>', '<|emo:sad|>']:
        hypothesis = hypothesis.replace(special, '')
    hypothesis = hypothesis.strip()

print(f"Ground truth: \"{ground_truth}\"")
if has_tokenizer:
    print(f"Hypothesis:   \"{hypothesis}\"")

    # Calculate WER
    def calculate_wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,
                        d[i][j-1] + 1,
                        d[i-1][j-1] + 1
                    )

        distance = d[len(ref_words)][len(hyp_words)]
        wer = distance / len(ref_words) if len(ref_words) > 0 else 0.0
        return wer * 100

    if hypothesis:
        wer = calculate_wer(ground_truth, hypothesis)
        print(f"WER:          {wer:.2f}%")
else:
    print(f"Token IDs: {tokens}")

print(f"\nTokens generated: {len(tokens)-1}")
print(f"Stopped at EOS: {tokens[-1] == eos_token_id}")

print("\n" + "="*70)
print("HYBRID TEST COMPLETE")
print("="*70)
print("\nConclusion:")
if tokens[-1] != eos_token_id or len(tokens) >= max_new_tokens:
    print("  ❌ Our decoder fails even with perfect encoder output")
    print("  ✅ Confirms issue is 100% in our decoder export")
else:
    print("  ⚠️  Unexpected - decoder worked correctly")
print()
