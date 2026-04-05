#!/usr/bin/env python3
"""Test Cohere models with LibriSpeech test-clean ground truth."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram

print("="*70)
print("Testing with LibriSpeech test-clean Ground Truth")
print("="*70)

# Load LibriSpeech sample
print("\n[1/6] Loading LibriSpeech test-clean sample...")
try:
    from datasets import load_dataset

    # Load just 3 samples from test-clean
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    samples = []
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        samples.append(sample)

    print(f"   ✓ Loaded {len(samples)} samples")
    for i, sample in enumerate(samples):
        print(f"   Sample {i+1}:")
        print(f"     Audio: {len(sample['audio']['array'])} samples @ {sample['audio']['sampling_rate']}Hz")
        print(f"     Ground truth: \"{sample['text']}\"")
        print(f"     Duration: {len(sample['audio']['array'])/sample['audio']['sampling_rate']:.2f}s")

except Exception as e:
    print(f"   ❌ Error loading LibriSpeech: {e}")
    exit(1)

# Load models
print("\n[2/6] Loading CoreML models...")
try:
    encoder = ct.models.MLModel(
        "build/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    decoder = ct.models.MLModel(
        "build/cohere_decoder_cached.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print(f"   ✓ Models loaded")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    exit(1)

# Try to load tokenizer
print("\n[3/6] Loading tokenizer...")
try:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("../tokenizer.model")
    print(f"   ✓ Tokenizer loaded")
    has_tokenizer = True
except Exception as e:
    print(f"   ⚠️  Could not load tokenizer: {e}")
    has_tokenizer = False

# Process samples
print("\n[4/6] Processing samples...")
mel_processor = CohereMelSpectrogram()
decoder_start_token_id = 13764
eos_token_id = 3
max_new_tokens = 200

results = []

for sample_idx, sample in enumerate(samples):
    print(f"\n   Sample {sample_idx + 1}/{len(samples)}:")

    # Get audio
    audio = sample['audio']['array'].astype(np.float32)
    sr = sample['audio']['sampling_rate']

    # Resample if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Compute mel
    mel = mel_processor(audio)
    mel_padded = np.pad(
        mel,
        ((0, 0), (0, 0), (0, 3001 - mel.shape[2])),
        mode='constant',
        constant_values=0
    )

    # Run encoder
    encoder_output = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([mel.shape[2]], dtype=np.int32)
    })

    encoder_hidden = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = value
            break

    # Run autoregressive decoding
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

        decoder_output = decoder.predict(decoder_input)

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
            break

    # Decode tokens
    hypothesis = ""
    if has_tokenizer:
        # Skip first token (start token) and decode
        hypothesis = sp.DecodeIds(tokens[1:])
        # Remove special tokens manually
        for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                       '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                       '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                       '<|endoftext|>', '<|en|>', '<|ar|>', '<|eo|>', '<|tt|>', '<|ay|>',
                       '<|af|>', '<|am|>', '<|audioseparator|>', '<|emo:happy|>', '<|emo:sad|>']:
            hypothesis = hypothesis.replace(special, '')
        hypothesis = hypothesis.strip()

    results.append({
        'sample_idx': sample_idx,
        'ground_truth': sample['text'].lower(),
        'hypothesis': hypothesis,
        'tokens': tokens,
        'num_tokens': len(tokens) - 1,
        'stopped_at_eos': tokens[-1] == eos_token_id
    })

    print(f"     Generated {len(tokens)-1} tokens (EOS: {tokens[-1] == eos_token_id})")
    print(f"     Tokens: {tokens[:20]}...")

# Calculate WER
print("\n[5/6] Calculating WER...")

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Levenshtein distance for words
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
                    d[i-1][j] + 1,    # deletion
                    d[i][j-1] + 1,    # insertion
                    d[i-1][j-1] + 1   # substitution
                )

    distance = d[len(ref_words)][len(hyp_words)]
    wer = distance / len(ref_words) if len(ref_words) > 0 else 0.0
    return wer * 100

for result in results:
    if has_tokenizer and result['hypothesis']:
        wer = calculate_wer(result['ground_truth'], result['hypothesis'])
        result['wer'] = wer
    else:
        result['wer'] = None

# Print results
print("\n[6/6] Results:")
print("\n" + "="*70)

for result in results:
    print(f"\nSample {result['sample_idx'] + 1}:")
    print(f"  Ground truth: \"{result['ground_truth']}\"")

    if has_tokenizer:
        print(f"  Hypothesis:   \"{result['hypothesis']}\"")
        if result['wer'] is not None:
            print(f"  WER: {result['wer']:.2f}%")
    else:
        print(f"  Token IDs: {result['tokens'][:30]}...")

    print(f"  Tokens generated: {result['num_tokens']}")
    print(f"  Stopped at EOS: {result['stopped_at_eos']}")

if has_tokenizer and all(r['wer'] is not None for r in results):
    avg_wer = sum(r['wer'] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"Average WER: {avg_wer:.2f}%")

print("\n" + "="*70)
print("LIBRISPEECH TEST COMPLETE")
print("="*70)
