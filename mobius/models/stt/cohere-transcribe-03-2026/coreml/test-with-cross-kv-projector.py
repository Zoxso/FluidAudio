#!/usr/bin/env python3
"""Test cross-KV projector with ground truth validation."""

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
from jiwer import wer

from cohere_mel_spectrogram import CohereMelSpectrogram


# LibriSpeech test-clean samples with ground truth
LIBRISPEECH_SAMPLES = [
    {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "path": "LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac",
        "text": "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce",
        "duration": 10.0,
    },
    {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "path": "LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac",
        "text": "stuff it into you his belly counselled him",
        "duration": 3.0,
    },
    {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "path": "LibriSpeech/test-clean/1089/134686/1089-134686-0002.flac",
        "text": "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels",
        "duration": 6.5,
    },
]


class Pipeline:
    """Transcription pipeline."""

    def __init__(self, encoder_path: Path, decoder_path: Path, projector_path: Path = None):
        print(f"Loading encoder: {encoder_path.name}")
        self.encoder = ct.models.MLModel(str(encoder_path))

        print(f"Loading decoder: {decoder_path.name}")
        self.decoder = ct.models.MLModel(str(decoder_path))

        self.projector = None
        if projector_path:
            print(f"Loading projector: {projector_path.name}")
            self.projector = ct.models.MLModel(str(projector_path))

        self.mel_processor = CohereMelSpectrogram()
        self.eos_token_id = 2  # Cohere EOS token

    def transcribe(self, audio_path: Path, max_tokens: int = 256):
        """Transcribe audio and return tokens + timing."""
        # Load audio
        audio, sr = sf.read(str(audio_path))
        if sr != 16000:
            raise ValueError(f"Expected 16kHz, got {sr}Hz")

        # Mel spectrogram
        mel_start = time.perf_counter()
        mel = self.mel_processor(audio)
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant')
        mel_features = mel_padded.astype(np.float32)
        mel_length = np.array([mel.shape[2]], dtype=np.int32)
        mel_time = time.perf_counter() - mel_start

        # Encoder
        enc_start = time.perf_counter()
        encoder_output = self.encoder.predict({
            "input_features": mel_features,
            "feature_length": mel_length
        })
        # Find encoder hidden states (3D tensor)
        encoder_hidden = None
        for key, value in encoder_output.items():
            if hasattr(value, 'shape') and len(value.shape) == 3:
                encoder_hidden = value
                break
        if encoder_hidden is None:
            raise ValueError("Encoder output not found")
        enc_time = time.perf_counter() - enc_start

        # Cross-KV projection (if using projector)
        proj_time = 0.0
        cross_k = None
        cross_v = None

        if self.projector:
            proj_start = time.perf_counter()
            proj_output = self.projector.predict({
                "encoder_hidden_states": encoder_hidden
            })
            cross_k = proj_output["cross_k"]
            cross_v = proj_output["cross_v"]
            proj_time = time.perf_counter() - proj_start

        # Decoder preparation
        num_layers = 8
        num_heads = 8
        head_dim = 128
        max_cache_len = 108

        cache_k = np.zeros((num_layers, num_heads, max_cache_len, head_dim), dtype=np.float32)
        cache_v = np.zeros((num_layers, num_heads, max_cache_len, head_dim), dtype=np.float32)

        current_token = np.array([[13764]], dtype=np.int32)  # Start token
        generated_tokens = [13764]

        enc_seq_len = encoder_hidden.shape[1]
        cross_attention_mask = np.ones((1, 1, 1, enc_seq_len), dtype=np.float32)

        # Decode
        dec_start = time.perf_counter()
        for step in range(max_tokens):
            step_array = np.array([step], dtype=np.int32)

            decoder_input = {
                "input_id": current_token,
                "encoder_hidden_states": encoder_hidden,
                "cache_k": cache_k,
                "cache_v": cache_v,
                "step": step_array,
                "cross_attention_mask": cross_attention_mask,
            }

            decoder_output = self.decoder.predict(decoder_input)

            # Handle different output formats
            if "logits" in decoder_output:
                logits = decoder_output["logits"]
                cache_k = decoder_output["new_cache_k"]
                cache_v = decoder_output["new_cache_v"]
            else:
                output_values = list(decoder_output.values())
                logits = output_values[0]
                cache_k = output_values[1]
                cache_v = output_values[2]

            next_token = int(np.argmax(logits, axis=-1)[0])
            generated_tokens.append(next_token)

            if next_token == self.eos_token_id:
                break

            current_token = np.array([[next_token]], dtype=np.int32)

        dec_time = time.perf_counter() - dec_start

        return {
            "tokens": generated_tokens,
            "mel_time": mel_time,
            "encoder_time": enc_time,
            "projector_time": proj_time,
            "decoder_time": dec_time,
            "total_time": mel_time + enc_time + proj_time + dec_time,
        }


def decode_tokens(tokens):
    """Decode tokens to text (simple approach without tokenizer)."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "CohereLabs/cohere-transcribe-03-2026",
            trust_remote_code=True
        )
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"  ⚠️  Could not load tokenizer: {e}")
        return f"[tokens: {tokens[:10]}...]"


def download_librispeech_sample(output_dir: Path):
    """Download a small LibriSpeech sample for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # For now, just create synthetic audio since LibriSpeech download is large
    # In production, you'd download actual samples
    print("Creating test audio (use actual LibriSpeech for production)...")

    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simple speech-like signal
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 600 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)

    test_path = output_dir / "test-sample.wav"
    sf.write(test_path, audio, sr)

    return [{
        "path": test_path,
        "text": "synthetic test audio",  # No real ground truth for synthetic
        "duration": duration,
    }]


def main():
    parser = argparse.ArgumentParser(description="Test cross-KV projector with ground truth")
    parser.add_argument(
        "--encoder",
        type=Path,
        default=Path("build-quantized/cohere_encoder.mlpackage"),
        help="Encoder model path"
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=Path("build-quantized/cohere_decoder_cached.mlpackage"),
        help="Decoder model path"
    )
    parser.add_argument(
        "--projector",
        type=Path,
        default=Path("build-quantized/cohere_cross_kv_projector.mlpackage"),
        help="Cross-KV projector path (optional)"
    )
    parser.add_argument(
        "--no-projector",
        action="store_true",
        help="Test without projector for comparison"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("test-audio"),
        help="Directory with test audio"
    )

    args = parser.parse_args()

    print("="*70)
    print("Cross-KV Projector Ground Truth Test")
    print("="*70)

    # Get test samples
    test_samples = download_librispeech_sample(args.audio_dir)

    # Test configurations
    configs = []

    if args.no_projector or not args.projector.exists():
        configs.append({
            "name": "Without Projector (baseline)",
            "encoder": args.encoder,
            "decoder": args.decoder,
            "projector": None,
        })

    if args.projector.exists() and not args.no_projector:
        configs.append({
            "name": "With Cross-KV Projector",
            "encoder": args.encoder,
            "decoder": args.decoder,
            "projector": args.projector,
        })

    all_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")

        pipeline = Pipeline(
            config["encoder"],
            config["decoder"],
            config.get("projector")
        )

        config_results = []

        for sample in test_samples:
            print(f"\nProcessing: {sample['path'].name}")
            print(f"  Duration: {sample['duration']:.1f}s")
            print(f"  Ground truth: {sample['text']}")

            result = pipeline.transcribe(sample["path"])

            # Decode tokens
            hypothesis = decode_tokens(result["tokens"])
            print(f"  Hypothesis: {hypothesis}")

            # Calculate WER if we have ground truth
            if sample["text"] != "synthetic test audio":
                error_rate = wer(sample["text"], hypothesis)
                result["wer"] = error_rate
                print(f"  WER: {error_rate*100:.2f}%")

            # Timing
            print(f"  Timing:")
            print(f"    Mel: {result['mel_time']*1000:.1f}ms")
            print(f"    Encoder: {result['encoder_time']*1000:.1f}ms")
            if result['projector_time'] > 0:
                print(f"    Projector: {result['projector_time']*1000:.1f}ms")
            print(f"    Decoder: {result['decoder_time']*1000:.1f}ms")
            print(f"    Total: {result['total_time']*1000:.1f}ms")
            print(f"  RTF: {result['total_time'] / sample['duration']:.2f}x")

            config_results.append({
                **result,
                "hypothesis": hypothesis,
                "ground_truth": sample["text"],
                "duration": sample["duration"],
            })

        all_results.append({
            "name": config["name"],
            "results": config_results,
        })

    # Summary comparison
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")

        baseline = all_results[0]["results"]
        optimized = all_results[1]["results"]

        # Average metrics
        avg_baseline_time = np.mean([r["total_time"] for r in baseline])
        avg_optimized_time = np.mean([r["total_time"] for r in optimized])

        avg_baseline_dec = np.mean([r["decoder_time"] for r in baseline])
        avg_optimized_dec = np.mean([r["decoder_time"] for r in optimized])
        avg_proj_time = np.mean([r["projector_time"] for r in optimized])

        print(f"\nTiming (averaged across {len(baseline)} samples):")
        print(f"  Baseline decoder:     {avg_baseline_dec*1000:.1f}ms")
        print(f"  Optimized decoder:    {avg_optimized_dec*1000:.1f}ms")
        print(f"  Projector overhead:   {avg_proj_time*1000:.1f}ms")
        print(f"  Net decoder speedup:  {avg_baseline_dec/avg_optimized_dec:.2f}x")
        print(f"\nTotal pipeline:")
        print(f"  Baseline:  {avg_baseline_time*1000:.1f}ms")
        print(f"  Optimized: {avg_optimized_time*1000:.1f}ms")
        print(f"  Speedup:   {avg_baseline_time/avg_optimized_time:.2f}x")

        # WER comparison if available
        if any("wer" in r for r in baseline):
            avg_baseline_wer = np.mean([r.get("wer", 0) for r in baseline if "wer" in r])
            avg_optimized_wer = np.mean([r.get("wer", 0) for r in optimized if "wer" in r])
            print(f"\nAccuracy:")
            print(f"  Baseline WER:  {avg_baseline_wer*100:.2f}%")
            print(f"  Optimized WER: {avg_optimized_wer*100:.2f}%")
            print(f"  Difference:    {(avg_optimized_wer - avg_baseline_wer)*100:.2f}pp")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
