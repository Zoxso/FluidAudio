#!/usr/bin/env python3
"""Full CoreML inference pipeline for Canary-1B v2 with decoding."""

import coremltools as ct
import numpy as np
import soundfile as sf
from pathlib import Path
import time
import json
from typing import List, Tuple, Optional, Dict
from canary_tokenizer import CanaryTokenizer


class CanaryCoreMLInference:
    """CoreML inference pipeline for Canary-1B v2."""

    def __init__(
        self,
        model_dir: str = "canary_coreml_fp32",
        max_audio_seconds: float = 30.0,
        max_decoder_steps: int = 256
    ):
        """Initialize the CoreML models and tokenizer.

        Args:
            model_dir: Directory containing CoreML models
            max_audio_seconds: Maximum audio duration in seconds
            max_decoder_steps: Maximum decoder sequence length
        """
        self.model_dir = Path(model_dir)
        self.max_audio_seconds = max_audio_seconds
        self.max_decoder_steps = max_decoder_steps
        self.sample_rate = 16000

        # Load models
        print(f"Loading CoreML models from {model_dir}...")
        self.preprocessor = ct.models.MLModel(str(self.model_dir / "canary_preprocessor.mlpackage"))
        self.encoder = ct.models.MLModel(str(self.model_dir / "canary_encoder.mlpackage"))
        self.decoder = ct.models.MLModel(str(self.model_dir / "canary_decoder.mlpackage"))
        self.proj_weights, self.proj_bias = self._load_projection()

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = CanaryTokenizer()

        # Load metadata
        with open(self.model_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.vocab_size = self.metadata.get("vocab_size", 16384)
        # Respect exported decoder length to avoid shape mismatches.
        self.max_decoder_steps = min(
            max_decoder_steps, int(self.metadata.get("max_decoder_seq_length", max_decoder_steps))
        )

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and prepare audio for inference.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio array, actual length in samples)
        """
        audio, sr = sf.read(audio_path, dtype='float32')

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample if needed (simplified - proper resampling would use librosa)
        if sr != self.sample_rate:
            print(f"Warning: Sample rate {sr} != {self.sample_rate}")

        # Truncate or pad to max duration
        max_samples = int(self.sample_rate * self.max_audio_seconds)
        actual_length = len(audio)

        if len(audio) > max_samples:
            audio = audio[:max_samples]
            actual_length = max_samples
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        return audio, actual_length

    def _load_projection(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load projection weights that map decoder hidden state -> vocab logits."""
        proj_path = Path(__file__).parent / "projection_weights.npz"
        if not proj_path.exists():
            raise FileNotFoundError(f"Missing projection weights at {proj_path}")
        data = np.load(proj_path)
        weights = data["weights"]  # shape: [vocab, hidden]
        bias = data["bias"]        # shape: [vocab]
        # For matmul hidden_state[1, hidden] @ W[hidden, vocab]
        return weights.T, bias

    def greedy_decode(
        self,
        encoder_embeddings: np.ndarray,
        encoder_mask: np.ndarray,
        language: str = "en",
        task: str = "transcribe",
        pnc: bool = True,
        temperature: float = 1.0
    ) -> Tuple[List[int], List[float]]:
        """Perform greedy decoding with the CoreML decoder.

        Args:
            encoder_embeddings: Encoder output embeddings [B, T, D]
            encoder_mask: Encoder attention mask [B, T]
            language: Target language
            task: Task type (transcribe or translate)
            pnc: Whether to include punctuation
            temperature: Sampling temperature

        Returns:
            Tuple of (token IDs, probabilities)
        """
        batch_size = 1

        # Initialize with prompt tokens
        prompt_tokens = self.tokenizer.create_prompt_tokens(language, task, pnc)
        generated_tokens = prompt_tokens.copy()
        token_probs = []

        # Create input arrays
        input_ids = np.zeros((batch_size, self.max_decoder_steps), dtype=np.int32)
        decoder_mask = np.zeros((batch_size, self.max_decoder_steps), dtype=np.float32)

        # Fill in prompt tokens
        for i, token in enumerate(prompt_tokens):
            input_ids[0, i] = token
            decoder_mask[0, i] = 1.0

        current_pos = len(prompt_tokens)

        # Autoregressive decoding loop
        print(f"Decoding with prompt: {prompt_tokens}")

        for step in range(self.max_decoder_steps - current_pos):
            # Run decoder
            decoder_output = self.decoder.predict({
                'input_ids': input_ids,
                'decoder_mask': decoder_mask,
                'encoder_embeddings': encoder_embeddings,
                'encoder_mask': encoder_mask
            })

            # Hidden state -> logits
            hidden_state = decoder_output['decoder'][0, current_pos - 1, :]  # [hidden]
            logits = np.dot(hidden_state, self.proj_weights) + self.proj_bias

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            probs = exp_logits / np.sum(exp_logits)

            # Greedy selection (could also do beam search or sampling)
            next_token = np.argmax(probs)
            token_prob = probs[next_token]

            # Add to generated tokens
            generated_tokens.append(int(next_token))
            token_probs.append(float(token_prob))

            # Check for end of sequence
            if next_token == self.tokenizer.special_tokens['eos']:
                print(f"End of sequence detected at position {current_pos}")
                break

            # Update inputs for next step
            input_ids[0, current_pos] = next_token
            decoder_mask[0, current_pos] = 1.0
            current_pos += 1

            # Progress indicator
            if step % 10 == 0:
                print(f"  Decoded {step} tokens...")

        return generated_tokens, token_probs

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        task: str = "transcribe",
        pnc: bool = True
    ) -> Dict:
        """Transcribe audio using the CoreML models.

        Args:
            audio_path: Path to audio file
            language: Target language
            task: Task type (transcribe or translate)
            pnc: Whether to include punctuation

        Returns:
            Dictionary with transcription results
        """
        print(f"\n{'='*60}")
        print(f"Transcribing: {audio_path}")
        print(f"Settings: language={language}, task={task}, pnc={pnc}")
        print(f"{'='*60}")

        # Load audio
        print("\n1. Loading audio...")
        start_time = time.time()
        audio, audio_length = self.load_audio(audio_path)
        audio_signal = audio.reshape(1, -1).astype(np.float32)
        audio_len = np.array([audio_length], dtype=np.int32)
        load_time = time.time() - start_time
        print(f"   Audio loaded in {load_time:.2f}s")

        # Preprocessor
        print("\n2. Running preprocessor...")
        start_time = time.time()
        preprocessor_out = self.preprocessor.predict({
            'audio_signal': audio_signal,
            'audio_length': audio_len
        })
        mel_features = preprocessor_out['processed']
        mel_length = preprocessor_out['processed_length']
        preprocess_time = time.time() - start_time
        print(f"   Preprocessor completed in {preprocess_time:.2f}s")
        print(f"   Mel features shape: {mel_features.shape}")

        # Encoder
        print("\n3. Running encoder...")
        start_time = time.time()
        encoder_out = self.encoder.predict({
            'features': mel_features,
            'features_length': mel_length
        })
        encoder_output = encoder_out['encoder']
        encoder_length = encoder_out['encoder_length']
        encode_time = time.time() - start_time
        print(f"   Encoder completed in {encode_time:.2f}s")
        print(f"   Encoder output shape: {encoder_output.shape}")

        # Prepare for decoder
        print("\n4. Preparing decoder inputs...")
        encoder_embeddings = np.transpose(encoder_output, (0, 2, 1))  # [B, T, D]
        encoder_seq_len = encoder_embeddings.shape[1]
        encoder_mask = np.ones((1, encoder_seq_len), dtype=np.float32)

        # Adjust mask based on actual length
        actual_length = int(encoder_length[0])
        if actual_length < encoder_seq_len:
            encoder_mask[0, actual_length:] = 0.0

        # Decode
        print("\n5. Running decoder (greedy decoding)...")
        start_time = time.time()
        generated_tokens, token_probs = self.greedy_decode(
            encoder_embeddings=encoder_embeddings,
            encoder_mask=encoder_mask,
            language=language,
            task=task,
            pnc=pnc
        )
        decode_time = time.time() - start_time
        print(f"   Decoder completed in {decode_time:.2f}s")
        print(f"   Generated {len(generated_tokens)} tokens")

        # Decode tokens to text
        print("\n6. Converting tokens to text...")
        # Skip prompt tokens for decoding
        prompt_length = len(self.tokenizer.create_prompt_tokens(language, task, pnc))
        output_tokens = generated_tokens[prompt_length:]
        text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

        # Calculate statistics
        total_time = load_time + preprocess_time + encode_time + decode_time
        audio_duration = audio_length / self.sample_rate
        rtf = audio_duration / total_time

        results = {
            'text': text,
            'tokens': output_tokens,
            'token_probs': token_probs,
            'timings': {
                'load': load_time,
                'preprocess': preprocess_time,
                'encode': encode_time,
                'decode': decode_time,
                'total': total_time
            },
            'audio_duration': audio_duration,
            'rtf': rtf,
            'settings': {
                'language': language,
                'task': task,
                'pnc': pnc
            }
        }

        print(f"\n{'='*60}")
        print(f"TRANSCRIPTION RESULTS")
        print(f"{'='*60}")
        print(f"Text: {text}")
        print(f"\nPerformance:")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Real-time factor: {rtf:.2f}x")
        print(f"{'='*60}")

        return results


def main():
    """Test the CoreML inference pipeline."""
    import sys

    # Use provided audio or default test file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "/Users/kikow/brandon/FluidAudioSwift/mobius/models/stt/parakeet-tdt-v3-0.6b/coreml/audio/yc_first_minute_16k_15s.wav"

    # Initialize inference pipeline
    pipeline = CanaryCoreMLInference(model_dir="canary_coreml_fp32")

    # Test different configurations
    configs = [
        {"language": "en", "task": "transcribe", "pnc": True},
        {"language": "en", "task": "transcribe", "pnc": False},
    ]

    all_results = []
    for config in configs:
        print(f"\n\n{'#'*70}")
        print(f"Configuration: {config}")
        print(f"{'#'*70}")

        result = pipeline.transcribe(
            audio_path,
            **config
        )
        all_results.append(result)

        # Save results
        output_file = f"coreml_transcription_{config['task']}_{config['language']}_pnc{config['pnc']}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_file}")

    # Compare with PyTorch results if available
    pytorch_file = "pytorch_transcriptions.json"
    if Path(pytorch_file).exists():
        print(f"\n\n{'='*70}")
        print("COMPARISON WITH PYTORCH")
        print(f"{'='*70}")

        with open(pytorch_file) as f:
            pytorch_results = json.load(f)

        for i, (config, result) in enumerate(zip(configs, all_results)):
            print(f"\nConfiguration {i+1}: {config}")
            print(f"CoreML: {result['text'][:100]}...")

            # Find matching PyTorch result
            if config['pnc']:
                pytorch_key = 'en_asr_pnc'
            else:
                pytorch_key = 'en_asr_nopnc'

            if pytorch_key in pytorch_results:
                print(f"PyTorch: {pytorch_results[pytorch_key]['text'][:100]}...")


if __name__ == "__main__":
    main()
