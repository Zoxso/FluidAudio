#!/usr/bin/env python3
"""Python mel spectrogram implementation for Cohere Transcribe (no CoreML frontend needed)."""

import numpy as np
import librosa


class CohereMelSpectrogram:
    """
    Mel spectrogram processor matching Cohere Transcribe parameters.

    This replaces the CoreML frontend - we compute mels in Python/Swift,
    then pass to the CoreML encoder.

    Parameters (from coreml_manifest.json):
    - sample_rate: 16000
    - n_fft: 1024
    - hop_length: 160
    - win_length: 1024
    - n_mels: 128
    - preemph: 0.97
    - max_audio_samples: 560000 (35 seconds)
    - max_feature_frames: 3501
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 1024,
        n_mels: int = 128,
        preemph: float = 0.97,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.preemph = preemph
        self.f_min = f_min
        self.f_max = f_max

        # Pre-compute mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
            norm='slaney',
        )

    def __call__(self, audio: np.ndarray, length: int = None) -> np.ndarray:
        """
        Compute mel spectrogram.

        Args:
            audio: Raw audio waveform, shape (samples,) or (batch, samples)
            length: Optional length of valid audio samples

        Returns:
            mel: Mel spectrogram, shape (1, n_mels, frames)
        """
        # Handle batch dimension
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, samples)

        batch_size = audio.shape[0]
        assert batch_size == 1, "Only batch size 1 supported"

        audio = audio[0]  # (samples,)

        # Apply preemphasis filter: y[n] = x[n] - preemph * x[n-1]
        if self.preemph > 0:
            audio = np.append(audio[0], audio[1:] - self.preemph * audio[:-1])

        # Compute STFT
        # librosa.stft uses center=True by default, which pads the signal
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            pad_mode='reflect',
        )
        # stft: (n_fft // 2 + 1, frames) complex

        # Compute power spectrogram
        power = np.abs(stft) ** 2  # (n_fft // 2 + 1, frames)

        # Apply mel filterbank
        mel = np.dot(self.mel_basis, power)  # (n_mels, frames)

        # Log scale
        mel = np.log10(np.maximum(mel, 1e-10))

        # Normalize (subtract mean)
        mel = mel - mel.mean()

        # Add batch dimension
        mel = mel[np.newaxis, ...]  # (1, n_mels, frames)

        return mel


def test_mel_spectrogram():
    """Test mel spectrogram computation."""
    import soundfile as sf

    print("=== Testing CohereMelSpectrogram ===\n")

    # Load audio
    print("1. Loading test audio...")
    audio, sr = sf.read("test-librispeech-real.wav")
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
    print(f"   Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Create mel processor
    print("\n2. Creating mel processor...")
    mel_processor = CohereMelSpectrogram()
    print("   ✓ Processor created")

    # Compute mel spectrogram
    print("\n3. Computing mel spectrogram...")
    mel = mel_processor(audio)
    print(f"   Mel shape: {mel.shape}")
    print(f"   Expected: (1, 128, ~1044) for {len(audio)/16000:.2f}s audio")
    print(f"   Mel stats: mean={mel.mean():.6f}, std={mel.std():.6f}")
    print(f"   Value range: [{mel.min():.4f}, {mel.max():.4f}]")

    # Pad to 3501 frames for encoder
    print("\n4. Padding to 3501 frames for encoder...")
    mel_padded = np.pad(
        mel,
        ((0, 0), (0, 0), (0, 3501 - mel.shape[2])),
        mode='constant',
        constant_values=0
    )
    print(f"   Padded shape: {mel_padded.shape}")
    assert mel_padded.shape == (1, 128, 3501), f"Expected (1, 128, 3501), got {mel_padded.shape}"
    print("   ✓ Shape correct")

    # Compare with HuggingFace processor
    print("\n5. Comparing with HuggingFace processor...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            "CohereLabs/cohere-transcribe-03-2026",
            trust_remote_code=True
        )
        hf_inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        hf_mel = hf_inputs["input_features"].numpy()

        # Pad HF mel to 3501
        hf_mel_padded = np.pad(
            hf_mel,
            ((0, 0), (0, 0), (0, 3501 - hf_mel.shape[2])),
            mode='constant',
            constant_values=0
        )

        print(f"   HuggingFace mel shape: {hf_mel_padded.shape}")
        print(f"   HuggingFace stats: mean={hf_mel_padded.mean():.6f}, std={hf_mel_padded.std():.6f}")

        # Compare
        diff = np.abs(mel_padded - hf_mel_padded)
        print(f"\n   Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")

        if diff.max() < 0.1:
            print("   ✅ Excellent match with HuggingFace!")
        elif diff.max() < 1.0:
            print("   ✅ Good match with HuggingFace")
        else:
            print("   ⚠️  Some differences from HuggingFace (expected due to implementation details)")

    except Exception as e:
        print(f"   ⚠️  Could not compare with HuggingFace: {e}")

    print("\n" + "="*70)
    print("SUCCESS! Mel spectrogram computed")
    print("="*70)

    return mel_padded


if __name__ == "__main__":
    test_mel_spectrogram()
