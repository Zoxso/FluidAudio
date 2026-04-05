#!/usr/bin/env python3
"""PyTorch streaming WER benchmark - preprocess FULL audio, then chunk mel.

This validates whether the approach "preprocess full audio, chunk mel" works
with NeMo's native streaming encoder.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from jiwer import wer

import nemo.collections.asr as nemo_asr

sys.path.insert(0, '.')
from conversion_scripts.individual_components import PreprocessorWrapper, EncoderStreamingWrapper, JointDecisionSingleStep, DecoderWrapper, JointWrapper


def tokens_to_text(tokens: list, tokenizer) -> str:
    """Convert token IDs to text using NeMo tokenizer."""
    return tokenizer.ids_to_text(tokens)


def greedy_decode_streaming(encoder_out: torch.Tensor, decoder_wrapper, joint_wrapper,
                           blank_id: int, h: torch.Tensor, c: torch.Tensor,
                           dec_hidden: torch.Tensor) -> tuple:
    """Greedy RNNT decoding for a single chunk, maintaining decoder state."""
    T = encoder_out.shape[2]
    tokens = []
    t = 0
    max_symbols_per_step = 10

    while t < T:
        enc_frame = encoder_out[:, :, t:t+1]
        symbols_emitted = 0

        while symbols_emitted < max_symbols_per_step:
            with torch.no_grad():
                token_ids, _, _, _ = joint_wrapper(enc_frame, dec_hidden)
            label = int(token_ids.view(-1)[0].item())

            if label == blank_id:
                t += 1
                break
            else:
                tokens.append(label)
                symbols_emitted += 1

                with torch.no_grad():
                    targets = torch.tensor([[label]], dtype=torch.int32)
                    dec_hidden_new, h, c = decoder_wrapper(
                        targets,
                        torch.tensor([1], dtype=torch.int32),
                        h, c
                    )
                    dec_hidden = dec_hidden_new

    return tokens, h, c, dec_hidden


def normalize_text(text: str) -> str:
    """Normalize text for WER computation."""
    text = text.replace("<EOU>", "")
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = " ".join(text.split())
    return text.strip()


def transcribe_pytorch_full_preprocess(audio: np.ndarray, preprocessor,
                                       encoder_wrapper, decoder_wrapper, joint_wrapper,
                                       tokenizer, encoder, blank_id: int,
                                       chunk_mel_frames: int = 17) -> str:
    """Transcribe audio using PyTorch with FULL audio preprocessing, then chunking mel.

    Key difference from chunk-by-chunk: preprocess full audio first, then chunk mel.
    """
    audio = audio.astype(np.float32)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    # STEP 1: Preprocess FULL audio at once
    audio_len = torch.tensor([len(audio)], dtype=torch.long)
    with torch.no_grad():
        full_mel, full_mel_len = preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )
        # full_mel: [1, mel_dim, T_mel]
        full_mel_len = int(full_mel_len[0])

    # Get initial cache from encoder
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        encoder.get_initial_cache_state(batch_size=1, device='cpu')
    )
    # Transpose to [B, L, ...] format
    cache_channel = cache_last_channel.transpose(0, 1)
    cache_time = cache_last_time.transpose(0, 1)
    cache_len = cache_last_channel_len.to(torch.int32)

    # Initialize decoder states
    D_dec = 640
    h = torch.zeros((1, 1, D_dec), dtype=torch.float32)
    c = torch.zeros((1, 1, D_dec), dtype=torch.float32)

    # Initialize decoder with blank
    with torch.no_grad():
        targets = torch.tensor([[blank_id]], dtype=torch.int32)
        dec_hidden, h, c = decoder_wrapper(
            targets,
            torch.tensor([1], dtype=torch.int32),
            h, c
        )

    all_tokens = []

    # STEP 2: Chunk mel features (not raw audio)
    num_chunks = (full_mel_len + chunk_mel_frames - 1) // chunk_mel_frames

    for i in range(num_chunks):
        start_frame = i * chunk_mel_frames
        end_frame = min(start_frame + chunk_mel_frames, full_mel_len)
        mel_chunk = full_mel[:, :, start_frame:end_frame]

        # Pad if needed
        if mel_chunk.shape[2] < chunk_mel_frames:
            pad_width = chunk_mel_frames - mel_chunk.shape[2]
            mel_chunk = torch.nn.functional.pad(mel_chunk, (0, pad_width))

        mel_len = torch.tensor([chunk_mel_frames], dtype=torch.int32)

        # Run streaming encoder with cache
        with torch.no_grad():
            enc_out, enc_len, cache_channel, cache_time, cache_len = encoder_wrapper(
                mel_chunk, mel_len, cache_channel, cache_time, cache_len
            )

        # Decode this chunk
        tokens, h, c, dec_hidden = greedy_decode_streaming(
            enc_out, decoder_wrapper, joint_wrapper, blank_id, h, c, dec_hidden
        )
        all_tokens.extend(tokens)

    # Convert to text
    text = tokens_to_text(all_tokens, tokenizer)
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--chunk-mel-frames", type=int, default=17)
    args = parser.parse_args()

    print("Loading Parakeet EOU model...")
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        "nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu"
    )
    asr_model.eval()

    # Set up streaming
    encoder = asr_model.encoder
    encoder.setup_streaming_params()

    # Get preprocessor directly (not wrapped)
    preprocessor = asr_model.preprocessor.eval()

    # Create wrappers for encoder/decoder/joint
    encoder_wrapper = EncoderStreamingWrapper(encoder.eval())
    decoder_wrapper = DecoderWrapper(asr_model.decoder.eval())
    joint_base = JointWrapper(asr_model.joint.eval())
    vocab_size = asr_model.tokenizer.vocab_size
    joint_wrapper = JointDecisionSingleStep(joint_base, vocab_size)

    tokenizer = asr_model.tokenizer
    blank_id = asr_model.decoder.blank_idx
    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)

    print(f"Sample rate: {sample_rate}")
    print(f"Mel chunk size: {args.chunk_mel_frames} frames")
    print("PyTorch streaming: FULL audio preprocessing, then chunking mel")

    print(f"Loading LibriSpeech test-clean (first {args.num_files} files)...")
    dataset = load_dataset("librispeech_asr", "clean", split="test")

    filtered = []
    for item in dataset:
        filtered.append(item)
        if len(filtered) >= args.num_files:
            break

    print(f"Using {len(filtered)} files")

    hypotheses = []
    references = []

    for i, item in enumerate(filtered):
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]
        reference = item["text"]

        # Resample if needed
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Transcribe using PyTorch streaming with full preprocessing
        hypothesis = transcribe_pytorch_full_preprocess(
            audio, preprocessor, encoder_wrapper,
            decoder_wrapper, joint_wrapper, tokenizer, encoder, blank_id,
            args.chunk_mel_frames
        )

        # Normalize
        hyp_norm = normalize_text(hypothesis)
        ref_norm = normalize_text(reference)

        hypotheses.append(hyp_norm)
        references.append(ref_norm)

        if (i + 1) % 10 == 0:
            current_wer = wer(references, hypotheses) * 100
            audio_duration = len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
            print(f"[{i+1}/{len(filtered)}] Running WER: {current_wer:.2f}% (last: {audio_duration:.1f}s)")
            print(f"  REF: {ref_norm[:80]}...")
            print(f"  HYP: {hyp_norm[:80]}...")

    # Final WER
    final_wer = wer(references, hypotheses) * 100
    print(f"\n{'='*50}")
    print(f"PyTorch Full Preprocess Streaming WER on {len(filtered)} files: {final_wer:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
