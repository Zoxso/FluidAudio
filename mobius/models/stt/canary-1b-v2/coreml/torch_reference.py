#!/usr/bin/env python3
"""PyTorch reference decode for Canary-1B v2 to compare against CoreML outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr


def _prepare_audio(audio_path: Path, sample_rate: int, max_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    data, sr = sf.read(str(audio_path), dtype="float32")
    if sr != sample_rate:
        raise RuntimeError(f"Sample rate mismatch: expected {sample_rate}, got {sr}")
    if data.ndim > 1:
        data = data[:, 0]
    max_samples = int(round(sample_rate * max_seconds))
    if data.size < max_samples:
        data = np.pad(data, (0, max_samples - data.size))
    elif data.size > max_samples:
        data = data[:max_samples]
    audio = data.reshape(1, -1)
    audio_len = np.array([max_samples], dtype=np.int32)
    return audio, audio_len


def greedy_decode(asr_model, encoder_outputs: torch.Tensor, encoder_lengths: torch.Tensor, max_steps: int) -> list[int]:
    """Greedy decode using the NeMo transformer decoder."""
    if not hasattr(asr_model, "transf_decoder"):
        raise RuntimeError("Model has no transformer decoder")

    # Prompt tokens (mirror NeMo transcribe prompt - canary2 format)
    # <|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
    # IDs: 16053 (prefix), 7, 4, 16, 64, 64, 5, 9, 11, 13
    tokens = [16053, 7, 4, 16, 64, 64, 5, 9, 11, 13]
    decoder_seq_length = max_steps
    input_ids = torch.zeros((1, decoder_seq_length), dtype=torch.int64)
    decoder_mask = torch.zeros((1, decoder_seq_length), dtype=torch.bool)
    max_fill = min(decoder_seq_length, len(tokens))
    input_ids[0, :max_fill] = torch.tensor(tokens[:max_fill], dtype=torch.int64)
    decoder_mask[0, :max_fill] = True
    current_pos = max_fill

    encoder_embeddings = encoder_outputs.transpose(1, 2)
    encoder_mask = torch.ones(encoder_embeddings.shape[0], encoder_embeddings.shape[1], dtype=torch.bool)
    for batch in range(encoder_mask.shape[0]):
        if encoder_lengths[batch] < encoder_embeddings.shape[1]:
            encoder_mask[batch, encoder_lengths[batch] :] = False

    generated = tokens[:max_fill]
    with torch.no_grad():
        for step in range(max_fill, max_steps):
            decoder_out = asr_model.transf_decoder(
                input_ids=input_ids,
                decoder_mask=decoder_mask,
                encoder_embeddings=encoder_embeddings,
                encoder_mask=encoder_mask,
                decoder_mems=None,
            )
            hidden_states = decoder_out[0, step - 1, :]
            
            # Project to logits using the beam search projection layer
            # We need to find it first.
            if not hasattr(asr_model, "_projection_layer"):
                 # Find it once
                 beam_search = asr_model.decoding.decoding.beam_search
                 if hasattr(beam_search, "project"):
                     asr_model._projection_layer = beam_search.project
                 elif hasattr(beam_search.decoder, "project"):
                     asr_model._projection_layer = beam_search.decoder.project
                 elif hasattr(beam_search.decoder, "output_layer"):
                     asr_model._projection_layer = beam_search.decoder.output_layer
                 else:
                     # Search modules
                     for module in asr_model.modules():
                         if isinstance(module, torch.nn.Linear):
                             if module.out_features > 16000 and module.out_features < 17000:
                                 asr_model._projection_layer = module
                                 break
            
            logits = asr_model._projection_layer(hidden_states)
            next_token = int(torch.argmax(logits).item())
            generated.append(next_token)
            input_ids[0, step] = next_token
            decoder_mask[0, step] = True
            if next_token == asr_model.tokenizer.eos_id:
                break
    return generated


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch reference greedy decode for Canary-1B v2.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to 16 kHz mono wav")
    parser.add_argument("--max-seconds", type=float, default=15.0, help="Window length to trim/pad audio")
    parser.add_argument("--max-steps", type=int, default=256, help="Maximum decoder steps")
    parser.add_argument("--model-id", type=str, default="nvidia/canary-1b-v2", help="NeMo model id")
    args = parser.parse_args()

    asr_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(args.model_id, map_location="cpu")
    asr_model.eval()
    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)

    audio_np, audio_len_np = _prepare_audio(args.audio, sample_rate, args.max_seconds)
    audio_t = torch.from_numpy(audio_np).to(dtype=torch.float32)
    audio_len_t = torch.from_numpy(audio_len_np)

    with torch.no_grad():
        mel, mel_len = asr_model.preprocessor(input_signal=audio_t, length=audio_len_t)
        enc, enc_len = asr_model.encoder(audio_signal=mel, length=mel_len)
        tokens = greedy_decode(asr_model, enc, enc_len, args.max_steps)
        text = asr_model.tokenizer.ids_to_text(tokens)

    print("Reference transcript (greedy):")
    print(text)
    print("\nGenerated tokens:", tokens)
    print("\nToken count:", len(tokens))


if __name__ == "__main__":
    main()
