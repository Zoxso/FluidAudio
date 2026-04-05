#!/usr/bin/env python3
"""CLI for exporting Parakeet Realtime EOU 120M components to CoreML.

This model is a cache-aware streaming FastConformer-RNNT designed for
end-of-utterance detection with low latency.

Key differences from TDT:
- Cache-aware encoder for streaming (requires encoder cache states)
- Outputs EOU token for end-of-utterance detection
- No duration outputs (unlike TDT)
- 120M parameters (smaller than 0.6B TDT)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import (
    DecoderWrapper,
    EncoderInitialWrapper,
    EncoderStreamingWrapper,
    ExportSettings,
    JointWrapper,
    JointDecisionWrapper,
    JointDecisionSingleStep,
    PreprocessorWrapper,
    MelEncoderWrapper,
    MelEncoderStreamingWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "nvidia/parakeet_realtime_eou_120m-v1"
AUTHOR = "Fluid Inference"

# NeMo streaming configuration (from encoder.streaming_cfg)
# chunk_size=[9, 16] - First chunk 9 mel frames, subsequent 16 mel frames
# pre_encode_cache_size=[0, 9] - 9 cached mel frames prepended to subsequent chunks
# valid_out_len=2 - Only 2 encoder frames valid per chunk
#
# IMPORTANT: CoreML preprocessor produces different mel frame counts than expected:
# - 1440 samples (9*160) → 10 mel frames (not 9)
# - 2560 samples (16*160) → 17 mel frames (not 16)
# We must trace encoders with ACTUAL CoreML preprocessor output shapes.
FIRST_CHUNK_SAMPLES = 1440   # 90ms at 16kHz
SUBSEQUENT_CHUNK_SAMPLES = 2560  # 160ms at 16kHz
PRE_ENCODE_CACHE_SIZE = 9  # mel frames cached for subsequent chunks
VALID_OUT_LEN = 2


def _compute_length(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def _prepare_audio(
    validation_audio: Optional[Path],
    sample_rate: int,
    max_samples: int,
    seed: Optional[int],
) -> torch.Tensor:
    if validation_audio is None:
        if seed is not None:
            torch.manual_seed(seed)
        audio = torch.randn(1, max_samples, dtype=torch.float32)
        return audio

    data, sr = sf.read(str(validation_audio), dtype="float32")
    if sr != sample_rate:
        raise typer.BadParameter(
            f"Validation audio sample rate {sr} does not match model rate {sample_rate}"
        )

    if data.ndim > 1:
        data = data[:, 0]

    if data.size == 0:
        raise typer.BadParameter("Validation audio is empty")

    if data.size < max_samples:
        pad_width = max_samples - data.size
        data = np.pad(data, (0, pad_width))
    elif data.size > max_samples:
        data = data[:max_samples]

    audio = torch.from_numpy(data).unsqueeze(0).to(dtype=torch.float32)
    return audio


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _parse_compute_units(name: str) -> ct.ComputeUnit:
    normalized = str(name).strip().upper()
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_NEURALENGINE": ct.ComputeUnit.CPU_AND_NE,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute units '{name}'. Choose from: " + ", ".join(mapping.keys())
        )
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    if name is None:
        return None
    normalized = str(name).strip().upper()
    if normalized == "":
        return None
    mapping = {
        "FLOAT32": ct.precision.FLOAT32,
        "FLOAT16": ct.precision.FLOAT16,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute precision '{name}'. Choose from: "
            + ", ".join(mapping.keys())
        )
    return mapping[normalized]


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    nemo_path: Optional[Path] = typer.Option(
        None,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to parakeet_realtime_eou_120m .nemo checkpoint (skip to auto-download)",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Model identifier to download when --nemo-path is omitted",
    ),
    output_dir: Path = typer.Option(
        Path("parakeet_eou_coreml"),
        help="Directory where mlpackages and metadata will be written",
    ),
    preprocessor_cu: str = typer.Option(
        "CPU_ONLY",
        "--preprocessor-cu",
        help="Compute units for preprocessor (default CPU_ONLY)",
    ),
    encoder_cu: str = typer.Option(
        "CPU_ONLY",
        "--encoder-cu",
        help="Compute units for encoder (default CPU_ONLY)",
    ),
    compute_precision: Optional[str] = typer.Option(
        None,
        "--compute-precision",
        help="Export precision: FLOAT32 (default) or FLOAT16.",
    ),
    chunk_seconds: float = typer.Option(
        0.16,
        "--chunk-seconds",
        help="Chunk size in seconds for streaming (default 0.16s = 160ms)",
    ),
    max_audio_seconds: float = typer.Option(
        2.0,
        "--max-audio-seconds",
        help="Maximum audio length in seconds for initial encoder (default 2.0s)",
    ),
) -> None:
    """Export all Parakeet EOU sub-modules to CoreML for streaming inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_cu = _parse_compute_units(preprocessor_cu)
    enc_cu = _parse_compute_units(encoder_cu)

    if nemo_path is not None:
        typer.echo(f"Loading NeMo model from {nemo_path}...")
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            str(nemo_path), map_location="cpu"
        )
        checkpoint_meta = {
            "type": "file",
            "path": str(nemo_path),
        }
    else:
        typer.echo(f"Downloading NeMo model via {model_id}...")
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_id, map_location="cpu"
        )
        checkpoint_meta = {
            "type": "pretrained",
            "model_id": model_id,
        }
    asr_model.eval()

    # Get model configuration
    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = _compute_length(max_audio_seconds, sample_rate)
    chunk_samples = _compute_length(chunk_seconds, sample_rate)
    
    # Update global constants dynamically
    # We use the same size for first and subsequent chunks for simplicity in this custom export
    FIRST_CHUNK_SAMPLES = chunk_samples
    SUBSEQUENT_CHUNK_SAMPLES = chunk_samples

    # Get encoder streaming configuration
    encoder = asr_model.encoder
    encoder_cfg = asr_model.cfg.encoder

    # Get cache sizes from streaming config
    # For EOU model: att_context_size = [70, 1] means 70 frames left context
    att_context_size = encoder_cfg.get("att_context_size", [-1, -1])
    if isinstance(att_context_size, list) and len(att_context_size) >= 2:
        left_context = att_context_size[0] if att_context_size[0] >= 0 else 70
    else:
        left_context = 70  # Default for EOU model

    # Get model dimensions
    d_model = int(encoder_cfg.d_model)
    n_layers = int(encoder_cfg.n_layers)
    subsampling_factor = int(encoder_cfg.get("subsampling_factor", 4))

    # Convolution context size
    conv_kernel_size = int(encoder_cfg.get("conv_kernel_size", 31))
    conv_context_size = (conv_kernel_size - 1) // 2

    typer.echo(f"Model config:")
    typer.echo(f"  Sample rate: {sample_rate}")
    typer.echo(f"  Attention context: {att_context_size}")
    typer.echo(f"  D_model: {d_model}")
    typer.echo(f"  N_layers: {n_layers}")
    typer.echo(f"  Subsampling factor: {subsampling_factor}")
    typer.echo(f"  Conv context size: {conv_context_size}")

    # Setup streaming configuration for the encoder
    encoder.setup_streaming_params()

    # Get initial cache state to determine shapes
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        encoder.get_initial_cache_state(batch_size=1, device="cpu")
    )
    # Convert cache_len to int32 for CoreML compatibility
    cache_last_channel_len = cache_last_channel_len.to(dtype=torch.int32)
    cache_size = cache_last_channel.shape[2]  # [L, B, cache_size, d_model]

    typer.echo(f"Cache shapes:")
    typer.echo(f"  cache_last_channel: {cache_last_channel.shape}")
    typer.echo(f"  cache_last_time: {cache_last_time.shape}")
    typer.echo(f"  cache_last_channel_len: {cache_last_channel_len.shape}")

    # Calculate frame counts
    chunk_frames = chunk_samples // (
        sample_rate // 100
    )  # Approximate mel frames at 10ms stride
    chunk_frames_subsampled = chunk_frames // subsampling_factor

    export_settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=_parse_compute_precision(compute_precision),
        max_audio_seconds=max_audio_seconds,
        max_symbol_steps=1,
        chunk_size_frames=chunk_frames_subsampled,
        cache_size=cache_size,
    )

    typer.echo(f"Export configuration:")
    typer.echo(f"  Chunk seconds: {chunk_seconds}")
    typer.echo(f"  Chunk samples: {chunk_samples}")
    typer.echo(f"  Cache size: {cache_size}")

    # Prepare test audio
    typer.echo("Preparing test audio...")
    audio_tensor = _prepare_audio(None, sample_rate, max_samples, seed=42)
    audio_length = torch.tensor([max_samples], dtype=torch.int32)

    # Create wrappers
    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder_initial = EncoderInitialWrapper(encoder.eval())
    encoder_streaming = EncoderStreamingWrapper(encoder.eval())
    decoder = DecoderWrapper(asr_model.decoder.eval())
    joint = JointWrapper(asr_model.joint.eval())

    decoder_export_flag = getattr(asr_model.decoder, "_rnnt_export", False)
    asr_model.decoder._rnnt_export = True

    try:
        with torch.no_grad():
            # Run preprocessor
            mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)
            mel_length_ref = mel_length_ref.to(dtype=torch.int32)

            # Run initial encoder (no cache)
            encoder_ref, encoder_length_ref = encoder_initial(mel_ref, mel_length_ref)
            encoder_length_ref = encoder_length_ref.to(dtype=torch.int32)

            typer.echo(f"Reference shapes:")
            typer.echo(f"  mel: {mel_ref.shape}")
            typer.echo(f"  encoder: {encoder_ref.shape}")

            # Clone tensors
            mel_ref = mel_ref.clone().detach()
            mel_length_ref = mel_length_ref.clone().detach()
            encoder_ref = encoder_ref.clone().detach()
            encoder_length_ref = encoder_length_ref.clone().detach()

        vocab_size = int(asr_model.tokenizer.vocab_size)
        decoder_hidden = int(asr_model.decoder.pred_hidden)
        decoder_layers = int(asr_model.decoder.pred_rnn_layers)

        targets = torch.full(
            (1, 1),
            fill_value=asr_model.decoder.blank_idx,
            dtype=torch.int32,
        )
        target_lengths = torch.tensor([1], dtype=torch.int32)
        zero_state = torch.zeros(
            decoder_layers,
            1,
            decoder_hidden,
            dtype=torch.float32,
        )

        with torch.no_grad():
            decoder_ref, h_ref, c_ref = decoder(
                targets, target_lengths, zero_state, zero_state
            )
            joint_ref = joint(encoder_ref, decoder_ref)

        decoder_ref = decoder_ref.clone()
        joint_ref = joint_ref.clone()

        # === Export Preprocessor ===
        typer.echo("Tracing and converting preprocessor...")
        preprocessor = preprocessor.cpu()
        audio_tensor = audio_tensor.cpu()
        audio_length = audio_length.cpu()
        traced_preprocessor = torch.jit.trace(
            preprocessor, (audio_tensor, audio_length), strict=False
        )
        traced_preprocessor.eval()
        preprocessor_inputs = [
            ct.TensorType(
                name="audio_signal",
                shape=(1, ct.RangeDim(1, max_samples)),
                dtype=np.float32,
            ),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ]
        preprocessor_outputs = [
            ct.TensorType(name="mel", dtype=np.float32),
            ct.TensorType(name="mel_length", dtype=np.int32),
        ]
        preprocessor_model = _coreml_convert(
            traced_preprocessor,
            preprocessor_inputs,
            preprocessor_outputs,
            export_settings,
            compute_units_override=pre_cu,
        )
        preprocessor_path = output_dir / "parakeet_eou_preprocessor.mlpackage"
        _save_mlpackage(
            preprocessor_model,
            preprocessor_path,
            "Parakeet EOU preprocessor",
        )

        # === Export Initial Encoder (no cache) ===
        typer.echo("Tracing and converting initial encoder (no cache)...")
        traced_encoder_initial = torch.jit.trace(
            encoder_initial, (mel_ref, mel_length_ref), strict=False
        )
        traced_encoder_initial.eval()
        encoder_initial_inputs = [
            ct.TensorType(
                name="mel", shape=_tensor_shape(mel_ref), dtype=np.float32
            ),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ]
        encoder_initial_outputs = [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
        ]
        encoder_initial_model = _coreml_convert(
            traced_encoder_initial,
            encoder_initial_inputs,
            encoder_initial_outputs,
            export_settings,
            compute_units_override=enc_cu,
        )
        encoder_initial_path = output_dir / "parakeet_eou_encoder_initial.mlpackage"
        _save_mlpackage(
            encoder_initial_model,
            encoder_initial_path,
            "Parakeet EOU encoder (initial, no cache)",
        )

        # === Export Streaming Encoder for First Chunk ===
        # NOTE: CoreML preprocessor gives 10 mel frames for 1440 samples (not 9!)
        # We must trace the encoder with the ACTUAL mel shape that CoreML preprocessor outputs.
        typer.echo("Tracing and converting streaming encoder for FIRST chunk...")

        # Prepare cache tensors in [B, L, ...] format
        cache_channel_b = cache_last_channel.transpose(0, 1)  # [B, L, cache, d_model]
        cache_time_b = cache_last_time.transpose(0, 1)  # [B, L, d_model, conv_ctx]

        # First chunk: 1440 samples → preprocessor outputs actual mel frame count
        first_chunk_audio = _prepare_audio(None, sample_rate, FIRST_CHUNK_SAMPLES, seed=43)
        first_chunk_length = torch.tensor([FIRST_CHUNK_SAMPLES], dtype=torch.int32)
        with torch.no_grad():
            first_chunk_mel, first_chunk_mel_len = preprocessor(first_chunk_audio, first_chunk_length)
            first_chunk_mel_len = first_chunk_mel_len.to(dtype=torch.int32)

        first_chunk_actual_mel_frames = first_chunk_mel.shape[2]
        typer.echo(f"  First chunk: {FIRST_CHUNK_SAMPLES} samples → {first_chunk_actual_mel_frames} mel frames")

        traced_encoder_first = torch.jit.trace(
            encoder_streaming,
            (
                first_chunk_mel,
                first_chunk_mel_len,
                cache_channel_b,
                cache_time_b,
                cache_last_channel_len,
            ),
            strict=False,
        )
        traced_encoder_first.eval()

        encoder_first_inputs = [
            ct.TensorType(
                name="mel", shape=_tensor_shape(first_chunk_mel), dtype=np.float32
            ),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
            ct.TensorType(
                name="cache_last_channel",
                shape=_tensor_shape(cache_channel_b),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_last_time",
                shape=_tensor_shape(cache_time_b),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_last_channel_len", shape=(1,), dtype=np.int32
            ),
        ]
        encoder_streaming_outputs = [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
            ct.TensorType(name="cache_last_channel_next", dtype=np.float32),
            ct.TensorType(name="cache_last_time_next", dtype=np.float32),
            ct.TensorType(name="cache_last_channel_len_next", dtype=np.int32),
        ]
        encoder_first_model = _coreml_convert(
            traced_encoder_first,
            encoder_first_inputs,
            encoder_streaming_outputs,
            export_settings,
            compute_units_override=enc_cu,
        )
        encoder_first_path = (
            output_dir / "parakeet_eou_encoder_streaming_first.mlpackage"
        )
        _save_mlpackage(
            encoder_first_model,
            encoder_first_path,
            f"Parakeet EOU encoder (streaming, first chunk - {first_chunk_actual_mel_frames} mel frames)",
        )

        # === Export Streaming Encoder for Subsequent Chunks ===
        # NOTE: CoreML preprocessor gives 17 mel frames for 2560 samples (not 16!)
        # Plus 9 mel frames from cache = 26 total (not 25!)
        typer.echo("Tracing and converting streaming encoder for SUBSEQUENT chunks...")

        # Subsequent chunks: 2560 samples → preprocessor outputs 17 mel frames
        # Plus pre_encode_cache = 9 frames = 26 total mel frames
        subsequent_chunk_audio = _prepare_audio(None, sample_rate, SUBSEQUENT_CHUNK_SAMPLES, seed=44)
        subsequent_chunk_length = torch.tensor([SUBSEQUENT_CHUNK_SAMPLES], dtype=torch.int32)
        with torch.no_grad():
            subsequent_new_mel, subsequent_new_mel_len = preprocessor(subsequent_chunk_audio, subsequent_chunk_length)
            subsequent_new_mel_len = subsequent_new_mel_len.to(dtype=torch.int32)

        subsequent_new_mel_frames = subsequent_new_mel.shape[2]
        # Create the full subsequent mel by prepending the cache (last 9 frames from first chunk)
        mel_cache_for_subsequent = first_chunk_mel[:, :, -PRE_ENCODE_CACHE_SIZE:]
        subsequent_chunk_mel = torch.cat([mel_cache_for_subsequent, subsequent_new_mel], dim=2)
        subsequent_total_mel_frames = subsequent_chunk_mel.shape[2]
        subsequent_chunk_mel_len = torch.tensor([subsequent_total_mel_frames], dtype=torch.int32)

        typer.echo(f"  Subsequent chunk: {SUBSEQUENT_CHUNK_SAMPLES} samples → {subsequent_new_mel_frames} new mel frames + {PRE_ENCODE_CACHE_SIZE} cached = {subsequent_total_mel_frames} total")

        traced_encoder_streaming = torch.jit.trace(
            encoder_streaming,
            (
                subsequent_chunk_mel,
                subsequent_chunk_mel_len,
                cache_channel_b,
                cache_time_b,
                cache_last_channel_len,
            ),
            strict=False,
        )
        traced_encoder_streaming.eval()

        encoder_streaming_inputs = [
            ct.TensorType(
                name="mel", shape=_tensor_shape(subsequent_chunk_mel), dtype=np.float32
            ),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
            ct.TensorType(
                name="cache_last_channel",
                shape=_tensor_shape(cache_channel_b),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_last_time",
                shape=_tensor_shape(cache_time_b),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_last_channel_len", shape=(1,), dtype=np.int32
            ),
        ]
        encoder_streaming_model = _coreml_convert(
            traced_encoder_streaming,
            encoder_streaming_inputs,
            encoder_streaming_outputs,
            export_settings,
            compute_units_override=enc_cu,
        )
        encoder_streaming_path = (
            output_dir / "parakeet_eou_encoder_streaming.mlpackage"
        )
        _save_mlpackage(
            encoder_streaming_model,
            encoder_streaming_path,
            f"Parakeet EOU encoder (streaming, subsequent chunks - {subsequent_total_mel_frames} mel frames)",
        )

        # === Export Decoder ===
        typer.echo("Tracing and converting decoder...")
        traced_decoder = torch.jit.trace(
            decoder,
            (targets, target_lengths, zero_state, zero_state),
            strict=False,
        )
        traced_decoder.eval()
        decoder_inputs = [
            ct.TensorType(
                name="targets", shape=_tensor_shape(targets), dtype=np.int32
            ),
            ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
            ct.TensorType(
                name="h_in", shape=_tensor_shape(zero_state), dtype=np.float32
            ),
            ct.TensorType(
                name="c_in", shape=_tensor_shape(zero_state), dtype=np.float32
            ),
        ]
        decoder_outputs = [
            ct.TensorType(name="decoder", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ]
        decoder_model = _coreml_convert(
            traced_decoder,
            decoder_inputs,
            decoder_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        decoder_path = output_dir / "parakeet_eou_decoder.mlpackage"
        _save_mlpackage(
            decoder_model,
            decoder_path,
            "Parakeet EOU decoder (RNNT prediction network)",
        )

        # === Export Joint ===
        typer.echo("Tracing and converting joint...")
        traced_joint = torch.jit.trace(
            joint,
            (encoder_ref, decoder_ref),
            strict=False,
        )
        traced_joint.eval()
        joint_inputs = [
            ct.TensorType(
                name="encoder", shape=_tensor_shape(encoder_ref), dtype=np.float32
            ),
            ct.TensorType(
                name="decoder", shape=_tensor_shape(decoder_ref), dtype=np.float32
            ),
        ]
        joint_outputs = [
            ct.TensorType(name="logits", dtype=np.float32),
        ]
        joint_model = _coreml_convert(
            traced_joint,
            joint_inputs,
            joint_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        joint_path = output_dir / "parakeet_eou_joint.mlpackage"
        _save_mlpackage(
            joint_model,
            joint_path,
            "Parakeet EOU joint network (RNNT)",
        )

        # === Export Joint Decision ===
        typer.echo("Tracing and converting joint decision head...")
        joint_decision = JointDecisionWrapper(joint, vocab_size=vocab_size)
        traced_joint_decision = torch.jit.trace(
            joint_decision,
            (encoder_ref, decoder_ref),
            strict=False,
        )
        traced_joint_decision.eval()
        joint_decision_inputs = [
            ct.TensorType(
                name="encoder", shape=_tensor_shape(encoder_ref), dtype=np.float32
            ),
            ct.TensorType(
                name="decoder", shape=_tensor_shape(decoder_ref), dtype=np.float32
            ),
        ]
        joint_decision_outputs = [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
        ]
        joint_decision_model = _coreml_convert(
            traced_joint_decision,
            joint_decision_inputs,
            joint_decision_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        joint_decision_path = output_dir / "parakeet_eou_joint_decision.mlpackage"
        _save_mlpackage(
            joint_decision_model,
            joint_decision_path,
            "Parakeet EOU joint + decision head",
        )

        # === Export Single-Step Joint Decision ===
        typer.echo("Tracing and converting single-step joint decision...")
        jd_single = JointDecisionSingleStep(joint, vocab_size=vocab_size)
        enc_step = encoder_ref[:, :, :1].contiguous()
        dec_step = decoder_ref[:, :, :1].contiguous()
        traced_jd_single = torch.jit.trace(
            jd_single,
            (enc_step, dec_step),
            strict=False,
        )
        traced_jd_single.eval()
        jd_single_inputs = [
            ct.TensorType(
                name="encoder_step", shape=(1, enc_step.shape[1], 1), dtype=np.float32
            ),
            ct.TensorType(
                name="decoder_step", shape=(1, dec_step.shape[1], 1), dtype=np.float32
            ),
        ]
        jd_single_outputs = [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
            ct.TensorType(name="top_k_ids", dtype=np.int32),
            ct.TensorType(name="top_k_logits", dtype=np.float32),
        ]
        jd_single_model = _coreml_convert(
            traced_jd_single,
            jd_single_inputs,
            jd_single_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        jd_single_path = output_dir / "parakeet_eou_joint_decision_single_step.mlpackage"
        _save_mlpackage(
            jd_single_model,
            jd_single_path,
            "Parakeet EOU single-step joint decision",
        )

        # === Save Metadata ===
        metadata: Dict[str, object] = {
            "model_id": model_id,
            "model_type": "parakeet_realtime_eou",
            "sample_rate": sample_rate,
            "max_audio_seconds": max_audio_seconds,
            "max_audio_samples": max_samples,
            "chunk_seconds": chunk_seconds,
            "chunk_samples": chunk_samples,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "subsampling_factor": subsampling_factor,
            "att_context_size": list(att_context_size) if hasattr(att_context_size, '__iter__') else att_context_size,
            "conv_context_size": conv_context_size,
            "decoder": {
                "hidden_size": decoder_hidden,
                "num_layers": decoder_layers,
                "blank_idx": int(asr_model.decoder.blank_idx),
            },
            "cache": {
                "cache_last_channel_shape": list(cache_channel_b.shape),
                "cache_last_time_shape": list(cache_time_b.shape),
                "cache_size": cache_size,
            },
            "streaming": {
                "first_chunk_mel_frames": first_chunk_actual_mel_frames,
                "subsequent_new_mel_frames": subsequent_new_mel_frames,
                "pre_encode_cache_size": PRE_ENCODE_CACHE_SIZE,
                "subsequent_total_mel_frames": subsequent_total_mel_frames,
                "valid_out_len": VALID_OUT_LEN,
                "first_chunk_samples": FIRST_CHUNK_SAMPLES,
                "subsequent_chunk_samples": SUBSEQUENT_CHUNK_SAMPLES,
            },
            "checkpoint": checkpoint_meta,
            "coreml": {
                "compute_units": export_settings.compute_units.name,
                "compute_precision": (
                    export_settings.compute_precision.name
                    if export_settings.compute_precision is not None
                    else "FLOAT32"
                ),
            },
            "components": {
                "preprocessor": {
                    "path": preprocessor_path.name,
                },
                "encoder_initial": {
                    "path": encoder_initial_path.name,
                    "description": "Encoder without cache (for batch inference)",
                },
                "encoder_streaming_first": {
                    "path": encoder_first_path.name,
                    "description": f"Streaming encoder for first chunk ({first_chunk_actual_mel_frames} mel frames)",
                    "mel_frames": first_chunk_actual_mel_frames,
                },
                "encoder_streaming": {
                    "path": encoder_streaming_path.name,
                    "description": f"Streaming encoder for subsequent chunks ({subsequent_total_mel_frames} mel frames = {PRE_ENCODE_CACHE_SIZE} cached + {subsequent_new_mel_frames} new)",
                    "mel_frames": subsequent_total_mel_frames,
                },
                "decoder": {
                    "path": decoder_path.name,
                },
                "joint": {
                    "path": joint_path.name,
                },
                "joint_decision": {
                    "path": joint_decision_path.name,
                },
                "joint_decision_single_step": {
                    "path": jd_single_path.name,
                },
            },
        }

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        typer.echo(f"Export complete. Metadata written to {metadata_path}")

        # Save tokenizer
        tokenizer_path = output_dir / "tokenizer.json"
        tokenizer_vocab = {
            str(i): asr_model.tokenizer.ids_to_tokens([i])[0]
            for i in range(vocab_size)
        }
        tokenizer_path.write_text(json.dumps(tokenizer_vocab, indent=2))
        typer.echo(f"Tokenizer saved to {tokenizer_path}")

    finally:
        asr_model.decoder._rnnt_export = decoder_export_flag


if __name__ == "__main__":
    app()
