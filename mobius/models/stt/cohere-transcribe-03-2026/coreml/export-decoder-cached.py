#!/usr/bin/env python3
"""Export Cohere Transcribe decoder (cached version) to CoreML.

Based on reverse engineering of BarathwajAnandan's cached decoder.
This version supports autoregressive decoding with KV cache.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class SimplifiedCachedDecoderWrapper(nn.Module):
    """
    Simplified decoder wrapper that uses raw tensors for cache instead of transformers Cache objects.
    This matches BarathwajAnandan's export format.
    """

    def __init__(self, full_model, max_seq_len=108):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        self.config = full_model.config

        # Cache dimensions
        dec_config = self.config.transf_decoder["config_dict"]
        self.num_layers = dec_config["num_layers"]
        self.num_heads = dec_config["num_attention_heads"]
        self.hidden_size = dec_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        # Use provided max_seq_len (practical inference limit, not model's max capacity)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_id,
        encoder_hidden_states,
        cache_k,
        cache_v,
        step,
        cross_attention_mask,
    ):
        """
        Single-step autoregressive decoding with simplified cache.

        Args:
            input_id: (1, 1) int64 - current token
            encoder_hidden_states: (1, encoder_frames, decoder_hidden) - encoder output (already projected)
            cache_k: (num_layers, num_heads, max_seq_len, head_dim) - key cache
            cache_v: (num_layers, num_heads, max_seq_len, head_dim) - value cache
            step: (1,) int32 - current decoding step (0-indexed)
            cross_attention_mask: (1, 1, 1, encoder_frames) - encoder attention mask

        Returns:
            logits: (1, vocab_size) - next token logits
            new_cache_k: (num_layers, num_heads, max_seq_len, head_dim)
            new_cache_v: (num_layers, num_heads, max_seq_len, head_dim)
        """
        batch_size = 1
        current_step = step.item() if step.dim() > 0 else step.int().item()

        # Encoder hidden states are already projected to decoder dimension by the encoder export

        # Convert tensor cache to EncoderDecoderCache format
        # The model uses EncoderDecoderCache which has separate self and cross attention caches
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        for layer_idx in range(self.num_layers):
            # Get this layer's cache (num_heads, max_seq_len, head_dim)
            layer_k = cache_k[layer_idx]  # (num_heads, max_seq_len, head_dim)
            layer_v = cache_v[layer_idx]

            # Add batch dimension and truncate to current step
            layer_k = layer_k.unsqueeze(0)  # (1, num_heads, max_seq_len, head_dim)
            layer_v = layer_v.unsqueeze(0)

            if current_step > 0:
                # Truncate to valid length
                layer_k = layer_k[:, :, :current_step, :]
                layer_v = layer_v[:, :, :current_step, :]
            else:
                # First step - empty cache
                layer_k = layer_k[:, :, :0, :]
                layer_v = layer_v[:, :, :0, :]

            # Update the self-attention cache
            self_attention_cache.update(layer_k, layer_v, layer_idx)

        # Create EncoderDecoderCache
        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Create positions tensor for current step
        positions = torch.tensor([[current_step]], dtype=torch.long, device=input_id.device)

        # Create self-attention mask (causal mask for current position)
        # For single-step decoding with cache, we attend to all previous positions
        past_len = current_step
        total_len = past_len + 1

        # Create causal mask
        query_positions = torch.tensor([[past_len]], device=input_id.device)  # Current position
        key_positions = torch.arange(total_len, device=input_id.device)[None, :]  # All positions up to current
        causal_bool = key_positions > query_positions
        self_attention_mask = torch.zeros((batch_size, 1, 1, total_len), device=input_id.device, dtype=encoder_hidden_states.dtype)
        self_attention_mask.masked_fill_(causal_bool[None, None, :, :], float("-inf"))

        # Reshape cross attention mask from (1, 1, 1, enc_len) to (1, enc_len)
        if cross_attention_mask is not None:
            # Convert from additive mask format to boolean format
            # The input is (1, 1, 1, enc_len) with 0s for valid positions
            cross_mask_reshaped = cross_attention_mask.squeeze(1).squeeze(1)  # (1, enc_len)
        else:
            cross_mask_reshaped = None

        # Call decoder
        decoder_outputs, updated_cache = self.decoder(
            input_ids=input_id,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_mask_reshaped,
            past_key_values=past_key_values,
            cache_position=None,
            kv_seq_len=None,
        )

        # Project to vocab
        logits = self.log_softmax(decoder_outputs)  # (1, 1, vocab_size)
        logits = logits.squeeze(1)  # (1, vocab_size)

        # Extract updated cache back to tensor format
        # EncoderDecoderCache has self_attention_cache which is a DynamicCache
        self_attn_cache = updated_cache.self_attention_cache
        new_cache_k_list = []
        new_cache_v_list = []

        for layer_idx in range(self.num_layers):
            layer_k = self_attn_cache.key_cache[layer_idx]  # (1, num_heads, seq_len, head_dim)
            layer_v = self_attn_cache.value_cache[layer_idx]

            # Remove batch dimension
            layer_k = layer_k.squeeze(0)  # (num_heads, seq_len, head_dim)
            layer_v = layer_v.squeeze(0)

            # Pad to max_seq_len
            current_len = layer_k.shape[1]
            if current_len < self.max_seq_len:
                pad_len = self.max_seq_len - current_len
                layer_k = torch.cat([
                    layer_k,
                    torch.zeros(self.num_heads, pad_len, self.head_dim, dtype=layer_k.dtype, device=layer_k.device)
                ], dim=1)
                layer_v = torch.cat([
                    layer_v,
                    torch.zeros(self.num_heads, pad_len, self.head_dim, dtype=layer_v.dtype, device=layer_v.device)
                ], dim=1)

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)  # (num_layers, num_heads, max_seq_len, head_dim)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_cached(output_dir: Path, precision: str = "float16"):
    """Export the cached Cohere decoder to CoreML."""
    print("="*70)
    print("Cohere Transcribe Decoder (Cached) Export")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full model
    print("\n[1/5] Loading model from HuggingFace...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("   ✓ Model loaded")

    # Get dimensions from config
    dec_config = model.config.transf_decoder["config_dict"]
    num_layers = dec_config["num_layers"]
    num_heads = dec_config["num_attention_heads"]
    # Use practical inference limit (from manifest), not model's max capacity
    max_seq_len = 108  # decoder_max_len from manifest (not 1024 from config)
    hidden_size = dec_config["hidden_size"]
    head_dim = hidden_size // num_heads

    # Wrap decoder
    print("\n[2/5] Wrapping decoder...")
    wrapped_decoder = SimplifiedCachedDecoderWrapper(model, max_seq_len=max_seq_len)
    wrapped_decoder.eval()
    print(f"   ✓ Decoder wrapped (max_seq_len={max_seq_len})")

    # Create example inputs
    print("\n[3/5] Creating example inputs...")
    batch_size = 1
    encoder_frames = 376  # From manifest
    decoder_hidden_size = dec_config["hidden_size"]  # 1024 - encoder output after projection
    vocab_size = model.config.head["num_classes"]

    example_input_id = torch.tensor([[13764]], dtype=torch.long)  # decoder_start_token_id
    example_encoder_hidden = torch.randn(batch_size, encoder_frames, decoder_hidden_size)  # Already projected
    example_cache_k = torch.zeros(num_layers, num_heads, max_seq_len, head_dim)
    example_cache_v = torch.zeros(num_layers, num_heads, max_seq_len, head_dim)
    example_step = torch.tensor([0], dtype=torch.int32)
    example_cross_mask = torch.ones(batch_size, 1, 1, encoder_frames)

    print(f"   input_id: {example_input_id.shape}")
    print(f"   encoder_hidden_states: {example_encoder_hidden.shape}")
    print(f"   cache_k/v: {example_cache_k.shape}")
    print(f"   step: {example_step.shape}")
    print(f"   cross_attention_mask: {example_cross_mask.shape}")
    print(f"   Num layers: {num_layers}, Num heads: {num_heads}, Head dim: {head_dim}")

    # Trace the model
    print("\n[4/5] Tracing decoder...")
    with torch.no_grad():
        traced_decoder = torch.jit.trace(
            wrapped_decoder,
            (
                example_input_id,
                example_encoder_hidden,
                example_cache_k,
                example_cache_v,
                example_step,
                example_cross_mask,
            ),
            check_trace=False,  # Disable sanity checks due to conditional logic in model
        )

    # Test traced model
    logits, new_k, new_v = traced_decoder(
        example_input_id,
        example_encoder_hidden,
        example_cache_k,
        example_cache_v,
        example_step,
        example_cross_mask,
    )
    print(f"   Logits: {logits.shape}")
    print(f"   New cache_k: {new_k.shape}")
    print(f"   New cache_v: {new_v.shape}")

    # Convert to CoreML
    print(f"\n[5/5] Converting to CoreML ({precision})...")

    # Define inputs
    inputs = [
        ct.TensorType(name="input_id", shape=example_input_id.shape, dtype=np.int32),
        ct.TensorType(name="encoder_hidden_states", shape=example_encoder_hidden.shape, dtype=np.float32),
        ct.TensorType(name="cache_k", shape=example_cache_k.shape, dtype=np.float32),
        ct.TensorType(name="cache_v", shape=example_cache_v.shape, dtype=np.float32),
        ct.TensorType(name="step", shape=example_step.shape, dtype=np.int32),
        ct.TensorType(name="cross_attention_mask", shape=example_cross_mask.shape, dtype=np.float32),
    ]

    # Set compute precision
    compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    # Convert
    mlmodel = ct.convert(
        traced_decoder,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="new_cache_k"),
            ct.TensorType(name="new_cache_v"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
    )

    # Save
    output_path = output_dir / "cohere_decoder_cached.mlpackage"
    mlmodel.save(str(output_path))

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Model size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2:.1f} MB")

    print("\n" + "="*70)
    print("DECODER (CACHED) EXPORT COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"\nModel configuration:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Attention heads: {num_heads}")
    print(f"  - Head dimension: {head_dim}")
    print(f"  - Max sequence length: {max_seq_len}")
    print(f"  - Vocab size: {vocab_size}")
    print("\nInputs:")
    print(f"  - input_id: {example_input_id.shape} int32 - current token")
    print(f"  - encoder_hidden_states: {example_encoder_hidden.shape} float32")
    print(f"  - cache_k/v: {example_cache_k.shape} float32 - KV cache")
    print(f"  - step: {example_step.shape} int32 - decoding step")
    print(f"  - cross_attention_mask: {example_cross_mask.shape} float32")
    print("\nOutputs:")
    print(f"  - logits: (1, {vocab_size}) float16/32 - next token logits")
    print(f"  - new_cache_k/v: {example_cache_k.shape} float16/32")
    print()


def main():
    parser = argparse.ArgumentParser(description="Export Cohere decoder (cached) to CoreML")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build"),
        help="Output directory for CoreML models"
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Model precision (default: float16)"
    )

    args = parser.parse_args()

    try:
        export_decoder_cached(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
