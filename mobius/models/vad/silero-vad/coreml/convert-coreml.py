import os
import argparse
import torch
import torch.nn as nn
import coremltools as ct
from silero_vad import load_silero_vad, __version__ as silero_version
from convert_model_components import COREML_AUTHOR, STFTModel, EncoderModel, DecoderModel, convert_stft_model, convert_encoder_model, convert_decoder_model

class UnifiedVADModel(nn.Module):
    """Unified VAD model combining STFT, Encoder, and Decoder"""

    def __init__(self, state_dict, stft_hop_length=128, stft_pad=64, encoder_configs=None):
        super().__init__()

        # Initialize all three sub-models
        stft_weights = state_dict["_model.stft.forward_basis_buffer"]
        self.stft = STFTModel(stft_weights, hop_length=stft_hop_length, pad=stft_pad)
        self.encoder = EncoderModel(state_dict, layer_configs=encoder_configs)
        self.decoder = DecoderModel(state_dict)

    def forward(self, audio_input, hidden_state, cell_state):
        # Pipeline: audio → STFT → Encoder → Decoder
        stft_out = self.stft(audio_input)
        encoder_out = self.encoder(stft_out)
        vad_out, new_hidden, new_cell = self.decoder(encoder_out, hidden_state, cell_state)
        return vad_out, new_hidden, new_cell


class UnifiedVADModel256ms(nn.Module):
    """Unified VAD model for 256ms processing (4160 samples: 64 context + 4096 current at 16kHz)"""

    def __init__(self, state_dict, stft_hop_length=128, stft_pad=64, encoder_configs=None):
        super().__init__()

        # Initialize all three sub-models (same as standard model)
        stft_weights = state_dict["_model.stft.forward_basis_buffer"]
        self.stft = STFTModel(stft_weights, hop_length=stft_hop_length, pad=stft_pad)
        self.encoder = EncoderModel(state_dict, layer_configs=encoder_configs)
        self.decoder = DecoderModel(state_dict)

    def forward(self, audio_input, hidden_state, cell_state):
        # Expect 4160 samples: 64 context + 4096 current
        # Process as 8 chunks of 512 samples with proper context handling
        # This maintains the model's training regime of 576-sample inputs

        # Extract initial context and current audio
        initial_context = audio_input[:, :64]  # First 64 samples as context
        current_audio = audio_input[:, 64:]    # Remaining 4096 samples

        # Pre-compute all chunks with overlapping context
        chunks = []
        context = initial_context  # Start with provided context

        for i in range(8):
            start_idx = i * 512
            chunk = current_audio[:, start_idx:start_idx + 512]
            chunk_with_context = torch.cat([context, chunk], dim=1)
            chunks.append(chunk_with_context)
            context = chunk[:, -64:]  # Update context for next iteration

        # Process all chunks through pipeline
        outputs = []
        h, c = hidden_state, cell_state

        for chunk_with_context in chunks:
            stft_out = self.stft(chunk_with_context)
            encoder_out = self.encoder(stft_out)
            vad_out, h, c = self.decoder(encoder_out, h, c)
            outputs.append(vad_out)

        # Aggregate outputs using noisy-OR (more nuanced probability combination)
        stacked_outputs = torch.cat(outputs, dim=2)  # Shape: [batch, 1, 8]
        # Noisy-OR: 1 - (1-p1) * (1-p2) * ... * (1-p8)
        # Implement using sequential multiplication to avoid torch.prod
        one_minus_probs = 1.0 - stacked_outputs
        product = one_minus_probs[:, :, 0:1]  # Start with first probability
        for i in range(1, 8):
            product = product * one_minus_probs[:, :, i:i+1]
        final_output = 1.0 - product

        return final_output, h, c

def convert_unified_model(state_dict, output_dir, version_suffix, stft_hop_length, stft_pad, encoder_configs):
    """Convert unified VAD model (STFT + Encoder + Decoder) to CoreML"""
    print("Converting Unified VAD model...")

    model = UnifiedVADModel(
        state_dict,
        stft_hop_length=stft_hop_length,
        stft_pad=stft_pad,
        encoder_configs=encoder_configs,
    )
    model.eval()

    # Create example inputs 64 samples of  context from previous audio, 512 of current audio
    example_audio = torch.randn(1, 576)  # Audio input (batch_size=1, samples=576)
    example_hidden = torch.randn(1, 128)  # Hidden state
    example_cell = torch.randn(1, 128)    # Cell state

    # Trace the model
    traced_model = torch.jit.trace(model, (example_audio, example_hidden, example_cell))

    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="audio_input", shape=(1, 576)),
            ct.TensorType(name="hidden_state", shape=(1, 128)),
            ct.TensorType(name="cell_state", shape=(1, 128))
        ],
        outputs=[
            ct.TensorType(name="vad_output"),
            ct.TensorType(name="new_hidden_state"),
            ct.TensorType(name="new_cell_state")
        ],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # Set metadata
    coreml_model.author = COREML_AUTHOR
    coreml_model.short_description = "Silero VAD Unified Model (STFT + Encoder + Decoder)"
    coreml_model.version = silero_version

    output_path = os.path.join(output_dir, f"silero-vad-unified{version_suffix}.mlpackage")
    coreml_model.save(output_path)
    print(f"Unified model saved to: {output_path}")
    return output_path


def convert_unified_model_256ms(state_dict, output_dir, version_suffix, stft_hop_length, stft_pad, encoder_configs):
    """Convert unified VAD model for 256ms processing (4160 samples: 64 context + 4096 current) to CoreML"""
    print("Converting Unified VAD model (256ms)...")

    model = UnifiedVADModel256ms(
        state_dict,
        stft_hop_length=stft_hop_length,
        stft_pad=stft_pad,
        encoder_configs=encoder_configs,
    )
    model.eval()

    # Create example inputs for 256ms (4160 samples: 64 context + 4096 current at 16kHz)
    example_audio = torch.randn(1, 4160)  # Audio input (batch_size=1, samples=4160)
    example_hidden = torch.randn(1, 128)  # Hidden state
    example_cell = torch.randn(1, 128)    # Cell state

    # Trace the model
    traced_model = torch.jit.trace(model, (example_audio, example_hidden, example_cell))

    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="audio_input", shape=(1, 4160)),
            ct.TensorType(name="hidden_state", shape=(1, 128)),
            ct.TensorType(name="cell_state", shape=(1, 128))
        ],
        outputs=[
            ct.TensorType(name="vad_output"),
            ct.TensorType(name="new_hidden_state"),
            ct.TensorType(name="new_cell_state")
        ],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # Set metadata
    coreml_model.author = COREML_AUTHOR
    coreml_model.short_description = "Silero VAD Unified Model 256ms (STFT + Encoder + Decoder) with noisy-OR aggregation"
    coreml_model.version = silero_version

    output_path = os.path.join(output_dir, f"silero-vad-unified-256ms{version_suffix}.mlpackage")
    coreml_model.save(output_path)
    print(f"Unified 256ms model saved to: {output_path}")
    return output_path


def convert_silero_vad_coreml(output_dir="./coreml_models", include_256ms=False):
    """Convert Silero VAD to CoreML format"""
    print(f"Loading Silero VAD model version {silero_version}")

    # Load the PyTorch model
    model = load_silero_vad()
    state_dict = model.state_dict()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Version suffix for filenames
    version_suffix = f"-v{silero_version}"

    # Clean up state dict - only keep 16k model (skip 8k as requested)
    cleaned_dict = {}
    for key, value in state_dict.items():
        if "_8k" not in key:  # Skip 8k model
            cleaned_dict[key] = value

    print(f"\nConverting model with {len(cleaned_dict)} tensors (16k model only)")

    # Print tensor info for debugging
    print("\nTensor info:")
    for key, tensor in cleaned_dict.items():
        print(f"  - {key}: {tensor.shape} ({tensor.dtype})")

    converted_models = []

    # Extract architectural parameters used during conversion
    stft_module = model._model.stft
    hop_length = int(stft_module.hop_length)
    pad_tuple = tuple(stft_module.padding.padding) if hasattr(stft_module, "padding") else (0, 0)
    stft_pad = int(pad_tuple[1] if len(pad_tuple) > 1 else pad_tuple[0])

    encoder_configs = []
    for _, block in model._model.encoder.named_children():
        stride = int(block.reparam_conv.stride[0])
        padding = int(block.reparam_conv.padding[0])
        encoder_configs.append({"stride": stride, "padding": padding})

    try:
        # Convert STFT model
        stft_path = convert_stft_model(cleaned_dict, output_dir, version_suffix)
        converted_models.append(stft_path)

        # Convert Encoder model
        encoder_path = convert_encoder_model(cleaned_dict, output_dir, version_suffix)
        converted_models.append(encoder_path)

        # Convert Decoder model
        decoder_path = convert_decoder_model(cleaned_dict, output_dir, version_suffix)
        converted_models.append(decoder_path)

        # Convert Unified model
        unified_path = convert_unified_model(
            cleaned_dict,
            output_dir,
            version_suffix,
            stft_hop_length=hop_length,
            stft_pad=stft_pad,
            encoder_configs=encoder_configs,
        )
        converted_models.append(unified_path)

        # Convert Unified 256ms model if requested
        if include_256ms:
            unified_256ms_path = convert_unified_model_256ms(
                cleaned_dict,
                output_dir,
                version_suffix,
                stft_hop_length=hop_length,
                stft_pad=stft_pad,
                encoder_configs=encoder_configs,
            )
            converted_models.append(unified_256ms_path)

        print(f"\n✅ Successfully converted Silero VAD to CoreML!")
        print(f"Models saved in: {output_dir}")
        print("\nConverted models:")
        for model_path in converted_models:
            model_name = os.path.basename(model_path)
            model_size = sum(os.path.getsize(os.path.join(model_path, f))
                           for f in os.listdir(model_path)
                           if os.path.isfile(os.path.join(model_path, f)))
            print(f"  - {model_name}: {model_size / (1024*1024):.1f} MB")

        print(f"\nUsage Options:")
        print(f"  Option 1 - Individual models (3-stage pipeline):")
        print(f"    1. STFT: audio → frequency features")
        print(f"    2. Encoder: frequency features → encoded features")
        print(f"    3. Decoder: encoded features + state → VAD output + new state")
        print(f"  Option 2 - Unified model (single inference):")
        print(f"    Unified: audio + state → VAD output + new state")

        return converted_models

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Silero VAD PyTorch model to CoreML format")
    parser.add_argument("--output-dir", type=str, default="./coreml_models",
                       help="Output directory for CoreML models")
    parser.add_argument("--include-256ms", action="store_true",
                       help="Include 256ms unified model variant (processes 4160 samples: 64 context + 4096 current)")
    args = parser.parse_args()

    convert_silero_vad_coreml(args.output_dir, args.include_256ms)
