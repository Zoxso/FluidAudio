import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from silero_vad import __version__ as silero_version

COREML_AUTHOR = "Fluid Infernece + Silero Team"

class STFTModel(nn.Module):
    """STFT preprocessing model for CoreML"""

    def __init__(self, stft_weights, hop_length=128, pad=64):
        super().__init__()
        # Register the STFT basis as a parameter
        self.register_buffer('forward_basis', stft_weights)
        self.hop_length = hop_length
        self.pad = pad

    def forward(self, x):
        # x shape: (batch_size, samples)
        # Mirror the TorchScript STFT: reflection pad then conv with provided basis
        # TorchScript applies reflection padding only on the right side (0, pad)
        x = F.pad(x, (0, self.pad), mode="reflect")
        x = x.unsqueeze(1)  # Add channel dimension: [batch, 1, samples]

        stft_out = torch.conv1d(x, self.forward_basis, stride=self.hop_length, padding=0)

        # stft_out shape: [batch, 258, time_steps]
        # 258 = 129 real + 129 imaginary components
        real_part = stft_out[:, :129, :]
        imag_part = stft_out[:, 129:, :]

        # Compute magnitude (match TorchScript pipeline)
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-12)

        return magnitude


class EncoderModel(nn.Module):
    """Encoder model with 4 convolutional layers"""

    def __init__(self, state_dict, layer_configs=None):
        super().__init__()
        self.layers = nn.ModuleList()

        # TorchScript encoder uses strides [1, 2, 2, 1] with padding 1
        default_configs = [
            {"stride": 1, "padding": 1},
            {"stride": 2, "padding": 1},
            {"stride": 2, "padding": 1},
            {"stride": 1, "padding": 1},
        ]

        if layer_configs is None:
            layer_configs = default_configs

        # Build 4 encoder layers
        for i in range(4):
            weight_key = f"_model.encoder.{i}.reparam_conv.weight"
            bias_key = f"_model.encoder.{i}.reparam_conv.bias"

            if weight_key in state_dict and bias_key in state_dict:
                weight = state_dict[weight_key]
                bias = state_dict[bias_key]

                stride = layer_configs[i]["stride"]
                padding = layer_configs[i]["padding"]

                # Create Conv1d layer
                out_channels, in_channels, kernel_size = weight.shape
                conv = nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )
                conv.weight.data = weight
                conv.bias.data = bias

                self.layers.append(conv)
                # Add activation (assuming ReLU based on typical VAD models)
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # Clip to prevent overflow in CoreML
            if isinstance(layer, nn.ReLU):
                x = torch.clamp(x, max=10000.0)
        return x


class DecoderModel(nn.Module):
    """Decoder model with LSTM and final conv"""

    def __init__(self, state_dict):
        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Load LSTM weights
        lstm_weight_ih = state_dict["_model.decoder.rnn.weight_ih"]
        lstm_weight_hh = state_dict["_model.decoder.rnn.weight_hh"]
        lstm_bias_ih = state_dict["_model.decoder.rnn.bias_ih"]
        lstm_bias_hh = state_dict["_model.decoder.rnn.bias_hh"]

        self.lstm.weight_ih_l0.data = lstm_weight_ih
        self.lstm.weight_hh_l0.data = lstm_weight_hh
        self.lstm.bias_ih_l0.data = lstm_bias_ih
        self.lstm.bias_hh_l0.data = lstm_bias_hh

        # Final conv layer
        final_weight = state_dict["_model.decoder.decoder.2.weight"]
        final_bias = state_dict["_model.decoder.decoder.2.bias"]

        # Dropout is disabled in eval mode, but keep the op for structural parity
        self.dropout = nn.Dropout(p=0.0)
        self.activation = nn.ReLU()
        self.final_conv = nn.Conv1d(128, 1, 1)
        self.final_conv.weight.data = final_weight
        self.final_conv.bias.data = final_bias

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state=None, cell_state=None):
        # x shape: (batch, channels, sequence)
        # Convert to (batch, sequence, channels) for LSTM
        x = x.transpose(1, 2)

        if hidden_state is not None and cell_state is not None:
            h0 = hidden_state.unsqueeze(0)  # Add num_layers dimension
            c0 = cell_state.unsqueeze(0)
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            lstm_out, (hn, cn) = self.lstm(x)

        # Convert back to (batch, channels, sequence)
        lstm_out = lstm_out.transpose(1, 2)

        # Match TorchScript decoder: dropout -> ReLU -> conv -> sigmoid
        out = self.dropout(lstm_out)
        out = self.activation(out)
        out = self.final_conv(out)
        out = self.sigmoid(out)

        # Return output and new states (remove num_layers dimension)
        return out, hn.squeeze(0), cn.squeeze(0)

def convert_stft_model(state_dict, output_dir, version_suffix):
    """Convert STFT preprocessing to CoreML"""
    print("Converting STFT model...")

    stft_weights = state_dict["_model.stft.forward_basis_buffer"]
    model = STFTModel(stft_weights)
    model.eval()

    # Create example input (batch_size=1, samples=576 for 16k)
    example_input = torch.randn(1, 576)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="audio_input", shape=(1, 576))],
        outputs=[ct.TensorType(name="stft_output")],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # Set metadata
    coreml_model.author = COREML_AUTHOR
    coreml_model.short_description = "Silero VAD STFT preprocessing"
    coreml_model.version = silero_version

    output_path = os.path.join(output_dir, f"silero-vad-stft{version_suffix}.mlpackage")
    coreml_model.save(output_path)
    print(f"STFT model saved to: {output_path}")
    return output_path


def convert_encoder_model(state_dict, output_dir, version_suffix):
    """Convert encoder to CoreML"""
    print("Converting Encoder model...")

    model = EncoderModel(state_dict)
    model.eval()

    # Create example input (batch_size=1, channels=129, sequence_length varies)
    example_input = torch.randn(1, 129, 64)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML with flexible input shape
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="stft_features", shape=(1, 129, ct.RangeDim(1, 1024)))],
        outputs=[ct.TensorType(name="encoder_output")],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # Set metadata
    coreml_model.author = COREML_AUTHOR
    coreml_model.short_description = "Silero VAD Encoder (4 conv layers)"
    coreml_model.version = silero_version

    output_path = os.path.join(output_dir, f"silero-vad-encoder{version_suffix}.mlpackage")
    coreml_model.save(output_path)
    print(f"Encoder model saved to: {output_path}")
    return output_path


def convert_decoder_model(state_dict, output_dir, version_suffix):
    """Convert decoder (LSTM + final conv) to CoreML"""
    print("Converting Decoder model...")

    model = DecoderModel(state_dict)
    model.eval()

    # Create example inputs
    example_input = torch.randn(1, 128, 64)
    example_hidden = torch.randn(1, 128)
    example_cell = torch.randn(1, 128)

    # Trace the model - we'll create a wrapper that handles optional states
    class DecoderWrapper(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, x, h, c):
            return self.decoder(x, h, c)

    wrapper = DecoderWrapper(model)
    traced_model = torch.jit.trace(wrapper, (example_input, example_hidden, example_cell))

    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="encoder_output", shape=(1, 128, ct.RangeDim(1, 1024))),
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
    coreml_model.short_description = "Silero VAD Decoder (LSTM + output layer)"
    coreml_model.version = silero_version

    output_path = os.path.join(output_dir, f"silero-vad-decoder{version_suffix}.mlpackage")
    coreml_model.save(output_path)
    print(f"Decoder model saved to: {output_path}")
    return output_path
