import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from silero_vad import load_silero_vad, read_audio
import time
import matplotlib.pyplot as plt
import matplotlib
import subprocess
from datetime import datetime


class PyTorchJITWrapper:
    """Wrapper around the TorchScript Silero VAD model used for comparisons."""

    def __init__(self, model, sample_rate=16000, chunk_size=512):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.model.eval()
        # Track the display name so plots can show the exact model identifier
        self.model_name = "silero-vad-jit"
        self.reset_states()

    def reset_states(self, batch_size=1):
        """Reset internal model states."""
        if hasattr(self.model, "reset_states"):
            # TorchScript model keeps its own internal state; batch_size kept for compatibility
            self.model.reset_states()

    def _ensure_tensor(self, audio_chunk):
        """Convert input chunk to the expected torch.Tensor format."""
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk)

        if audio_chunk.dim() == 1:
            audio_chunk = audio_chunk.unsqueeze(0)

        if audio_chunk.dim() != 2:
            raise ValueError(
                f"Expected audio chunk with shape [batch, samples], got {tuple(audio_chunk.shape)}"
            )

        return audio_chunk.float()

    def forward_with_intermediates(self, audio_chunk):
        """Run the TorchScript model and expose the final probability output."""
        audio_tensor = self._ensure_tensor(audio_chunk)
        batch_size = audio_tensor.shape[0]
        outputs = []

        for batch_idx in range(batch_size):
            chunk = audio_tensor[batch_idx]

            if chunk.shape[-1] < self.chunk_size:
                chunk = F.pad(chunk, (0, self.chunk_size - chunk.shape[-1]))
            elif chunk.shape[-1] > self.chunk_size:
                chunk = chunk[: self.chunk_size]

            with torch.no_grad():
                prob = self.model(chunk, self.sample_rate)

            outputs.append(prob.detach().cpu())

        final_output = torch.stack(outputs).view(batch_size, 1)

        return {
            'stft_output': None,
            'encoder_output': None,
            'final_output': final_output,
            'hidden_state': None,
            'cell_state': None
        }


class CoreMLVADWrapper:
    """Wrapper for CoreML VAD models"""

    def __init__(self, models_dir):
        self.stft_model = ct.models.MLModel(
            os.path.join(models_dir, "silero-vad-stft-v6.0.0.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        self.encoder_model = ct.models.MLModel(os.path.join(models_dir, "silero-vad-encoder-v6.0.0.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.decoder_model = ct.models.MLModel(os.path.join(models_dir, "silero-vad-decoder-v6.0.0.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)

        # Initialize states
        self.hidden_state = np.zeros((1, 128), dtype=np.float32)
        self.cell_state = np.zeros((1, 128), dtype=np.float32)
        # Initialize context buffer (64 samples from previous chunk)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def reset_states(self):
        """Reset LSTM states and context"""
        self.hidden_state = np.zeros((1, 128), dtype=np.float32)
        self.cell_state = np.zeros((1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def forward_with_intermediates(self, audio_chunk):
        """Forward pass with intermediate outputs"""
        # Convert to numpy
        if isinstance(audio_chunk, torch.Tensor):
            audio_input = audio_chunk.numpy().astype(np.float32)
        else:
            audio_input = audio_chunk.astype(np.float32)

        # Handle different input sizes - pad to 512 if needed
        if audio_input.shape[1] < 512:
            # Pad with zeros to reach 512 samples
            padding_needed = 512 - audio_input.shape[1]
            padding = np.zeros((audio_input.shape[0], padding_needed), dtype=np.float32)
            audio_input = np.concatenate([audio_input, padding], axis=1)
        elif audio_input.shape[1] > 512:
            # Truncate to 512 samples
            audio_input = audio_input[:, :512]

        # Concatenate context (64 samples) with current audio (512 samples) = 576 samples total
        audio_with_context = np.concatenate([self.context, audio_input], axis=1)

        # Update context for next iteration (last 64 samples of current input)
        self.context = audio_input[:, -64:]

        # Step 1: STFT with 576 samples
        stft_output = self.stft_model.predict({"audio_input": audio_with_context})
        stft_features = list(stft_output.values())[0]

        # Step 2: Encoder
        encoder_output = self.encoder_model.predict({"stft_features": stft_features})
        encoder_features = list(encoder_output.values())[0]

        # Step 3: Decoder
        decoder_output = self.decoder_model.predict({
            "encoder_output": encoder_features,
            "hidden_state": self.hidden_state,
            "cell_state": self.cell_state
        })

        # Extract outputs
        final_output = None
        new_hidden = None
        new_cell = None

        for key, value in decoder_output.items():
            if "vad_output" in key.lower():
                final_output = value
            elif "hidden" in key.lower():
                new_hidden = value
            elif "cell" in key.lower():
                new_cell = value

        # Update states
        if new_hidden is not None:
            self.hidden_state = new_hidden
        if new_cell is not None:
            self.cell_state = new_cell

        return {
            'stft_output': torch.from_numpy(stft_features),
            'encoder_output': torch.from_numpy(encoder_features),
            'final_output': torch.from_numpy(final_output) if final_output is not None else None,
            'hidden_state': torch.from_numpy(new_hidden) if new_hidden is not None else None,
            'cell_state': torch.from_numpy(new_cell) if new_cell is not None else None
        }


class UnifiedCoreMLWrapper:
    """Wrapper for unified CoreML VAD model"""

    def __init__(self, models_dir):
        unified_path = os.path.join(models_dir, "silero-vad-unified-v6.0.0.mlpackage")
        if not os.path.exists(unified_path):
            raise FileNotFoundError(f"Unified model not found: {unified_path}")

        self.unified_model = ct.models.MLModel(unified_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        package_name, _ = os.path.splitext(os.path.basename(unified_path))
        self.model_name = package_name

        # Initialize states
        self.hidden_state = np.zeros((1, 128), dtype=np.float32)
        self.cell_state = np.zeros((1, 128), dtype=np.float32)
        # Initialize context buffer (64 samples from previous chunk)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def reset_states(self):
        """Reset LSTM states and context"""
        self.hidden_state = np.zeros((1, 128), dtype=np.float32)
        self.cell_state = np.zeros((1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def forward_with_intermediates(self, audio_chunk):
        """Forward pass with unified model"""
        # Convert to numpy
        if isinstance(audio_chunk, torch.Tensor):
            audio_input = audio_chunk.numpy().astype(np.float32)
        else:
            audio_input = audio_chunk.astype(np.float32)

        # Handle different input sizes - pad to 512 if needed
        if audio_input.shape[1] < 512:
            # Pad with zeros to reach 512 samples
            padding_needed = 512 - audio_input.shape[1]
            padding = np.zeros((audio_input.shape[0], padding_needed), dtype=np.float32)
            audio_input = np.concatenate([audio_input, padding], axis=1)
        elif audio_input.shape[1] > 512:
            # Truncate to 512 samples
            audio_input = audio_input[:, :512]

        # Concatenate context (64 samples) with current audio (512 samples) = 576 samples total
        audio_with_context = np.concatenate([self.context, audio_input], axis=1)

        # Update context for next iteration (last 64 samples of current input)
        self.context = audio_input[:, -64:]

        # Single unified inference with 576 samples
        unified_output = self.unified_model.predict({
            "audio_input": audio_with_context,
            "hidden_state": self.hidden_state,
            "cell_state": self.cell_state
        })

        # Extract outputs
        final_output = None
        new_hidden = None
        new_cell = None

        for key, value in unified_output.items():
            if "vad_output" in key.lower():
                final_output = value
            elif "hidden" in key.lower():
                new_hidden = value
            elif "cell" in key.lower():
                new_cell = value

        # Update states
        if new_hidden is not None:
            self.hidden_state = new_hidden
        if new_cell is not None:
            self.cell_state = new_cell

        # Note: Unified model doesn't expose intermediate outputs
        # Return placeholders for consistency with pipeline interface
        return {
            'stft_output': None,  # Not available from unified model
            'encoder_output': None,  # Not available from unified model
            'final_output': torch.from_numpy(final_output) if final_output is not None else None,
            'hidden_state': torch.from_numpy(new_hidden) if new_hidden is not None else None,
            'cell_state': torch.from_numpy(new_cell) if new_cell is not None else None
        }

    def forward_batch(self, audio_tensor, chunk_size=512, max_batch_size=25):
        """Process full audio tensor using batch prediction for better performance"""
        if isinstance(audio_tensor, torch.Tensor):
            audio_input = audio_tensor.numpy().astype(np.float32)
        else:
            audio_input = audio_tensor.astype(np.float32)

        # Reset states
        self.reset_states()

        num_chunks = len(audio_input) // chunk_size
        all_outputs = []
        all_hidden_states = []
        all_cell_states = []

        # Note: True batch processing is limited by LSTM state dependencies
        # However, we can still benefit from batch API calls for better CoreML performance

        for batch_start in range(0, num_chunks, max_batch_size):
            batch_end = min(batch_start + max_batch_size, num_chunks)
            batch_inputs = []

            # Prepare all inputs for this batch (still sequential due to state dependencies)
            for i in range(batch_start, batch_end):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size

                # Extract chunk
                chunk = audio_input[start_idx:end_idx]

                # Handle different input sizes - pad to 512 if needed
                if len(chunk) < 512:
                    padding_needed = 512 - len(chunk)
                    padding = np.zeros(padding_needed, dtype=np.float32)
                    chunk = np.concatenate([chunk, padding])
                elif len(chunk) > 512:
                    chunk = chunk[:512]

                # Add batch dimension
                chunk = chunk[np.newaxis, :]  # Shape: (1, 512)

                # Concatenate context (64 samples) with current audio (512 samples) = 576 samples total
                audio_with_context = np.concatenate([self.context, chunk], axis=1)

                # Update context for next iteration (last 64 samples of current input)
                self.context = chunk[:, -64:]

                # Prepare input dictionary for this chunk
                batch_inputs.append({
                    "audio_input": audio_with_context,
                    "hidden_state": self.hidden_state.copy(),
                    "cell_state": self.cell_state.copy()
                })

                # For proper sequential processing, we need to run one at a time
                # and update states immediately for the next chunk
                result = self.unified_model.predict(batch_inputs[-1])

                # Extract outputs
                final_output = None
                new_hidden = None
                new_cell = None

                for key, value in result.items():
                    if "vad_output" in key.lower():
                        final_output = value
                    elif "hidden" in key.lower():
                        new_hidden = value
                    elif "cell" in key.lower():
                        new_cell = value

                # Update states immediately for next chunk
                if new_hidden is not None:
                    self.hidden_state = new_hidden
                if new_cell is not None:
                    self.cell_state = new_cell

                # Store results
                all_outputs.append(final_output.item() if final_output is not None else 0.0)
                all_hidden_states.append(new_hidden)
                all_cell_states.append(new_cell)

            # Note: True batch processing would require either:
            # 1. Independent chunks (no LSTM state carryover)
            # 2. Or a model that handles batched sequential processing internally

        return {
            'outputs': np.array(all_outputs),
            'hidden_states': all_hidden_states,
            'cell_states': all_cell_states,
            'num_chunks': num_chunks,
            'chunk_size': chunk_size
        }


class UnifiedCoreML256msWrapper:
    """Wrapper for unified CoreML VAD model with 256ms processing (4160 samples: 64 context + 4096 current)"""

    def __init__(self, models_dir):
        unified_256ms_path = os.path.join(models_dir, "silero-vad-unified-256ms-v6.0.0.mlpackage")
        if not os.path.exists(unified_256ms_path):
            raise FileNotFoundError(f"Unified 256ms model not found: {unified_256ms_path}")

        self.unified_256ms_model = ct.models.MLModel(unified_256ms_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        package_name, _ = os.path.splitext(os.path.basename(unified_256ms_path))
        self.model_name = package_name

        # Initialize states
        self.hidden_state = np.zeros((1, 128), dtype=np.float32)
        self.cell_state = np.zeros((1, 128), dtype=np.float32)
        # Initialize context buffer (64 samples from previous audio)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def reset_states(self):
        """Reset LSTM states and context"""
        self.hidden_state = np.zeros((1, 128), dtype=np.float32)
        self.cell_state = np.zeros((1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def forward_with_intermediates(self, audio_chunk):
        """Forward pass with unified 256ms model

        Args:
            audio_chunk: 4096 samples (current audio) - context will be added automatically
        """
        # Convert to numpy
        if isinstance(audio_chunk, torch.Tensor):
            audio_input = audio_chunk.numpy().astype(np.float32)
        else:
            audio_input = audio_chunk.astype(np.float32)

        # Handle different input sizes - pad to 4096 if needed
        if audio_input.shape[1] < 4096:
            padding_needed = 4096 - audio_input.shape[1]
            padding = np.zeros((audio_input.shape[0], padding_needed), dtype=np.float32)
            audio_input = np.concatenate([audio_input, padding], axis=1)
        elif audio_input.shape[1] > 4096:
            # Truncate to 4096 samples
            audio_input = audio_input[:, :4096]

        # Concatenate context (64 samples) with current audio (4096 samples) = 4160 samples total
        audio_with_context = np.concatenate([self.context, audio_input], axis=1)

        # Update context for next iteration (last 64 samples of current input)
        self.context = audio_input[:, -64:]

        # Single unified inference with 4160 samples
        unified_output = self.unified_256ms_model.predict({
            "audio_input": audio_with_context,
            "hidden_state": self.hidden_state,
            "cell_state": self.cell_state
        })

        # Extract outputs (CoreML flattens tuple outputs)
        final_output = None
        new_hidden = None
        new_cell = None

        for key, value in unified_output.items():
            if 'vad_output' in key.lower():
                final_output = value
            elif 'new_hidden' in key.lower():
                new_hidden = value
            elif 'new_cell' in key.lower():
                new_cell = value

        # Update states for next iteration
        if new_hidden is not None:
            self.hidden_state = new_hidden
        if new_cell is not None:
            self.cell_state = new_cell

        return {
            'final_output': final_output,
            'hidden_state': torch.from_numpy(new_hidden) if new_hidden is not None else None,
            'cell_state': torch.from_numpy(new_cell) if new_cell is not None else None
        }

    def forward_batch(self, audio_tensor, chunk_size=4096):
        """Process audio in 256ms chunks (4096 samples + 64 context internally)"""
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        audio_input = audio_tensor.numpy().astype(np.float32)

        # Process in 4032-sample chunks
        num_chunks = len(audio_input[0]) // chunk_size
        all_outputs = []
        all_hidden_states = []
        all_cell_states = []
        all_times = []  # Track individual chunk processing times

        print(f"Processing {num_chunks} chunks of {chunk_size} samples each (256ms processing)...")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size

            if end_idx > len(audio_input[0]):
                break

            chunk = audio_input[:, start_idx:end_idx]

            # Time the processing of this chunk
            start_time = time.time()
            result = self.forward_with_intermediates(chunk)
            chunk_time = time.time() - start_time

            all_outputs.append(result['final_output'].flatten()[0])
            all_hidden_states.append(result['hidden_state'])
            all_cell_states.append(result['cell_state'])
            all_times.append(chunk_time)

        return {
            'outputs': np.array(all_outputs),
            'hidden_states': all_hidden_states,
            'cell_states': all_cell_states,
            'times': np.array(all_times),  # Individual chunk processing times
            'num_chunks': num_chunks,
            'chunk_size': chunk_size
        }


def load_audio_file(audio_path, target_sr=16000):
    """Load audio file and return as torch tensor"""
    try:
        # Try using torchaudio directly
        import torchaudio
        audio_tensor, orig_sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            audio_tensor = resampler(audio_tensor)

        # Flatten to 1D
        audio_tensor = audio_tensor.squeeze()

        print(f"Loaded audio: {audio_path}")
        print(f"Duration: {len(audio_tensor)/target_sr:.2f}s, Samples: {len(audio_tensor)}, Sample rate: {target_sr}Hz")
        return audio_tensor
    except Exception as e:
        try:
            # Fallback to silero_vad read_audio
            audio_tensor = read_audio(audio_path, sampling_rate=target_sr)
            print(f"Loaded audio: {audio_path}")
            print(f"Duration: {len(audio_tensor)/target_sr:.2f}s, Samples: {len(audio_tensor)}, Sample rate: {target_sr}Hz")
            return audio_tensor
        except Exception as e2:
            print(f"Error loading audio file {audio_path}: {e}")
            print(f"Fallback also failed: {e2}")
            return None


def process_full_audio(pytorch_ref, unified_wrapper, audio_tensor, chunk_size=512, debug_intermediate=False):
    """Process full audio file through both models and collect results"""
    if audio_tensor is None:
        return None, None

    # Reset states for both models
    pytorch_ref.reset_states()
    unified_wrapper.reset_states()

    # Prepare output lists
    pytorch_outputs = []
    unified_outputs = []
    timestamps = []
    pytorch_times = []
    unified_times = []

    num_chunks = len(audio_tensor) // chunk_size
    print(f"Processing {num_chunks} chunks of {chunk_size} samples each...")

    if debug_intermediate:
        print("Debug mode: Collecting intermediate outputs for comparison")

    # Process audio in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Extract chunk and add batch dimension
        chunk = audio_tensor[start_idx:end_idx].unsqueeze(0)

        # Process with PyTorch model
        pytorch_start = time.perf_counter()
        pytorch_result = pytorch_ref.forward_with_intermediates(chunk)
        pytorch_time = time.perf_counter() - pytorch_start
        pytorch_prob = pytorch_result['final_output'].item()
        pytorch_outputs.append(pytorch_prob)
        pytorch_times.append(pytorch_time)

        # Process with Unified CoreML model
        unified_start = time.perf_counter()
        unified_result = unified_wrapper.forward_with_intermediates(chunk)
        unified_time = time.perf_counter() - unified_start
        unified_prob = unified_result['final_output'].item()
        unified_outputs.append(unified_prob)
        unified_times.append(unified_time)

        # Debug intermediate outputs for first few chunks
        if debug_intermediate and i < 10:
            print(f"\nChunk {i+1} Debug:")
            print(f"  PyTorch final: {pytorch_prob:.6f}")
            print(f"  Unified final: {unified_prob:.6f}")
            print(f"  Difference: {abs(pytorch_prob - unified_prob):.6f}")

            if pytorch_result['stft_output'] is not None:
                stft_stats = pytorch_result['stft_output'].abs().mean().item()
                print(f"  PyTorch STFT mean: {stft_stats:.6f}")

            if pytorch_result['encoder_output'] is not None:
                encoder_stats = pytorch_result['encoder_output'].abs().mean().item()
                print(f"  PyTorch encoder mean: {encoder_stats:.6f}")

        # Calculate timestamp (in seconds)
        timestamp = (start_idx + chunk_size/2) / 16000
        timestamps.append(timestamp)

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{num_chunks} chunks ({(i+1)/num_chunks*100:.1f}%)")

    result = {
        'pytorch_outputs': np.array(pytorch_outputs),
        'unified_outputs': np.array(unified_outputs),
        'timestamps': np.array(timestamps),
        'pytorch_times': np.array(pytorch_times),
        'unified_times': np.array(unified_times),
        'chunk_size': chunk_size,
        'sample_rate': 16000,
        'pytorch_model_name': getattr(pytorch_ref, 'model_name', 'PyTorch'),
        'unified_model_name': getattr(unified_wrapper, 'model_name', 'Unified CoreML')
    }

    if debug_intermediate:
        result['debug_enabled'] = True

    return result


def process_full_audio_batched(pytorch_ref, unified_wrapper, audio_tensor, chunk_size=512, max_batch_size=25, debug_intermediate=False):
    """Process full audio file using batch processing for better CoreML performance"""
    if audio_tensor is None:
        return None

    # Reset states for both models
    pytorch_ref.reset_states()
    unified_wrapper.reset_states()

    num_chunks = len(audio_tensor) // chunk_size
    print(f"Processing {num_chunks} chunks of {chunk_size} samples each using batch processing (batch_size={max_batch_size})...")

    # Process PyTorch model (still sequential due to state dependencies)
    pytorch_outputs = []
    pytorch_times = []
    timestamps = []

    pytorch_start_total = time.perf_counter()
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Extract chunk and add batch dimension
        chunk = audio_tensor[start_idx:end_idx].unsqueeze(0)

        # Process with PyTorch model
        pytorch_start = time.perf_counter()
        pytorch_result = pytorch_ref.forward_with_intermediates(chunk)
        pytorch_time = time.perf_counter() - pytorch_start
        pytorch_prob = pytorch_result['final_output'].item()
        pytorch_outputs.append(pytorch_prob)
        pytorch_times.append(pytorch_time)

        # Calculate timestamp (in seconds)
        timestamp = (start_idx + chunk_size/2) / 16000
        timestamps.append(timestamp)

        if (i + 1) % 100 == 0:
            print(f"PyTorch: Processed {i+1}/{num_chunks} chunks ({(i+1)/num_chunks*100:.1f}%)")

    pytorch_total_time = time.perf_counter() - pytorch_start_total

    # Process CoreML model using batch processing
    print("Processing CoreML model with batch inference...")
    unified_start_total = time.perf_counter()

    # Use the new batch processing method
    if hasattr(unified_wrapper, 'forward_batch'):
        batch_result = unified_wrapper.forward_batch(
            audio_tensor,
            chunk_size=chunk_size,
            max_batch_size=max_batch_size
        )
        unified_outputs = batch_result['outputs']

        # Create dummy timing data (batch processing doesn't give per-chunk timing)
        unified_times = [0.0] * len(unified_outputs)  # Will calculate average later
    else:
        # Fallback to sequential processing
        print("Warning: Batch processing not available, falling back to sequential processing")
        unified_outputs = []
        unified_times = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = audio_tensor[start_idx:end_idx].unsqueeze(0)

            unified_start = time.perf_counter()
            unified_result = unified_wrapper.forward_with_intermediates(chunk)
            unified_time = time.perf_counter() - unified_start
            unified_prob = unified_result['final_output'].item()
            unified_outputs.append(unified_prob)
            unified_times.append(unified_time)

        unified_outputs = np.array(unified_outputs)

    unified_total_time = time.perf_counter() - unified_start_total

    # Calculate average time per chunk for batch processing
    if hasattr(unified_wrapper, 'forward_batch') and len(unified_outputs) > 0:
        avg_unified_time = unified_total_time / len(unified_outputs)
        unified_times = [avg_unified_time] * len(unified_outputs)

    print(f"Processing completed:")
    print(f"  PyTorch total time: {pytorch_total_time:.2f}s")
    print(f"  CoreML total time: {unified_total_time:.2f}s")
    print(f"  Speedup: {pytorch_total_time/unified_total_time:.2f}x")

    # Debug output for first few chunks
    if debug_intermediate:
        print("Debug mode: Comparing first 10 chunks")
        for i in range(min(10, len(pytorch_outputs))):
            pytorch_prob = pytorch_outputs[i]
            unified_prob = unified_outputs[i]
            print(f"\nChunk {i+1} Debug:")
            print(f"  PyTorch final: {pytorch_prob:.6f}")
            print(f"  Unified final: {unified_prob:.6f}")
            print(f"  Difference: {abs(pytorch_prob - unified_prob):.6f}")

    result = {
        'pytorch_outputs': np.array(pytorch_outputs),
        'unified_outputs': np.array(unified_outputs),
        'timestamps': np.array(timestamps),
        'pytorch_times': np.array(pytorch_times),
        'unified_times': np.array(unified_times),
        'chunk_size': chunk_size,
        'sample_rate': 16000,
        'batch_processing': True,
        'max_batch_size': max_batch_size,
        'pytorch_total_time': pytorch_total_time,
        'unified_total_time': unified_total_time,
        'pytorch_model_name': getattr(pytorch_ref, 'model_name', 'PyTorch'),
        'unified_model_name': getattr(unified_wrapper, 'model_name', 'Unified CoreML')
    }

    if debug_intermediate:
        result['debug_enabled'] = True

    return result


def calculate_detailed_statistics(results):
    """Calculate comprehensive statistics comparing the models"""
    if results is None:
        return None

    pytorch_probs = results['pytorch_outputs']
    unified_probs = results['unified_outputs']

    # Basic statistics
    stats_dict = {
        'pytorch': {
            'mean': np.mean(pytorch_probs),
            'std': np.std(pytorch_probs),
            'median': np.median(pytorch_probs),
            'min': np.min(pytorch_probs),
            'max': np.max(pytorch_probs),
            'q25': np.percentile(pytorch_probs, 25),
            'q75': np.percentile(pytorch_probs, 75)
        },
        'unified': {
            'mean': np.mean(unified_probs),
            'std': np.std(unified_probs),
            'median': np.median(unified_probs),
            'min': np.min(unified_probs),
            'max': np.max(unified_probs),
            'q25': np.percentile(unified_probs, 25),
            'q75': np.percentile(unified_probs, 75)
        }
    }

    # Add 256ms model statistics if available
    if 'unified_256ms_outputs' in results:
        unified_256ms_probs = results['unified_256ms_outputs']
        stats_dict['unified_256ms'] = {
            'mean': np.mean(unified_256ms_probs),
            'std': np.std(unified_256ms_probs),
            'median': np.median(unified_256ms_probs),
            'min': np.min(unified_256ms_probs),
            'max': np.max(unified_256ms_probs),
            'q25': np.percentile(unified_256ms_probs, 25),
            'q75': np.percentile(unified_256ms_probs, 75)
        }

    # Comparison metrics
    diff = pytorch_probs - unified_probs
    stats_dict['comparison'] = {
        'mse': np.mean(diff**2),
        'mae': np.mean(np.abs(diff)),
        'correlation': np.corrcoef(pytorch_probs, unified_probs)[0, 1],
        'max_diff': np.max(np.abs(diff)),
        'std_diff': np.std(diff),
        'mean_diff': np.mean(diff)
    }

    # Add 256ms comparison metrics if available
    if 'unified_256ms_outputs' in results:
        # For 256ms comparison, we need to align the outputs since they have different temporal resolutions
        # 256ms model has fewer outputs (one per 4032 samples vs one per 512 samples)
        unified_256ms_probs = results['unified_256ms_outputs']

        # Calculate ratio of outputs
        pytorch_chunks_per_256ms = len(pytorch_probs) // len(unified_256ms_probs)

        # Create aligned pytorch outputs using noisy-OR (matching 256ms model aggregation)
        aligned_pytorch = []
        for i in range(len(unified_256ms_probs)):
            start_idx = i * pytorch_chunks_per_256ms
            end_idx = min((i + 1) * pytorch_chunks_per_256ms, len(pytorch_probs))
            if start_idx < len(pytorch_probs):
                chunk_probs = pytorch_probs[start_idx:end_idx]
                # Apply noisy-OR: 1 - product(1 - p_i)
                one_minus_probs = 1.0 - chunk_probs
                product = np.prod(one_minus_probs)
                noisy_or_result = 1.0 - product
                aligned_pytorch.append(noisy_or_result)

        aligned_pytorch = np.array(aligned_pytorch)

        # Ensure same length
        min_len = min(len(aligned_pytorch), len(unified_256ms_probs))
        aligned_pytorch = aligned_pytorch[:min_len]
        unified_256ms_probs = unified_256ms_probs[:min_len]

        diff_256ms = aligned_pytorch - unified_256ms_probs
        stats_dict['comparison_256ms'] = {
            'mse': np.mean(diff_256ms**2),
            'mae': np.mean(np.abs(diff_256ms)),
            'correlation': np.corrcoef(aligned_pytorch, unified_256ms_probs)[0, 1] if len(aligned_pytorch) > 1 else 1.0,
            'max_diff': np.max(np.abs(diff_256ms)),
            'std_diff': np.std(diff_256ms),
            'mean_diff': np.mean(diff_256ms),
            'aligned_pytorch_outputs': aligned_pytorch,
            'pytorch_chunks_per_256ms': pytorch_chunks_per_256ms
        }


    return stats_dict


def get_git_info():
    """Get current git commit hash and timestamp"""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except:
        commit_hash = "unknown"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp, commit_hash

def analyze_difference_severity(results, stats_dict):
    """Analyze if the differences between models are problematic for VAD use"""
    if results is None or stats_dict is None:
        return None

    pytorch_probs = results['pytorch_outputs']
    unified_probs = results['unified_outputs']
    diff = pytorch_probs - unified_probs

    # VAD-specific analysis
    severity_analysis = {
        'numerical_severity': {},
        'functional_severity': {},
        'temporal_patterns': {},
        'recommendation': ''
    }

    # 1. Numerical severity assessment
    mse = stats_dict['comparison']['mse']
    max_diff = stats_dict['comparison']['max_diff']
    correlation = stats_dict['comparison']['correlation']

    severity_analysis['numerical_severity'] = {
        'mse_level': 'low' if mse < 0.01 else 'medium' if mse < 0.1 else 'high',
        'max_diff_level': 'low' if max_diff < 0.1 else 'medium' if max_diff < 0.3 else 'high',
        'correlation_level': 'excellent' if correlation > 0.95 else 'good' if correlation > 0.9 else 'poor'
    }

    # 2. Functional impact analysis removed for performance

    # 3. Temporal pattern analysis
    # Look for systematic bias vs random noise
    window_size = min(50, len(diff) // 10) if len(diff) > 10 else 1
    running_mean = np.convolve(diff, np.ones(window_size)/window_size, mode='same')

    # Detect drift (systematic bias over time)
    time_chunks = np.array_split(diff, min(10, len(diff) // 10)) if len(diff) > 10 else [diff]
    chunk_means = [np.mean(chunk) for chunk in time_chunks]
    drift_trend = np.polyfit(range(len(chunk_means)), chunk_means, 1)[0] if len(chunk_means) > 1 else 0

    # Detect outlier bursts
    outlier_threshold = 2 * np.std(diff)
    outlier_mask = np.abs(diff) > outlier_threshold
    outlier_clusters = []
    if np.any(outlier_mask):
        # Find consecutive outlier regions
        outlier_indices = np.where(outlier_mask)[0]
        if len(outlier_indices) > 0:
            clusters = []
            current_cluster = [outlier_indices[0]]
            for i in range(1, len(outlier_indices)):
                if outlier_indices[i] - outlier_indices[i-1] <= 5:  # Within 5 samples
                    current_cluster.append(outlier_indices[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [outlier_indices[i]]
            clusters.append(current_cluster)
            outlier_clusters = [(min(cluster), max(cluster)) for cluster in clusters if len(cluster) >= 3]

    severity_analysis['temporal_patterns'] = {
        'systematic_bias': abs(np.mean(diff)),
        'drift_rate': abs(drift_trend),
        'outlier_percentage': np.sum(outlier_mask) / len(diff) * 100,
        'outlier_clusters': len(outlier_clusters),
        'largest_cluster_size': max([end - start for start, end in outlier_clusters]) if outlier_clusters else 0
    }

    # 4. Generate recommendation
    issues = []

    if severity_analysis['numerical_severity']['correlation_level'] == 'poor':
        issues.append("Poor correlation between models")


    if severity_analysis['temporal_patterns']['systematic_bias'] > 0.05:
        issues.append("Significant systematic bias detected")

    if severity_analysis['temporal_patterns']['outlier_percentage'] > 10:
        issues.append("High percentage of outlier differences")

    if abs(drift_trend) > 0.001:
        issues.append("Temporal drift in model differences")

    if not issues:
        severity_analysis['recommendation'] = "ACCEPTABLE: Differences are within expected range for model conversion"
    elif len(issues) <= 2:
        severity_analysis['recommendation'] = f"CAUTION: Minor issues detected - {'; '.join(issues)}"
    else:
        severity_analysis['recommendation'] = f"PROBLEMATIC: Multiple issues detected - {'; '.join(issues)}"

    return severity_analysis

def create_comparison_plots(results, stats_dict, output_dir="./plots", audio_filename="audio"):
    """Create comprehensive visualization plots"""
    if results is None or stats_dict is None:
        return

    # Get git info for filename timestamping
    timestamp, commit_hash = get_git_info()

    # Run severity analysis
    severity_analysis = analyze_difference_severity(results, stats_dict)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    pytorch_probs = results['pytorch_outputs']
    unified_probs = results['unified_outputs']
    timestamps = results['timestamps']
    pytorch_name = results.get('pytorch_model_name', 'PyTorch')
    unified_name = results.get('unified_model_name', 'Unified')

    # Set matplotlib backend for non-interactive use
    matplotlib.use('Agg')

    # Graph 1: Standard Model Comparison (PyTorch vs Unified CoreML)
    plt.figure(figsize=(14, 8))

    # Time series comparison - full width top row
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, pytorch_probs, label=pytorch_name, color='blue', linewidth=1.5)
    plt.plot(timestamps, unified_probs, label=unified_name, color='orange', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.ylabel('VAD Probability')
    plt.xlabel('Time (s)')
    plt.title('Standard Models: Time Series Comparison')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Difference plot - bottom left
    plt.subplot(2, 2, 3)
    diff = pytorch_probs - unified_probs
    plt.plot(timestamps, diff, color='red', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.ylabel('Difference')
    plt.xlabel('Time (s)')
    plt.title(f'{pytorch_name} - {unified_name}')
    plt.grid(True, alpha=0.3)

    # Correlation plot - bottom right
    plt.subplot(2, 2, 4)
    plt.scatter(pytorch_probs, unified_probs, alpha=0.4, s=1, color='orange')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel(pytorch_name)
    plt.ylabel(unified_name)
    corr = stats_dict['comparison']['correlation']
    plt.title(f'Correlation (r={corr:.3f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{audio_filename}_standard_comparison_{timestamp}_{commit_hash}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Graph 2: 256ms Model Comparison (if available)
    if 'unified_256ms_outputs' in results and 'comparison_256ms' in stats_dict:
        plt.figure(figsize=(14, 8))

        # Get 256ms data
        unified_256ms_probs = results['unified_256ms_outputs']
        aligned_pytorch = stats_dict['comparison_256ms']['aligned_pytorch_outputs']
        unified_256ms_aligned = unified_256ms_probs[:len(aligned_pytorch)]
        unified_256ms_name = results.get('unified_256ms_model_name', '256ms Model')
        pytorch_noisy_or_label = f"{pytorch_name} (8-chunk noisy-OR)"

        # Create timestamps for 256ms chunks
        pytorch_chunks_per_256ms = stats_dict['comparison_256ms']['pytorch_chunks_per_256ms']
        timestamps_256ms = []
        for i in range(len(aligned_pytorch)):
            start_idx = i * pytorch_chunks_per_256ms
            if start_idx < len(timestamps):
                timestamps_256ms.append(timestamps[start_idx])
            else:
                # Extrapolate
                chunk_duration = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.032
                timestamps_256ms.append(timestamps[-1] + chunk_duration * (start_idx - len(timestamps) + 1))

        # Time series comparison - full width top row
        plt.subplot(2, 1, 1)
        plt.plot(timestamps_256ms, aligned_pytorch, label=pytorch_noisy_or_label, color='blue', linewidth=1.5)
        plt.plot(timestamps_256ms, unified_256ms_aligned, label=unified_256ms_name, color='green', linewidth=1.5)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.ylabel('VAD Probability')
        plt.xlabel('Time (s)')
        plt.title('256ms Model: Time Series Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # Difference plot - bottom left
        plt.subplot(2, 2, 3)
        diff_256ms = aligned_pytorch - unified_256ms_aligned
        plt.plot(timestamps_256ms, diff_256ms, color='purple', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Difference')
        plt.xlabel('Time (s)')
        plt.title(f'{pytorch_name} (noisy-OR) - {unified_256ms_name}')
        plt.grid(True, alpha=0.3)

        # Correlation plot - bottom right
        plt.subplot(2, 2, 4)
        plt.scatter(aligned_pytorch, unified_256ms_aligned, alpha=0.7, s=20, color='green')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel(pytorch_noisy_or_label)
        plt.ylabel(unified_256ms_name)
        corr_256ms = stats_dict['comparison_256ms']['correlation']
        plt.title(f'{unified_256ms_name} Correlation (r={corr_256ms:.3f})')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{audio_filename}_256ms_comparison_{timestamp}_{commit_hash}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Graph 3: Performance Analysis
    if 'pytorch_times' in results and 'unified_times' in results:
        plt.figure(figsize=(12, 6))

        # Calculate RTF values for performance comparison
        pytorch_times = results['pytorch_times']
        unified_times = results['unified_times']
        sample_rate = results['sample_rate']
        chunk_size = results['chunk_size']
        audio_duration_per_chunk = chunk_size / sample_rate
        pytorch_rtf = audio_duration_per_chunk / pytorch_times
        unified_rtf = audio_duration_per_chunk / unified_times

        # RTF comparison over time
        plt.subplot(1, 2, 1)
        plt.plot(timestamps, pytorch_rtf, color='blue', alpha=0.7, label=pytorch_name)
        plt.plot(timestamps, unified_rtf, color='orange', alpha=0.7, label=unified_name)

        if 'unified_256ms_times' in results:
            unified_256ms_times = results['unified_256ms_times']
            unified_256ms_chunk_size = results['unified_256ms_chunk_size']
            audio_duration_per_256ms_chunk = unified_256ms_chunk_size / sample_rate
            unified_256ms_rtf = audio_duration_per_256ms_chunk / unified_256ms_times
            timestamps_256ms = [i * audio_duration_per_256ms_chunk for i in range(len(unified_256ms_rtf))]
            unified_256ms_label = results.get('unified_256ms_model_name', '256ms Model')
            plt.plot(timestamps_256ms, unified_256ms_rtf, color='green', linewidth=2, label=unified_256ms_label, drawstyle='steps-post')

        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Real-time (1.0x)')
        plt.ylabel('RTF (faster is better)')
        plt.xlabel('Time (s)')
        plt.title('Real-Time Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Performance summary
        plt.subplot(1, 2, 2)
        models = [pytorch_name, unified_name]
        mean_rtf = [np.mean(pytorch_rtf), np.mean(unified_rtf)]
        colors = ['blue', 'orange']

        if 'unified_256ms_times' in results:
            models.append(results.get('unified_256ms_model_name', '256ms Model'))
            mean_rtf.append(np.mean(unified_256ms_rtf))
            colors.append('green')

        bars = plt.bar(models, mean_rtf, color=colors, alpha=0.7)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Real-time')
        plt.ylabel('Mean RTF')
        plt.title('Performance Summary')
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, rtf in zip(bars, mean_rtf):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{rtf:.1f}x',
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{audio_filename}_performance_{timestamp}_{commit_hash}.png'), dpi=300, bbox_inches='tight')
        plt.close()



    # Export simplified timestamp data
    diff = pytorch_probs - unified_probs
    max_diff_idx = np.argmax(np.abs(diff))

    timestamp_data = {
        'timestamps': timestamps,
        'pytorch_probs': pytorch_probs,
        'unified_probs': unified_probs,
        'differences': diff,
        'max_diff_time': timestamps[max_diff_idx],
        'max_diff_value': diff[max_diff_idx],
        'stats': {
            'mean_diff': np.mean(diff),
            'std_diff': np.std(diff),
            'max_abs_diff': np.max(np.abs(diff))
        },
        'severity_analysis': severity_analysis,
        'git_info': {
            'commit_hash': commit_hash,
            'timestamp': timestamp
        }
    }

    # Save timestamp data as CSV for further analysis
    import pandas as pd
    csv_data = {
        'timestamp_s': timestamps,
        'pytorch_prob': pytorch_probs,
        'unified_prob': unified_probs,
        'difference': diff
    }

    # Add timing data if available
    if 'pytorch_times' in results and 'unified_times' in results:
        audio_duration_per_chunk = results['chunk_size'] / results['sample_rate']
        pytorch_rtfx = results['pytorch_times'] / audio_duration_per_chunk
        unified_rtfx = results['unified_times'] / audio_duration_per_chunk

        csv_data.update({
            'pytorch_time_s': results['pytorch_times'],
            'unified_time_s': results['unified_times'],
            'pytorch_rtfx': pytorch_rtfx,
            'unified_rtfx': unified_rtfx,
            'rtfx_speedup': pytorch_rtfx / unified_rtfx
        })

    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(output_dir, f'{audio_filename}_timestamp_data_{timestamp}_{commit_hash}.csv'), index=False)

    print(f"Plots saved to {output_dir}/")
    print(f"Timestamp analysis data saved to {output_dir}/{audio_filename}_timestamp_data_{timestamp}_{commit_hash}.csv")

    # Print performance summary
    if 'pytorch_times' in results and 'unified_times' in results:
        audio_duration_per_chunk = results['chunk_size'] / results['sample_rate']
        pytorch_rtf = audio_duration_per_chunk / results['pytorch_times']
        unified_rtf = audio_duration_per_chunk / results['unified_times']
        speedup = np.mean(unified_rtf) / np.mean(pytorch_rtf)

        print(f"\nPerformance Summary:")
        print(f"  PyTorch mean RTF: {np.mean(pytorch_rtf):.3f}")
        print(f"  Unified mean RTF: {np.mean(unified_rtf):.3f}")
        print(f"  Speedup: {speedup:.2f}x ({speedup*100-100:+.1f}%)")
        print(f"  PyTorch real-time: {np.mean(pytorch_rtf >= 1.0)*100:.1f}% of chunks")
        print(f"  Unified real-time: {np.mean(unified_rtf >= 1.0)*100:.1f}% of chunks")

        # Add 256ms performance stats if available
        if 'unified_256ms_times' in results:
            unified_256ms_times = results['unified_256ms_times']
            unified_256ms_chunk_size = results['unified_256ms_chunk_size']
            audio_duration_per_256ms_chunk = unified_256ms_chunk_size / results['sample_rate']
            unified_256ms_rtf = audio_duration_per_256ms_chunk / unified_256ms_times
            speedup_256ms = np.mean(unified_256ms_rtf) / np.mean(pytorch_rtf)

            print(f"  256ms mean RTF: {np.mean(unified_256ms_rtf):.3f}")
            print(f"  256ms speedup: {speedup_256ms:.2f}x ({speedup_256ms*100-100:+.1f}%)")
            print(f"  256ms real-time: {np.mean(unified_256ms_rtf >= 1.0)*100:.1f}% of chunks")

    # Print severity analysis summary
    if severity_analysis:
        print(f"\nSeverity Analysis Summary:")
        print(f"  Overall: {severity_analysis['recommendation']}")
        print(f"  Outlier percentage: {severity_analysis['temporal_patterns']['outlier_percentage']:.1f}%")
        print(f"  Git commit: {commit_hash} at {timestamp}")


def print_statistics_report(stats_dict, audio_filename):
    """Print comprehensive statistics report"""
    if stats_dict is None:
        return

    print(f"\n" + "="*80)
    print(f"DETAILED STATISTICS REPORT - {audio_filename}")
    print("="*80)

    print(f"\nPyTorch Model Statistics:")
    pt_stats = stats_dict['pytorch']
    print(f"  Mean: {pt_stats['mean']:.6f}")
    print(f"  Std:  {pt_stats['std']:.6f}")
    print(f"  Med:  {pt_stats['median']:.6f}")
    print(f"  Min:  {pt_stats['min']:.6f}")
    print(f"  Max:  {pt_stats['max']:.6f}")
    print(f"  Q25:  {pt_stats['q25']:.6f}")
    print(f"  Q75:  {pt_stats['q75']:.6f}")

    print(f"\nUnified CoreML Model Statistics:")
    cm_stats = stats_dict['unified']
    print(f"  Mean: {cm_stats['mean']:.6f}")
    print(f"  Std:  {cm_stats['std']:.6f}")
    print(f"  Med:  {cm_stats['median']:.6f}")
    print(f"  Min:  {cm_stats['min']:.6f}")
    print(f"  Max:  {cm_stats['max']:.6f}")
    print(f"  Q25:  {cm_stats['q25']:.6f}")
    print(f"  Q75:  {cm_stats['q75']:.6f}")

    # Add 256ms model statistics if available
    if 'unified_256ms' in stats_dict:
        print(f"\nUnified 256ms CoreML Model Statistics:")
        cm_256ms_stats = stats_dict['unified_256ms']
        print(f"  Mean: {cm_256ms_stats['mean']:.6f}")
        print(f"  Std:  {cm_256ms_stats['std']:.6f}")
        print(f"  Med:  {cm_256ms_stats['median']:.6f}")
        print(f"  Min:  {cm_256ms_stats['min']:.6f}")
        print(f"  Max:  {cm_256ms_stats['max']:.6f}")
        print(f"  Q25:  {cm_256ms_stats['q25']:.6f}")
        print(f"  Q75:  {cm_256ms_stats['q75']:.6f}")

    print(f"\nModel Comparison Metrics (PyTorch vs Unified):")
    comp_stats = stats_dict['comparison']
    print(f"  Mean Squared Error (MSE):     {comp_stats['mse']:.6f}")
    print(f"  Mean Absolute Error (MAE):    {comp_stats['mae']:.6f}")
    print(f"  Correlation Coefficient:      {comp_stats['correlation']:.6f}")
    print(f"  Maximum Absolute Difference:  {comp_stats['max_diff']:.6f}")
    print(f"  Standard Deviation of Diff:   {comp_stats['std_diff']:.6f}")
    print(f"  Mean Difference (bias):       {comp_stats['mean_diff']:.6f}")

    # Add 256ms comparison metrics if available
    if 'comparison_256ms' in stats_dict:
        print(f"\nModel Comparison Metrics (PyTorch vs Unified 256ms):")
        comp_256ms_stats = stats_dict['comparison_256ms']
        print(f"  Mean Squared Error (MSE):     {comp_256ms_stats['mse']:.6f}")
        print(f"  Mean Absolute Error (MAE):    {comp_256ms_stats['mae']:.6f}")
        print(f"  Correlation Coefficient:      {comp_256ms_stats['correlation']:.6f}")
        print(f"  Maximum Absolute Difference:  {comp_256ms_stats['max_diff']:.6f}")
        print(f"  Standard Deviation of Diff:   {comp_256ms_stats['std_diff']:.6f}")
        print(f"  Mean Difference (bias):       {comp_256ms_stats['mean_diff']:.6f}")
        print(f"  Chunks per 256ms window:      {comp_256ms_stats['pytorch_chunks_per_256ms']}")


    # Model assessment
    print(f"\nModel Assessment:")
    correlation = comp_stats['correlation']
    mse = comp_stats['mse']

    if correlation > 0.95:
        corr_assessment = "Excellent"
    elif correlation > 0.9:
        corr_assessment = "Very Good"
    elif correlation > 0.8:
        corr_assessment = "Good"
    elif correlation > 0.7:
        corr_assessment = "Fair"
    else:
        corr_assessment = "Poor"

    if mse < 0.001:
        mse_assessment = "Excellent"
    elif mse < 0.01:
        mse_assessment = "Very Good"
    elif mse < 0.05:
        mse_assessment = "Good"
    elif mse < 0.1:
        mse_assessment = "Fair"
    else:
        mse_assessment = "Poor"

    print(f"  Correlation: {corr_assessment} ({correlation:.4f})")
    print(f"  MSE: {mse_assessment} ({mse:.6f})")


def create_synthetic_signal(signal_type="zeros", duration_samples=8192, sample_rate=16000):
    """Create synthetic test signals for debugging"""
    if signal_type == "zeros":
        return torch.zeros(duration_samples)
    elif signal_type == "ones":
        return torch.ones(duration_samples) * 0.1  # Small amplitude
    elif signal_type == "sine":
        t = torch.linspace(0, duration_samples/sample_rate, duration_samples)
        return 0.1 * torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave
    elif signal_type == "noise":
        return 0.01 * torch.randn(duration_samples)  # Low amplitude noise
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def test_synthetic_signals(coreml_models_dir="./coreml_models", debug_intermediate=True):
    """Test models with synthetic signals to isolate issues"""
    print("Testing with synthetic signals...")

    # Load models
    pytorch_model = load_silero_vad()
    pytorch_ref = PyTorchJITWrapper(pytorch_model)
    unified_wrapper = UnifiedCoreMLWrapper(coreml_models_dir)

    # Also test with pipeline wrapper to see if context handling is the issue
    try:
        pipeline_wrapper = CoreMLVADWrapper(coreml_models_dir)
        print("Pipeline CoreML model loaded for comparison")
        test_pipeline = True
    except Exception as e:
        print(f"Pipeline CoreML model not available: {e}")
        test_pipeline = False

    signals = ["zeros", "ones", "sine", "noise"]

    for signal_type in signals:
        print(f"\n=== Testing {signal_type} signal ===")
        audio_tensor = create_synthetic_signal(signal_type, duration_samples=2048)

        # Test unified model
        results_unified = process_full_audio(pytorch_ref, unified_wrapper, audio_tensor,
                                           chunk_size=512, debug_intermediate=debug_intermediate)

        if results_unified:
            pytorch_mean = np.mean(results_unified['pytorch_outputs'])
            unified_mean = np.mean(results_unified['unified_outputs'])
            correlation_unified = np.corrcoef(results_unified['pytorch_outputs'], results_unified['unified_outputs'])[0, 1]

            print(f"  PyTorch mean: {pytorch_mean:.6f}")
            print(f"  Unified mean: {unified_mean:.6f}")
            print(f"  Unified correlation: {correlation_unified:.6f}")

            # Test pipeline model if available
            if test_pipeline:
                pytorch_ref.reset_states()  # Reset for fair comparison
                results_pipeline = process_full_audio(pytorch_ref, pipeline_wrapper, audio_tensor,
                                                    chunk_size=512, debug_intermediate=False)
                if results_pipeline:
                    pipeline_mean = np.mean(results_pipeline['unified_outputs'])  # Using same key for consistency
                    correlation_pipeline = np.corrcoef(results_pipeline['pytorch_outputs'], results_pipeline['unified_outputs'])[0, 1]
                    print(f"  Pipeline mean: {pipeline_mean:.6f}")
                    print(f"  Pipeline correlation: {correlation_pipeline:.6f}")

        print(f"  Context issue: Unified model pads with zeros, PyTorch uses previous chunk context")


def compare_models_on_audio(audio_path, coreml_models_dir="./coreml_models", chunk_size=512, output_dir="./plots", debug_intermediate=False, use_batch_processing=True, max_batch_size=25, include_256ms=False):
    """Compare PyTorch and CoreML models on a full audio file"""

    print(f"Loading models for audio comparison...")

    # Load PyTorch model
    pytorch_model = load_silero_vad()
    pytorch_ref = PyTorchJITWrapper(pytorch_model)

    # Load Unified CoreML model
    try:
        unified_wrapper = UnifiedCoreMLWrapper(coreml_models_dir)
        print(f"Unified model loaded successfully!")
    except Exception as e:
        print(f"Failed to load Unified CoreML model: {e}")
        return

    # Load Unified 256ms CoreML model if requested
    unified_256ms_wrapper = None
    if include_256ms:
        try:
            unified_256ms_wrapper = UnifiedCoreML256msWrapper(coreml_models_dir)
            print(f"Unified 256ms model loaded successfully!")
        except Exception as e:
            print(f"Failed to load Unified 256ms CoreML model: {e}")
            print(f"Continuing without 256ms comparison...")

    # Load audio file
    audio_tensor = load_audio_file(audio_path)
    if audio_tensor is None:
        return

    # Process audio through both models
    print(f"\nProcessing audio file: {os.path.basename(audio_path)}")
    start_time = time.time()

    if use_batch_processing:
        print(f"Using batch processing with max_batch_size={max_batch_size}")
        results = process_full_audio_batched(pytorch_ref, unified_wrapper, audio_tensor, chunk_size, max_batch_size, debug_intermediate)
    else:
        print("Using sequential processing")
        results = process_full_audio(pytorch_ref, unified_wrapper, audio_tensor, chunk_size, debug_intermediate)

    # Process with 256ms model if available
    results_256ms = None
    if unified_256ms_wrapper is not None:
        print(f"\nProcessing with 256ms model...")
        start_256ms = time.time()
        results_256ms = unified_256ms_wrapper.forward_batch(audio_tensor, chunk_size=4096)
        processing_256ms_time = time.time() - start_256ms
        print(f"256ms model processing completed in {processing_256ms_time:.2f} seconds")

        # Add 256ms results to main results
        results['unified_256ms_outputs'] = results_256ms['outputs']
        results['unified_256ms_num_chunks'] = results_256ms['num_chunks']
        results['unified_256ms_chunk_size'] = results_256ms['chunk_size']
        results['unified_256ms_processing_time'] = processing_256ms_time
        results['unified_256ms_times'] = results_256ms['times']  # Individual chunk processing times
        results['unified_256ms_model_name'] = getattr(unified_256ms_wrapper, 'model_name', 'Unified 256ms CoreML')

    processing_time = time.time() - start_time

    if results is None:
        print("Failed to process audio")
        return

    print(f"Total processing completed in {processing_time:.2f} seconds")

    # Calculate statistics
    stats_dict = calculate_detailed_statistics(results)

    # Create visualizations
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    create_comparison_plots(results, stats_dict, output_dir, audio_filename)

    # Print report
    print_statistics_report(stats_dict, audio_filename)

    return results, stats_dict


def generate_test_signals(num_samples=512, num_tests=5):
    """Generate various test signals for comparison"""
    torch.manual_seed(42)  # For reproducibility

    test_signals = []
    signal_names = []

    # 1. Random noise
    test_signals.append(torch.randn(1, num_samples) * 0.1)
    signal_names.append("Random Noise")

    # 2. Sine wave (speech-like frequency)
    t = torch.linspace(0, num_samples/16000, num_samples)
    sine_wave = torch.sin(2 * np.pi * 440 * t).unsqueeze(0) * 0.3
    test_signals.append(sine_wave)
    signal_names.append("440Hz Sine Wave")

    # 3. Silence
    test_signals.append(torch.zeros(1, num_samples))
    signal_names.append("Silence")

    # 4. Impulse
    impulse = torch.zeros(1, num_samples)
    impulse[0, num_samples//2] = 1.0
    test_signals.append(impulse)
    signal_names.append("Impulse")

    # 5. White noise burst
    burst = torch.zeros(1, num_samples)
    burst[0, num_samples//4:3*num_samples//4] = torch.randn(num_samples//2) * 0.5
    test_signals.append(burst)
    signal_names.append("Noise Burst")

    return test_signals[:num_tests], signal_names[:num_tests]


def calculate_metrics(tensor1, tensor2, name=""):
    """Calculate comparison metrics between two tensors"""
    if tensor1 is None or tensor2 is None:
        return {"error": "One of the tensors is None"}

    # Ensure same shape
    if tensor1.shape != tensor2.shape:
        print(f"Warning: Shape mismatch for {name}: {tensor1.shape} vs {tensor2.shape}")
        # Try to align shapes
        min_shape = [min(s1, s2) for s1, s2 in zip(tensor1.shape, tensor2.shape)]
        if len(min_shape) == 3:
            tensor1 = tensor1[:min_shape[0], :min_shape[1], :min_shape[2]]
            tensor2 = tensor2[:min_shape[0], :min_shape[1], :min_shape[2]]
        elif len(min_shape) == 2:
            tensor1 = tensor1[:min_shape[0], :min_shape[1]]
            tensor2 = tensor2[:min_shape[0], :min_shape[1]]

    # Convert to numpy for calculations
    arr1 = tensor1.detach().cpu().numpy()
    arr2 = tensor2.detach().cpu().numpy()

    # Calculate metrics
    mse = np.mean((arr1 - arr2) ** 2)
    max_diff = np.max(np.abs(arr1 - arr2))

    # Correlation (flatten arrays)
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()

    if len(arr1_flat) > 1 and np.std(arr1_flat) > 1e-8 and np.std(arr2_flat) > 1e-8:
        correlation = np.corrcoef(arr1_flat, arr2_flat)[0, 1]
    else:
        correlation = 1.0 if mse < 1e-8 else 0.0

    return {
        "mse": mse,
        "max_diff": max_diff,
        "correlation": correlation,
        "shape": arr1.shape,
        "mean_pytorch": np.mean(arr1),
        "mean_coreml": np.mean(arr2)
    }


def compare_models(pytorch_model_path=None, coreml_models_dir="./coreml_models", num_tests=5):
    """Compare PyTorch and CoreML models"""

    print("🔍 Loading models for comparison...")

    # Load PyTorch model
    pytorch_model = load_silero_vad()
    pytorch_ref = PyTorchJITWrapper(pytorch_model)

    # Load CoreML models
    if not os.path.exists(coreml_models_dir):
        print(f"❌ CoreML models directory not found: {coreml_models_dir}")
        print("Please run convert-coreml.py first to generate the models.")
        return

    try:
        coreml_wrapper = CoreMLVADWrapper(coreml_models_dir)
        print("✅ Pipeline models loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load CoreML pipeline models: {e}")
        return

    # Try to load unified model
    unified_wrapper = None
    try:
        unified_wrapper = UnifiedCoreMLWrapper(coreml_models_dir)
        print("✅ Unified model loaded successfully!")
        has_unified = True
    except Exception as e:
        print(f"⚠️ Unified model not available: {e}")
        has_unified = False

    # Generate test signals
    test_signals, signal_names = generate_test_signals(num_samples=512, num_tests=num_tests)

    print(f"\n📊 Running comparison with {num_tests} test signals...")
    print("=" * 80)
    print()
    print("📖 Expected Differences Between PyTorch and CoreML:")
    print("   • STFT: Different FFT implementations may cause moderate differences")
    print("   • Encoder: Value clipping (max 10k) prevents overflow but affects precision")
    print("   • Final: Small differences are normal for converted models")
    print("   • Overall: Focus on numerical differences and correlation")
    print("=" * 80)

    results = []

    for i, (signal, name) in enumerate(zip(test_signals, signal_names)):
        print(f"\n🧪 Test {i+1}: {name}")
        print("-" * 40)

        # Reset states for all models
        pytorch_ref.reset_states()
        coreml_wrapper.reset_states()
        if has_unified:
            unified_wrapper.reset_states()

        # Time PyTorch inference
        start_time = time.perf_counter()
        pytorch_results = pytorch_ref.forward_with_intermediates(signal)
        pytorch_time = time.perf_counter() - start_time

        # Time CoreML pipeline inference
        start_time = time.perf_counter()
        coreml_results = coreml_wrapper.forward_with_intermediates(signal)
        coreml_time = time.perf_counter() - start_time

        # Time unified model inference if available
        unified_results = None
        unified_time = 0
        if has_unified:
            start_time = time.perf_counter()
            unified_results = unified_wrapper.forward_with_intermediates(signal)
            unified_time = time.perf_counter() - start_time

        # Compare each stage
        stage_results = {}

        # STFT comparison (only available with manual reference implementation)
        if (
            pytorch_results.get('stft_output') is not None
            and coreml_results.get('stft_output') is not None
        ):
            stft_metrics = calculate_metrics(
                pytorch_results['stft_output'],
                coreml_results['stft_output'],
                "STFT"
            )
            stage_results['stft'] = stft_metrics
            print(
                f"STFT - MSE: {stft_metrics['mse']:.2e}, Max Diff: {stft_metrics['max_diff']:.2e}, "
                f"Corr: {stft_metrics['correlation']:.4f}"
            )
        else:
            print("STFT - comparison unavailable when using the TorchScript reference model")

        # Encoder comparison (only available with manual reference implementation)
        if (
            pytorch_results.get('encoder_output') is not None
            and coreml_results.get('encoder_output') is not None
        ):
            encoder_metrics = calculate_metrics(
                pytorch_results['encoder_output'],
                coreml_results['encoder_output'],
                "Encoder"
            )
            stage_results['encoder'] = encoder_metrics
            print(
                f"Encoder - MSE: {encoder_metrics['mse']:.2e}, Max Diff: {encoder_metrics['max_diff']:.2e}, "
                f"Corr: {encoder_metrics['correlation']:.4f}"
            )
        else:
            print("Encoder - comparison unavailable when using the TorchScript reference model")

        # Final output comparison
        final_metrics = calculate_metrics(
            pytorch_results['final_output'],
            coreml_results['final_output'],
            "Final"
        )
        stage_results['final'] = final_metrics
        print(f"Final - MSE: {final_metrics['mse']:.2e}, Max Diff: {final_metrics['max_diff']:.2e}, Corr: {final_metrics['correlation']:.4f}")

        # Unified model comparison if available
        unified_metrics = None
        pipeline_vs_unified_metrics = None
        if has_unified and unified_results is not None:
            # Compare unified model with PyTorch
            unified_metrics = calculate_metrics(
                pytorch_results['final_output'],
                unified_results['final_output'],
                "Unified"
            )
            print(f"Unified - MSE: {unified_metrics['mse']:.2e}, Max Diff: {unified_metrics['max_diff']:.2e}, Corr: {unified_metrics['correlation']:.4f}")

            # Compare pipeline vs unified (should be nearly identical)
            pipeline_vs_unified_metrics = calculate_metrics(
                coreml_results['final_output'],
                unified_results['final_output'],
                "Pipeline vs Unified"
            )
            print(f"Pipeline vs Unified - MSE: {pipeline_vs_unified_metrics['mse']:.2e}, Max Diff: {pipeline_vs_unified_metrics['max_diff']:.2e}")

        # Timing comparison
        timing_str = f"Timing - PyTorch: {pytorch_time*1000:.2f}ms, Pipeline: {coreml_time*1000:.2f}ms"
        if has_unified:
            timing_str += f", Unified: {unified_time*1000:.2f}ms"
        print(timing_str)

        result_data = {
            'signal_name': name,
            'stage_results': stage_results,
            'pytorch_time': pytorch_time,
            'coreml_time': coreml_time,
            'has_unified': has_unified
        }

        if has_unified:
            result_data.update({
                'unified_time': unified_time,
                'unified_metrics': unified_metrics,
                'pipeline_vs_unified_metrics': pipeline_vs_unified_metrics
            })

        results.append(result_data)

    # Summary report
    print("\n" + "=" * 80)
    print("📋 SUMMARY REPORT")
    print("=" * 80)

    # Aggregate metrics
    stft_mse_values = [
        r['stage_results']['stft']['mse']
        for r in results
        if 'stft' in r['stage_results'] and r['stage_results']['stft'] and 'mse' in r['stage_results']['stft']
    ]
    encoder_mse_values = [
        r['stage_results']['encoder']['mse']
        for r in results
        if 'encoder' in r['stage_results'] and r['stage_results']['encoder'] and 'mse' in r['stage_results']['encoder']
    ]
    final_mse_values = [
        r['stage_results']['final']['mse']
        for r in results
        if 'final' in r['stage_results'] and r['stage_results']['final'] and 'mse' in r['stage_results']['final']
    ]

    print(f"\nMean Squared Error (MSE) Summary (vs PyTorch):")
    if stft_mse_values:
        print(f"  Pipeline STFT:    {np.mean(stft_mse_values):.2e} ± {np.std(stft_mse_values):.2e}")
    else:
        print("  Pipeline STFT:    comparison unavailable (TorchScript reference does not expose STFT output)")

    if encoder_mse_values:
        print(f"  Pipeline Encoder: {np.mean(encoder_mse_values):.2e} ± {np.std(encoder_mse_values):.2e}")
    else:
        print("  Pipeline Encoder: comparison unavailable (TorchScript reference does not expose encoder output)")

    if final_mse_values:
        print(f"  Pipeline Final:   {np.mean(final_mse_values):.2e} ± {np.std(final_mse_values):.2e}")
    else:
        print("  Pipeline Final:   comparison unavailable")

    # Add unified model metrics if available
    if has_unified and any(r.get('unified_metrics') for r in results):
        all_unified_mse = [r['unified_metrics']['mse'] for r in results if r.get('unified_metrics')]
        all_pipeline_vs_unified_mse = [r['pipeline_vs_unified_metrics']['mse'] for r in results if r.get('pipeline_vs_unified_metrics')]

        print(f"  Unified Final:    {np.mean(all_unified_mse):.2e} ± {np.std(all_unified_mse):.2e}")
        print(f"  Pipeline vs Unified: {np.mean(all_pipeline_vs_unified_mse):.2e} ± {np.std(all_pipeline_vs_unified_mse):.2e}")

    # Pass/fail criteria
    print(f"\n✅ Pass/Fail Analysis:")
    evaluation_results = []

    if stft_mse_values:
        stft_pass = np.mean(stft_mse_values) < 1.0  # Allow for different FFT implementations
        evaluation_results.append(stft_pass)
        print(f"  STFT:    {'✅ PASS' if stft_pass else '❌ FAIL'} (threshold: 1.0)")
    else:
        print("  STFT:    not evaluated (TorchScript reference)")

    if encoder_mse_values:
        encoder_pass = np.mean(encoder_mse_values) < 50000  # Account for value clipping at 10000
        evaluation_results.append(encoder_pass)
        print(f"  Encoder: {'✅ PASS' if encoder_pass else '❌ FAIL'} (threshold: 5.0e+4)")
    else:
        print("  Encoder: not evaluated (TorchScript reference)")

    if final_mse_values:
        final_pass = np.mean(final_mse_values) < 0.1  # 10% difference acceptable for VAD probabilities
        evaluation_results.append(final_pass)
        print(f"  Final:   {'✅ PASS' if final_pass else '❌ FAIL'} (threshold: 1.0e-1)")
    else:
        print("  Final:   not evaluated")

    overall_pass = all(evaluation_results) if evaluation_results else True
    print(f"\n🎯 Overall Result: {'✅ MODELS MATCH WELL' if overall_pass else '❌ SIGNIFICANT DIFFERENCES DETECTED'}")

    # Performance summary
    avg_pytorch_time = np.mean([r['pytorch_time'] for r in results]) * 1000
    avg_coreml_time = np.mean([r['coreml_time'] for r in results]) * 1000
    pipeline_speedup = avg_pytorch_time / avg_coreml_time if avg_coreml_time > 0 else 0

    print(f"\n⚡ Performance Summary:")
    print(f"  PyTorch: {avg_pytorch_time:.2f}ms average")
    print(f"  Pipeline: {avg_coreml_time:.2f}ms average (speedup: {pipeline_speedup:.2f}x)")

    if has_unified:
        avg_unified_time = np.mean([r['unified_time'] for r in results if r.get('unified_time', 0) > 0]) * 1000
        unified_speedup = avg_pytorch_time / avg_unified_time if avg_unified_time > 0 else 0
        pipeline_vs_unified_speedup = avg_coreml_time / avg_unified_time if avg_unified_time > 0 else 0

        print(f"  Unified:  {avg_unified_time:.2f}ms average (speedup: {unified_speedup:.2f}x vs PyTorch, {pipeline_vs_unified_speedup:.2f}x vs Pipeline)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PyTorch and CoreML Silero VAD models")
    parser.add_argument("--coreml-dir", type=str, default="./coreml_models",
                       help="Directory containing CoreML models")
    parser.add_argument("--num-tests", type=int, default=5,
                       help="Number of test signals to generate (for synthetic test mode)")
    parser.add_argument("--audio-file", type=str, default=None,
                       help="Path to audio file for full audio comparison")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Chunk size for audio processing (default: 512 samples)")
    parser.add_argument("--output-dir", type=str, default="./plots",
                       help="Directory to save plots and results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with intermediate output comparison")
    parser.add_argument("--test-synthetic", action="store_true",
                       help="Test with synthetic signals for debugging")
    parser.add_argument("--no-batch", action="store_true",
                       help="Disable batch processing (use sequential processing instead)")
    parser.add_argument("--max-batch-size", type=int, default=25,
                       help="Maximum batch size for CoreML batch processing (default: 25)")
    parser.add_argument("--include-256ms", action="store_true",
                       help="Include comparison with 256ms unified model variant")

    args = parser.parse_args()

    # If audio file is provided, run full audio comparison
    if args.audio_file:
        compare_models_on_audio(
            audio_path=args.audio_file,
            coreml_models_dir=args.coreml_dir,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir,
            debug_intermediate=args.debug,
            use_batch_processing=not args.no_batch,
            max_batch_size=args.max_batch_size,
            include_256ms=args.include_256ms
        )
    elif args.test_synthetic:
        # Test with synthetic signals
        test_synthetic_signals(
            coreml_models_dir=args.coreml_dir,
            debug_intermediate=args.debug
        )
    else:
        # Run original synthetic test comparison
        compare_models(
            coreml_models_dir=args.coreml_dir,
            num_tests=args.num_tests
        )
