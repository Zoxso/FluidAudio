# How We Built the CoreML Decoder Algorithm

This document explains the process of reverse-engineering the NeMo decoding logic and reconstructing it for CoreML.

## 1. The Challenge
We started with a set of raw CoreML models (`canary_encoder`, `canary_decoder`) exported from NeMo, but we lacked the "glue code" to make them work together. The NeMo codebase uses a complex `BeamSearchSequenceGenerator` that wraps the decoder, handling masking, caching, and token generation. Our goal was to replicate this logic in a simple, standalone script (`coreml_inference.py`).

## 2. The Reverse Engineering Process

To understand exactly how to feed the CoreML decoder, we used a **"Ground Truth Capture"** strategy.

### Step 1: Spying on the Black Box
We couldn't easily trace the execution flow of NeMo's massive codebase. Instead, we **monkeypatched** the running PyTorch model to intercept data at the exact moment it entered the decoder.
*   **Tool**: `debug_transcribe_hooks.py`
*   **What we captured**:
    *   `encoder_embeddings`: The output of the encoder (which feeds the decoder).
    *   `decoder_input_ids`: The exact sequence of tokens fed to the decoder.
    *   `decoder_mask`: The attention mask used to hide future tokens.

### Step 2: The "Hidden State" Discovery
When we ran the CoreML decoder with these captured inputs, the output shape was `(Batch, Time, 1024)`.
*   **Problem**: Our vocabulary size is ~16,384. We expected logits (probabilities) of shape `(..., 16384)`, but got `1024`.
*   **Insight**: The CoreML model was only the **Transformer Decoder body**. It outputted *hidden states* (internal representations), not final token probabilities. The "Token Classifier" (Projection Layer) was missing!

### Step 3: Finding the Missing Layer
We searched the NeMo model structure for a linear layer that maps `1024 -> 16384`.
*   **Found**: `model.decoding.decoding.beam_search.project` (or similar depending on config).
*   **Action**: We exported these weights to `projection_weights.npz`.

### Step 4: The Tokenizer Mismatch
Even with the projection layer, the decoder entered an infinite loop of `<|endoftext|>`.
*   **Investigation**: We checked the tokenizer configuration.
*   **Finding**: The config said `eos_id=2`. But in reality, ID `2` was `<pad>` and ID `3` was `<|endoftext|>`. The model was generating `3`, but our loop was waiting for `2`.
*   **Fix**: Updated the tokenizer to use the correct EOS ID.

## 3. The Final Algorithm

We reconstructed the **Greedy Decoding** algorithm by assembling these pieces. Here is the logic flow implemented in `coreml_inference.py`:

### Phase 1: Encoding
1.  **Audio -> Mel Spectrogram**: Use `canary_preprocessor.mlpackage`.
2.  **Mel -> Embeddings**: Use `canary_encoder.mlpackage`.
    *   *Result*: `encoder_embeddings` (Sequence of 1024-dim vectors).

### Phase 2: Initialization
1.  **Prompt Construction**: Create the initial token sequence using the `canary2` format:
    `[BOS, <|startofcontext|>, <|startoftranscript|>, ..., <|nodiarize|>]`
2.  **State Setup**: Initialize `input_ids` with the prompt and `decoder_mask` with 1s for the prompt.

### Phase 3: The Autoregressive Loop
For each step `t` from `len(prompt)` to `max_steps`:

1.  **Run Decoder**:
    *   Feed `input_ids` (current history) and `encoder_embeddings` to `canary_decoder.mlpackage`.
    *   *Output*: `hidden_states` (Sequence of 1024-dim vectors).

2.  **Project to Logits**:
    *   Take the hidden state at the last valid position: `h = hidden_states[t-1]`.
    *   Apply the exported projection weights: `logits = h @ W.T + b`.
    *   *Result*: A probability distribution over the 16k vocabulary.

3.  **Select Token (Greedy)**:
    *   `next_token = argmax(logits)`.

4.  **Update & Check**:
    *   Append `next_token` to `input_ids`.
    *   Update `decoder_mask` to include the new token.
    *   **Stop Condition**: If `next_token == EOS_ID (3)`, break the loop.

## Summary
The "Algorithm" is essentially a manual implementation of the standard Transformer generation loop, but with two critical external dependencies that were stripped from the CoreML model:
1.  **The Projection Weights** (restored via `projection_weights.npz`).
2.  **The Tokenizer Logic** (reimplemented in `CanaryTokenizer`).
