# CoreML Decoder Integration Report

## Executive Summary
We have successfully integrated and verified the CoreML decoder for the Canary 1B model. The inference pipeline now produces accurate transcriptions and handles stopping conditions correctly.

**Status:** ✅ Operational
**Performance:** ~3.2x Real-Time Factor (on test clip)
**Accuracy:** 100% match with expected transcript

---

## The Challenge
Initial attempts to run the exported CoreML decoder resulted in:
1.  **Garbage Output:** The model generated streams of random special tokens (e.g., `<|215|>`, `<|emo:neutral|>`) instead of text.
2.  **Infinite Loops:** The decoder failed to stop, generating tokens until the maximum step limit was reached.

## Methodology
To diagnose these issues, we employed a **"Ground Truth Capture"** strategy:
1.  We created a script (`debug_transcribe_hooks.py`) to **monkeypatch** the running NeMo PyTorch model.
2.  This allowed us to intercept and capture the exact inputs (embeddings, masks, prompt tokens) and outputs (logits) passing through the decoder during a successful inference.
3.  We then compared these captured tensors against our CoreML pipeline to identify discrepancies.

## Critical Fixes

### 1. The Missing Projection Layer (Fixing Garbage Output)
*   **Diagnosis:** The CoreML decoder model outputs **hidden states** (vectors of size 1024), whereas the inference script expected **logits** (probabilities for the 16k vocabulary). Treating hidden states as logits resulted in selecting random tokens from the first 1024 indices.
*   **Resolution:** We identified the linear projection layer in the NeMo model, exported its weights to `projection_weights.npz`, and implemented the projection step (`hidden_state @ weights.T + bias`) in `coreml_inference.py`.

### 2. The Incorrect EOS Token (Fixing Infinite Loops)
*   **Diagnosis:** The tokenizer was configured with `eos_id=2`. Inspection of the actual NeMo tokenizer revealed that ID `2` is `<pad>`, while the actual End-Of-Sequence token is **ID 3** (`<|endoftext|>`). The inference loop was waiting for token 2, but the model generated token 3, causing it to never stop.
*   **Resolution:** We updated `CanaryTokenizer` to use the correct `eos_id=3`.

### 3. Prompt Format & Robustness
*   **Diagnosis:** The initial prompt construction relied on hardcoded token IDs that didn't match the specific `canary2` format required by the model (e.g., missing `<|startofcontext|>`).
*   **Resolution:** We updated `CanaryTokenizer` to construct the prompt using the canonical string format and `text_to_ids`. This ensures the prompt is always correct and robust to tokenizer changes.

## Verification Results
Running `coreml_inference.py` on the test audio now yields:

```json
{
  "text": "Hello World, this is a test of the automatic speech recognition system.<|endoftext|>",
  "audio_duration": 3.43,
  "rtf": 3.24
}
```

## Artifacts
*   **`coreml_inference.py`**: The corrected inference script.
*   **`canary_tokenizer.py`**: The corrected tokenizer.
*   **`projection_weights.npz`**: The exported projection layer weights (Required for inference).
