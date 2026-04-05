#!/usr/bin/env python3
"""Canary BPE tokenizer wrapper for CoreML inference."""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import nemo.collections.asr as nemo_asr


class CanaryTokenizer:
    """Wrapper for Canary's BPE tokenizer with 16,384 tokens."""

    def __init__(self, model_path: str = "nvidia/canary-1b-v2"):
        """Initialize the tokenizer from the NeMo model."""
        print("Loading Canary tokenizer...")

        # Load the model to get the tokenizer
        self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
            model_path, map_location="cpu"
        )
        self.tokenizer = self.model.tokenizer

        # Get special tokens
        self.vocab_size = 16384

        # Special token IDs (inferred from Canary-1B v2 / canary2 format)
        self.special_tokens = {
            'pad': 0,
            'bos': 16053, # Used as prefix in text_to_ids
            'eos': 3,     # <|endoftext|>
            
            # Canary2 special tokens
            'startofcontext': 7,
            'startoftranscript': 4,
            'emo_undefined': 16,
            'pnc': 5,
            'nopnc': 6,
            'noitn': 9,
            'notimestamp': 11,
            'nodiarize': 13,
        }

        # Language tokens
        self.language_tokens = {
            'en': 64,
            'es': 65, # Guessing, need to verify if needed
            'de': 66,
            'fr': 67,
        }

        # Build reverse vocab for decoding
        self._build_reverse_vocab()

    def _build_reverse_vocab(self):
        """Build ID to token mapping."""
        self.id_to_token = {}

        # The tokenizer is a list of tokenizers, get the actual tokenizer
        if isinstance(self.tokenizer, list):
            actual_tokenizer = self.tokenizer[0] if self.tokenizer else None
        else:
            actual_tokenizer = self.tokenizer

        if actual_tokenizer:
            # Try to get the vocabulary
            if hasattr(actual_tokenizer, 'vocab'):
                if hasattr(actual_tokenizer.vocab, 'items'):
                    for token, token_id in actual_tokenizer.vocab.items():
                        self.id_to_token[token_id] = token
                else:
                    print("Warning: vocab attribute is not a dictionary")
            elif hasattr(actual_tokenizer, 'get_vocab'):
                vocab = actual_tokenizer.get_vocab()
                for token, token_id in vocab.items():
                    self.id_to_token[token_id] = token
            else:
                # Fallback: use the model's decoding method
                print("Warning: Could not access tokenizer vocabulary directly")
        else:
            print("Warning: Could not access tokenizer")

    def create_prompt_tokens(
        self,
        language: str = "en",
        task: str = "transcribe",
        pnc: bool = True,
        include_timestamps: bool = False,
    ) -> List[int]:
        """Create the prompt tokens for the decoder using the canary2 format."""
        # Format: <|startofcontext|><|startoftranscript|><|emo:undefined|><|source_lang|><|target_lang|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
        
        # We assume source_lang == target_lang for transcribe
        # For translate, we might need to handle it differently, but for now assume en->en
        
        # Construct the prompt string based on canary2 format
        # <|startofcontext|><|startoftranscript|><|emo:undefined|><|source_lang|><|target_lang|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
        
        # Map lang code to token string if needed, but we have IDs.
        # Actually, using text_to_ids is safer as it handles prefixing and special tokens correctly.
        
        # We need to map 'en' to '<|en|>' etc.
        # The tokenizer vocab should have these.
        
        lang_token = f"<|{language}|>"
        pnc_token = "<|pnc|>" if pnc else "<|nopnc|>"
        timestamp_token = "<|notimestamp|>" if not include_timestamps else "" # Timestamps not fully supported in this snippet
        
        prompt_str = (
            "<|startofcontext|>"
            "<|startoftranscript|>"
            "<|emo:undefined|>"
            f"{lang_token}"
            f"{lang_token}" # Target lang same as source for transcribe
            f"{pnc_token}"
            "<|noitn|>"
            f"{timestamp_token}"
            "<|nodiarize|>"
        )
        
        # Encode
        tokens = self.encode(prompt_str)
        
        # Verify it starts with 16053 (BOS) if expected
        # The encode method uses text_to_ids which usually adds it.
        
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        # Get the actual tokenizer if it's a list
        if isinstance(self.tokenizer, list):
            actual_tokenizer = self.tokenizer[0] if self.tokenizer else None
        else:
            actual_tokenizer = self.tokenizer

        # Use the model's tokenizer decode method
        if actual_tokenizer:
            if hasattr(actual_tokenizer, 'ids_to_text'):
                return actual_tokenizer.ids_to_text(token_ids)
            elif hasattr(actual_tokenizer, 'decode'):
                return actual_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        # Fallback: manual decoding
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token.startswith('<|') and token.endswith('|>'):
                    continue
                tokens.append(token)
        return ''.join(tokens).replace('▁', ' ').strip()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Get the actual tokenizer if it's a list
        if isinstance(self.tokenizer, list):
            actual_tokenizer = self.tokenizer[0] if self.tokenizer else None
        else:
            actual_tokenizer = self.tokenizer

        if actual_tokenizer:
            if hasattr(actual_tokenizer, 'text_to_ids'):
                return actual_tokenizer.text_to_ids(text)
            elif hasattr(actual_tokenizer, 'encode'):
                return actual_tokenizer.encode(text)

        raise NotImplementedError("Tokenizer encode method not available")


def save_tokenizer_vocab(tokenizer: CanaryTokenizer, output_path: str):
    """Save tokenizer vocabulary to JSON for reference."""
    vocab_data = {
        'vocab_size': tokenizer.vocab_size,
        'special_tokens': tokenizer.special_tokens,
        'language_tokens': tokenizer.language_tokens,
        'id_to_token_sample': {
            str(k): v for k, v in list(tokenizer.id_to_token.items())[:100]
        }
    }

    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)

    print(f"Tokenizer vocabulary saved to {output_path}")


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = CanaryTokenizer()

    # Create prompt tokens
    prompt = tokenizer.create_prompt_tokens(
        language="en",
        task="transcribe",
        pnc=True
    )
    print(f"Prompt tokens: {prompt}")

    # Test encoding/decoding
    test_text = "Hello, this is a test."
    encoded = tokenizer.encode(test_text)
    print(f"Encoded '{test_text}': {encoded[:20]}...")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded back: '{decoded}'")

    # Save vocabulary for reference
    save_tokenizer_vocab(tokenizer, "canary_tokenizer_vocab.json")
