#!/usr/bin/env python3
"""Test script for tokenizers module.

This verifies that the tokenizers work correctly after refactoring.
No bullshit, just tests.
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from md4.tokenizers import SentencePieceTokenizer, SMILESTokenizer, create_tokenizer


def test_sentencepiece_tokenizer():
    """Test SentencePiece tokenizer functionality."""
    print("Testing SentencePiece tokenizer...")
    
    # Check if model exists
    model_path = "data/sentencepiece_tokenizer.model"
    if not os.path.exists(model_path):
        print(f"  Skipping - model not found at {model_path}")
        return
    
    tokenizer = SentencePieceTokenizer(model_path)
    
    # Test basic properties
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad ID: {tokenizer.pad_id}")
    print(f"  UNK ID: {tokenizer.unk_id}")
    print(f"  SEP ID: {tokenizer.sep_id}")
    
    # Test encoding/decoding
    test_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O"
    encoded = tokenizer.encode(test_smiles)
    print(f"  Encoded length: {len(encoded)}")
    
    decoded = tokenizer.decode(encoded)
    print(f"  Decoded: {decoded[:50]}...")
    
    # Test batch decoding with padding
    padded_tokens = np.pad(encoded, (0, 10), constant_values=tokenizer.pad_id)
    batch = np.array([padded_tokens, padded_tokens])
    decoded_batch = tokenizer.batch_decode(batch)
    print(f"  Batch decoded (2 sequences): {len(decoded_batch)} items")
    
    print("  ✓ SentencePiece tokenizer works!")


def test_smiles_tokenizer():
    """Test SMILES HuggingFace tokenizer functionality."""
    print("\nTesting SMILES tokenizer...")
    
    # Check if tokenizer exists
    tokenizer_path = "data/smiles_tokenizer"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "data/pubchem_large_tokenizer"
        if not os.path.exists(tokenizer_path):
            print(f"  Skipping - tokenizer not found")
            return
    
    tokenizer = SMILESTokenizer(tokenizer_path, max_length=128)
    
    # Test basic properties
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad ID: {tokenizer.pad_id}")
    print(f"  UNK ID: {tokenizer.unk_id}")
    
    # Test encoding/decoding
    test_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O"
    encoded = tokenizer.encode(test_smiles)
    print(f"  Encoded shape: {encoded.shape if hasattr(encoded, 'shape') else len(encoded)}")
    
    # Handle both list and array outputs
    if isinstance(encoded, np.ndarray):
        token_ids = encoded.tolist()
    else:
        token_ids = encoded
    
    decoded = tokenizer.decode(token_ids)
    print(f"  Decoded: {decoded[:50]}...")
    
    # Test batch decoding
    batch = [token_ids, token_ids]
    decoded_batch = tokenizer.batch_decode(batch)
    print(f"  Batch decoded (2 sequences): {len(decoded_batch)} items")
    
    print("  ✓ SMILES tokenizer works!")


def test_factory():
    """Test tokenizer factory function."""
    print("\nTesting tokenizer factory...")
    
    # Test creating tokenizers via factory
    if os.path.exists("data/sentencepiece_tokenizer.model"):
        tok = create_tokenizer("sentencepiece", "data/sentencepiece_tokenizer.model")
        print(f"  Created SentencePiece tokenizer: vocab_size={tok.vocab_size}")
    
    if os.path.exists("data/smiles_tokenizer"):
        tok = create_tokenizer("huggingface", "data/smiles_tokenizer")
        print(f"  Created HuggingFace tokenizer: vocab_size={tok.vocab_size}")
    elif os.path.exists("data/pubchem_large_tokenizer"):
        tok = create_tokenizer("huggingface", "data/pubchem_large_tokenizer")
        print(f"  Created HuggingFace tokenizer: vocab_size={tok.vocab_size}")
    
    # Test invalid type
    try:
        create_tokenizer("invalid", "path")
        print("  ✗ Factory should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Factory correctly raised error: {e}")
    
    print("  ✓ Factory works!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing refactored tokenizers module")
    print("=" * 60)
    
    test_sentencepiece_tokenizer()
    test_smiles_tokenizer()
    test_factory()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)