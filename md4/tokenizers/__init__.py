"""Tokenizers module for molecular data processing.

This module provides tokenizers for SMILES strings using different backends:
- SentencePiece (TensorFlow-based)
- HuggingFace Transformers

Keep it simple, keep it modular.
"""

from .base import BaseTokenizer
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .smiles_tokenizer import SMILESTokenizer


def create_tokenizer(tokenizer_type: str, tokenizer_path: str, **kwargs) -> BaseTokenizer:
    """Factory function to create tokenizers.
    
    Args:
        tokenizer_type: Type of tokenizer ("sentencepiece" or "huggingface")
        tokenizer_path: Path to tokenizer model
        **kwargs: Additional tokenizer-specific arguments
        
    Returns:
        Tokenizer instance
        
    Raises:
        ValueError: If tokenizer_type is not recognized
    """
    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(tokenizer_path, **kwargs)
    elif tokenizer_type == "huggingface":
        return SMILESTokenizer(tokenizer_path, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Use 'sentencepiece' or 'huggingface'.")


__all__ = [
    "BaseTokenizer",
    "SentencePieceTokenizer", 
    "SMILESTokenizer",
    "create_tokenizer"
]