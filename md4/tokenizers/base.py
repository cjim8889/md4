"""Base tokenizer protocol for molecular tokenizers.

This module defines the interface that all tokenizers must implement.
Keep it simple, keep it clean.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence, Optional, Union
import numpy as np


class BaseTokenizer(ABC):
    """Abstract base class for all molecular tokenizers.
    
    This defines the minimal interface that every tokenizer must implement.
    No fancy shit, just what's needed.
    """
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        pass
    
    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Return the padding token ID."""
        pass
    
    @property
    @abstractmethod
    def unk_id(self) -> int:
        """Return the unknown token ID."""
        pass
    
    @abstractmethod
    def encode(self, text: str, **kwargs) -> Union[List[int], np.ndarray]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            **kwargs: Additional encoding parameters
            
        Returns:
            Token IDs as list or numpy array
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: Sequence[int], **kwargs) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: Sequence of token IDs
            **kwargs: Additional decoding parameters
            
        Returns:
            Decoded text string
        """
        pass
    
    @abstractmethod
    def batch_decode(self, token_ids: Sequence[Sequence[int]], **kwargs) -> List[str]:
        """Decode multiple sequences of token IDs.
        
        Args:
            token_ids: Batch of token ID sequences
            **kwargs: Additional decoding parameters
            
        Returns:
            List of decoded text strings
        """
        pass