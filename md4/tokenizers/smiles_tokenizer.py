"""HuggingFace-based tokenizer for SMILES strings.

This tokenizer wraps HuggingFace's PreTrainedTokenizerFast
for compatibility with existing SMILES tokenizers.
"""

from typing import List, Sequence, Union, Optional
import numpy as np
import transformers

from .base import BaseTokenizer


class SMILESTokenizer(BaseTokenizer):
    """SMILES tokenizer using HuggingFace transformers.
    
    This wraps a PreTrainedTokenizerFast for SMILES tokenization.
    Simple wrapper, no magic.
    """
    
    def __init__(self, 
                 tokenizer_path: str,
                 max_length: Optional[int] = None,
                 padding: str = "max_length",
                 truncation: bool = True,
                 add_special_tokens: bool = True):
        """Initialize SMILES tokenizer.
        
        Args:
            tokenizer_path: Path to pretrained tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add special tokens
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.padding = padding if max_length else False
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        
        # Cache special token IDs
        self._vocab_size = self.tokenizer.vocab_size
        self._pad_id = self.tokenizer.pad_token_id or 0
        self._unk_id = self.tokenizer.unk_token_id or 1
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size
    
    @property
    def pad_id(self) -> int:
        """Return padding token ID."""
        return self._pad_id
    
    @property
    def unk_id(self) -> int:
        """Return unknown token ID."""
        return self._unk_id
    
    def encode(self, 
               text: str, 
               max_length: Optional[int] = None,
               padding: Optional[str] = None,
               truncation: Optional[bool] = None,
               add_special_tokens: Optional[bool] = None,
               return_tensors: Optional[str] = "np") -> Union[List[int], np.ndarray]:
        """Encode SMILES string to token IDs.
        
        Args:
            text: Input SMILES string (or bytes)
            max_length: Override default max_length
            padding: Override default padding strategy
            truncation: Override default truncation
            add_special_tokens: Override default special tokens
            return_tensors: Return type ("np" for numpy, None for list)
            
        Returns:
            Token IDs as list or numpy array
        """
        # Handle bytes input
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        
        # Use provided values or defaults
        max_length = max_length or self.max_length
        padding = padding if padding is not None else self.padding
        truncation = truncation if truncation is not None else self.truncation
        add_special_tokens = add_special_tokens if add_special_tokens is not None else self.add_special_tokens
        
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding if max_length else False,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
        
        # Flatten if numpy array
        if isinstance(encoded, np.ndarray):
            encoded = encoded.reshape(-1)
        
        return encoded
    
    def decode(self, 
               token_ids: Sequence[int],
               skip_special_tokens: bool = True,
               **kwargs) -> str:
        """Decode token IDs to SMILES string.
        
        Args:
            token_ids: Sequence of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded SMILES string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(self, 
                     token_ids: Sequence[Sequence[int]],
                     skip_special_tokens: bool = True,
                     **kwargs) -> List[str]:
        """Decode multiple sequences of token IDs.
        
        Args:
            token_ids: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded SMILES strings
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    # Expose underlying tokenizer for compatibility
    def __getattr__(self, name):
        """Forward attribute access to underlying tokenizer."""
        return getattr(self.tokenizer, name)