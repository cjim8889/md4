"""SentencePiece tokenizer for molecular SMILES.

This tokenizer uses TensorFlow's SentencePiece implementation.
Clean, efficient, no bullshit.
"""

from typing import List, Sequence, Union
import numpy as np
import tensorflow as tf
import tensorflow_text as tftxt
import jax.numpy as jnp

from .base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece tokenizer using TensorFlow implementation.
    
    This is the heavy-duty tokenizer for when you need proper
    subword tokenization with BPE or unigram models.
    """

    def __init__(self, model_path: str, add_bos: bool = True, add_eos: bool = True):
        """Initialize SentencePiece tokenizer.
        
        Args:
            model_path: Path to the SentencePiece model file
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
        """
        print(f"Loading SentencePiece model from: {model_path}")
        with tf.io.gfile.GFile(model_path, "rb") as model_fp:
            sp_model = model_fp.read()
        
        self.sp_tokenizer = tftxt.SentencepieceTokenizer(
            model=sp_model, 
            add_bos=add_bos, 
            add_eos=add_eos, 
            reverse=False
        )
        
        # Cache special token IDs
        self._vocab_size = self.sp_tokenizer.vocab_size().numpy()
        self._pad_id = self.sp_tokenizer.string_to_id("[PAD]").numpy()
        self._unk_id = self.sp_tokenizer.string_to_id("[UNK]").numpy()
        self._sep_id = self.sp_tokenizer.string_to_id("[SEP]").numpy()
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return int(self._vocab_size)
    
    @property
    def pad_id(self) -> int:
        """Return padding token ID."""
        return int(self._pad_id)
    
    @property
    def unk_id(self) -> int:
        """Return unknown token ID."""
        return int(self._unk_id)
    
    @property
    def sep_id(self) -> int:
        """Return separator token ID."""
        return int(self._sep_id)
    
    def encode(self, text: str, **kwargs) -> np.ndarray:
        """Encode text to token IDs.
        
        Args:
            text: Input SMILES string
            
        Returns:
            Token IDs as numpy array or TensorFlow tensor
        """
        tokens = self.sp_tokenizer.tokenize(text)
        # If in eager mode, convert to numpy, otherwise return tensor
        if tf.executing_eagerly():
            return tokens.numpy()
        return tokens
    
    def decode(self, token_ids: Sequence[int], **kwargs) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: Sequence of token IDs
            
        Returns:
            Decoded SMILES string
        """
        # Convert to int32 for SentencePiece
        token_ids = np.array(token_ids, dtype=np.int32)
        decoded = self.sp_tokenizer.detokenize(token_ids)
        if tf.executing_eagerly():
            return decoded.numpy().decode('utf-8')
        return decoded
    
    def batch_decode(self, token_ids: Sequence[Sequence[int]], **kwargs) -> List[str]:
        """Decode multiple sequences with padding removal.
        
        Args:
            token_ids: Batch of token ID sequences
            
        Returns:
            List of decoded SMILES strings
        """
        results = []
        for seq in self._decode_with_padding_removal(token_ids):
            if isinstance(seq, bytes):
                results.append(seq.decode('utf-8'))
            else:
                results.append(str(seq))
        return results
    
    def _decode_with_padding_removal(self, token_ids):
        """Decode tokens after removing padding tokens.
        
        This handles multiple input types: numpy arrays, JAX arrays, lists.
        """
        # Handle multi-dimensional numpy/JAX arrays
        if isinstance(token_ids, (np.ndarray, jnp.ndarray)):
            if token_ids.ndim > 1:
                # Process each sequence separately
                results = []
                for seq in token_ids:
                    # Remove padding tokens
                    seq_no_padding = seq[seq != self.pad_id]
                    # Convert to int32 for SentencePiece
                    seq_no_padding = seq_no_padding.astype(np.int32)
                    decoded = self.sp_tokenizer.detokenize(seq_no_padding)
                    if tf.executing_eagerly():
                        decoded = decoded.numpy()
                    results.append(decoded)
                return results
            else:
                # 1D array
                t_no_padding = token_ids[token_ids != self.pad_id]
                t_no_padding = t_no_padding.astype(np.int32)
                decoded = self.sp_tokenizer.detokenize(t_no_padding)
                if tf.executing_eagerly():
                    return decoded.numpy()
                return decoded
        else:
            # Handle lists or other sequences
            if hasattr(token_ids, "__len__") and len(token_ids) > 0 and hasattr(token_ids[0], "__len__"):
                # Nested sequences (batch)
                results = []
                for seq in token_ids:
                    seq_no_padding = [token for token in seq if token != self.pad_id]
                    seq_no_padding = np.array(seq_no_padding, dtype=np.int32)
                    decoded = self.sp_tokenizer.detokenize(seq_no_padding)
                    if tf.executing_eagerly():
                        decoded = decoded.numpy()
                    results.append(decoded)
                return results
            else:
                # Single sequence
                t_no_padding = [token for token in token_ids if token != self.pad_id]
                t_no_padding = np.array(t_no_padding, dtype=np.int32)
                decoded = self.sp_tokenizer.detokenize(t_no_padding)
                if tf.executing_eagerly():
                    return decoded.numpy()
                return decoded