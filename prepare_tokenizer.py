#!/usr/bin/env python3
"""
Script to train a custom tokenizer on SMILES strings using the transformers library.
Includes special tokens for padding, masking, and end of sentence.
"""

import os
import argparse
from typing import List, Iterator
import pandas as pd
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace, Sequence, Split
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


def load_smiles_from_csv(csv_files: List[str]) -> Iterator[str]:
    """Load SMILES strings from CSV files."""
    for csv_file in csv_files:
        print(f"Loading SMILES from {csv_file}...")
        df = pd.read_parquet(csv_file)
        if 'smiles' in df.columns:
            for smiles in df['smiles'].to_list():
                if smiles and isinstance(smiles, str):
                    yield smiles
        else:
            print(f"Warning: 'smiles' column not found in {csv_file}")


def create_smiles_pre_tokenizer():
    """Create a pre-tokenizer suitable for SMILES strings."""
    # SMILES use specific characters - we'll split on common atom/bond boundaries
    # This helps the tokenizer learn meaningful chemical substructures
    return Sequence([
        Split(pattern=r'(\[|\]|\(|\)|=|#|@|\+|\-|%|\d+)', behavior="isolated"),
        Whitespace()
    ])


def train_smiles_tokenizer(
    csv_files: List[str],
    vocab_size: int = 1000,
    output_dir: str = "data",
    tokenizer_name: str = "smiles_tokenizer"
) -> PreTrainedTokenizerFast:
    """
    Train a BPE tokenizer on SMILES strings with special tokens.
    
    Args:
        csv_files: List of CSV files containing SMILES strings
        vocab_size: Target vocabulary size
        output_dir: Directory to save the tokenizer
        tokenizer_name: Name for the tokenizer files
        
    Returns:
        Trained tokenizer
    """
    
    # Define special tokens
    special_tokens = [
        "[PAD]",    # Padding token
        "[UNK]",    # Unknown token  
        "[CLS]",    # Classification token (start of sequence)
        "[SEP]",    # Separator token (end of sequence)
    ]
    
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Set up pre-tokenizer for SMILES
    tokenizer.pre_tokenizer = create_smiles_pre_tokenizer()
    
    # Optional: normalize to lowercase (uncomment if needed)
    # tokenizer.normalizer = Lowercase()
    
    # Set up trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )
    
    # Collect SMILES strings for training
    print("Collecting SMILES strings for tokenizer training...")
    smiles_texts = list(load_smiles_from_csv(csv_files))
    
    if not smiles_texts:
        raise ValueError("No SMILES strings found in the provided CSV files!")
    
    print(f"Found {len(smiles_texts)} SMILES strings for training")
    print("Sample SMILES:")
    for i, smiles in enumerate(smiles_texts[:5]):
        print(f"  {i+1}: {smiles}")
    
    # Train the tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(smiles_texts, trainer=trainer)
    
    # Add post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Create transformers tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, tokenizer_name)
    fast_tokenizer.save_pretrained(tokenizer_path)
    
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {fast_tokenizer.vocab_size}")
    
    return fast_tokenizer


def test_tokenizer(tokenizer: PreTrainedTokenizerFast, test_smiles: List[str]):
    """Test the trained tokenizer on sample SMILES strings."""
    print("\n" + "="*50)
    print("TESTING TOKENIZER")
    print("="*50)
    
    for i, smiles in enumerate(test_smiles[:3]):
        print(f"\nTest {i+1}: {smiles}")
        
        # Tokenize
        encoded = tokenizer(smiles, return_tensors="np", padding=True, truncation=True)
        tokens = encoded['input_ids'][0].tolist()
        
        print(f"  Tokens: {tokens}")
        print(f"  Token strings: {[tokenizer.decode([t]) for t in tokens]}")
        print(f"  Attention mask: {encoded['attention_mask'][0].tolist()}")
        
        # Decode back
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Decoded: {decoded}")
        print(f"  Match original: {decoded.replace(' ', '') == smiles.replace(' ', '')}")


def main():
    parser = argparse.ArgumentParser(description="Train a SMILES tokenizer")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSV files")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--output-dir", default="data", help="Output directory for tokenizer")
    parser.add_argument("--tokenizer-name", default="smiles_tokenizer", help="Name for the tokenizer")
    
    args = parser.parse_args()
    
    # CSV files to process
    splits = {'train': 'data/train-00000-of-00001-e9b227f8c7259c8b.parquet', 'validation': 'data/validation-00000-of-00001-9368b7243ba1bff8.parquet'}
    hf_prefix = "hf://datasets/sagawa/pubchem-10m-canonicalized/"
    csv_files = [hf_prefix + splits["train"], hf_prefix + splits["validation"]]
    
    # Train tokenizer
    tokenizer = train_smiles_tokenizer(
        csv_files=csv_files,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name
    )
    
    # Test with sample SMILES
    test_smiles = list(load_smiles_from_csv([csv_files[0]]))[:5]  # Get first 5 from train
    if test_smiles:
        test_tokenizer(tokenizer, test_smiles)
    
    print(f"\nTokenizer training complete!")
    print(f"To use the tokenizer:")
    print(f"  from transformers import PreTrainedTokenizerFast")
    print(f"  tokenizer = PreTrainedTokenizerFast.from_pretrained('{os.path.join(args.output_dir, args.tokenizer_name)}')")


if __name__ == "__main__":
    main() 