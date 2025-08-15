#!/usr/bin/env python3
"""
Script to train a custom tokenizer on SMILES strings using the transformers library.
Includes special tokens for padding, masking, and end of sentence.
"""

import os
import argparse
from typing import List, Optional
import datasets
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


def create_smiles_pre_tokenizer():
    """Create a pre-tokenizer suitable for SMILES strings."""
    # SMILES use specific characters - we'll split on common atom/bond boundaries
    # This helps the tokenizer learn meaningful chemical substructures
    return Sequence([
        Split(pattern=r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])', behavior="isolated"),
    ])


def train_smiles_tokenizer(
    dataset_name: str = "jablonkagroup/pubchem-smiles-molecular-formula",
    vocab_size: Optional[int] = None,
    min_frequency: int = 2000,
    output_dir: str = "data",
    tokenizer_name: str = "pubchem_large_tokenizer"
) -> PreTrainedTokenizerFast:
    """
    Train a BPE tokenizer on SMILES strings and molecular formulas with special tokens.
    
    Args:
        dataset_name: HuggingFace dataset name containing SMILES strings and molecular formulas
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for tokens
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
    
    # Set up trainer
    if vocab_size is not None:
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True
        )
    else:
        trainer = BpeTrainer(
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True
        )
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    ds = datasets.load_dataset(dataset_name, split="train")
    print(f"Dataset loaded with {len(ds)} examples")
    
    # Create iterator that yields both SMILES and molecular formulas
    def text_iterator():
        for example in ds:
            # Always yield SMILES if available
            if "smiles" in example and example["smiles"]:
                yield example["smiles"]
            
            # Yield molecular formula only if it exists and is not empty/None
            if "molecular_formula" in example and example["molecular_formula"]:
                molecular_formula = example["molecular_formula"].strip()
                if molecular_formula and molecular_formula.lower() not in ["none", "null", ""]:
                    yield molecular_formula
    
    # Train the tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}...")
    print("Training on both SMILES and molecular formulas...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    
    # Add post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP] $B [SEP]",
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


def test_tokenizer(tokenizer: PreTrainedTokenizerFast, test_texts: List[str]):
    """Test the trained tokenizer on sample SMILES strings and molecular formulas."""
    print("\n" + "="*50)
    print("TESTING TOKENIZER")
    print("="*50)
    
    for i, text in enumerate(test_texts[:6]):  # Show more examples since we have both types
        print(f"\nTest {i+1}: {text}")
        
        # Tokenize
        encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        tokens = encoded['input_ids'][0].tolist()
        
        print(f"  Tokens: {tokens}")
        print(f"  Token strings: {[tokenizer.decode([t]) for t in tokens]}")
        print(f"  Attention mask: {encoded['attention_mask'][0].tolist()}")
        
        # Decode back
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Decoded: {decoded}")
        print(f"  Match original: {decoded.replace(' ', '') == text.replace(' ', '')}")


def main():
    parser = argparse.ArgumentParser(description="Train a SMILES tokenizer")
    parser.add_argument("--dataset-name", default="jablonkagroup/pubchem-smiles-molecular-formula", 
                       help="HuggingFace dataset name")
    parser.add_argument("--vocab-size", type=int, default=2048, help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=4000, help="Minimum frequency for tokens")
    parser.add_argument("--output-dir", default="data", help="Output directory for tokenizer")
    parser.add_argument("--tokenizer-name", default="pubchem_large_tokenizer_2048", help="Name for the tokenizer")
    
    args = parser.parse_args()
    
    # Train tokenizer
    tokenizer = train_smiles_tokenizer(
        dataset_name=args.dataset_name,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name
    )
    
    # Load dataset for testing
    print("Loading test samples...")
    ds = datasets.load_dataset(args.dataset_name, split="train")
    test_examples = ds[:5]  # Get first 5 examples for testing
    
    # Test with both SMILES and molecular formulas
    test_texts = []
    for example in test_examples:
        if "smiles" in example and example["smiles"]:
            test_texts.append(example["smiles"])
        
        if "molecular_formula" in example and example["molecular_formula"]:
            molecular_formula = example["molecular_formula"].strip()
            if molecular_formula and molecular_formula.lower() not in ["none", "null", ""]:
                test_texts.append(molecular_formula)
    
    if test_texts:
        test_tokenizer(tokenizer, test_texts)
    
    print(f"\nTokenizer training complete!")
    print(f"To use the tokenizer:")
    print(f"  from transformers import PreTrainedTokenizerFast")
    print(f"  tokenizer = PreTrainedTokenizerFast.from_pretrained('{os.path.join(args.output_dir, args.tokenizer_name)}')")


if __name__ == "__main__":
    main() 