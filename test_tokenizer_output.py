#!/usr/bin/env python3
"""
Test the trained SentencePiece tokenizer
"""

import sentencepiece as spm
import datasets
import random

def get_real_dataset_examples(n_samples=10):
    """Load real examples from the actual HuggingFace dataset"""
    print("Loading real examples from the PubChem dataset...")
    
    try:
        # Load the dataset
        ds = datasets.load_dataset("jablonkagroup/pubchem-smiles-molecular-formula", split="train")
        print("Dataset loaded successfully")
        
        # Create a list of valid examples first
        valid_examples = []
        print("Scanning dataset for valid examples...")
        
        # Scan through dataset to collect valid examples (limit scan to avoid memory issues)
        max_scan = 10000  # Scan first 10k examples to find valid ones
        for i, example in enumerate(ds):
            if i >= max_scan:
                break
                
            try:
                # Extract SMILES and molecular formula
                smiles = example["smiles"]
                molecular_formula = example["molecular_formula"]
                
                # Skip if either field is None, empty, or invalid
                if not smiles or not molecular_formula:
                    continue
                if smiles in [None, "None", ""] or molecular_formula in [None, "None", ""]:
                    continue
                    
                # Convert to strings and strip whitespace
                smiles = str(smiles).strip()
                molecular_formula = str(molecular_formula).strip()
                
                # Skip if still empty after stripping
                if not smiles or not molecular_formula:
                    continue
                    
                # Store valid example
                formatted_example = f"{molecular_formula}[SEP]{smiles}"
                valid_examples.append((formatted_example, molecular_formula, smiles, i))
                
            except (KeyError, TypeError, AttributeError):
                continue
        
        print(f"Found {len(valid_examples)} valid examples")
        
        # Randomly sample from valid examples
        if len(valid_examples) < n_samples:
            n_samples = len(valid_examples)
            
        sampled_examples = random.sample(valid_examples, n_samples)
        
        real_examples = []
        for i, (formatted_example, molecular_formula, smiles, dataset_idx) in enumerate(sampled_examples):
            real_examples.append(formatted_example)
            print(f"Example {i + 1} (dataset index {dataset_idx}): {molecular_formula} -> {smiles}")
        
        print(f"\nSuccessfully loaded {len(real_examples)} randomly sampled examples from dataset")
        return real_examples
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to hardcoded examples...")
        
        # Fallback examples
        return [
            "C2H6O[SEP]CCO",  # ethanol
            "C6H6[SEP]c1ccccc1",  # benzene  
            "H2O[SEP]O",  # water
            "C2H4O2[SEP]CC(=O)O",  # acetic acid
            "C8H10[SEP]Cc1ccccc1C"  # xylene
        ]

def test_tokenizer():
    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load('data/sentencepiece_tokenizer_unigram_2048.model')

    print("SentencePiece tokenizer loaded successfully.")
    print(f"Vocab size: {sp.vocab_size()}")
    
    # Test sentences with real molecular formulas and SMILES from dataset
    test_sentences = get_real_dataset_examples()
    
    print("Testing SentencePiece tokenizer:")
    print(f"Vocabulary size: {sp.vocab_size()}")
    try:
        sep_id = sp.PieceToId('[SEP]')
        print(f"[SEP] token ID: {sep_id}")
    except Exception:
        print("[SEP] token not found in vocabulary")
    print()
    
    for sentence in test_sentences:
        print(f"Input: {sentence}")
        
        # Encode to pieces
        pieces = sp.EncodeAsPieces(sentence)
        print(f"Pieces: {pieces}")
        
        # Encode to IDs
        ids = sp.EncodeAsIds(sentence)
        print(f"IDs: {ids}")
        
        # Decode back
        decoded = sp.DecodePieces(pieces)
        print(f"Decoded: {decoded}")
        
        # Check if [SEP] is preserved
        sep_preserved = "[SEP]" in pieces
        print(f"[SEP] preserved: {sep_preserved}")
        print("-" * 50)

if __name__ == "__main__":
    test_tokenizer()
