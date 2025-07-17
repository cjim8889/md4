"""PubChem molecular dataset preprocessing with SAFE encoding."""

import multiprocessing as mp
import os
from functools import partial
from typing import Any, Iterator, List

import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers
from datasets import load_dataset
from tqdm import tqdm

# Add SAFE import
try:
    import safe
    SAFE_AVAILABLE = True
except ImportError:
    SAFE_AVAILABLE = False
    print("Warning: SAFE library not available. Install with: pip install safe-mol")

# Add tokenizer imports
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from md4 import rdkit_utils

FlatFeatures = dict[str, Any]


def smiles_to_safe(smiles):
    """Convert SMILES to SAFE representation with error handling."""
    if not SAFE_AVAILABLE:
        return None
    
    try:
        safe_repr = safe.encode(smiles)
        return safe_repr
    except Exception:
        return None


def process_smiles_to_features(smiles, fp_radius=2, fp_bits=2048, pad_to_length=128):
    """Process a single SMILES to extract SMILES, SAFE, and molecular features."""
    # Get molecular features
    features = rdkit_utils.get_molecule_features(
        smiles, radius=fp_radius, n_bits=fp_bits, pad_to_length=pad_to_length
    )
    if features is not None:
        # Convert SMILES to SAFE
        safe_repr = smiles_to_safe(smiles)
        if safe_repr is not None:
            return {
                "smiles": smiles,
                "safe": safe_repr,
                "fingerprint": features["fingerprint"],
                "atom_types": features["atom_types"],
            }
    return None


def load_safe_from_data(data_list: List[dict]) -> Iterator[str]:
    """Load SAFE strings from processed data list."""
    for item in data_list:
        if item and 'safe' in item and isinstance(item['safe'], str):
            yield item['safe']


def create_safe_pre_tokenizer():
    """Create a pre-tokenizer suitable for SAFE strings."""
    # SAFE uses specific tokens - split on common boundaries
    return Sequence([
        Split(pattern=r'(\[|\]|\(|\)|=|#|@|\+|\-|%|\d+|\.)', behavior="isolated"),
        Whitespace()
    ])


def train_safe_tokenizer(
    safe_data: List[str],
    vocab_size: int = 1024,
    output_dir: str = "data",
    tokenizer_name: str = "safe_tokenizer",
    min_frequency: int = 200
) -> transformers.PreTrainedTokenizerFast:
    """
    Train a BPE tokenizer on SAFE strings with special tokens.
    
    Args:
        safe_data: List of SAFE strings
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
    
    # Set up pre-tokenizer for SAFE
    tokenizer.pre_tokenizer = create_safe_pre_tokenizer()
    
    # Set up trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True  
    )
    
    if not safe_data:
        raise ValueError("No SAFE strings found for training!")
    
    print(f"Found {len(safe_data)} SAFE strings for training")
    print("Sample SAFE strings:")
    for i, safe_str in enumerate(safe_data[:5]):
        print(f"  {i+1}: {safe_str}")
    
    # Train the tokenizer
    print(f"Training SAFE tokenizer with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(safe_data, trainer=trainer)
    
    # Add post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Create transformers tokenizer
    fast_tokenizer = transformers.PreTrainedTokenizerFast(
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
    
    print(f"SAFE tokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {fast_tokenizer.vocab_size}")
    
    return fast_tokenizer


def preprocess_pubchem(data_dir, fp_radius=2, fp_bits=4096, vocab_size=1000, min_frequency=200, pad_to_length=128):
    """Load and preprocess PubChem dataset with SAFE encoding and tokenizer training."""

    if not SAFE_AVAILABLE:
        raise ImportError("SAFE library is required. Install with: pip install safe-mol")

    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pubchem_train_path = os.path.join(data_dir, "pubchem_train.parquet")
    pubchem_val_path = os.path.join(data_dir, "pubchem_val.parquet")
    tokenizer_path = os.path.join(data_dir, "safe_tokenizer")

    # Check if preprocessed data already exists
    if os.path.exists(pubchem_train_path) and os.path.exists(pubchem_val_path):
        print("Loading existing preprocessed PubChem data...")
        pubchem_train_df = pl.read_parquet(pubchem_train_path)
        pubchem_val_df = pl.read_parquet(pubchem_val_path)
        
        # Extract SAFE data for tokenizer training if tokenizer doesn't exist
        if not os.path.exists(tokenizer_path):
            print("Training SAFE tokenizer on existing data...")
            safe_data = pubchem_train_df["safe"].to_list()
            train_safe_tokenizer(
                safe_data=safe_data,
                vocab_size=vocab_size,
                output_dir=data_dir,
                tokenizer_name="safe_tokenizer",
                min_frequency=min_frequency
            )
    else:
        print("Downloading and preprocessing PubChem data...")

        # Load dataset from HuggingFace
        ds = load_dataset("sagawa/pubchem-10m-canonicalized")
        df_train = ds["train"].to_pandas()

        # Process training data
        print("Processing training SMILES...")
        pubchem_set_raw = set(df_train["smiles"])
        pubchem_smiles_list = list(pubchem_set_raw)

        # Shuffle and split SMILES directly
        import random

        random.seed(42)
        random.shuffle(pubchem_smiles_list)

        split_idx = int(0.95 * len(pubchem_smiles_list))
        pubchem_train_smiles = pubchem_smiles_list[:split_idx]
        pubchem_val_smiles = pubchem_smiles_list[split_idx:]

        # Get number of CPU cores for multiprocessing
        num_cores = mp.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing")

        # Create partial function with fixed parameters
        process_func = partial(
            process_smiles_to_features, fp_radius=fp_radius, fp_bits=fp_bits, pad_to_length=pad_to_length
        )

        # Generate features for training data using multiprocessing
        print("Generating features for training data...")
        with mp.Pool(processes=num_cores) as pool:
            results = list(
                tqdm(
                    pool.map(process_func, pubchem_train_smiles),
                    total=len(pubchem_train_smiles),
                    desc="Processing training data",
                )
            )
        # Filter out None results
        train_data = [result for result in results if result is not None]

        # Generate features for validation data using multiprocessing
        print("Generating features for validation data...")
        with mp.Pool(processes=num_cores) as pool:
            results = list(
                tqdm(
                    pool.map(process_func, pubchem_val_smiles),
                    total=len(pubchem_val_smiles),
                    desc="Processing validation data",
                )
            )
        # Filter out None results
        val_data = [result for result in results if result is not None]

        # Train tokenizer on SAFE representations
        print("Training SAFE tokenizer...")
        safe_data = [item["safe"] for item in train_data if item.get("safe")]
        train_safe_tokenizer(
            safe_data=safe_data,
            vocab_size=vocab_size,
            output_dir=data_dir,
            tokenizer_name="safe_tokenizer",
            min_frequency=min_frequency
        )

        # Save to parquet
        train_data_dict = {
            "smiles": [item["smiles"] for item in train_data],
            "safe": [item["safe"] for item in train_data],
            "fingerprint": [item["fingerprint"] for item in train_data],
            "atom_types": [item["atom_types"] for item in train_data],
        }
        val_data_dict = {
            "smiles": [item["smiles"] for item in val_data],
            "safe": [item["safe"] for item in val_data],
            "fingerprint": [item["fingerprint"] for item in val_data],
            "atom_types": [item["atom_types"] for item in val_data],
        }
        
        pubchem_train_df = pl.DataFrame(train_data_dict)
        pubchem_val_df = pl.DataFrame(val_data_dict)

        pubchem_train_df.write_parquet(pubchem_train_path)
        pubchem_val_df.write_parquet(pubchem_val_path)

        print(
            f"Saved {len(train_data)} training examples and {len(val_data)} validation examples"
        )

    def load_pubchem_split(split_df):
        """Convert DataFrame to TensorFlow dataset."""
        # Convert fingerprint arrays to proper 2D numpy array
        fingerprints = np.stack(split_df["fingerprint"]).astype(np.int32)
        smiles = split_df["smiles"].to_numpy().astype(str)
        safe_reprs = split_df["safe"].to_numpy().astype(str)
        atom_types = np.stack(split_df["atom_types"]).astype(np.int32)

        ds = tf.data.Dataset.from_tensor_slices(
            {
                "smiles": smiles,
                "safe": safe_reprs,
                "fingerprint": fingerprints,
                "atom_types": atom_types,
            }
        )

        return ds

    # Try to load existing dataset, create if doesn't exist
    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder("pubchem_safe", data_dir=data_dir, config="pubchem_safe")
        print("Loading existing TFDS dataset...")
    except Exception:
        # Create dataset builder if it doesn't exist
        print("Creating new TFDS dataset...")
        pubchem_builder = tfds.dataset_builders.store_as_tfds_dataset(
            name="pubchem_safe",
            version="1.0.0",
            features=tfds.features.FeaturesDict(
                {
                    "smiles": tfds.features.Text(),
                    "safe": tfds.features.Text(),
                    "fingerprint": tfds.features.Tensor(
                        shape=(fp_bits,), dtype=tf.int32
                    ),
                    "atom_types": tfds.features.Tensor(
                        shape=(128,), dtype=tf.int32
                    ),
                }
            ),
            split_datasets={
                "train": load_pubchem_split(pubchem_train_df),
                "validation": load_pubchem_split(pubchem_val_df),
            },
            config="pubchem_safe",
            data_dir=data_dir,
            description=f"PubChem dataset with SAFE encoding, Morgan fingerprints (radius={fp_radius}, bits={fp_bits}) and atom types",
            file_format="array_record",
            disable_shuffling=True,
        )

    return pubchem_builder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/pubchem")
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--min_frequency", type=int, default=200)
    parser.add_argument("--pad_to_length", type=int, default=128)
    args = parser.parse_args()
    preprocess_pubchem(data_dir=args.data_dir, fp_bits=args.fp_bits, vocab_size=args.vocab_size, min_frequency=args.min_frequency, pad_to_length=args.pad_to_length)