"""PubChem molecular dataset input pipeline with SAFE encoding."""

import dataclasses
import multiprocessing as mp
import os
from functools import partial
from typing import Any

import grain.python as grain
import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers
from datasets import load_dataset
from ml_collections import config_dict
from tqdm import tqdm

# Add SAFE import
try:
    import safe
    SAFE_AVAILABLE = True
except ImportError:
    SAFE_AVAILABLE = False
    print("Warning: SAFE library not available. Install with: pip install safe-mol")

# Add tokenizer imports for SAFE
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from md4 import rdkit_utils

FlatFeatures = dict[str, Any]

_SAFE_TOKENIZER = "data/pubchem_safe/safe_tokenizer"


def smiles_to_safe(smiles):
    """Convert SMILES to SAFE representation with error handling."""
    if not SAFE_AVAILABLE:
        return None
    
    try:
        safe_repr = safe.encode(smiles)
        return safe_repr
    except Exception:
        return None


def create_safe_pre_tokenizer():
    """Create a pre-tokenizer suitable for SAFE strings."""
    # SAFE uses specific tokens - split on common boundaries
    return Sequence([
        Split(pattern=r'(\[|\]|\(|\)|=|#|@|\+|\-|%|\d+|\.)', behavior="isolated"),
        Whitespace()
    ])


def train_safe_tokenizer(
    safe_data: list[str],
    vocab_size: int = 1024,
    output_dir: str = "data",
    tokenizer_name: str = "safe_tokenizer",
    min_frequency: int = 200
) -> transformers.PreTrainedTokenizerFast:
    """Train a BPE tokenizer on SAFE strings with special tokens."""
    
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


@dataclasses.dataclass
class TokenizeSafe(grain.MapTransform):
    tokenizer: transformers.PreTrainedTokenizerFast
    max_length: int = 128

    def map(self, features):
        safe_repr = features["safe"]
        features["safe"] = self.tokenizer.encode(
            safe_repr.decode() if isinstance(safe_repr, bytes) else safe_repr,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        ).reshape(-1)
        return features


@dataclasses.dataclass
class ProcessMolecular(grain.MapTransform):
    """Process molecular data with fingerprints and atom types."""

    def map(self, features):
        # Fingerprint is already in correct format (numpy array from rdkit_utils)
        if "fingerprint" in features:
            features["fingerprint"] = features["fingerprint"].astype(np.int32)

        if "safe" in features:
            features["safe"] = features["safe"].astype(np.int32)
            
        if "atom_types" in features:
            features["atom_types"] = features["atom_types"].astype(np.int32)
        
        if "smiles" in features:
            del features["smiles"]

        return features


def preprocess_or_load_pubchem(data_dir, fp_radius=2, fp_bits=2048, pad_to_length=128):
    """Load and preprocess PubChem dataset with SAFE encoding and tokenizer training."""

    if not SAFE_AVAILABLE:
        raise ImportError("SAFE library is required. Install with: pip install safe-mol")

    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder("pubchem_large", data_dir=data_dir, config="pubchem_large")
        print("Loading existing TFDS dataset...")
    except Exception:
        # Create dataset builder if it doesn't exist
        print("Creating new TFDS dataset...")
        ds = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula", split="train", cache_dir=data_dir, streaming=True)

        # Create partial function with fixed parameters
        process_func = partial(
            rdkit_utils.process_smiles, fp_radius=fp_radius, fp_bits=fp_bits, pad_to_length=pad_to_length
        )

        train_ds = ds.select(range(int(len(ds) * 0.95)))
        val_ds = ds.select(range(int(len(ds) * 0.95), len(ds)))

        def iterator_fn(ds):
            for smi in ds["smiles"]:
                features = process_func(smi)
                if features is not None:
                    yield features["smiles"], features

        pubchem_builder = tfds.dataset_builders.store_as_tfds_dataset(
            name="pubchem_large",
            version="1.0.1",
            features=tfds.features.FeaturesDict(
                {
                    "smiles": tfds.features.Text(),
                    "safe": tfds.features.Text(),
                    "fingerprint": tfds.features.Tensor(
                        shape=(fp_bits,), dtype=tf.bool
                    ),
                    "atom_types": tfds.features.Tensor(
                        shape=(pad_to_length,), dtype=tf.int8
                    ),
                }
            ),
            split_datasets={
                "train": iterator_fn(train_ds),
                "validation": iterator_fn(val_ds),
            },
            config="pubchem_large",
            data_dir=data_dir,
            description=f"PubChem dataset with SAFE, SMILES, Morgan fingerprints (radius={fp_radius}, bits={fp_bits}) and atom types",
            file_format="array_record",
            disable_shuffling=True,
        )

        return pubchem_builder


def create_pubchem_datasets(config: config_dict.ConfigDict, seed: int):
    """Create PubChem datasets with SAFE encoding and molecular features."""

    if not SAFE_AVAILABLE:
        raise ImportError("SAFE library is required. Install with: pip install safe-mol")

    # Molecular dataset with SAFE, SMILES and Morgan fingerprints
    fp_radius = config.get("fp_radius", 2)
    fp_bits = config.get("fp_bits", 2048)
    max_length = config.get("max_length", 160)
    vocab_size = config.get("vocab_size", 1024)
    pad_to_length = config.get("pad_to_length", 160)

    # Use preprocess_pubchem to get the dataset builder
    pubchem_builder = preprocess_or_load_pubchem(
        data_dir=os.path.join("./data", "pubchem_safe"), 
        fp_radius=fp_radius, 
        fp_bits=fp_bits,
        pad_to_length=pad_to_length
    )
    data_source = pubchem_builder.as_data_source()
    # Load SAFE tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(_SAFE_TOKENIZER)
    
    train_transformations = [
        TokenizeSafe(tokenizer, max_length=max_length),
        ProcessMolecular(),
    ]
    train_source = data_source["train"]
    eval_transformations = [
        TokenizeSafe(tokenizer, max_length=max_length),
        ProcessMolecular(),
    ]
    eval_source = {k: v for k, v in data_source.items() if k != "train"}

    info = {
        "fp_radius": fp_radius,
        "fp_bits": fp_bits,
        "atom_types": rdkit_utils.ATOM_TYPES,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "pad_to_length": pad_to_length,
    }

    return train_source, train_transformations, eval_source, eval_transformations, info


if __name__ == "__main__":
    pubchem_builder = preprocess_or_load_pubchem(
        data_dir="./data/pubchem_large",
        fp_radius=2,
        fp_bits=2048,
        pad_to_length=160,
    )