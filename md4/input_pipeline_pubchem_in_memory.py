"""Streaming input pipeline for PubChem molecular dataset with SAFE encoding.

This module provides a streaming pipeline that reads directly from HuggingFace PubChem
dataset and generates molecular features on-the-fly without requiring pre-processing.
"""

import dataclasses
import threading
from typing import Any, Dict, Tuple

import grain.python as grain
import jax
import numpy as np
import polars as pl
import transformers
from absl import logging
from datasets import load_dataset
from ml_collections import config_dict

from md4 import rdkit_utils

# Constants
_SAFE_TOKENIZER = "data/pubchem_safe/safe_tokenizer"
_PUBCHEM_DATASET = "jablonkagroup/pubchem-smiles-molecular-formula"


class PubChemHFDataSource(grain.RandomAccessDataSource):
    """A streaming data source for PubChem molecular data from HuggingFace."""
    
    def __init__(
        self,
        dataset: pl.DataFrame,
        split: str = "train",
    ):
        self.dataset = dataset
        self.split = split

    def __len__(self) -> int:
        """Return fake length since streaming datasets don't have fixed length."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get next item from the streaming dataset."""
        smiles = self.dataset["smiles"][index]
        return {"smiles": smiles}

@dataclasses.dataclass
class ProcessMolecular(grain.MapTransform):
    """Process molecular data with fingerprints and atom types."""

    def map(self, features):
        smiles = features["smiles"]
        processed = rdkit_utils.process_smiles(smiles)  
        return processed

@dataclasses.dataclass
class FilterNoneMolecules(grain.FilterTransform):
    """Process molecular data with SAFE encoding, fingerprints, and atom types in streaming mode."""
    
    def filter(self, features: Dict[str, Any]) -> bool:
        """Filter out molecules that are None."""
        return features is not None

def load_pubchem_streaming_source(
    split: str = "train",
) -> PubChemHFDataSource:
    """Load PubChem dataset as a streaming data source."""
    dataset = load_dataset(
        _PUBCHEM_DATASET,
        split=split,
    )

    return PubChemHFDataSource(
        dataset=dataset,
        split=split,
    )


def compile_molecular_transformations(
) -> list:
    """Compile transformations for molecular data processing."""
    operations = []
    
    operations.append(ProcessMolecular())
    operations.append(FilterNoneMolecules())
    return operations


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> Tuple[grain.DataLoader, Dict[str, grain.DataLoader], Dict[str, Any]]:
    """Create streaming molecular datasets for training and evaluation."""
    info = {}
    
    # Create streaming data sources
    train_source = load_pubchem_streaming_source(
        split="train",
    )
    
    # Create transformations
    train_transformations = compile_molecular_transformations(
    )
    
    # Create training data loader
    train_loader = grain.load(
        source=train_source,
        shuffle=True,
        seed=seed,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        transformations=train_transformations,
        batch_size=128,
        worker_count=15,
        read_options=grain.ReadOptions(num_threads=4, prefetch_buffer_size=1024),
        drop_remainder=True,
    )
    
    return train_loader, info

if __name__ == "__main__":
    from tqdm import tqdm
    from rich import print
    
    train_loader, info = create_datasets(
        config=config_dict.ConfigDict(),
        seed=42,
    )
    print(f"Train loader: {train_loader}")
    print(f"Info: {info}")
    
    # Test throughput with tqdm
    print("\nTesting data loading throughput...")
    try:
        with tqdm(desc="Fetching molecules", unit="mol", smoothing=0.1) as pbar:
            for i, batch in enumerate(train_loader):
                pbar.update(batch["fingerprint"].shape[0])
                pbar.set_postfix({"batch_size": batch["fingerprint"].shape[0]})
                
                # Stop after a reasonable number of samples for testing
                if i >= 100000:
                    break
                    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError during data loading: {e}")