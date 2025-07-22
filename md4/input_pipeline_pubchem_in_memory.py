"""Streaming input pipeline for PubChem molecular dataset with SAFE encoding.

This module provides a streaming pipeline that reads directly from HuggingFace PubChem
dataset and generates molecular features on-the-fly without requiring pre-processing.
"""

import dataclasses
import glob
from pathlib import Path
# import threading
from typing import Any, Dict, Tuple

import grain.python as grain
grain.config.update("py_debug_mode", True)
grain.config.update("py_dataset_visualization_output_dir", "")

import jax
# import numpy as np
import polars as pl
# import transformers
# from absl import logging
# from datasets import load_dataset
# import transformers
# from absl import logging
# from datasets import load_dataset
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

def find_data_files(data_file_pattern):
  data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
  return data_files

def get_pubchem_dataset(
    shuffle: bool = False,
    shuffle_seed: int = 42,
    num_epochs: int = 1,
    dataloading_host_index: int = 0,
    dataloading_host_count: int = 1,
    grain_worker_count: int = 15,
) -> grain.IterDataset:
    """Load PubChem dataset as a streaming data source."""
    data_files = find_data_files("data/pubchem_large/data/train-*.parquet")
    dataset = grain.MapDataset.source(data_files)
    if shuffle:
        dataset = dataset.shuffle(shuffle_seed)
    
    dataset = dataset.repeat(num_epochs)
    dataset = dataset[dataloading_host_index::dataloading_host_count]

    assert grain_worker_count <= len(dataset), (
        f"grain worker count is currently {grain_worker_count}, exceeding the max allowable value {len(dataset)} "
        f"(file shard count of a data loading host) for your dataset. "
        f"Please lower grain_worker_count or increase file shard count."
    )

    dataset = dataset.map(grain.experimental.ParquetIterDataset)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=len(dataset))
    dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=100, seed=shuffle_seed)

    return dataset

def create_dataloading_pipeline(
    dataset: grain.IterDataset,
    batch_size: int = 128,
    worker_count: int = 15,
):
    dataset = dataset.map(ProcessMolecular())
    dataset = dataset.filter(FilterNoneMolecules())
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.mp_prefetch(grain.MultiprocessingOptions(num_workers=worker_count, per_worker_buffer_size=16, enable_profiling=False))
    return dataset





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
    import time

    dataset = get_pubchem_dataset()

    train_loader = create_dataloading_pipeline(dataset, worker_count=15, batch_size=512)
    print(f"Train loader: {train_loader}")
    
    # Test throughput with tqdm
    print("\nTesting data loading throughput...")
    try:
        with tqdm(desc="Fetching molecules", unit="mol", smoothing=0.1) as pbar:
            for i, batch in enumerate(train_loader):
                pbar.update(batch["fingerprint"].shape[0])
                pbar.set_postfix({"batch_size": batch["fingerprint"].shape[0]})
                
                # Stop after a reasonable number of samples for testing
                    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError during data loading: {e}")
