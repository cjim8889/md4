"""PubChem molecular dataset input pipeline with SAFE encoding."""

import dataclasses
import multiprocessing as mp
import os
from functools import partial
from typing import Any

import grain.python as grain
import numpy as np
import tensorflow_datasets as tfds
import transformers
from datasets import load_dataset
from ml_collections import config_dict
from tqdm import tqdm

from md4 import rdkit_utils

FlatFeatures = dict[str, Any]

_SAFE_TOKENIZER = "data/pubchem_safe/safe_tokenizer"


def process_smiles_worker(smi, fp_radius=2, fp_bits=2048, pad_to_length=128):
    """Process a single SMILES string - optimized for chunksize batching."""
    return rdkit_utils.process_smiles(smi, fp_radius=fp_radius, fp_bits=fp_bits, pad_to_length=pad_to_length)

@dataclasses.dataclass
class TokenizeSafe(grain.MapTransform):
    tokenizer: "transformers.PreTrainedTokenizerFast"
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


def preprocess_or_load_pubchem(data_dir, fp_radius=2, fp_bits=2048, pad_to_length=128, 
                               chunk_size=1000, num_processes=None):
    """Load and preprocess PubChem dataset with SAFE encoding and tokenizer training.
    
    Args:
        data_dir: Directory to store the dataset
        fp_radius: Morgan fingerprint radius
        fp_bits: Number of bits for fingerprint
        pad_to_length: Length to pad atom types to
        chunk_size: Chunk size for multiprocessing (default: 1000)
        num_processes: Number of processes to use (default: min(cpu_count(), 8))
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder("pubchem_large", version="1.0.2", data_dir=data_dir, config="pubchem_large")
        print("Loading existing TFDS dataset...")
    except Exception:
        # Parameters for multiprocessing batch function
        # (removed process_func since we're using batch processing now)

        print("Loading full dataset without streaming...")
        ds_full = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula", split="train", cache_dir=data_dir, streaming=False)
        print(f"Loaded {len(ds_full)} samples")
        train_size = int(len(ds_full) * 0.95)
        train_ds = ds_full.select(range(train_size))
        val_ds = ds_full.select(range(train_size, len(ds_full)))

        def iterator_closure(ds):
            def iterator():
                # Use multiprocessing with chunksize for optimal performance
                _num_processes = num_processes if num_processes is not None else mp.cpu_count()
                
                smiles_data = ds["smiles"]
                total_samples = len(smiles_data)
                
                # Calculate optimal chunksize: typically total_samples / (4 * num_processes)
                # This ensures good load balancing without too much overhead
                chunksize = chunk_size
                update_interval = 1000
                print(f"Processing {total_samples} SMILES with {_num_processes} processes (chunksize={chunksize})...")
                
                with mp.Pool(processes=_num_processes) as pool:
                    # Create partial function with fixed parameters
                    worker_func = partial(
                        process_smiles_worker,
                        fp_radius=fp_radius,
                        fp_bits=fp_bits,
                        pad_to_length=pad_to_length
                    )
                    
                    # Use imap_unordered with chunksize for optimal throughput
                    processed_count = 0
                    with tqdm(total=total_samples, desc="Processing SMILES") as pbar:
                        for result in pool.imap_unordered(worker_func, smiles_data, chunksize=chunksize):
                            if result is not None:  # Filter out failed processing
                                yield result["smiles"], result
                            processed_count += 1
                            if processed_count % update_interval == 0:  # Update progress every chunk
                                pbar.update(update_interval)
                        
                        # Update progress for remaining items
                        remaining = processed_count % update_interval
                        if remaining > 0:
                            pbar.update(remaining)
            return iterator

        tfds.dataset_builders.store_as_tfds_dataset
        pubchem_builder = tfds.dataset_builders.store_as_tfds_dataset(
            name="pubchem_large",
            version="1.0.2",
            features=tfds.features.FeaturesDict(
                {
                    "smiles": tfds.features.Text(),
                    "safe": tfds.features.Text(),
                    "fingerprint": tfds.features.Tensor(
                        shape=(fp_bits,), dtype=np.bool_
                    ),
                    "atom_types": tfds.features.Tensor(
                        shape=(pad_to_length,), dtype=np.int8
                    ),
                }
            ),
            split_datasets={
                "train": iterator_closure(train_ds)(),
                "validation": iterator_closure(val_ds)(),
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

    # Molecular dataset with SAFE, SMILES and Morgan fingerprints
    fp_radius = config.get("fp_radius", 2)
    fp_bits = config.get("fp_bits", 2048)
    max_length = config.get("max_length", 160)
    vocab_size = config.get("vocab_size", 1024)
    pad_to_length = config.get("pad_to_length", 160)
    tokenizer_path = config.get("tokenizer", _SAFE_TOKENIZER)    

    # Use preprocess_pubchem to get the dataset builder
    num_processes = config.get("num_processes", None)
    chunk_size = config.get("chunk_size", 64)
    pubchem_builder = preprocess_or_load_pubchem(
        data_dir=os.path.join("./data", "pubchem_safe"), 
        fp_radius=fp_radius, 
        fp_bits=fp_bits,
        pad_to_length=pad_to_length,
        chunk_size=chunk_size,
        num_processes=num_processes
    )
    data_source = pubchem_builder.as_data_source()
    # Load SAFE tokenizer
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
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
        pad_to_length=160,  # Adjust pad_to_length as needed
        chunk_size=64,  # Adjust batch size as needed
        num_processes=16  # Use default (min(cpu_count(), 8))
    )