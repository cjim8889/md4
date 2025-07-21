"""Streaming input pipeline for PubChem molecular dataset with SAFE encoding.

This module provides a streaming pipeline that reads directly from HuggingFace PubChem
dataset and generates molecular features on-the-fly without requiring pre-processing.
"""

import dataclasses
import threading
from typing import Any, Dict, Optional, Tuple

import grain.python as grain
import jax
import numpy as np
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
        dataset,
        dataloading_host_index: int,
        dataloading_host_count: int,
        num_threads: int,
        generate_padding_example: bool,
        max_target_length: int,
        data_column_name: str,
        split: str = "train",
    ):
        self.dataset = dataset
        self.num_threads = num_threads
        self.dataloading_host_count = dataloading_host_count
        self.dataloading_host_index = dataloading_host_index
        self.generate_padding_example = generate_padding_example
        self.max_target_length = max_target_length
        self.data_column_name = data_column_name
        self.split = split
        
        # Configure dataset for streaming
        self.n_shards = max(dataset.n_shards, dataloading_host_count * num_threads)
        self._check_shard_count()
        
        self.dataset_shards = [
            dataloading_host_index * self.num_threads + i
            for i in range(self.num_threads)
        ]
        
        # Split dataset by node for distributed processing
        self.datasets = [
            dataset.shard(num_shards=self.n_shards, index=x)
            for x in self.dataset_shards
        ]
        self.data_iters = []
        self.out_of_data = False

    def _check_shard_count(self):
        """Check if we have enough shards for efficient processing."""
        if self.n_shards < (self.dataloading_host_count * self.num_threads):
            logging.warning(
                f"Inefficient dataloading. Dataset contains {self.n_shards} shards, "
                f"smaller than number of host loading data ({self.dataloading_host_count * self.num_threads}). "
                f"This may lead to inefficient dataloading."
            )
            self.n_shards = self.dataloading_host_count * self.num_threads

    def _update_shard(self, idx: int):
        """Update to next shard when current shard is exhausted."""
        new_shard = (
            self.dataset_shards[idx]
            + self.dataloading_host_count * self.num_threads
        )
        if new_shard < self.n_shards:
            logging.info(
                f"Updating host {self.dataloading_host_index} dataset {idx}, "
                f"was on shard {self.dataset_shards[idx]}, new shard is {new_shard}"
            )
            self.dataset_shards[idx] = new_shard
            self.datasets[idx] = self.dataset.shard(
                num_shards=self.n_shards, index=self.dataset_shards[idx]
            )
            self.data_iters[idx] = iter(self.datasets[idx])
        else:
            logging.info(
                f"Run out of shards on host {self.dataloading_host_index}, "
                f"shard {self.dataset_shards[idx]} is not available"
            )
            self.out_of_data = True
            if self.generate_padding_example:
                logging.info(
                    f"Host {self.dataloading_host_index} will start generating "
                    f"all-0 padding examples until step number is met."
                )

    def __len__(self) -> int:
        """Return fake length since streaming datasets don't have fixed length."""
        return 10_000_000_000

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get next item from the streaming dataset."""
        if not self.data_iters:
            self.data_iters = [iter(x) for x in self.datasets]
        
        idx = int(threading.current_thread().name.split("_")[1])
        
        while True:
            try:
                if self.out_of_data:
                    if self.generate_padding_example:
                        return {
                            self.data_column_name: np.zeros(
                                self.max_target_length, dtype=np.int32
                            )
                        }
                    else:
                        return None
                
                data = next(self.data_iters[idx])
                return data
                
            except StopIteration:
                self._update_shard(idx)


@dataclasses.dataclass
class ProcessMolecularStreaming(grain.MapTransform):
    """Process molecular data with SAFE encoding, fingerprints, and atom types in streaming mode."""
    
    fp_radius: int = 2
    fp_bits: int = 2048
    pad_to_length: int = 160
    
    def map(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Process molecular features from SMILES string."""
        smiles = features.get("smiles", "")
        if isinstance(smiles, bytes):
            smiles = smiles.decode()
        
        if not smiles:
            return self._create_padding_example()
        
        # Process the SMILES string using rdkit_utils
        processed = rdkit_utils.process_smiles(
            smiles, 
            fp_radius=self.fp_radius, 
            fp_bits=self.fp_bits, 
            pad_to_length=self.pad_to_length
        )
        
        if processed is None:
            return self._create_padding_example()
        
        # Convert to appropriate data types
        result = {
            "fingerprint": processed["fingerprint"].astype(np.int32),
            "atom_types": processed["atom_types"].astype(np.int32),
            "safe": processed["safe"],
        }
        
        return result
    
    def _create_padding_example(self) -> Dict[str, Any]:
        """Create a padding example for invalid molecules."""
        return {
            "fingerprint": np.zeros(self.fp_bits, dtype=np.int32),
            "atom_types": np.full(self.pad_to_length, rdkit_utils.ATOM_TYPES['PAD'], dtype=np.int32),
            "safe": "",
        }


@dataclasses.dataclass
class TokenizeSafeStreaming(grain.MapTransform):
    """Tokenize SAFE strings using the pretrained SAFE tokenizer."""
    
    tokenizer: transformers.PreTrainedTokenizerFast
    max_length: int = 128
    
    def map(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize SAFE representation."""
        safe_repr = features.get("safe", "")
        if isinstance(safe_repr, bytes):
            safe_repr = safe_repr.decode()
        
        if not safe_repr:
            features["safe"] = np.zeros(self.max_length, dtype=np.int32)
            return features
        
        # Tokenize the SAFE string
        tokenized = self.tokenizer.encode(
            safe_repr,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        ).reshape(-1)
        
        features["safe"] = tokenized.astype(np.int32)
        return features


def load_pubchem_streaming_source(
    split: str = "train",
    dataloading_host_index: int = 0,
    dataloading_host_count: int = 1,
    num_threads: int = 1,
    generate_padding_example: bool = True,
    max_target_length: int = 128,
) -> PubChemHFDataSource:
    """Load PubChem dataset as a streaming data source."""
    dataset = load_dataset(
        _PUBCHEM_DATASET,
        split=split,
        streaming=True,
    )
    
    dataset = dataset.select_columns(["smiles"])
    
    return PubChemHFDataSource(
        dataset=dataset,
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        num_threads=num_threads,
        generate_padding_example=generate_padding_example,
        max_target_length=max_target_length,
        data_column_name="smiles",
        split=split,
    )


def compile_molecular_transformations(
    tokenizer: transformers.PreTrainedTokenizerFast,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    pad_to_length: int = 160,
    max_length: int = 128,
    process_batch_size: int = 32,
    packing: bool = True,
    drop_remainder: bool = True,
) -> list:
    """Compile transformations for molecular data processing."""
    operations = []
    
    # Process molecular features
    operations.append(ProcessMolecularStreaming(
        fp_radius=fp_radius,
        fp_bits=fp_bits,
        pad_to_length=pad_to_length,
    ))
    
    # Tokenize SAFE representations
    operations.append(TokenizeSafeStreaming(
        tokenizer=tokenizer,
        max_length=max_length,
    ))
    
    # Pack and batch examples
    if packing:
        operations.append(
            grain.experimental.PackAndBatchOperation(
                batch_size=process_batch_size,
                length_struct={"safe": max_length},
            )
        )
    else:
        operations.append(
            grain.Batch(
                batch_size=process_batch_size,
                drop_remainder=drop_remainder,
            )
        )
    
    return operations


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> Tuple[grain.DataLoader, Dict[str, grain.DataLoader], Dict[str, Any]]:
    """Create streaming molecular datasets for training and evaluation."""
    info = {}
    
    # Validate configuration
    assert config.batch_size % jax.process_count() == 0
    process_batch_size = config.batch_size // jax.process_count()
    eval_batch_size = config.get("eval_batch_size", config.batch_size)
    process_eval_batch_size = eval_batch_size // jax.process_count()
    
    # Molecular dataset parameters
    fp_radius = config.get("fp_radius", 2)
    fp_bits = config.get("fp_bits", 2048)
    max_length = config.get("max_length", 160)
    pad_to_length = config.get("pad_to_length", 160)
    tokenizer_path = config.get("tokenizer", _SAFE_TOKENIZER)
    
    # Load SAFE tokenizer
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Create streaming data sources
    train_source = load_pubchem_streaming_source(
        split="train",
        dataloading_host_index=jax.process_index(),
        dataloading_host_count=jax.process_count(),
        num_threads=config.grain_num_workers,
        generate_padding_example=True,
        max_target_length=max_length,
    )
    
    eval_source = load_pubchem_streaming_source(
        split="validation",
        dataloading_host_index=jax.process_index(),
        dataloading_host_count=jax.process_count(),
        num_threads=config.grain_num_workers,
        generate_padding_example=True,
        max_target_length=max_length,
    )
    
    # Create transformations
    train_transformations = compile_molecular_transformations(
        tokenizer=tokenizer,
        fp_radius=fp_radius,
        fp_bits=fp_bits,
        pad_to_length=pad_to_length,
        max_length=max_length,
        process_batch_size=process_batch_size,
        packing=True,
    )
    
    eval_transformations = compile_molecular_transformations(
        tokenizer=tokenizer,
        fp_radius=fp_radius,
        fp_bits=fp_bits,
        pad_to_length=pad_to_length,
        max_length=max_length,
        process_batch_size=process_eval_batch_size,
        packing=True,
    )
    
    # Create info dictionary
    info = {
        "fp_radius": fp_radius,
        "fp_bits": fp_bits,
        "atom_types": rdkit_utils.ATOM_TYPES,
        "tokenizer": tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "pad_to_length": pad_to_length,
    }
    
    # Create index sampler for training
    index_sampler = grain.IndexSampler(
        num_records=len(train_source),
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        seed=seed,
    )
    
    # Create training data loader
    train_loader = grain.DataLoader(
        data_source=train_source,
        operations=train_transformations,
        sampler=index_sampler,
        worker_count=config.grain_num_workers,
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1024),
    )
    
    # Create evaluation data loader
    eval_loaders = {}
    eval_loader = grain.load(
        source=eval_source,
        num_epochs=1,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        transformations=eval_transformations,
        worker_count=0,
        read_options=grain.ReadOptions(prefetch_buffer_size=1024),
    )
    eval_loaders["validation"] = eval_loader
    
    return train_loader, eval_loaders, info