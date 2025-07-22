"""PubChem molecular dataset input pipeline with SAFE encoding.

Memory Optimizations:
- Worker functions moved to separate minimal module to avoid loading heavy dependencies
- Heavy imports (tensorflow_datasets, transformers, polars, grain) are now conditional
- Worker processes only load numpy, shared_memory, and rdkit_utils (when needed)
- This reduces worker memory footprint by avoiding unnecessary module imports
"""

import dataclasses
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import shared_memory
import os
import glob
from pathlib import Path
import glob
from pathlib import Path

import numpy as np
from ml_collections import config_dict
from tqdm import tqdm

from md4 import rdkit_utils

# Heavy imports are now imported conditionally within functions to reduce memory usage

_SAFE_TOKENIZER = "data/pubchem_safe/safe_tokenizer"


def find_data_files(data_file_pattern):
  data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
  return data_files

# Import worker functions from separate module to avoid loading heavy dependencies
# This prevents spawned worker processes from importing grain, tensorflow_datasets, 
# transformers, polars and other heavy modules that are not needed for SMILES processing
from md4.pubchem_worker import init_worker_pubchem, process_individual_smiles_pubchem

@dataclasses.dataclass
class TokenizeSafe:
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
class ProcessMolecular:
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


def preprocess_or_load_pubchem(data_dir, fp_radius=2, fp_bits=2048, 
def preprocess_or_load_pubchem(data_dir, fp_radius=2, fp_bits=2048, 
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
    # Import heavy modules only when needed
    import tensorflow_datasets as tfds
    import polars as pl

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder("pubchem_large", version="1.0.3", data_dir=data_dir, config="pubchem_large")
        print("Loading existing TFDS dataset...")
        return pubchem_builder
    except Exception:
        # Parameters for multiprocessing batch function
        # (removed process_func since we're using batch processing now)

        print("Loading full dataset without streaming...")
        ds_full = pl.read_parquet(find_data_files("data/pubchem_large/data/train-*.parquet"))
        ds_full = pl.read_parquet(find_data_files("data/pubchem_large/data/train-*.parquet"))
        print(f"Loaded {len(ds_full)} samples")
        train_size = int(len(ds_full) * 0.98)
        train_ds = ds_full[:train_size]
        val_ds = ds_full[train_size:]
        train_size = int(len(ds_full) * 0.98)
        train_ds = ds_full[:train_size]
        val_ds = ds_full[train_size:]

        def iterator_closure(ds: pl.DataFrame):
        def iterator_closure(ds: pl.DataFrame):
            def iterator():
                # Use optimized multiprocessing with minimal memory exchange
                # Use optimized multiprocessing with minimal memory exchange
                _num_processes = num_processes if num_processes is not None else mp.cpu_count()
                
                smiles_data = ds["smiles"]
                total_samples = len(smiles_data)
                batch_size = chunk_size  # Use chunk_size as batch_size
                
                print(f"Processing {total_samples} SMILES with {_num_processes} processes (batch_size={batch_size})...")
                
                print(f"Creating shared memory for fingerprints...")
                # Create shared memory for fingerprints
                fingerprint_size = total_samples * fp_bits
                fingerprint_shm = shared_memory.SharedMemory(create=True, size=fingerprint_size)
                fingerprint_array = np.ndarray((total_samples, fp_bits), dtype=np.bool_, buffer=fingerprint_shm.buf)
                
                try:
                    # Create individual tasks as (smiles, index) tuples - using generator for memory efficiency
                    tasks = ((smiles_data[i], i) for i in range(total_samples))
                    
                    with mp.get_context("spawn").Pool(
                            processes=_num_processes,
                            initializer=init_worker_pubchem,
                            initargs=(fingerprint_shm.name, fingerprint_array.shape)) as pool:
                        
                        # Process individual SMILES
                        total_processed = 0
                        since_last_update = 0
                        
                        with tqdm(total=total_samples, desc="Processing SMILES") as pbar:
                            for result in pool.imap_unordered(process_individual_smiles_pubchem, tasks, chunksize=chunk_size):
                                if result != -1:
                                    total_processed += 1
                                    since_last_update += 1
                                    yield smiles_data[result], {
                                        "fingerprint": fingerprint_array[result, :],
                                        "smiles": smiles_data[result]
                                    }


                                if since_last_update >= 2000:
                                    pbar.update(since_last_update)
                                    since_last_update = 0
                        
                        # Update progress for remaining items
                        if since_last_update > 0:
                            pbar.update(since_last_update)
                except Exception as e:
                    print(f"Error: {e}")
                    raise e
                finally:
                    # Clean up shared memory
                    try:
                        fingerprint_shm.close()
                        fingerprint_shm.unlink()
                    except Exception as e:
                        print(f"Cleanup error: {e}")
            
            return iterator

        # pubchem_builder = tfds.dataset_builders.AdhocBuilder(
        #     name="pubchem_large",
        #     version="1.0.3",
        #     features=tfds.features.FeaturesDict(
        #         {
        #             "smiles": tfds.features.Text(),
        #             "fingerprint": tfds.features.Tensor(
        #                 shape=(fp_bits,), dtype=np.bool_
        #             ),
        #         }
        #     ),
        #     split_datasets={
        #         "train": iterator_closure(train_ds)(),
        #         "validation": iterator_closure(val_ds)(),
        #     },
        #     config="pubchem_large",
        #     data_dir=data_dir,
        #     description=f"PubChem dataset with SAFE, SMILES, Morgan fingerprints (radius={fp_radius}, bits={fp_bits}) and atom types",
        #     file_format="array_record",
        #     disable_shuffling=True,
        #     download_config=tfds.download.DownloadConfig(
        #         num_shards=32,
        #     )
        # )

        # NUM_SHARDS = 32
        # dataset_info = tfds.core.DatasetInfo(
        #     builder=tfds.dataset_builders.AdhocBuilder,
        #     name="pubchem_large",
        #     version="1.0.3",
        #     features=tfds.features.FeaturesDict(
        #         {
        #             "smiles": tfds.features.Text(),
        #             "fingerprint": tfds.features.Tensor(
        #                 shape=(fp_bits,), dtype=np.bool_
        #             ),
        #         }
        #     ),
        #     split_dict={
        #         "train": tfds.core.SplitInfo(name="train", num_examples=len(train_ds), num_shards=NUM_SHARDS),
        #         "validation": tfds.core.SplitInfo(name="validation", num_examples=len(val_ds), num_shards=NUM_SHARDS),
        #     },
        # )


        pubchem_builder = tfds.dataset_builders.store_as_tfds_dataset(
            name="pubchem_large",
            version="1.0.3",
            features=tfds.features.FeaturesDict(
                {
                    "smiles": tfds.features.Text(),
                    "fingerprint": tfds.features.Tensor(
                        shape=(fp_bits,), dtype=np.bool_
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
            # download_config=tfds.download.DownloadConfig(
                # num_shards=64,
            # )
        )

        return pubchem_builder


def create_pubchem_datasets(config: config_dict.ConfigDict, seed: int):
    """Create PubChem datasets with SAFE encoding and molecular features."""
    # Import heavy modules only when needed
    import transformers

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
    data_loader =preprocess_or_load_pubchem(
        data_dir="./data/pubchem_large",
        fp_radius=2,
        fp_bits=2048,
        chunk_size=64,
        num_processes=16
    )

    print(data_loader.info)