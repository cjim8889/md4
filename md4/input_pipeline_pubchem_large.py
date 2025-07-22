"""PubChem molecular dataset input pipeline with SAFE encoding."""

import dataclasses
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import glob
from pathlib import Path

import grain.python as grain
import numpy as np
import polars as pl
import tensorflow_datasets as tfds
import transformers
from ml_collections import config_dict

from tqdm import tqdm

from md4 import rdkit_utils

_SAFE_TOKENIZER = "data/pubchem_safe/safe_tokenizer"


def find_data_files(data_file_pattern):
  data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
  return data_files

# ---------------- pool bootstrap ----------------
def _init_worker_pubchem(smiles_name, smiles_shape, smiles_dtype,
                         fp_name, fp_shape, status_name):
    global SMILES, FINGERPRINTS, STATUS, SMILES_SHM, FP_SHM, STATUS_SHM        # be explicit
    # Keep references to shared memory objects to prevent garbage collection
    SMILES_SHM   = shared_memory.SharedMemory(smiles_name)
    FP_SHM       = shared_memory.SharedMemory(fp_name)
    STATUS_SHM   = shared_memory.SharedMemory(status_name)
    
    SMILES       = np.ndarray(smiles_shape,     dtype=smiles_dtype,
                              buffer=SMILES_SHM.buf)
    FINGERPRINTS = np.ndarray(fp_shape,         dtype=np.bool_,
                              buffer=FP_SHM.buf)
    STATUS       = np.ndarray((smiles_shape[0],), dtype=np.uint8,
                              buffer=STATUS_SHM.buf)

# ---------------- worker proper -----------------
def optimized_batch_worker_pubchem(task):                        # accepts tuple (start, stop)
    """Process a batch of SMILES with minimal overhead using global shared arrays."""
    start, stop = task  # unpack the tuple
    processed = 0
    for i in range(start, stop):
        if rdkit_utils.process_smiles_with_shared_memory(
                SMILES[i], FINGERPRINTS, i):
            STATUS[i] = 1
            processed += 1
        else:
            STATUS[i] = 0
    return processed


def process_smiles_worker(args, shape):
    """Process a single SMILES string - optimized for chunksize batching."""
    smi, i = args  # Unpack the tuple
    shm_fingerprint = shared_memory.SharedMemory(name="fingerprint_shm")
    shm_fingerprint_array = np.ndarray(shape, dtype=np.bool_, buffer=shm_fingerprint.buf)
    if rdkit_utils.process_smiles_with_shared_memory(smi, shm_fingerprint_array, i):
        return i
    else:
        return None

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

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder("pubchem_large", version="1.0.3", data_dir=data_dir, config="pubchem_large")
        print("Loading existing TFDS dataset...")
    except Exception:
        # Parameters for multiprocessing batch function
        # (removed process_func since we're using batch processing now)

        print("Loading full dataset without streaming...")
        ds_full = pl.read_parquet(find_data_files("data/pubchem_large/data/train-*.parquet"))
        print(f"Loaded {len(ds_full)} samples")
        train_size = int(len(ds_full) * 0.98)
        train_ds = ds_full[:train_size]
        val_ds = ds_full[train_size:]

        def iterator_closure(ds: pl.DataFrame):
            def iterator():
                # Use optimized multiprocessing with minimal memory exchange
                _num_processes = num_processes if num_processes is not None else mp.cpu_count()
                
                smiles_data = ds["smiles"]
                total_samples = len(smiles_data)
                batch_size = chunk_size  # Use chunk_size as batch_size
                
                print(f"Processing {total_samples} SMILES with {_num_processes} processes (batch_size={batch_size})...")
                
                # Convert to numpy string array - much more efficient than encoding/decoding
                smiles_array = smiles_data.to_numpy()
                
                # Create shared memory for fixed-size string array
                smiles_shm = shared_memory.SharedMemory(create=True, size=smiles_array.nbytes)
                smiles_shared_array = np.ndarray(smiles_array.shape, dtype=smiles_array.dtype, buffer=smiles_shm.buf)
                
                # Copy data to shared memory
                smiles_shared_array[:] = smiles_array
                
                # Create shared memory for fingerprints and status
                fingerprint_size = total_samples * fp_bits
                fingerprint_shm = shared_memory.SharedMemory(create=True, size=fingerprint_size)
                fingerprint_array = np.ndarray((total_samples, fp_bits), dtype=np.bool_, buffer=fingerprint_shm.buf)
                
                status_shm = shared_memory.SharedMemory(create=True, size=total_samples)  # uint8
                status_array = np.ndarray((total_samples,), dtype=np.uint8, buffer=status_shm.buf)
                status_array.fill(255)  # Initialize with "not processed" value
                
                try:
                    # Create batch ranges as simple (start, stop) tuples
                    tasks = [(i, min(i + batch_size, total_samples))
                             for i in range(0, total_samples, batch_size)]
                    
                    with mp.get_context("fork").Pool(
                            processes=_num_processes,
                            initializer=_init_worker_pubchem,
                            initargs=(smiles_shm.name, smiles_shared_array.shape,
                                      smiles_shared_array.dtype,
                                      fingerprint_shm.name, fingerprint_array.shape,
                                      status_shm.name)) as pool:
                        
                        # Process batches
                        total_processed = 0
                        since_last_update = 0
                        
                        with tqdm(total=total_samples, desc="Processing batches") as pbar:
                            for result in pool.imap_unordered(optimized_batch_worker_pubchem, tasks, chunksize=16):
                                total_processed += result
                                since_last_update += result
                                if since_last_update >= 2000:
                                    pbar.update(since_last_update)
                                    since_last_update = 0
                        
                        # Update progress for remaining items
                        if since_last_update > 0:
                            pbar.update(since_last_update)
                        
                        # Yield successful results
                        for i in range(total_samples):
                            if status_array[i] == 1:
                                yield smiles_data[i], {
                                    "fingerprint": fingerprint_array[i, :],
                                    "smiles": smiles_data[i]
                                }
                
                finally:
                    # Clean up shared memory
                    try:
                        smiles_shm.close()
                        smiles_shm.unlink()
                        fingerprint_shm.close()
                        fingerprint_shm.unlink()
                        status_shm.close()
                        status_shm.unlink()
                    except Exception as e:
                        print(f"Cleanup error: {e}")
            
            return iterator

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
    preprocess_or_load_pubchem(
        data_dir="./data/pubchem_large",
        fp_radius=2,
        fp_bits=2048,
        chunk_size=128,
        num_processes=16
    )