"""PubChem molecular dataset input pipeline with SAFE encoding.

Memory Optimizations:
- Worker functions moved to separate minimal module to avoid loading heavy dependencies
- Heavy imports (tensorflow_datasets, transformers, polars, grain) are now conditional
- Worker processes only load numpy, shared_memory, and rdkit_utils (when needed)
- This reduces worker memory footprint by avoiding unnecessary module imports
"""

import glob
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers
from ml_collections import config_dict

from md4.utils.pubchem_worker import process_and_write_shard_tfrecord

# Heavy imports are now imported conditionally within functions to reduce memory usage

_SMILES_TOKENIZER = "data/pubchem_large_tokenizer"

def find_data_files(data_file_pattern):
  data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
  return data_files

def preprocess_or_load_pubchem(
        data_dir, 
        tokenizer: "transformers.PreTrainedTokenizerFast",
        version="1.0.3",
        fp_radius=2, 
        fp_bits=2048, 
        training_shards=16,
        validation_shards=4,
        max_length=160,
        num_workers=None,
        include_formula=False
    ):
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

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tfds_data_dir = Path(data_dir) / version
    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder_from_directory(tfds_data_dir)
        print("Loading existing TFDS dataset...")
        return pubchem_builder
    except Exception:
        # Parameters for multiprocessing batch function
        # (removed process_func since we're using batch processing now)

        print("Loading full dataset without streaming...")
        ds_full = pl.read_parquet(find_data_files("data/pubchem_large/data/train-*.parquet"))
        print(f"Loaded {len(ds_full)} samples")
        train_size = int(len(ds_full) * 0.98)
        train_ds = ds_full[:train_size]
        val_ds = ds_full[train_size:]


        _num_workers = num_workers if num_workers is not None else mp.cpu_count()
        print(f"Using {_num_workers} workers for processing...")

        training_shard_size = len(train_ds) // training_shards
        if training_shard_size < 1:
            training_shard_size = 1
        training_shards_tasks = [train_ds["smiles"][i * training_shard_size:(i + 1) * training_shard_size] for i in range(training_shards)]

        validation_shard_size = len(val_ds) // validation_shards
        if validation_shard_size < 1:
            validation_shard_size = 1
        validation_shards_tasks = [val_ds["smiles"][i * validation_shard_size:(i + 1) * validation_shard_size] for i in range(validation_shards)]

        # Include formula shards if include_formula is True
        # Use the same indexing as SMILES shards to ensure perfect alignment
        if include_formula:
            training_formula_shards = [train_ds["molecular_formula"][i * training_shard_size:(i + 1) * training_shard_size] for i in range(len(training_shards_tasks))]
            validation_formula_shards = [val_ds["molecular_formula"][i * validation_shard_size:(i + 1) * validation_shard_size] for i in range(len(validation_shards_tasks))]
        else:
            training_formula_shards = [None] * len(training_shards_tasks)
            validation_formula_shards = [None] * len(validation_shards_tasks)


        from tqdm.contrib.concurrent import process_map

        features = tfds.features.FeaturesDict(
            {
                "smiles": tfds.features.Tensor(
                    shape=(max_length,), dtype=np.int32
                ),
                "fingerprint": tfds.features.Tensor(
                    shape=(fp_bits,), dtype=np.int8
                ),
            }
        )

        if not tfds_data_dir.exists():
            tfds_data_dir.mkdir(parents=True, exist_ok=True)

        valid_training_counts = process_map(
            process_and_write_shard_tfrecord,
            [(i, len(training_shards_tasks), shard, "train", tfds_data_dir, features, fp_bits, tokenizer, max_length, include_formula, training_formula_shards[i]) for i, shard in enumerate(training_shards_tasks)],
            max_workers=_num_workers,
        )

        valid_validation_counts = process_map(
            process_and_write_shard_tfrecord,
            [(i, len(validation_shards_tasks), shard, "validation", tfds_data_dir, features, fp_bits, tokenizer, max_length, include_formula, validation_formula_shards[i]) for i, shard in enumerate(validation_shards_tasks)],
            max_workers=_num_workers,
        )

        tfds.folder_dataset.write_metadata(
            data_dir=str(tfds_data_dir),
            features=features,
            split_infos=[
                tfds.core.SplitInfo(name="train", shard_lengths=valid_training_counts, num_bytes=0),
                tfds.core.SplitInfo(name="validation", shard_lengths=valid_validation_counts, num_bytes=0)
            ],
            description="PubChem dataset with SMILES, Morgan fingerprints (radius={fp_radius}, bits={fp_bits})",
            check_data=False,
        )

        pubchem_builder = tfds.builder_from_directory(tfds_data_dir)

        return pubchem_builder


def create_pubchem_datasets(config: config_dict.ConfigDict, seed: int):
    """Create PubChem datasets with SAFE encoding and molecular features."""
    # Import heavy modules only when needed

    # Molecular dataset with SAFE, SMILES and Morgan fingerprints
    fp_radius = config.get("fp_radius", 2)
    fp_bits = config.get("fp_bits", 2048)
    max_length = config.get("max_length", 160)
    tokenizer_path = config.get("tokenizer", _SMILES_TOKENIZER)  
    batch_size = config.get("batch_size", 512)  

    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Use preprocess_pubchem to get the dataset builder
    num_processes = config.get("num_processes", 64)
    include_formula = config.get("include_formula", False)
    pubchem_builder = preprocess_or_load_pubchem(
        data_dir=os.path.join("./data", "pubchem_large"), 
        tokenizer=tokenizer,
        max_length=max_length,
        version=config.get("version", "1.0.5"),
        fp_radius=fp_radius, 
        fp_bits=fp_bits,
        training_shards=config.get("training_shards", 64),
        validation_shards=config.get("validation_shards", 2),
        num_workers=num_processes,
        include_formula=include_formula,
    )

    # Load SMILES tokenizer
    vocab_size = tokenizer.vocab_size

    # Load Datasets
    train_split = tfds.split_for_jax_process('train', drop_remainder=True)
    train_dataset = pubchem_builder.as_dataset(
        split=train_split,
        shuffle_files=True,
    )
    train_dataset = train_dataset.repeat()  # Repeat for continuous training
    train_dataset = train_dataset.shuffle(batch_size * 8)  # Shuffle with larger buffer
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.as_numpy_iterator()

    validation_split = tfds.split_for_jax_process('validation', drop_remainder=True)
    eval_dataset = pubchem_builder.as_dataset(
        split=validation_split,
        shuffle_files=True,
    )
    eval_dataset = eval_dataset.repeat()  # Repeat for continuous evaluation
    eval_dataset = eval_dataset.shuffle(batch_size * 8)  # Shuffle with larger buffer
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=True)
    eval_dataset = eval_dataset.prefetch(tf.data.AUTOTUNE)
    eval_dataset = eval_dataset.as_numpy_iterator()

    info = {
        "fp_radius": fp_radius,
        "fp_bits": fp_bits,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
    }

    return train_dataset, {
        "validation": eval_dataset,
    }, info

def calculate_sequence_lengths(dataset_iterator, num_batches=100):
    """Calculate average sequence lengths before padding (excluding token id 0)."""
    all_lengths = []
    
    for i, batch in enumerate(dataset_iterator):
        if i >= num_batches:
            break
            
        smiles_batch = batch['smiles']  # Shape: (batch_size, max_length)
        
        # Vectorized approach: find first occurrence of padding token (0) for each sequence
        # Create a mask where True indicates padding tokens
        padding_mask = (smiles_batch == 0)
        
        # Find the first True (padding token) along axis 1 for each sequence
        # If no padding found, argmax returns 0, so we need to handle that case
        first_padding_idx = np.argmax(padding_mask, axis=1)
        
        # Handle sequences with no padding (where argmax returns 0 but position 0 is not padding)
        no_padding_mask = ~padding_mask[:, 0] & (first_padding_idx == 0)
        lengths_before_padding = np.where(no_padding_mask, smiles_batch.shape[1], first_padding_idx)
        
        all_lengths.extend(lengths_before_padding)
    
    return np.array(all_lengths)

if __name__ == "__main__":
    train_dataset, eval_datasets, info = create_pubchem_datasets(
        config_dict.ConfigDict({
            "fp_radius": 2,
            "fp_bits": 4096,
            "max_length": 128,
            "tokenizer": "data/pubchem_large_tokenizer_2048",
            "batch_size": 512,
            "version": "1.0.7",
            "training_shards": 256,
            "validation_shards": 8,
            "num_processes": 160,
            "include_formula": True,  # Set to True to include molecular formulas
        }),
        seed=42
    )

    features = next(train_dataset)

    tokenizer = info["tokenizer"]
    decoded = tokenizer.decode(features["smiles"][0])  # Decode smiles of the first sample
    print("Decoded SMILES:", decoded)

    decoded_2 = tokenizer.decode(features["smiles"][1])  # Decode smiles of the second sample
    print("Decoded SMILES 2:", decoded_2)
    print("Sample features:", features["smiles"][:2])  # Print first 10 tokens of the first sample
    # Calculate average sequence lengths before padding
    # print("Calculating average sequence lengths before padding...")
    # lengths = calculate_sequence_lengths(train_dataset, num_batches=10000)
    
    # avg_length = np.mean(lengths)
    # median_length = np.median(lengths)
    # std_length = np.std(lengths)
    # min_length = np.min(lengths)
    # max_length = np.max(lengths)
    
    # print(f"Average length before padding: {avg_length:.2f}")
    # print(f"Median length before padding: {median_length:.2f}")
    # print(f"Standard deviation: {std_length:.2f}")
    # print(f"Min length: {min_length}")
    # print(f"Max length: {max_length}")
    # print(f"Total sequences analyzed: {len(lengths)}")

    results = tfds.benchmark(train_dataset, num_iter=1000, batch_size=512)
    # print(results)

    # Example usage