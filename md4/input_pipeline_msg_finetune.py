"""PubChem molecular dataset input pipeline with SAFE encoding.

Memory Optimizations:
- Worker functions moved to separate minimal module to avoid loading heavy dependencies
- Heavy imports (tensorflow_datasets, transformers, polars, grain) are now conditional
- Worker processes only load numpy, shared_memory, and rdkit_utils (when needed)
- This reduces worker memory footprint by avoiding unnecessary module imports
"""

from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers
from ml_collections import config_dict

from md4.pubchem_worker import process_and_write_msg_tfrecord

# Heavy imports are now imported conditionally within functions to reduce memory usage

_SMILES_TOKENIZER = "data/pubchem_large_tokenizer"

def preprocess_or_load_msg_finetune(
        data_dir, 
        tokenizer: "transformers.PreTrainedTokenizerFast",
        version="1.0.0",
        max_length=160,
        num_workers=None
    ):
    """Load and preprocess MSG finetune dataset with INCHI to SMILES conversion.
    
    Args:
        data_dir: Directory to store the dataset
        tokenizer: Pretrained tokenizer for SMILES encoding
        version: Version of the dataset
        max_length: Maximum sequence length for SMILES
        num_workers: Number of worker processes for parallel processing
    """
    # Import heavy modules only when needed
    import multiprocessing as mp
    import os

    import numpy as np
    import polars as pl
    import tensorflow_datasets as tfds
    
    fp_bits = 4096  # MSG dataset uses 4096-bit fingerprints

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tfds_data_dir = Path(data_dir) / version
    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder_from_directory(tfds_data_dir)
        print("Loading existing TFDS dataset...")
        return pubchem_builder
    except Exception:
        print("Loading MSG finetune dataset...")
        ds_train = pl.read_parquet("data/msg_finetune/train_predictions.parquet")
        ds_test = pl.read_parquet("data/msg_finetune/test_predictions.parquet")
        print(f"Loaded {len(ds_train)} train and {len(ds_test)} test samples")

        # Create tuples of (inchi, fingerprint) for processing
        train_data = list(zip(ds_train["inchi"].to_list(), ds_train["predicted_fingerprint"].to_list()))
        test_data = list(zip(ds_test["inchi"].to_list(), ds_test["predicted_fingerprint"].to_list()))

        # Print Data Shape
        print(f"Train data: {len(train_data)} samples")
        print(f"Test data: {len(test_data)} samples")
        print(f"Sample train data: {train_data[:2][1]}")

        from tqdm import tqdm

        features = tfds.features.FeaturesDict(
            {
                "smiles": tfds.features.Tensor(
                    shape=(max_length,), dtype=np.int32
                ),
                "fingerprint": tfds.features.Tensor(
                    shape=(fp_bits,), dtype=np.float32  # MSG fingerprints are float32
                ),
                "true_fingerprint": tfds.features.Tensor(
                    shape=(2048,), dtype=np.float32  # True fingerprint for loss calculation
                ),
            }
        )

        if not tfds_data_dir.exists():
            tfds_data_dir.mkdir(parents=True)

        # Process as single shards since dataset is smaller
        print("Processing training data...")
        valid_training_counts = []
        for args in tqdm([(0, 1, train_data, "train", tfds_data_dir, features, tokenizer, max_length)], desc="Processing train"):
            count = process_and_write_msg_tfrecord(args)
            valid_training_counts.append(count)

        print("Processing validation data...")
        valid_validation_counts = []
        for args in tqdm([(0, 1, test_data, "validation", tfds_data_dir, features, tokenizer, max_length)], desc="Processing validation"):
            count = process_and_write_msg_tfrecord(args)
            valid_validation_counts.append(count)

        tfds.folder_dataset.write_metadata(
            data_dir=str(tfds_data_dir),
            features=features,
            split_infos=[
                tfds.core.SplitInfo(name="train", shard_lengths=valid_training_counts, num_bytes=0),
                tfds.core.SplitInfo(name="validation", shard_lengths=valid_validation_counts, num_bytes=0)
            ],
            description=f"MSG finetune dataset with SMILES and fingerprints (bits={fp_bits})",
            check_data=False,
        )

        pubchem_builder = tfds.builder_from_directory(tfds_data_dir)

        return pubchem_builder


def create_msg_finetune_datasets(config: config_dict.ConfigDict, seed: int):
    """Create PubChem datasets with SAFE encoding and molecular features."""
    # Import heavy modules only when needed

    # Molecular dataset with SAFE, SMILES and Morgan fingerprints
    max_length = config.get("max_length", 160)
    tokenizer_path = config.get("tokenizer", _SMILES_TOKENIZER)  
    batch_size = config.get("batch_size", 512)  

    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Use preprocess_pubchem to get the dataset builder
    pubchem_builder = preprocess_or_load_msg_finetune(
        data_dir=config.get("data_dir", "data/msg_finetune"),
        tokenizer=tokenizer,
        version="1.0.1",
        max_length=max_length,
        num_workers=config.get("num_workers", 4)
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
        "fp_radius": 4096,
        "fp_bits": 2,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
    }

    return train_dataset, {
        "validation": eval_dataset,
    }, info

if __name__ == "__main__":
    import numpy as np
    train_dataset, eval_datasets, info = create_msg_finetune_datasets(
        config_dict.ConfigDict({
            "fp_radius": 2,
            "fp_bits": 4096,
            "max_length": 128,
            "tokenizer": _SMILES_TOKENIZER,
            "batch_size": 512,
            "version": "1.0.0",
            "training_shards": 1,
            "validation_shards": 1,
            "num_processes": 4,
        }),
        seed=42
    )

    features = next(train_dataset)

    tokenizer = info["tokenizer"]

    fingerprint = features["fingerprint"][0]  # Get fingerprint of the first sample
    print("Fingerprint shape:", fingerprint.shape)
    print("Fingerprint sum:", np.sum(fingerprint))  # Check if it has non-zero values
    print(fingerprint)

    true_fingerprint = features["true_fingerprint"][0]  # Get true fingerprint of the first sample
    print("True fingerprint shape:", true_fingerprint.shape)
    print("True fingerprint sum:", np.sum(true_fingerprint))  # Check if it has non-zero values
    print(true_fingerprint)


    decoded = tokenizer.decode(features["smiles"][0])  # Decode smiles of the first sample
    print("Decoded SMILES:", decoded)

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

    # results = tfds.benchmark(train_dataset, num_iter=1000, batch_size=512)
    # print(results)

    # Example usage