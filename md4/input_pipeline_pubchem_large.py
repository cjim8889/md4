"""PubChem molecular dataset input pipeline with SAFE encoding.

Memory Optimizations:
- Worker functions moved to separate minimal module to avoid loading heavy dependencies
- Heavy imports (tensorflow_datasets, transformers, polars, grain) are now conditional
- Worker processes only load numpy, shared_memory, and rdkit_utils (when needed)
- This reduces worker memory footprint by avoiding unnecessary module imports
"""

import dataclasses
import glob
import multiprocessing as mp
import os
from pathlib import Path

import grain.python as grain
import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers
from ml_collections import config_dict

from md4.pubchem_worker import process_and_write_shard_tfrecord

# Heavy imports are now imported conditionally within functions to reduce memory usage

_SMILES_TOKENIZER = "data/pubchem_large_tokenizer"

def find_data_files(data_file_pattern):
  data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
  return data_files

def tokenize(tokenizer: "transformers.PreTrainedTokenizerFast", max_length: int = 128):
    def _tokenize_py_func(features):
        
        def _py_tokenize(smiles_bytes):
            # Convert bytes to string if needed
            smiles_str = smiles_bytes.numpy().decode('utf-8') if hasattr(smiles_bytes, 'numpy') else smiles_bytes.decode('utf-8')
            
            # Tokenize using the tokenizer
            tokens = tokenizer.encode(
                smiles_str,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            ).reshape(-1)
            
            return tokens.astype(np.int32)
        
        # Use tf.py_function to wrap the Python function
        tokenized_smiles = tf.py_function(
            func=_py_tokenize,
            inp=[features["smiles"]],
            Tout=tf.int32
        )
        tokenized_smiles.set_shape([max_length])
        
        # Update features dict
        features["smiles"] = tokenized_smiles
        return features
    
    return _tokenize_py_func


def tokenize1(tokenizer: "transformers.PreTrainedTokenizerFast", max_length: int = 128):
    def _tokenize_py_func(features):
        
        # Tokenize using the tokenizer
        tokens = tokenizer.encode(
            str(features["smiles"]),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        ).reshape(-1)
        
        features["smiles"] = tokens

        return features
        
    return _tokenize_py_func

@dataclasses.dataclass
class Tokenize(grain.MapTransform):
    tokenizer: "transformers.PreTrainedTokenizerFast"
    max_length: int = 128

    def map(self, features):
        smiles_repr = features["smiles"]
        features["smiles"] = self.tokenizer.encode(
            smiles_repr.decode() if isinstance(smiles_repr, bytes) else smiles_repr,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        ).reshape(-1)
        return features

def preprocess_or_load_pubchem(
        data_dir, 
        version="1.0.3",
        fp_radius=2, 
        fp_bits=2048, 
        training_shards=16,
        validation_shards=4,
        num_workers=None
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

        from tqdm.contrib.concurrent import process_map

        features = tfds.features.FeaturesDict(
            {
                "smiles": tfds.features.Text(),
                "fingerprint": tfds.features.Tensor(
                    shape=(fp_bits,), dtype=np.int8
                ),
            }
        )

        if not tfds_data_dir.exists():
            tfds_data_dir.mkdir(parents=True, exist_ok=True)

        valid_training_counts = process_map(
            process_and_write_shard_tfrecord,
            [(i, len(training_shards_tasks), shard, "train", tfds_data_dir, features, fp_bits) for i, shard in enumerate(training_shards_tasks)],
            max_workers=_num_workers,
        )

        valid_validation_counts = process_map(
            process_and_write_shard_tfrecord,
            [(i, len(validation_shards_tasks), shard, "validation", tfds_data_dir, features, fp_bits) for i, shard in enumerate(validation_shards_tasks)],
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

    # Use preprocess_pubchem to get the dataset builder
    num_processes = config.get("num_processes", None)
    pubchem_builder = preprocess_or_load_pubchem(
        data_dir=os.path.join("./data", "pubchem_large"), 
        version=config.get("version", "1.0.5"),
        fp_radius=fp_radius, 
        fp_bits=fp_bits,
        training_shards=config.get("training_shards", 64),
        validation_shards=config.get("validation_shards", 2),
        num_workers=num_processes,
    )

    # Load SAFE tokenizer
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenization = tokenize1(tokenizer=tokenizer, max_length=max_length)
    vocab_size = tokenizer.vocab_size

    # Load Datasets
    train_split = tfds.split_for_jax_process('train', drop_remainder=True)
    train_dataset = pubchem_builder.as_dataset(
        split=train_split,
        shuffle_files=True,
    )
    train_dataset = train_dataset.map(tokenization, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(128)
    train_dataset = train_dataset.batch(512, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.as_numpy_iterator()

    validation_split = tfds.split_for_jax_process('validation', drop_remainder=True)
    eval_dataset = pubchem_builder.as_dataset(
        split=validation_split,
        shuffle_files=True,
    )
    eval_dataset = eval_dataset.map(tokenization, num_parallel_calls=tf.data.AUTOTUNE)
    eval_dataset = eval_dataset.shuffle(128)
    eval_dataset = eval_dataset.batch(512, drop_remainder=True)
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


if __name__ == "__main__":
    data_loader =preprocess_or_load_pubchem(
        data_dir="./data/pubchem_large",
        version="1.0.5",
        fp_radius=2,
        fp_bits=2048,
        num_workers=64,
        training_shards=64,
        validation_shards=2,
    )
    print(data_loader.info)
    import tensorflow_datasets as tfds

    ds = data_loader.as_dataset(split=split, shuffle_files=True)


    import tensorflow as tf
    import tensorflow_datasets as tfds
    import transformers
    # Example usage
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(_SMILES_TOKENIZER)

    ds = ds.shuffle(64)
    ds = ds.batch(512, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    # train_loader = grain.load(
    #     source=data_loader.as_data_source(split="train", decoders={"smiles": tfds.decode.SkipDecoding()}),
    #     num_epochs=1,
    #     shuffle=True,
    #     seed=42,
    #     transformations=[
    #         Tokenize(tokenizer=tokenizer, max_length=128),
    #     ],
    #     shard_options=grain.ShardByJaxProcess(drop_remainder=True),
    #     batch_size=512,
    #     worker_count=16,
    #     read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=1024),
    #     drop_remainder=True,
    # )

    import tensorflow_datasets as tfds

    results = tfds.benchmark(ds, num_iter=100, batch_size=512)
    print(results)

    # Example usage
    features = next(ds)
    print(features["smiles"].shape)
    print(features["fingerprint"].shape)
    print(features["smiles"][:5])
    print(features["fingerprint"][:5])