import functools
import glob
import multiprocessing as mp
import os
from pathlib import Path

import jax
import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_collections import config_dict

from md4.tokenizers import SentencePieceTokenizer
from md4.utils.pubchem_worker import process_and_write_shard_tfrecord

# Heavy imports are now imported conditionally within functions to reduce memory usage


_SENTENCEPIECE_TOKENIZER = "data/sentencepiece_tokenizer.model"


def _parse_tfexample_fn(tfrecord, fp_bits=2048):
    """Parse a single TFRecord example."""
    # Define feature description based on the features used in dataset creation
    # Note: fingerprint might be stored as int64 in TFRecord but we want int8
    # SMILES could be either string (raw SMILES) or tokenized integers depending on tokenizer usage
    feature_description = {
        "smiles": tf.io.FixedLenFeature(
            [], tf.string
        ),  # Use VarLenFeature to handle both cases
        "fingerprint": tf.io.FixedLenFeature(
            [fp_bits], tf.int64
        ),  # Read as int8 directly
    }

    # Parse the input tf.train.Example proto using the dictionary above
    example = tf.io.parse_single_example(tfrecord, feature_description)

    # Convert fingerprint from int64 to int8 to match expected format
    example["fingerprint"] = tf.cast(example["fingerprint"], tf.int8)

    return example


def _parse_smiles_with_rdkit(smiles_bytes):
    """Parse SMILES string using RDKit and return processed string.

    Args:
        smiles_bytes: SMILES string as bytes from TensorFlow

    Returns:
        Processed SMILES string as bytes
    """
    try:
        # Decode bytes to string
        smiles_str = (
            smiles_bytes.numpy().decode("utf-8")
            if hasattr(smiles_bytes, "numpy")
            else smiles_bytes.decode("utf-8")
        )

        # TODO: Add RDKit processing here
        # Example placeholder - you can fill this in:
        # from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None:
            randomized_smiles = Chem.MolToSmiles(mol, doRandom=True)
            return randomized_smiles.encode("utf-8")

        # For now, just return the original SMILES
        return smiles_str.encode("utf-8")
    except Exception as e:
        # Return original on error
        return smiles_bytes if isinstance(smiles_bytes, bytes) else smiles_bytes.numpy()


def _apply_rdkit_parsing(example):
    """Apply RDKit parsing to SMILES field using tf.py_function."""
    # Apply RDKit parsing to the SMILES string
    parsed_smiles = tf.py_function(
        func=_parse_smiles_with_rdkit, inp=[example["smiles"]], Tout=tf.string
    )
    # Set shape since py_function loses shape information
    parsed_smiles.set_shape([])

    # Update the example with parsed SMILES
    example["smiles"] = parsed_smiles
    return example


def create_high_entropy_dataset(
    tfrecord_pattern,
    fp_bits=2048,
    cycle_length=10,
    block_length=1,
    file_shuffle_buffer=1000,
    record_shuffle_buffer=2000,
    batch_shuffle_buffer=20,
    use_rdkit_parsing=True,
    seed=None,
):
    """Create a high-entropy dataset from TFRecord files with multiple levels of shuffling."""
    # First, list all file paths to the sharded tfrecord dataset
    dataset = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=True, seed=seed)

    # Make sure to fully shuffle the list of tfrecord files
    dataset = dataset.shuffle(
        buffer_size=file_shuffle_buffer, reshuffle_each_iteration=True, seed=seed
    )

    # Preprocesses files concurrently and interleaves records from each file into a single, unified dataset
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,  # Important for high entropy
    )

    # Parse raw protobufs into structs
    dataset = dataset.map(
        functools.partial(_parse_tfexample_fn, fp_bits=fp_bits),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Apply RDKit parsing if enabled
    if use_rdkit_parsing:
        dataset = dataset.map(_apply_rdkit_parsing, num_parallel_calls=tf.data.AUTOTUNE)

    # Add record-level shuffling for maximum entropy
    dataset = dataset.shuffle(
        record_shuffle_buffer, reshuffle_each_iteration=True, seed=seed
    )

    return dataset


def find_data_files(data_file_pattern):
    data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
    assert len(data_files) > 0, f"No file found with pattern {data_file_pattern}."
    return data_files


def preprocess_or_load_pubchem(
    tfrecord_dir,
    parquet_dir,
    version="1.0.3",
    fp_radius=2,
    fp_bits=2048,
    training_shards=16,
    validation_shards=4,
    max_length=160,
    num_workers=None,
    include_formula=False,
):
    """Load and preprocess PubChem dataset with SAFE encoding and tokenizer training.

    Args:
        tfrecord_dir: Directory to store/read TFRecord files
        parquet_dir: Directory containing parquet files
        fp_radius: Morgan fingerprint radius
        fp_bits: Number of bits for fingerprint
        pad_to_length: Length to pad atom types to
        chunk_size: Chunk size for multiprocessing (default: 1000)
        num_processes: Number of processes to use (default: min(cpu_count(), 8))
    """
    # Import heavy modules only when needed

    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    tfds_data_dir = Path(tfrecord_dir) / version
    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder_from_directory(tfds_data_dir)
        print("Loading existing TFDS dataset...")
        return pubchem_builder
    except Exception:
        # Parameters for multiprocessing batch function
        # (removed process_func since we're using batch processing now)

        print("Loading full dataset without streaming...")
        ds_full = pl.read_parquet(
            find_data_files(os.path.join(parquet_dir, "train-*.parquet"))
        )
        print(f"Loaded {len(ds_full)} samples")
        train_size = int(len(ds_full) * 0.98)
        train_ds = ds_full[:train_size]
        val_ds = ds_full[train_size:]

        _num_workers = num_workers if num_workers is not None else mp.cpu_count()
        print(f"Using {_num_workers} workers for processing...")

        training_shard_size = len(train_ds) // training_shards
        if training_shard_size < 1:
            training_shard_size = 1
        training_shards_tasks = [
            train_ds["smiles"][i * training_shard_size : (i + 1) * training_shard_size]
            for i in range(training_shards)
        ]

        validation_shard_size = len(val_ds) // validation_shards
        if validation_shard_size < 1:
            validation_shard_size = 1
        validation_shards_tasks = [
            val_ds["smiles"][
                i * validation_shard_size : (i + 1) * validation_shard_size
            ]
            for i in range(validation_shards)
        ]

        # Include formula shards if include_formula is True
        # Use the same indexing as SMILES shards to ensure perfect alignment
        if include_formula:
            training_formula_shards = [
                train_ds["molecular_formula"][
                    i * training_shard_size : (i + 1) * training_shard_size
                ]
                for i in range(len(training_shards_tasks))
            ]
            validation_formula_shards = [
                val_ds["molecular_formula"][
                    i * validation_shard_size : (i + 1) * validation_shard_size
                ]
                for i in range(len(validation_shards_tasks))
            ]
        else:
            training_formula_shards = [None] * len(training_shards_tasks)
            validation_formula_shards = [None] * len(validation_shards_tasks)

        from tqdm.contrib.concurrent import process_map

        features = tfds.features.FeaturesDict(
            {
                "smiles": tfds.features.Text(),
                "fingerprint": tfds.features.Tensor(shape=(fp_bits,), dtype=np.int8),
            }
        )

        if not tfds_data_dir.exists():
            tfds_data_dir.mkdir(parents=True, exist_ok=True)

        # Create partially applied function with fixed arguments for training
        process_training_shard = functools.partial(
            process_and_write_shard_tfrecord,
            total_shards=training_shards,
            split="train",
            output_dir=tfds_data_dir,
            features=features,
            fp_bits=fp_bits,
            fp_radius=fp_radius,
            canonical=config.get("canonical", True),
            randomize=config.get("randomize", True),
            isomeric=config.get("isomeric", False),
            include_formula=include_formula,
        )

        valid_training_counts = process_map(
            process_training_shard,
            [
                (i, shard, training_formula_shards[i])
                for i, shard in enumerate(training_shards_tasks)
            ],
            max_workers=_num_workers,
        )

        # Create partially applied function with fixed arguments for validation
        process_validation_shard = functools.partial(
            process_and_write_shard_tfrecord,
            total_shards=validation_shards,
            split="validation",
            output_dir=tfds_data_dir,
            features=features,
            fp_bits=fp_bits,
            fp_radius=fp_radius,
            canonical=config.get("canonical", True),
            randomize=config.get("randomize", True),
            isomeric=config.get("isomeric", False),
            include_formula=include_formula,
        )

        valid_validation_counts = process_map(
            process_validation_shard,
            [
                (i, shard, validation_formula_shards[i])
                for i, shard in enumerate(validation_shards_tasks)
            ],
            max_workers=_num_workers,
        )

        tfds.folder_dataset.write_metadata(
            data_dir=str(tfds_data_dir),
            features=features,
            split_infos=[
                tfds.core.SplitInfo(
                    name="train", shard_lengths=valid_training_counts, num_bytes=0
                ),
                tfds.core.SplitInfo(
                    name="validation",
                    shard_lengths=valid_validation_counts,
                    num_bytes=0,
                ),
            ],
            description="PubChem dataset with SMILES, Morgan fingerprints (radius={fp_radius}, bits={fp_bits})",
            check_data=False,
        )

        pubchem_builder = tfds.builder_from_directory(tfds_data_dir)

        return pubchem_builder


def random_pad_after_first_sep_ratio(
    tokens: tf.Tensor,  # 1D int tensor (unpadded/truncated), length L <= max_length
    sep_id: int,
    pad_id: int,
    max_length: int,
    interior_frac: float = 0.5,  # upper bound fraction of P moved after [SEP]
    seed: int | None = None,
) -> tf.Tensor:
    tokens = tf.convert_to_tensor(tokens)
    dtype = tokens.dtype
    sep_id = tf.cast(sep_id, dtype)
    pad_id = tf.cast(pad_id, dtype)

    L = tf.shape(tokens)[0]
    P = tf.maximum(max_length - L, 0)  # total PADs required

    def no_pad():
        out = tokens[:max_length]
        out.set_shape([max_length])
        return out

    # locate first [SEP]
    sep_pos_all = tf.where(tf.equal(tokens, sep_id))  # [?,1]
    has_sep = tf.greater(tf.shape(sep_pos_all)[0], 0)

    def only_trailing():
        out = tf.concat([tokens, tf.fill([P], pad_id)], axis=0)
        out = out[:max_length]
        out.set_shape([max_length])
        return out

    def with_sep():
        sep_pos = tf.reduce_min(tf.reshape(sep_pos_all, [-1]))
        pre = tokens[: sep_pos + 1]  # includes [SEP]
        post = tokens[sep_pos + 1 :]  # strictly after [SEP]
        L2 = tf.shape(post)[0]

        # sample interior uniformly from 0..floor(P*interior_frac)
        frac = tf.clip_by_value(tf.cast(interior_frac, tf.float32), 0.0, 1.0)
        interior_max = tf.cast(tf.floor(tf.cast(P, tf.float32) * frac), tf.int32)
        interior = tf.cond(
            tf.greater(interior_max, 0),
            lambda: tf.random.uniform(
                [], minval=0, maxval=interior_max + 1, dtype=tf.int32, seed=seed
            ),
            lambda: tf.zeros([], tf.int32),
        )
        end_pad = P - interior

        # build the post-[SEP] region by placing the L2 tokens into (L2+interior) slots
        n_total_post = L2 + interior
        base = tf.fill([n_total_post], pad_id)
        perm = tf.random.shuffle(tf.range(n_total_post, dtype=tf.int32), seed=seed)
        tok_pos = tf.sort(perm[:L2])  # preserves token order
        region = tf.tensor_scatter_nd_update(base, tf.expand_dims(tok_pos, 1), post)

        out = tf.concat([pre, region, tf.fill([end_pad], pad_id)], axis=0)
        out = out[:max_length]
        out.set_shape([max_length])
        return out

    return tf.cond(P <= 0, no_pad, lambda: tf.cond(has_sep, with_sep, only_trailing))


def create_pubchem_datasets(config: config_dict.ConfigDict, seed: int):
    """Create PubChem datasets with SAFE encoding and molecular features."""
    # Molecular dataset with SAFE, SMILES and Morgan fingerprints
    fp_radius = config.get("fp_radius", 2)
    fp_bits = config.get("fp_bits", 2048)
    max_length = config.get("max_length", 160)
    tokenizer_path = config.get("tokenizer", _SENTENCEPIECE_TOKENIZER)
    batch_size = config.get("batch_size", 512)

    # Adjust batch size for multi-host environments
    if config.get("initialize_multihost", False):
        # In multi-host mode, each process should handle batch_size // num_processes
        process_batch_size = (batch_size // jax.process_count()) * config.get(
            "process_batch_size_multiplier", 1
        )
        print(
            f"Multi-host detected: using process batch size {process_batch_size} (total: {batch_size})"
        )
    else:
        process_batch_size = batch_size

    tokenizer = SentencePieceTokenizer(
        model_path=tokenizer_path,
    )

    # Use preprocess_pubchem to get the dataset builder
    num_processes = config.get("num_processes", 64)
    include_formula = config.get("include_formula", False)
    # Get data directories from config or use defaults
    tfrecord_dir = config.get("tfrecord_data_dir", "./data/pubchem_large_text")
    parquet_dir = config.get("parquet_data_dir", "data/pubchem_large/data")

    pubchem_builder = preprocess_or_load_pubchem(
        tfrecord_dir=tfrecord_dir,
        parquet_dir=parquet_dir,
        max_length=max_length,
        version=config.get("version", "1.0.0"),
        fp_radius=fp_radius,
        fp_bits=fp_bits,
        training_shards=config.get("training_shards", 64),
        validation_shards=config.get("validation_shards", 2),
        num_workers=num_processes,
        include_formula=include_formula,
    )

    # Load SMILES tokenizer
    vocab_size = tokenizer.vocab_size

    # Create high-entropy datasets using TFRecord pattern loading
    tfds_data_dir = Path(tfrecord_dir) / config.get("version", "1.0.0")

    # Define TFRecord patterns for train and validation splits (matching pubchem_worker.py naming)
    train_pattern = str(tfds_data_dir / "pubchem_large-train.tfrecord-?????-of-?????")
    validation_pattern = str(
        tfds_data_dir / "pubchem_large-validation.tfrecord-?????-of-?????"
    )

    # Check if TFRecord files exist, otherwise fall back to TFDS builder
    use_high_entropy_loading = (
        len(glob.glob(train_pattern)) > 0 and len(glob.glob(validation_pattern)) > 0
    )

    def _tokenize_and_truncate(x):
        toks = tokenizer.encode(x["smiles"])
        toks = tf.cast(toks, tf.int32)
        x["smiles"] = random_pad_after_first_sep_ratio(
            tokens=toks,
            sep_id=int(tokenizer.sep_id),
            pad_id=int(tokenizer.pad_id),
            max_length=max_length,
            interior_frac=config.get("interior_frac", 0.1),
            seed=seed,
        )

        return x

    if use_high_entropy_loading:
        print("Using high-entropy TFRecord loading for maximum data diversity...")

        # Create high-entropy training dataset
        train_dataset = create_high_entropy_dataset(
            train_pattern,
            fp_bits=fp_bits,
            cycle_length=config.get("cycle_length", 16),
            block_length=config.get("block_length", 4),
            file_shuffle_buffer=config.get("file_shuffle_buffer", 1000),
            record_shuffle_buffer=config.get("record_shuffle_buffer", 10000),
            use_rdkit_parsing=config.get("use_rdkit_parsing", False),
            seed=seed,
        )

        # Add process-specific data sharding for multi-host
        if config.get("initialize_multihost", False):
            # Shard the dataset across processes
            train_dataset = train_dataset.shard(
                num_shards=jax.process_count(), index=jax.process_index()
            )

        train_dataset = train_dataset.map(
            _tokenize_and_truncate,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Apply multiple levels of shuffling and batching for maximum entropy
        train_dataset = train_dataset.shuffle(
            process_batch_size * 16, reshuffle_each_iteration=True, seed=seed
        )  # Large shuffle buffer
        train_dataset = train_dataset.repeat()  # Repeat for continuous training
        train_dataset = train_dataset.batch(process_batch_size, drop_remainder=True)
        train_dataset = train_dataset.shuffle(
            config.get("batch_shuffle_buffer", 50),
            reshuffle_each_iteration=True,
            seed=seed + 1 if seed is not None else None,
        )  # Batch-level shuffling
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.as_numpy_iterator()

        # Create high-entropy validation dataset
        eval_dataset = create_high_entropy_dataset(
            validation_pattern,
            fp_bits=fp_bits,
            cycle_length=config.get("cycle_length", 8),
            block_length=config.get("block_length", 2),
            file_shuffle_buffer=config.get("file_shuffle_buffer", 200),
            record_shuffle_buffer=config.get("record_shuffle_buffer", 2000),
            use_rdkit_parsing=config.get("use_rdkit_parsing", False),
            seed=seed + 2 if seed is not None else None,
        )

        # Add process-specific data sharding for multi-host
        if config.get("initialize_multihost", False):
            # Shard the dataset across processes
            eval_dataset = eval_dataset.shard(
                num_shards=jax.process_count(), index=jax.process_index()
            )

        eval_dataset = eval_dataset.map(
            _tokenize_and_truncate,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        eval_dataset = eval_dataset.shuffle(
            process_batch_size * 8,
            reshuffle_each_iteration=True,
            seed=seed + 3 if seed is not None else None,
        )  # Shuffle with larger buffer
        eval_dataset = eval_dataset.repeat()  # Repeat for continuous evaluation
        eval_dataset = eval_dataset.batch(process_batch_size, drop_remainder=True)
        eval_dataset = eval_dataset.shuffle(
            config.get("batch_shuffle_buffer", 20),
            reshuffle_each_iteration=True,
            seed=seed + 4 if seed is not None else None,
        )  # Batch-level shuffling
        eval_dataset = eval_dataset.prefetch(tf.data.AUTOTUNE)
        eval_dataset = eval_dataset.as_numpy_iterator()

    else:
        print(
            "TFRecord files not found, falling back to TFDS builder with enhanced shuffling..."
        )

        # Load Datasets using TFDS builder with enhanced shuffling
        train_split = tfds.split_for_jax_process("train", drop_remainder=True)
        train_dataset = pubchem_builder.as_dataset(
            split=train_split,
            shuffle_files=True,
        )

        train_dataset = train_dataset.map(
            _tokenize_and_truncate,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        train_dataset = train_dataset.repeat()  # Repeat for continuous training
        train_dataset = train_dataset.shuffle(
            batch_size * 64, reshuffle_each_iteration=True, seed=seed
        )  # Enhanced shuffle buffer
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        train_dataset = train_dataset.shuffle(
            config.get("batch_shuffle_buffer", 50),
            reshuffle_each_iteration=True,
            seed=seed + 1 if seed is not None else None,
        )  # Batch-level shuffling
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.as_numpy_iterator()

        validation_split = tfds.split_for_jax_process("validation", drop_remainder=True)
        eval_dataset = pubchem_builder.as_dataset(
            split=validation_split,
            shuffle_files=True,
        )
        eval_dataset = eval_dataset.map(
            _tokenize_and_truncate,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        eval_dataset = eval_dataset.repeat()  # Repeat for continuous evaluation
        eval_dataset = eval_dataset.shuffle(
            batch_size * 8,
            reshuffle_each_iteration=True,
            seed=seed + 2 if seed is not None else None,
        )  # Enhanced shuffle buffer
        eval_dataset = eval_dataset.batch(batch_size, drop_remainder=True)
        eval_dataset = eval_dataset.shuffle(
            config.get("batch_shuffle_buffer", 20),
            reshuffle_each_iteration=True,
            seed=seed + 3 if seed is not None else None,
        )  # Batch-level shuffling
        eval_dataset = eval_dataset.prefetch(tf.data.AUTOTUNE)
        eval_dataset = eval_dataset.as_numpy_iterator()

    info = {
        "fp_radius": fp_radius,
        "fp_bits": fp_bits,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "sep_id": tokenizer.sep_id,
        "pad_id": tokenizer.pad_id,
    }

    return (
        train_dataset,
        {
            "validation": eval_dataset,
        },
        info,
    )


if __name__ == "__main__":
    train_dataset, eval_datasets, info = create_pubchem_datasets(
        config_dict.ConfigDict(
            {
                "fp_radius": 2,
                "fp_bits": 4096,
                "max_length": 128,
                "tokenizer": "data/sentencepiece_tokenizer_bpe_3000_newcorpus.model",
                "batch_size": 512,
                "version": "1.1.0",
                "training_shards": 128,
                "validation_shards": 4,
                "num_processes": 128,
                "include_formula": True,  # Set to True to include molecular formulas
                # SMILES processing configuration
                "canonical": True,  # Whether to canonicalize SMILES
                "randomize": True,  # Whether to randomize SMILES output
                "isomeric": False,  # Whether to include stereochemistry
                # Data directory configuration
                "tfrecord_data_dir": "./data/pubchem_large_text",
                "parquet_data_dir": "data/pubchem_large/data",
                # High-entropy loading configuration
                "cycle_length": 16,  # Number of files to interleave concurrently
                "block_length": 4,  # Number of consecutive elements from each file
                "file_shuffle_buffer": 1000,  # File-level shuffle buffer
                "record_shuffle_buffer": 10000,  # Record-level shuffle buffer
                "batch_shuffle_buffer": 50,  # Batch-level shuffle buffer
                "use_rdkit_parsing": False,
            }
        ),
        seed=42,
    )

    features = next(train_dataset)

    tokenizer = info["tokenizer"]

    print(features["smiles"][0].shape)

    decoded = tokenizer.batch_decode(
        features["smiles"][:3]
    )  # Decode smiles of the first sample
    print("Decoded SMILES:", decoded)

    decoded_2 = tokenizer._decode_with_padding_removal(
        features["smiles"][1]
    )  # Decode smiles of the second sample
    print("Decoded SMILES 2:", decoded_2)
    print(
        "Sample features:", features["smiles"][:2]
    )  # Print first 10 tokens of the first sample
    print("Sample features dtype:", features["smiles"].dtype)
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
