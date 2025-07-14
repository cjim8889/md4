"""PubChem molecular dataset preprocessing."""

import dataclasses
import multiprocessing as mp
import os
from functools import partial
from typing import Any

import grain.python as grain
import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers
from datasets import load_dataset
from ml_collections import config_dict
from tqdm import tqdm

from md4 import rdkit_utils

FlatFeatures = dict[str, Any]


def process_inchi_to_features(inchi, fp_radius=2, fp_bits=4096):
    """Process a single InChI to extract SMILES and molecular features."""
    smiles = rdkit_utils.inchi_to_smiles(inchi)
    if smiles:
        features = rdkit_utils.get_molecule_features(
            smiles, radius=fp_radius, n_bits=fp_bits
        )
        if features is not None:
            return {
                "smiles": smiles,
                "fingerprint": features["fingerprint"],
            }
    return None


def preprocess_pubchem(data_dir, fp_radius=2, fp_bits=4096):
    """Load and preprocess PubChem dataset with Morgan fingerprints."""

    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pubchem_train_path = os.path.join(data_dir, "pubchem_train.parquet")
    pubchem_val_path = os.path.join(data_dir, "pubchem_val.parquet")

    # Check if preprocessed data already exists
    if os.path.exists(pubchem_train_path) and os.path.exists(pubchem_val_path):
        print("Loading existing preprocessed PubChem data...")
        pubchem_train_df = pl.read_parquet(pubchem_train_path)
        pubchem_val_df = pl.read_parquet(pubchem_val_path)
    else:
        print("Downloading and preprocessing PubChem data...")

        # Load dataset from HuggingFace
        ds = load_dataset("sagawa/pubchem-10m-canonicalized")
        df_train = ds["train"].to_pandas()

        # Process training data
        print("Processing training SMILES...")
        pubchem_set_raw = set(df_train["smiles"])
        pubchem_smiles_list = list(pubchem_set_raw)

        # Process SMILES to InChI for deduplication
        pubchem_inchis = rdkit_utils.process_pubchem_data(pubchem_smiles_list)
        pubchem_set = set(pubchem_inchis)
        pubchem_inchis = list(pubchem_set)

        # Shuffle and split
        import random

        random.seed(42)
        random.shuffle(pubchem_inchis)

        split_idx = int(0.95 * len(pubchem_inchis))
        pubchem_train_inchis = pubchem_inchis[:split_idx]
        pubchem_val_inchis = pubchem_inchis[split_idx:]

        # Get number of CPU cores for multiprocessing
        num_cores = mp.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing")

        # Create partial function with fixed parameters
        process_func = partial(
            process_inchi_to_features, fp_radius=fp_radius, fp_bits=fp_bits
        )

        # Generate features for training data using multiprocessing
        print("Generating features for training data...")
        with mp.Pool(processes=num_cores) as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, pubchem_train_inchis),
                    total=len(pubchem_train_inchis),
                    desc="Processing training data",
                )
            )
        # Filter out None results
        train_data = [result for result in results if result is not None]

        # Generate features for validation data using multiprocessing
        print("Generating features for validation data...")
        with mp.Pool(processes=num_cores) as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, pubchem_val_inchis),
                    total=len(pubchem_val_inchis),
                    desc="Processing validation data",
                )
            )
        # Filter out None results
        val_data = [result for result in results if result is not None]

        # Save to parquet


        train_data_dict = {
            "smiles": [item["smiles"] for item in train_data],
            "fingerprint": [item["fingerprint"] for item in train_data],
        }
        val_data_dict = {
            "smiles": [item["smiles"] for item in val_data],
            "fingerprint": [item["fingerprint"] for item in val_data],
        }
        
        pubchem_train_df = pl.DataFrame(train_data_dict)
        pubchem_val_df = pl.DataFrame(val_data_dict)

        pubchem_train_df.write_parquet(pubchem_train_path)
        pubchem_val_df.write_parquet(pubchem_val_path)

        print(
            f"Saved {len(train_data)} training examples and {len(val_data)} validation examples"
        )

    def load_pubchem_split(split_df):
        """Convert DataFrame to TensorFlow dataset."""
        # Convert fingerprint arrays to proper 2D numpy array
        fingerprints = split_df["fingerprint"].to_numpy().astype(np.int32)
        smiles = np.stack(split_df["smiles"].values).astype(np.dtype(str))
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "smiles": smiles,
                "fingerprint": fingerprints,
            }
        )

        return ds

    # Try to load existing dataset, create if doesn't exist
    try:
        # Check if dataset already exists
        pubchem_builder = tfds.builder("pubchem", data_dir=data_dir, config="pubchem")
        print("Loading existing TFDS dataset...")
    except Exception:
        # Create dataset builder if it doesn't exist
        print("Creating new TFDS dataset...")
        pubchem_builder = tfds.dataset_builders.store_as_tfds_dataset(
            name="pubchem",
            version="1.0.0",
            features=tfds.features.FeaturesDict(
                {
                    "smiles": tfds.features.Text(),
                    "fingerprint": tfds.features.Tensor(
                        shape=(fp_bits,), dtype=tf.int32
                    ),
                }
            ),
            split_datasets={
                "train": load_pubchem_split(pubchem_train_df),
                "validation": load_pubchem_split(pubchem_val_df),
            },
            config="pubchem",
            data_dir=data_dir,
            description=f"PubChem dataset with Morgan fingerprints (radius={fp_radius}, bits={fp_bits}) and atom types",
            file_format="array_record",
            disable_shuffling=True,
        )

    return pubchem_builder


if __name__ == "__main__":
    preprocess_pubchem(data_dir="data/pubchem")