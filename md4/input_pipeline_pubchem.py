"""PubChem molecular dataset input pipeline."""

import dataclasses
import os
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

_SMILES_TOKENIZER = "data/smiles_tokenizer"


@dataclasses.dataclass
class Tokenize(grain.MapTransform):
    tokenizer: transformers.PreTrainedTokenizerFast
    max_length: int = 128

    def map(self, features):
        smiles = features["smiles"]
        features["smiles"] = self.tokenizer.encode(
            smiles.decode() if isinstance(smiles, bytes) else smiles,
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

        if "smiles" in features:
            features["smiles"] = features["smiles"].astype(np.int32)
        return features


def preprocess_pubchem(data_dir, fp_radius=2, fp_bits=2048):
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

        # Convert InChI back to SMILES and generate features
        print("Generating features for training data...")
        train_data = []
        for inchi in tqdm(pubchem_train_inchis, desc="Processing training data"):
            smiles = rdkit_utils.inchi_to_smiles(inchi)
            if smiles:
                features = rdkit_utils.get_molecule_features(
                    smiles, radius=fp_radius, n_bits=fp_bits
                )
                if features is not None:
                    train_data.append(
                        {
                            "smiles": smiles,
                            "fingerprint": features["fingerprint"],
                        }
                    )

        print("Generating features for validation data...")
        val_data = []
        for inchi in tqdm(pubchem_val_inchis, desc="Processing validation data"):
            smiles = rdkit_utils.inchi_to_smiles(inchi)
            if smiles:
                features = rdkit_utils.get_molecule_features(
                    smiles, radius=fp_radius, n_bits=fp_bits
                )
                if features is not None:
                    val_data.append(
                        {
                            "smiles": smiles,
                            "fingerprint": features["fingerprint"],
                        }
                    )

        # Save to parquet
        pubchem_train_df = pl.DataFrame(train_data)
        pubchem_val_df = pl.DataFrame(val_data)

        pubchem_train_df.write_parquet(pubchem_train_path)
        pubchem_val_df.write_parquet(pubchem_val_path)

        print(
            f"Saved {len(train_data)} training examples and {len(val_data)} validation examples"
        )

    def load_pubchem_split(split_df):
        """Convert DataFrame to TensorFlow dataset."""
        # Convert fingerprint arrays to proper 2D numpy array
        fingerprints = np.stack(split_df["fingerprint"]).astype(np.int32)
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


def create_pubchem_datasets(config: config_dict.ConfigDict, seed: int):
    """Create PubChem datasets with molecular features."""

    # Molecular dataset with SMILES and Morgan fingerprints
    fp_radius = config.get("fp_radius", 2)
    fp_bits = config.get("fp_bits", 2048)
    max_length = config.get("max_length", 128)

    # Use preprocess_pubchem to get the dataset builder
    pubchem_builder = preprocess_pubchem(
        data_dir=os.path.join("./data", "pubchem"), fp_radius=fp_radius, fp_bits=fp_bits
    )
    data_source = pubchem_builder.as_data_source()

    tokenizer = transformers.AutoTokenizer.from_pretrained(_SMILES_TOKENIZER)
    train_transformations = [
        Tokenize(tokenizer, max_length=max_length),
        ProcessMolecular(),
    ]
    train_source = data_source["train"]
    eval_transformations = [
        Tokenize(tokenizer, max_length=max_length),
        ProcessMolecular(),
    ]
    eval_source = {k: v for k, v in data_source.items() if k != "train"}

    info = {
        "fp_radius": fp_radius,
        "fp_bits": fp_bits,
        "atom_types": rdkit_utils.ATOM_TYPES,
        "tokenizer": tokenizer,
    }

    return train_source, train_transformations, eval_source, eval_transformations, info
