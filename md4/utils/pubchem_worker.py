"""Minimal worker module for PubChem SMILES processing.

This module contains only the essential imports and functions needed
for multiprocessing workers to avoid loading heavy dependencies.
"""

from typing import List, Tuple, Any
import numpy as np


def process_and_write_shard(args):
    """Process a shard of SMILES and write results to shared memory.

    Args:
        args: Tuple of (shard_index, total_shards, shard, split, output_dir, features, fp_bits)
    """
    shard_index, total_shards, shard, split, output_dir, features, fp_bits = args[:7]

    from array_record.python import array_record_module as array_record
    from .rdkit_utils import process_smiles
    import os

    shard_filename = os.path.join(
        output_dir,
        f"pubchem_large-{split}.array_record-{shard_index:05d}-of-{total_shards:05d}",
    )  # args[4] = num_total_chunks
    writer = array_record.ArrayRecordWriter(shard_filename, "group_size:1")

    written_count = 0
    for smi in shard:
        result = process_smiles(smi, fp_radius=2, fp_bits=fp_bits)
        if result is not None:
            serialised = features.serialize_example(
                {
                    "smiles": smi,
                    "fingerprint": result,
                }
            )
            writer.write(serialised)
            written_count += 1

    writer.close()
    print(
        f"Shard {shard_index} processed: {written_count} entries written to {shard_filename}"
    )
    return written_count


def process_and_write_shard_tfrecord(
    shard_index: int,
    shard,
    formula_shard = None,
    *,
    total_shards: int,
    split: str,
    output_dir: str,
    features,
    fp_bits: int,
    tokenizer,
    max_length: int,
    include_formula: bool = False
):
    """Process a shard of SMILES and write results to TFRecord.

    Args:
        shard_index: Index of the current shard
        shard: SMILES data for this shard
        formula_shard: Formula data for this shard (if include_formula=True)
        total_shards: Total number of shards
        split: Dataset split name ('train', 'validation', etc.)
        output_dir: Directory to write TFRecord files
        features: TensorFlow features specification
        fp_bits: Number of bits for fingerprint
        tokenizer: Tokenizer for SMILES encoding
        max_length: Maximum sequence length
        include_formula: Whether to include molecular formulas
    """

    import tensorflow as tf
    from .rdkit_utils import process_smiles
    import os

    shard_filename = os.path.join(
        output_dir,
        f"pubchem_large-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}",
    )  # args[4] = num_total_chunks
    shard_txt_filename = os.path.join(
        output_dir, f"pubchem_large-{split}-{shard_index:05d}-of-{total_shards:05d}.txt"
    )

    with open(shard_txt_filename, "w") as txt_file:
        with tf.io.TFRecordWriter(shard_filename) as writer:
            written_count = 0
            for i, smi in enumerate(shard):
                result = process_smiles(smi, fp_radius=2, fp_bits=fp_bits)
                if result is not None:
                    _fp, _smiles = result

                    # Prepare input for tokenizer - either just SMILES or [formula, SMILES]
                    text = None
                    text_pair = None
                    if include_formula and formula_shard is not None:
                        # Use formula and SMILES as text pair for tokenizer
                        formula = formula_shard[i]

                        # Handle missing or invalid formulas
                        if formula is None:
                            # Skip this entry if formula is missing
                            continue

                        # Convert to string and validate
                        try:
                            formula_str = str(formula).strip()
                            smi_str = str(_smiles).strip()

                            # Skip if either is empty after conversion
                            if (
                                not formula_str
                                or not smi_str
                                or formula_str.lower() in ["nan", "none", "null"]
                            ):
                                continue

                            text = formula_str
                            text_pair = smi_str
                        except (ValueError, TypeError):
                            # Skip entries with conversion errors
                            continue
                    else:
                        # Backward compatibility: just use SMILES
                        try:
                            text = str(_smiles).strip()
                            text_pair = None
                            # Validate that text is a non-empty string
                            if not text or text.lower() in ["nan", "none", "null"]:
                                continue
                        except (ValueError, TypeError):
                            # Skip entries with conversion errors
                            continue

                    if tokenizer is not None:
                        try:
                            smiles = (
                                tokenizer.encode(
                                    text=text,
                                    text_pair=text_pair,
                                    add_special_tokens=True,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=max_length,
                                    return_tensors="np",
                                )
                                .reshape(-1)
                                .astype(np.int32)
                            )
                        except Exception as e:
                            # Skip entries that cause tokenization errors
                            print(
                                f"Tokenization error for shard {shard_index}, entry {i}: {e}"
                            )
                            print(f"  Text: {repr(text)}")
                            print(f"  Text_pair: {repr(text_pair)}")
                            continue
                    else:
                        smiles = f"{text}[SEP]{text_pair}"

                    serialised = features.serialize_example(
                        {
                            "smiles": smiles,
                            "fingerprint": _fp,
                        }
                    )
                    writer.write(serialised)
                    txt_file.write(f"{smiles}\n")

                    written_count += 1

    print(
        f"Shard {shard_index} processed: {written_count} entries written to {shard_filename}"
    )
    return written_count


def process_and_write_msg_tfrecord(
    shard_index: int,
    total_shards: int,
    data_tuples: List[Tuple[str, Any]],
    split: str,
    output_dir: str,
    features: Any,
    tokenizer: Any,
    max_length: int,
    fp_bits: int = 2048,
) -> int:
    """Process MSG finetune data (INCHI + fingerprints) and write to TFRecord.

    Args:
        shard_index: Index of the current shard being processed
        total_shards: Total number of shards
        data_tuples: List of (inchi, fingerprint) tuples to process
        split: Dataset split name ('train', 'validation', etc.)
        output_dir: Directory to write the TFRecord files
        features: TensorFlow features specification for serialization
        tokenizer: Tokenizer for converting SMILES to tokens
        max_length: Maximum sequence length for tokenized SMILES
        fp_bits: Size of the Morgan fingerprint (default: 2048)

    Returns:
        Number of successfully written entries
    """
    import tensorflow as tf
    from rdkit.Chem import MolFromInchi, MolToSmiles, rdFingerprintGenerator
    import os

    shard_filename = os.path.join(
        output_dir,
        f"msg_finetune-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}",
    )

    # Create Morgan fingerprint generator with configurable fp_bits (modern approach)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_bits)

    written_count = 0
    skipped_count = 0
    error_count = 0

    with tf.io.TFRecordWriter(shard_filename) as writer:
        for inchi, fingerprint in data_tuples:
            try:
                # Convert INCHI to RDKit molecule object
                mol = MolFromInchi(inchi)
                if mol is None:
                    skipped_count += 1
                    continue

                # Generate canonical SMILES (no stereochemistry, standardized)
                canonical_smiles = MolToSmiles(
                    mol, isomericSmiles=False, canonical=True
                )
                if not canonical_smiles:
                    skipped_count += 1
                    continue

                # Compute true fingerprint directly from molecule using modern generator (no filtering)
                try:
                    true_fingerprint = mfpgen.GetFingerprintAsNumPy(mol).astype(np.int8)
                except Exception:
                    skipped_count += 1
                    continue

                # Tokenize canonical SMILES
                smiles_tokens = (
                    tokenizer.encode(
                        canonical_smiles,
                        add_special_tokens=True,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="np",
                    )
                    .reshape(-1)
                    .astype(np.int32)
                )

                # Ensure fingerprints are in correct format and shape
                fingerprint_array = np.array(fingerprint, dtype=np.float32)
                true_fingerprint_array = np.array(true_fingerprint, dtype=np.float32)

                # Serialize and write to TFRecord
                serialised = features.serialize_example(
                    {
                        "smiles": smiles_tokens,
                        "fingerprint": fingerprint_array,
                        "true_fingerprint": true_fingerprint_array,
                    }
                )
                writer.write(serialised)
                written_count += 1

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only print first few errors to avoid spam
                    print(f"Error processing INCHI {inchi[:50]}...: {str(e)[:100]}")
                continue

    print(
        f"Shard {shard_index} processed: {written_count} entries written, {skipped_count} skipped, {error_count} errors"
    )
    print(f"Output file: {shard_filename}")
    return written_count
