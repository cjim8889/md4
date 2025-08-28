"""Minimal worker module for PubChem SMILES processing.

This module contains only the essential imports and functions needed
for multiprocessing workers to avoid loading heavy dependencies.
"""

from typing import Any, List, Tuple

import numpy as np


def process_and_write_shard(args):
    """Process a shard of SMILES and write results to shared memory.

    Args:
        args: Tuple of (shard_index, total_shards, shard, split, output_dir, features, fp_bits)
    """
    shard_index, total_shards, shard, split, output_dir, features, fp_bits = args[:7]

    import os

    from array_record.python import array_record_module as array_record

    from .rdkit_utils import process_smiles

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
    args: tuple,
    *,
    total_shards: int,
    split: str,
    output_dir: str,
    features,
    fp_bits: int,
    fp_radius: int = 2,
    num_variants: int = 1,
    canonical: bool = True,
    randomize: bool = True,
    isomeric: bool = False,
    include_formula: bool = False,
    use_safe: bool = False,
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
        fp_radius: Morgan fingerprint radius
        canonical: Whether to canonicalize SMILES
        randomize: Whether to randomize SMILES output
        isomeric: Whether to include stereochemistry in output SMILES
        include_formula: Whether to include molecular formulas
        use_safe: Whether to use SAFE encoding for SMILES
    """

    shard_index, shard, formula_shard = args

    import os

    import tensorflow as tf

    from .rdkit_utils import process_smiles

    shard_filename = os.path.join(
        output_dir,
        f"pubchem_large-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}",
    )  # args[4] = num_total_chunks
    shard_txt_filename = os.path.join(
        output_dir, f"pubchem_large-{split}-{shard_index:05d}-of-{total_shards:05d}.txt"
    )

    safe_encoder = None
    if use_safe:
        from .safe_utils import SAFEConverter

        safe_encoder = SAFEConverter(
            slicer="brics", require_hs=False, ignore_stereo=True
        )

    with open(shard_txt_filename, "w") as txt_file:
        with tf.io.TFRecordWriter(shard_filename) as writer:
            written_count = 0
            for i, smi in enumerate(shard):
                result = process_smiles(
                    smi,
                    fp_radius=fp_radius,
                    fp_bits=fp_bits,
                    canonical=canonical,
                    randomize=randomize,
                    isomeric=isomeric,
                    num_variants=num_variants,
                    safe_encoder=safe_encoder,
                )
                if result is not None:
                    _fp, _smiles_list = result

                    # Process each SMILES variant
                    _processed_smiles_list = []
                    for _smiles in _smiles_list:
                        # Prepare SMILES string - either just SMILES or [formula, SMILES]
                        if include_formula and formula_shard is not None:
                            # Use formula and SMILES as combined text
                            formula = formula_shard[i]

                            # Handle missing or invalid formulas
                            if formula is None:
                                # Skip this variant if formula is missing
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

                                smiles = f"{formula_str}[SEP]{smi_str}"
                            except (ValueError, TypeError):
                                # Skip entries with conversion errors
                                continue
                        else:
                            # Just use SMILES
                            try:
                                smi_str = str(_smiles).strip()
                                # Validate that text is a non-empty string
                                if not smi_str or smi_str.lower() in [
                                    "nan",
                                    "none",
                                    "null",
                                ]:
                                    continue
                                smiles = smi_str
                            except (ValueError, TypeError):
                                # Skip entries with conversion errors
                                continue

                        _processed_smiles_list.append(smiles)
                        txt_file.write(f"{smiles}\n")


                    if not _processed_smiles_list:
                        # Skip writing if no valid SMILES variants
                        continue

                    serialised = features.serialize_example(
                        {
                            "smiles": _processed_smiles_list,
                            "fingerprint": _fp,
                        }
                    )
                    writer.write(serialised)

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
    fp_radius: int = 2,
    canonical: bool = True,
    randomize: bool = True,
    isomeric: bool = False,
    num_variants: int = 1,
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
    import os

    import tensorflow as tf
    from rdkit.Chem import MolFromInchi, MolToSmiles, rdFingerprintGenerator

    shard_filename = os.path.join(
        output_dir,
        f"msg_finetune-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}",
    )

    # Create Morgan fingerprint generator with configurable fp_bits (modern approach)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_bits, fpSize=fp_bits)

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

                # Generate SMILES with specified options
                canonical_smiles = MolToSmiles(
                    mol,
                    isomericSmiles=isomeric,
                    canonical=canonical,
                    doRandom=randomize,
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
