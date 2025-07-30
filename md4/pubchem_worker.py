"""Minimal worker module for PubChem SMILES processing.

This module contains only the essential imports and functions needed 
for multiprocessing workers to avoid loading heavy dependencies.
"""

import numpy as np
from multiprocessing import shared_memory

# Global variables for worker processes
FINGERPRINTS = None
FP_SHM = None

def init_worker_pubchem(fp_name, fp_shape):
    """Initialize worker with shared memory for fingerprints."""
    global FINGERPRINTS, FP_SHM
    # Keep references to shared memory objects to prevent garbage collection
    FP_SHM = shared_memory.SharedMemory(fp_name)
    FINGERPRINTS = np.ndarray(fp_shape, dtype=np.int8, buffer=FP_SHM.buf)

def process_individual_smiles_pubchem(task):
    """Process individual SMILES with global shared arrays for fingerprints.
    
    Args:
        task: Tuple of (smiles, index)
        
    Returns:
        index if successful, -1 if failed
    """
    # Import rdkit_utils here to avoid loading it in main process
    from md4 import rdkit_utils
    
    smiles, index = task  # unpack the tuple
    return rdkit_utils.process_smiles_with_shared_memory(smiles, index)

def process_and_write_shard(args):
    """Process a shard of SMILES and write results to shared memory.
    
    Args:
        args: Tuple of (shard_index, total_shards, shard, split, output_dir, features, fp_bits)
    """
    shard_index, total_shards, shard, split, output_dir, features, fp_bits = args[:7]

    from array_record.python import array_record_module as array_record
    from md4.rdkit_utils import process_smiles
    import os

    shard_filename = os.path.join(output_dir, f"pubchem_large-{split}.array_record-{shard_index:05d}-of-{total_shards:05d}") # args[4] = num_total_chunks
    writer = array_record.ArrayRecordWriter(shard_filename, 'group_size:1')
    
    written_count = 0
    for smi in shard:
        result = process_smiles(smi, fp_radius=2, fp_bits=fp_bits)
        if result is not None:
            serialised = features.serialize_example({
                "smiles": smi,
                "fingerprint": result,
            })
            writer.write(serialised)
            written_count += 1

    writer.close()
    print(f"Shard {shard_index} processed: {written_count} entries written to {shard_filename}")
    return written_count

def process_and_write_shard_tfrecord(args):
    """Process a shard of SMILES and write results to shared memory.
    
    Args:
        args: Tuple of (shard_index, total_shards, shard, split, output_dir, features, fp_bits)
    """
    shard_index, total_shards, shard, split, output_dir, features, fp_bits, tokenizer, max_length = args[:9]

    import tensorflow as tf
    from md4.rdkit_utils import process_smiles
    import os

    shard_filename = os.path.join(output_dir, f"pubchem_large-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}") # args[4] = num_total_chunks
    
    with tf.io.TFRecordWriter(shard_filename) as writer:
        written_count = 0
        for smi in shard:
            result = process_smiles(smi, fp_radius=2, fp_bits=fp_bits)
            if result is not None:
                smiles = tokenizer.encode(
                    smi,
                    add_special_tokens=True,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np",
                ).reshape(-1).astype(np.int32)

                serialised = features.serialize_example({
                    "smiles": smiles,
                    "fingerprint": result,
                })
                writer.write(serialised)
                written_count += 1

    print(f"Shard {shard_index} processed: {written_count} entries written to {shard_filename}")
    return written_count

def process_and_write_msg_tfrecord(args):
    """Process MSG finetune data (INCHI + fingerprints) and write to TFRecord.
    
    Args:
        args: Tuple of (shard_index, total_shards, data_tuples, split, output_dir, features, tokenizer, max_length)
    """
    shard_index, total_shards, data_tuples, split, output_dir, features, tokenizer, max_length = args[:8]

    import tensorflow as tf
    from rdkit.Chem import MolFromInchi, MolToSmiles, rdFingerprintGenerator
    import os

    shard_filename = os.path.join(output_dir, f"msg_finetune-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}")
    
    # Create Morgan fingerprint generator (modern approach)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
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
                canonical_smiles = MolToSmiles(mol, isomericSmiles=False, canonical=True)
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
                smiles_tokens = tokenizer.encode(
                    canonical_smiles,
                    add_special_tokens=True,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np",
                ).reshape(-1).astype(np.int32)

                # Ensure fingerprints are in correct format and shape
                fingerprint_array = np.array(fingerprint, dtype=np.float32)
                true_fingerprint_array = np.array(true_fingerprint, dtype=np.float32)
                
                # Serialize and write to TFRecord
                serialised = features.serialize_example({
                    "smiles": smiles_tokens,
                    "fingerprint": fingerprint_array,
                    "true_fingerprint": true_fingerprint_array,
                })
                writer.write(serialised)
                written_count += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only print first few errors to avoid spam
                    print(f"Error processing INCHI {inchi[:50]}...: {str(e)[:100]}")
                continue

    print(f"Shard {shard_index} processed: {written_count} entries written, {skipped_count} skipped, {error_count} errors")
    print(f"Output file: {shard_filename}")
    return written_count