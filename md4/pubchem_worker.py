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
    from rdkit.Chem import MolFromInchi, MolToSmiles
    import os

    shard_filename = os.path.join(output_dir, f"msg_finetune-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}")
    
    with tf.io.TFRecordWriter(shard_filename) as writer:
        written_count = 0
        for inchi, fingerprint in data_tuples:
            # Convert INCHI to SMILES
            try:
                mol = MolFromInchi(inchi)
                if mol is not None:
                    smiles_str = MolToSmiles(mol)
                    
                    # Tokenize SMILES
                    smiles = tokenizer.encode(
                        smiles_str,
                        add_special_tokens=True,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="np",
                    ).reshape(-1).astype(np.int32)

                    # Convert fingerprint to float32 and ensure correct shape
                    fingerprint_array = np.array(fingerprint, dtype=np.float32)

                    serialised = features.serialize_example({
                        "smiles": smiles,
                        "fingerprint": fingerprint_array,
                    })
                    writer.write(serialised)
                    written_count += 1
            except Exception as e:
                print(f"Error processing INCHI: {e}")
                continue

    print(f"Shard {shard_index} processed: {written_count} entries written to {shard_filename}")
    return written_count