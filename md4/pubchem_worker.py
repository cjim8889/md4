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
    FINGERPRINTS = np.ndarray(fp_shape, dtype=np.bool_, buffer=FP_SHM.buf)

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
    if rdkit_utils.process_smiles_with_shared_memory(smiles, FINGERPRINTS, index):
        return index
    else:
        return -1 