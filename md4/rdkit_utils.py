"""RDKit utilities for molecular data processing."""

import functools
import random

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

# Set random seed and suppress RDKit warnings
random.seed(42)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Filter atoms allowed in molecules
FILTER_ATOMS = {
    "C",
    "N",
    "S",
    "O",
    "F",
    "Cl",
    "H",
    "P",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
    "Fe",
    "Mg",
    "Zn",
    "Ca",
    "Na",
    "K",
    "Al",
}

# Atom type mapping for molecular data
ATOM_TYPES = {
    "C": 0,
    "N": 1,
    "S": 2,
    "O": 3,
    "F": 4,
    "Cl": 5,
    "H": 6,
    "P": 7,
    "Br": 8,
    "I": 9,
    "B": 10,
    "Si": 11,
    "Se": 12,
    "Fe": 13,
    "Mg": 14,
    "Zn": 15,
    "Ca": 16,
    "Na": 17,
    "K": 18,
    "Al": 19,
    "PAD": 20,
    "UNK": 21,
}

# Inverse mapping from indices to atom symbols
INVERSE_ATOM_TYPES = {v: k for k, v in ATOM_TYPES.items()}


def atom_types_to_symbols(atom_types):
    """Convert atom type indices to symbols.

    Args:
        atom_types: numpy array of atom type indices

    Returns:
        List of atom symbols
    """
    symbols = []
    for idx in atom_types:
        if idx in INVERSE_ATOM_TYPES:
            symbol = INVERSE_ATOM_TYPES[idx]
            if symbol != "PAD":  # Skip padding tokens
                symbols.append(symbol)
        else:
            symbols.append(f"UNK({idx})")  # Unknown atom type
    return symbols


def format_atom_types_summary(atom_types):
    """Create a summary string of atom types for display.

    Args:
        atom_types: numpy array of atom type indices

    Returns:
        String summary of atom counts
    """
    symbols = atom_types_to_symbols(atom_types)

    # Count occurrences of each atom type
    from collections import Counter

    atom_counts = Counter(symbols)

    # Format as "C:6, N:2, O:1" etc.
    count_strs = [f"{atom}:{count}" for atom, count in sorted(atom_counts.items())]
    return ", ".join(count_strs)


def filter_molecule(mol):
    """Basic molecule filtering without atom type restrictions."""
    if Descriptors.MolWt(mol) >= 3000:  # Molecular weight filter
        return False

    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:  # No charged atoms
            return False
        # if atom.GetSymbol() not in FILTER_ATOMS:  # Only allowed atom types
        #     return False

    return True


def get_atom_type_indices(mol, types=None):
    """Get atom type indices for a SMILES string."""
    if types is None:
        types = ATOM_TYPES

    if mol is None:
        return None

    type_idx = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in types:
            type_idx.append(types["UNK"])
        else:
            type_idx.append(types[symbol])

    return np.array(type_idx, dtype=np.int8)


@functools.lru_cache(maxsize=10)
def get_generator(fp_radius=2, fp_bits=2048):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    return mfpgen


def process_smiles(smi, fp_radius=2, fp_bits=2048):
    """Process SMILES string to InChI with filtering."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if filter_molecule(mol):
            fingerprint = get_generator(fp_radius, fp_bits).GetFingerprintAsNumPy(mol)

            return np.asarray(fingerprint, dtype=np.int8)
    except Exception:
        pass
    return None


def process_smiles_with_shared_memory(smi, i, fp_radius=2, fp_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smi)
        if filter_molecule(mol):
            fingerprint = get_generator(fp_radius, fp_bits).GetFingerprintAsNumPy(mol)
            return i, fingerprint
    except Exception:
        pass

    return i, None


def calculate_smiles_validity(texts_array):
    """Calculate validity metrics for a batch of SMILES strings.

    Args:
        texts_array: numpy array of strings with shape (batch_size,) containing SMILES strings

    Returns:
        dict: Dictionary containing validity metrics:
            - validity_rate: float, fraction of valid SMILES
            - valid_count: int, number of valid SMILES
            - total_count: int, total number of SMILES
            - filtered_count: int, number of SMILES that pass filtering
            - filtered_rate: float, fraction of SMILES that pass filtering
    """
    if len(texts_array.shape) != 1:
        raise ValueError(f"Expected 1D array, got shape {texts_array.shape}")

    total_count = len(texts_array)
    valid_count = 0
    filtered_count = 0

    for smiles_str in texts_array:
        # Convert numpy string to Python string if needed
        if isinstance(smiles_str, np.bytes_):
            smiles_str = smiles_str.decode("utf-8")
        elif isinstance(smiles_str, np.str_):
            smiles_str = str(smiles_str)

        # Strip whitespace and handle empty strings
        smiles_str = smiles_str.strip()
        if not smiles_str:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is not None:
                valid_count += 1
                # Also check if it passes our molecule filtering
                if filter_molecule(mol):
                    filtered_count += 1
        except Exception:
            # Any exception during parsing means invalid SMILES
            pass

    validity_rate = valid_count / total_count if total_count > 0 else 0.0
    filtered_rate = filtered_count / total_count if total_count > 0 else 0.0

    return {
        "validity_rate": validity_rate,
        "valid_count": valid_count,
        "total_count": total_count,
        "filtered_count": filtered_count,
        "filtered_rate": filtered_rate,
    }
