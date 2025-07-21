"""RDKit utilities for molecular data processing."""
import random

import numpy as np
import safe
import safe.converter
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# Set random seed and suppress RDKit warnings
random.seed(42)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Filter atoms allowed in molecules
FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P'}

# Atom type mapping for molecular data
ATOM_TYPES = {
    'C': 0,
    'N': 1, 
    'S': 2,
    'O': 3,
    'F': 4,
    'Cl': 5,
    'H': 6,
    'P': 7
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
            if symbol != 'PAD':  # Skip padding tokens
                symbols.append(symbol)
        else:
            symbols.append(f'UNK({idx})')  # Unknown atom type
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
    if Descriptors.MolWt(mol) >= 2000:  # Molecular weight filter
        return False
    
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:  # No charged atoms
            return False
        if atom.GetSymbol() not in FILTER_ATOMS:  # Only allowed atom types
            return False
    
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
            return None  # Skip if unknown atom is encountered
        type_idx.append(types[symbol])
    
    return np.array(type_idx, dtype=np.int8)


def process_smiles(smi, fp_radius=2, fp_bits=2048, pad_to_length=160):
    """Process SMILES string to InChI with filtering."""
    try:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)
        safe_mol = safe.converter.encode(mol, ignore_stereo=True)
        if filter_molecule(mol):
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
            atom_types = get_atom_type_indices(mol)
            atom_types_padded = np.pad(atom_types, (0, pad_to_length - atom_types.shape[0]), 'constant', constant_values=ATOM_TYPES['PAD'])

            return {
                "fingerprint": np.asarray(fingerprint, dtype=np.int8),
                "atom_types": np.asarray(atom_types_padded, dtype=np.int8),
                "safe": safe_mol,
                "smiles": smi
            }
    except:
        pass
    return None