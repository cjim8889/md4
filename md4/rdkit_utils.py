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
FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P', 'Br', 'I', 'B', 'Si', 'Se', 'Fe', 'Mg', 'Zn', 'Ca', 'Na', 'K', 'Al'}

# Atom type mapping for molecular data
ATOM_TYPES = {
    'C': 0,
    'N': 1, 
    'S': 2,
    'O': 3,
    'F': 4,
    'Cl': 5,
    'H': 6,
    'P': 7,
    'Br': 8,
    'I': 9,
    'B': 10,
    'Si': 11,
    'Se': 12,
    'Fe': 13,
    'Mg': 14,
    'Zn': 15,
    'Ca': 16,
    'Na': 17,
    'K': 18,
    'Al': 19,
    'PAD': 20
}

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
                "fingerprint": np.asarray(fingerprint, dtype=np.bool_),
                "atom_types": np.asarray(atom_types_padded, dtype=np.int8),
                "safe": safe_mol,
                "smiles": smi
            }
    except:
        pass
    return None