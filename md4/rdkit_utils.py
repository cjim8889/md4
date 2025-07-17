"""RDKit utilities for molecular data processing."""

import random
import multiprocessing

import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

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
    'P': 7,
    'PAD': 8
}


def read_from_sdf(path):
    """Read SMILES from SDF file format."""
    res = []
    app = False
    with open(path, 'r') as f:
        for line in tqdm(f.readlines(), desc='Loading SDF structures', leave=False):
            if app:
                res.append(line.strip())
                app = False
            if line.startswith('> <SMILES>'):
                app = True
    return res


def filter_molecule(mol):
    """Basic molecule filtering without atom type restrictions."""
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:  # No multi-component molecules
            return False
        
        if Descriptors.MolWt(mol) >= 1500:  # Molecular weight filter
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:  # No charged atoms
                return False
    except:
        return False
    
    return True


def filter_with_atom_types(mol):
    """Filter molecules with atom type restrictions."""
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:  # No multi-component molecules
            return False
        
        if Descriptors.MolWt(mol) >= 1500:  # Molecular weight filter
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:  # No charged atoms
                return False
            if atom.GetSymbol() not in FILTER_ATOMS:  # Only allowed atom types
                return False
    except:
        return False
    
    return True


def process_smiles(smi):
    """Process SMILES string to InChI with filtering."""
    try:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)
        if filter_with_atom_types(mol):
            return Chem.MolToInchi(mol)
    except:
        pass
    return None


def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Generate Morgan fingerprint for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.asarray(fp, dtype=np.int8)
    except:
        return None


def get_atom_type_indices(smiles, types=None):
    """Get atom type indices for a SMILES string."""
    if types is None:
        types = ATOM_TYPES
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        type_idx = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in types:
                return None  # Skip if unknown atom is encountered
            type_idx.append(types[symbol])
        
        return np.array(type_idx, dtype=np.int32)
    except:
        return None


def inchi_to_smiles(inchi):
    """Convert InChI to canonical SMILES."""
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except:
        return None


def get_molecule_features(smiles, radius=2, n_bits=2048, types=None, pad_to_length=None):
    """Get both fingerprint and atom type indices for a SMILES string."""
    if types is None:
        types = ATOM_TYPES
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get fingerprint
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprint = np.asarray(fp, dtype=np.int8)
        
        # Get atom type indices
        type_idx = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in types:
                return None  # Skip if unknown atom is encountered
            type_idx.append(types[symbol])
        
        atom_types = np.array(type_idx, dtype=np.int32)

        if pad_to_length is not None:
            atom_types = np.pad(atom_types, (0, pad_to_length - atom_types.shape[0]), 'constant', constant_values=types['PAD'])
        
        return {
            "fingerprint": fingerprint,
            "atom_types": atom_types
        }
    except:
        return None


def process_pubchem_data(smiles_list, num_processes=None):
    """Process a list of SMILES using multiprocessing."""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_smiles, smiles_list), 
            total=len(smiles_list), 
            desc="Cleaning PubChem structures", 
            leave=False
        ))
    
    return [r for r in results if r is not None] 