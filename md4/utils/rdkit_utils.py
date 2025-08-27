"""RDKit utilities for molecular data processing."""

import functools
import random

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .safe_utils import SAFEConverter

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
    if Descriptors.MolWt(mol) >= 1500:  # Molecular weight filter
        return False

    # for atom in mol.GetAtoms():
    # if atom.GetFormalCharge() != 0:  # No charged atoms
    # return False
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


def process_smiles(
    smi, fp_radius=2, fp_bits=2048, canonical=True, randomize=True, isomeric=False, num_variants=1, safe_encoder: SAFEConverter | None = None
):
    """Process SMILES string with filtering and configurable output format.

    Args:
        smi: Input SMILES string
        fp_radius: Morgan fingerprint radius (default: 2)
        fp_bits: Number of bits for fingerprint (default: 2048)
        canonical: Whether to canonicalize SMILES (default: True)
        randomize: Whether to randomize SMILES output (default: True, ignored if canonical=False)
        isomeric: Whether to include stereochemistry in output SMILES (default: False)
        num_variants: Number of SMILES variants to generate (default: 1)

    Returns:
        Tuple of (fingerprint, smiles_list) or None if processing fails
        - fingerprint: numpy array of Morgan fingerprint
        - smiles_list: list of SMILES strings
          If canonical=True: first element is canonical, rest are randomized
          If canonical=False: all elements are original SMILES (repeated)
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if filter_molecule(mol):
            fingerprint = get_generator(fp_radius, fp_bits).GetFingerprintAsNumPy(mol)
            
            smiles_list = []
            
            if canonical:
                # First variant: canonical (non-randomized)
                if safe_encoder is not None:
                    # Use SAFE encoding for canonical SMILES
                    canonical_smiles = safe_encoder.encoder(mol, canonical=True, randomize=False)
                else:
                    canonical_smiles = Chem.MolToSmiles(
                        mol, isomericSmiles=isomeric, doRandom=False, canonical=True
                    )
                smiles_list.append(canonical_smiles)
            else:
                # No canonicalization, use original SMILES
                smiles_list.append(safe_encoder.encoder(mol, canonical=False, randomize=False))

            # Additional variants: randomized if requested
            if randomize and num_variants > 1:
                for _ in range(num_variants - 1):
                    if safe_encoder is not None:
                        # Use SAFE encoding for randomized SMILES
                        randomized_smiles = safe_encoder.encoder(mol, canonical=False, randomize=True)
                    else:
                        randomized_smiles = Chem.MolToSmiles(
                            mol, isomericSmiles=isomeric, doRandom=True, canonical=False
                        )
                    smiles_list.append(randomized_smiles)

            return np.asarray(fingerprint, dtype=np.int8), smiles_list
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


def get_molecular_formula(mol):
    """Get molecular formula from RDKit molecule object.

    Args:
        mol: RDKit molecule object

    Returns:
        str: Molecular formula (e.g., 'C6H10N2')
    """
    if mol is None:
        return None

    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def calculate_formula_smiles_validity(texts_list):
    """Calculate validity and matching metrics for formula-SMILES pairs.

    Args:
        texts_list: list of strings containing sequences in one of these formats:
                   - [CLS] formula [SEP] smiles [SEP] [PAD]...
                   - formula[SEP]smiles

    Returns:
        dict: Dictionary containing validity and matching metrics:
            - validity_rate: float, fraction of valid SMILES
            - valid_count: int, number of valid SMILES
            - total_count: int, total number of samples
            - filtered_count: int, number of SMILES that pass filtering
            - filtered_rate: float, fraction of SMILES that pass filtering
            - formula_match_count: int, number of samples where formula matches SMILES
            - formula_match_rate: float, fraction of samples where formula matches SMILES
    """
    if not texts_list:
        return {
            "validity_rate": 0.0,
            "valid_count": 0,
            "total_count": 0,
            "filtered_count": 0,
            "filtered_rate": 0.0,
            "formula_match_count": 0,
            "formula_match_rate": 0.0,
        }

    total_count = len(texts_list)
    valid_count = 0
    filtered_count = 0
    formula_match_count = 0

    for text_str in texts_list:
        # Convert numpy string to Python string if needed
        if isinstance(text_str, np.bytes_):
            text_str = text_str.decode("utf-8")
        elif isinstance(text_str, np.str_):
            text_str = str(text_str)

        # Strip whitespace and handle empty strings
        text_str = text_str.strip()
        if not text_str:
            continue

        formula_str = None
        smiles_str = None

        # Handle different formats
        if "[CLS]" in text_str:
            # Format: [CLS] formula [SEP] smiles [SEP] [PAD]...
            sep_splits = text_str.split("[SEP]")
            if len(sep_splits) < 2:
                continue

            # Extract formula (between [CLS] and first [SEP])
            formula_part = sep_splits[0].replace("[CLS]", "").strip()
            formula_str = formula_part

            # Extract SMILES (between first and second [SEP])
            smiles_str = sep_splits[1].strip()
        else:
            # Format: formula[SEP]smiles
            sep_splits = text_str.split("[SEP]")
            if len(sep_splits) < 2:
                continue

            formula_str = sep_splits[0].strip()
            smiles_str = sep_splits[1].strip()

        # Remove all whitespace from the SMILES string
        smiles_str = smiles_str.replace(" ", "").replace("\n", "").replace("\t", "")

        # Skip empty SMILES or formula strings
        if not smiles_str or not formula_str:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is not None:
                valid_count += 1

                # Check if it passes our molecule filtering
                if filter_molecule(mol):
                    filtered_count += 1

                # Check if formula matches SMILES
                calculated_formula = get_molecular_formula(mol)
                if calculated_formula and calculated_formula == formula_str:
                    formula_match_count += 1

        except Exception:
            # Any exception during parsing means invalid SMILES
            pass

    validity_rate = valid_count / total_count if total_count > 0 else 0.0
    filtered_rate = filtered_count / total_count if total_count > 0 else 0.0
    formula_match_rate = formula_match_count / total_count if total_count > 0 else 0.0

    return {
        "validity_rate": validity_rate,
        "valid_count": valid_count,
        "total_count": total_count,
        "filtered_count": filtered_count,
        "filtered_rate": filtered_rate,
        "formula_match_count": formula_match_count,
        "formula_match_rate": formula_match_rate,
    }


# Backward compatibility alias
calculate_smiles_validity = calculate_formula_smiles_validity
