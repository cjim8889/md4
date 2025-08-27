import copy as dcopy
import itertools
import random
import re
from collections import Counter
from typing import Callable, List, Optional, TypeAlias, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, rdchem, rdGeometry, rdmolfiles, rdmolops

Mol: TypeAlias = Chem.rdchem.Mol


def reorder_atoms(
    mol: Mol,
    break_ties: bool = True,
    include_chirality: bool = True,
    include_isotopes: bool = True,
) -> Optional[Mol]:
    """Reorder the atoms in a mol. It ensures a single atom order for the same molecule,
    regardless of its original representation.

    Args:
        mol: a molecule.
        break_ties: Force breaking of ranked ties.
        include_chirality: Use chiral information when computing rank.
        include_isotopes: Use isotope information when computing rank.

    Returns:
        mol: a molecule.
    """
    if mol.GetNumAtoms() == 0:
        return mol

    new_order = rdmolfiles.CanonicalRankAtoms(
        mol,
        breakTies=break_ties,
        includeChirality=include_chirality,
        includeIsotopes=include_isotopes,
    )
    new_order = sorted([(y, x) for x, y in enumerate(new_order)])
    return rdmolops.RenumberAtoms(mol, [y for (x, y) in new_order])


def to_mol(
    mol: Union[str, Mol],
    add_hs: bool = False,
    explicit_only: bool = False,
    ordered: bool = False,
    kekulize: bool = False,
    sanitize: bool = True,
    allow_cxsmiles: bool = True,
    parse_name: bool = True,
    remove_hs: bool = True,
    strict_cxsmiles: bool = True,
) -> Optional[Mol]:
    """Convert an input molecule (smiles representation) into a `Mol`.

    Args:
        mol: A SMILES or a molecule.
        add_hs: Whether hydrogens should be added the molecule after the SMILES has been parsed.
        explicit_only: Whether to only add explicit hydrogen or both
            (implicit and explicit). when `add_hs` is set to True.
        ordered: Whether the atom should be ordered. This option is
            important if you want to ensure that the features returned will always maintain
            a single atom order for the same molecule, regardless of its original SMILES representation.
        kekulize: Whether to perform kekulization of the input molecules.
        sanitize: Whether to apply rdkit sanitization when input is a SMILES.
        allow_cxsmiles: Recognize and parse CXSMILES.
        parse_name: Parse (and set) the molecule name as well.
        remove_hs: Wether to remove the hydrogens from the input SMILES.
        strict_cxsmiles: Throw an exception if the CXSMILES parsing fails.

    Returns:
        mol: the molecule if some conversion have been made. If the conversion fails
        None is returned so make sure that you handle this case on your own.
    """

    if not isinstance(mol, (str, Mol)):
        raise ValueError(f"Input should be a Mol or a string instead of '{type(mol)}'")

    if isinstance(mol, str):
        smiles_params = rdmolfiles.SmilesParserParams()
        smiles_params.sanitize = sanitize
        smiles_params.allowCXSMILES = allow_cxsmiles
        smiles_params.parseName = parse_name
        smiles_params.removeHs = remove_hs
        smiles_params.strictCXSMILES = strict_cxsmiles

        _mol = rdmolfiles.MolFromSmiles(mol, params=smiles_params)

        if not sanitize and _mol is not None:
            _mol.UpdatePropertyCache(False)
    else:
        _mol = mol

    # Add hydrogens
    if _mol is not None and add_hs:
        _mol = rdmolops.AddHs(_mol, explicitOnly=explicit_only, addCoords=True)

    # Reorder atoms
    if _mol is not None and ordered:
        _mol = reorder_atoms(_mol)

    if _mol is not None and kekulize:
        rdmolops.Kekulize(_mol, clearAromaticFlags=False)

    return _mol


def from_smarts(smarts: Optional[str]) -> Optional[Mol]:
    """Convert a SMARTS string to a molecule

    Args:
        smarts: a smarts string
    """

    if smarts is None:
        return None
    return Chem.MolFromSmarts(smarts)


def randomize_atoms(mol: Mol) -> Optional[Mol]:
    """Randomize the position of the atoms in a mol.

    Args:
        mol: a molecule.

    Returns:
        mol: a molecule.
    """
    if mol.GetNumAtoms() == 0:
        return mol

    atom_indices = list(range(mol.GetNumAtoms()))
    random.shuffle(atom_indices)
    return rdmolops.RenumberAtoms(mol, atom_indices)


def atom_indices_to_mol(mol: Mol, copy: bool = True):
    """Add the `molAtomMapNumber` property to each atoms.

    Args:
        mol: a molecule
        copy: Whether to copy the molecule.
    """

    if copy is True:
        mol = dcopy.deepcopy(mol)

    for atom in mol.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    return mol


def to_smiles(
    mol: Mol,
    canonical: bool = True,
    isomeric: bool = True,
    kekulize: bool = False,
    ordered: bool = False,
    explicit_bonds: bool = False,
    explicit_hs: bool = False,
    randomize: bool = False,
    cxsmiles: bool = False,
    allow_to_fail: bool = False,
    with_atom_indices: bool = False,
) -> Optional[str]:
    """Convert a mol to a SMILES.

    Args:
        mol: a molecule.
        canonical: if false no attempt will be made to canonicalize the molecule.
        isomeric: whether to include information about stereochemistry in the SMILES.
        kekulize: whether to return the kekule version of the SMILES.
        ordered: whether to force reordering of the atoms first.
        explicit_bonds: if true, all bond orders will be explicitly indicated in the output SMILES.
        explicit_hs: if true, all H counts will be explicitly indicated in the output SMILES.
        randomize: whether to randomize the generated smiles. Override `canonical`.
        cxsmiles: Whether to return a CXSMILES instead of a SMILES.
        allow_to_fail: Raise an error if the conversion to SMILES fails. Return None otherwise.
        with_atom_indices: Whether to add atom indices to the SMILES.
    """

    if ordered and canonical is False:
        mol = reorder_atoms(mol)

    if randomize:
        mol = randomize_atoms(mol)
        canonical = False

    if with_atom_indices:
        mol = atom_indices_to_mol(mol, copy=True)

    smiles = None
    try:
        if cxsmiles:
            smiles = rdmolfiles.MolToCXSmiles(
                mol,
                isomericSmiles=isomeric,
                canonical=canonical,
                allBondsExplicit=explicit_bonds,
                allHsExplicit=explicit_hs,
                kekuleSmiles=kekulize,
            )

        else:
            smiles = rdmolfiles.MolToSmiles(
                mol,
                isomericSmiles=isomeric,
                canonical=canonical,
                allBondsExplicit=explicit_bonds,
                allHsExplicit=explicit_hs,
                kekuleSmiles=kekulize,
            )

    except Exception as e:
        if allow_to_fail:
            raise e

        return None

    return smiles


def remove_stereochemistry(mol: Mol, copy: bool = True):
    """Removes all stereochemistry info from the molecule.

    Args:
        mol: a molecule
        copy: Whether to copy the molecule.
    """

    if copy is True:
        mol = dcopy.deepcopy(mol)
    rdmolops.RemoveStereochemistry(mol)
    return mol


def add_hs(
    mol: Mol,
    explicit_only: bool = False,
    add_coords: bool = False,
    only_on_atoms: Optional[List[int]] = None,
    add_residue_info: bool = False,
):
    """Adds hydrogens to the molecule.

    Args:
        mol: a molecule.
        explicit_only: whether to only add explicit hydrogens.
        add_coords: whether to add 3D coordinates to the hydrogens.
        only_on_atoms: a list of atoms to add hydrogens only on.
        add_residue_info: whether to add residue information to the hydrogens.
            Useful for PDB files.
    """
    mol = rdmolops.AddHs(
        mol,
        explicitOnly=explicit_only,
        addCoords=add_coords,
        onlyOnAtoms=only_on_atoms,
        addResidueInfo=add_residue_info,
    )

    return mol


class SAFEConverter:
    """Molecule line notation conversion from SMILES to SAFE

    A SAFE representation is a string based representation of a molecule decomposition into fragment components,
    separated by a dot ('.'). Note that each component (fragment) might not be a valid molecule by themselves,
    unless explicitely correct to add missing hydrogens.

    !!! note "Slicing algorithms"

        By default SAFE strings are generated using `BRICS`, however, the following alternative are supported:

        * [Hussain-Rea (`hr`)](https://pubs.acs.org/doi/10.1021/ci900450m)
        * [RECAP (`recap`)](https://pubmed.ncbi.nlm.nih.gov/9611787/)
        * [RDKit's MMPA (`mmpa`)](https://www.rdkit.org/docs/source/rdkit.Chem.rdMMPA.html)
        * Any possible attachment points (`attach`)

        Furthermore, you can also provide your own slicing algorithm, which should return a pair of atoms
        corresponding to the bonds to break.

    """

    SUPPORTED_SLICERS = ["hr", "rotatable", "recap", "mmpa", "attach", "brics"]
    __SLICE_SMARTS = {
        "hr": ["[*]!@-[*]"],  # any non ring single bond
        "recap": [
            "[$([C;!$(C([#7])[#7])](=!@[O]))]!@[$([#7;+0;!D1])]",
            "[$(C=!@O)]!@[$([O;+0])]",
            "[$([N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*]))]-!@[$([*])]",
            "[$(C(=!@O)([#7;+0;D2,D3])!@[#7;+0;D2,D3])]!@[$([#7;+0;D2,D3])]",
            "[$([O;+0](-!@[#6!$(C=O)])-!@[#6!$(C=O)])]-!@[$([#6!$(C=O)])]",
            "C=!@C",
            "[N;+1;D4]!@[#6]",
            "[$([n;+0])]-!@C",
            "[$([O]=[C]-@[N;+0])]-!@[$([C])]",
            "c-!@c",
            "[$([#7;+0;D2,D3])]-!@[$([S](=[O])=[O])]",
        ],
        "mmpa": ["[#6+0;!$(*=,#[!#6])]!@!=!#[*]"],  # classical mmpa slicing smarts
        "attach": [
            "[*]!@[*]"
        ],  # any potential attachment point, including hydrogens when explicit
        "rotatable": ["[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"],
    }

    def __init__(
        self,
        slicer: Optional[Union[str, List[str], Callable]] = "brics",
        require_hs: Optional[bool] = None,
        use_original_opener_for_attach: bool = True,
        ignore_stereo: bool = False,
    ):
        """Constructor for the SAFE converter

        Args:
            slicer: slicer algorithm to use for encoding.
                Can either be one of the supported slicing algorithm (SUPPORTED_SLICERS)
                or a custom callable that returns the bond ids that can be sliced.
            require_hs: whether the slicing algorithm require the molecule to have hydrogen explictly added.
                `attach` slicer requires adding hydrogens.
            use_original_opener_for_attach: whether to use the original branch opener digit when adding back
                mapping number to attachment points, or use simple enumeration.
            ignore_stereo: RDKIT does not support some particular SAFE subset when stereochemistry is defined.

        """
        self.slicer = slicer
        if isinstance(slicer, str) and slicer.lower() in self.SUPPORTED_SLICERS:
            self.slicer = self.__SLICE_SMARTS.get(slicer.lower(), slicer)
        if self.slicer != "brics" and isinstance(self.slicer, str):
            self.slicer = [self.slicer]
        if isinstance(self.slicer, (list, tuple)):
            self.slicer = [from_smarts(x) for x in self.slicer]
            if any(x is None for x in self.slicer):
                raise ValueError(f"Slicer: {slicer} cannot be valid")
        self.require_hs = require_hs or (slicer == "attach")
        self.use_original_opener_for_attach = use_original_opener_for_attach
        self.ignore_stereo = ignore_stereo

    @staticmethod
    def randomize(mol: Mol, rng: Optional[int] = None):
        """Randomize the position of the atoms in a mol.

        Args:
            mol: molecules to randomize
            rng: optional seed to use
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        if mol.GetNumAtoms() == 0:
            return mol
        atom_indices = list(range(mol.GetNumAtoms()))
        atom_indices = rng.permutation(atom_indices).tolist()
        return Chem.RenumberAtoms(mol, atom_indices)

    @classmethod
    def _find_branch_number(cls, inp: str):
        """Find the branch number and ring closure in the SMILES representation using regexp

        Args:
            inp: input smiles
        """
        inp = re.sub(r"\[.*?\]", "", inp)  # noqa
        matching_groups = re.findall(r"((?<=%)\d{2})|((?<!%)\d+)(?![^\[]*\])", inp)
        # first match is for multiple connection as multiple digits
        # second match is for single connections requiring 2 digits
        # SMILES does not support triple digits
        branch_numbers = []
        for m in matching_groups:
            if m[0] == "":
                branch_numbers.extend(int(mm) for mm in m[1])
            elif m[1] == "":
                branch_numbers.append(int(m[0].replace("%", "")))
        return branch_numbers

    def _ensure_valid(self, inp: str):
        """Ensure that the input SAFE string is valid by fixing the missing attachment points

        Args:
            inp: input SAFE string

        """
        missing_tokens = [inp]
        branch_numbers = self._find_branch_number(inp)
        # only use the set that have exactly 1 element
        # any branch number that is not pairwise should receive a dummy atom to complete the attachment point
        branch_numbers = Counter(branch_numbers)
        for i, (bnum, bcount) in enumerate(branch_numbers.items()):
            if bcount % 2 != 0:
                bnum_str = str(bnum) if bnum < 10 else f"%{bnum}"
                _tk = f"[*:{i + 1}]{bnum_str}"
                if self.use_original_opener_for_attach:
                    bnum_digit = bnum_str.strip("%")  # strip out the % sign
                    _tk = f"[*:{bnum_digit}]{bnum_str}"
                missing_tokens.append(_tk)
        return ".".join(missing_tokens)

    def _fragment(self, mol: Mol, allow_empty: bool = False):
        """
        Perform bond cutting in place for the input molecule, given the slicing algorithm

        Args:
            mol: input molecule to split
            allow_empty: whether to allow the slicing algorithm to return empty bonds
        Raises:
            SAFEFragmentationError: if the slicing algorithm return empty bonds
        """

        if self.slicer is None:
            matching_bonds = []

        elif callable(self.slicer):
            matching_bonds = self.slicer(mol)
            matching_bonds = list(matching_bonds)

        elif self.slicer == "brics":
            matching_bonds = BRICS.FindBRICSBonds(mol)
            matching_bonds = [brics_match[0] for brics_match in matching_bonds]

        else:
            matches = set()
            for smarts in self.slicer:
                matches |= {
                    tuple(sorted(match))
                    for match in mol.GetSubstructMatches(smarts, uniquify=True)
                }
            matching_bonds = list(matches)

        if matching_bonds is None or len(matching_bonds) == 0 and not allow_empty:
            raise ValueError(
                "Slicing algorithms did not return any bonds that can be cut !"
            )
        return matching_bonds or []

    def encoder(
        self,
        inp: Union[str, Mol],
        canonical: bool = True,
        randomize: Optional[bool] = False,
        seed: Optional[int] = None,
        constraints: Optional[List[Mol]] = None,
        allow_empty: bool = False,
        rdkit_safe: bool = True,
    ):
        """Convert input smiles to SAFE representation

        Args:
            inp: input smiles
            canonical: whether to return canonical smiles string. Defaults to True
            randomize: whether to randomize the safe string encoding. Will be ignored if canonical is provided
            seed: optional seed to use when allowing randomization of the SAFE encoding.
                Randomization happens at two steps:
                1. at the original smiles representation by randomization the atoms.
                2. at the SAFE conversion by randomizing fragment orders
            constraints: List of molecules or pattern to preserve during the SAFE construction. Any bond slicing would
                happen outside of a substructure matching one of the patterns.
            allow_empty: whether to allow the slicing algorithm to return empty bonds
            rdkit_safe: whether to apply rdkit-safe digit standardization to the output SAFE string.
        """
        rng = None
        if randomize:
            rng = np.random.default_rng(seed)
            if not canonical:
                inp = to_mol(inp, remove_hs=False)
                inp = self.randomize(inp, rng)

        if isinstance(inp, Mol):
            inp = to_smiles(inp, canonical=canonical, randomize=False, ordered=False)
        # EN: we first normalize the attachment if the molecule is a query:
        # inp = dm.reactions.convert_attach_to_isotope(inp, as_smiles=True)

        # TODO(maclandrol): RDKit supports some extended form of ring closure, up to 5 digits
        # https://www.rdkit.org/docs/RDKit_Book.html#ring-closures and I should try to include them
        branch_numbers = self._find_branch_number(inp)

        mol = to_mol(inp, remove_hs=False)
        potential_stereos = Chem.FindPotentialStereo(mol)
        has_stereo_bonds = any(
            x.type == Chem.StereoType.Bond_Double for x in potential_stereos
        )
        if self.ignore_stereo:
            mol = remove_stereochemistry(mol)

        bond_map_id = 1
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(0)
                atom.SetIsotope(bond_map_id)
                bond_map_id += 1

        if self.require_hs:
            mol = add_hs(mol)
        matching_bonds = self._fragment(mol, allow_empty=allow_empty)
        substructed_ignored = []
        if constraints is not None:
            substructed_ignored = list(
                itertools.chain(
                    *[
                        mol.GetSubstructMatches(constraint, uniquify=True)
                        for constraint in constraints
                    ]
                )
            )

        bonds = []
        for i_a, i_b in matching_bonds:
            # if both atoms of the bond are found in a disallowed substructure, we cannot consider them
            # on the other end, a bond between two substructure to preserved independently is perfectly fine
            if any(
                (i_a in ignore_x and i_b in ignore_x)
                for ignore_x in substructed_ignored
            ):
                continue
            obond = mol.GetBondBetweenAtoms(i_a, i_b)
            bonds.append(obond.GetIdx())

        if len(bonds) > 0:
            mol = Chem.FragmentOnBonds(
                mol,
                bonds,
                dummyLabels=[
                    (i + bond_map_id, i + bond_map_id) for i in range(len(bonds))
                ],
            )
        # here we need to be clever and disable rooted atom as the atom with mapping

        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if randomize:
            frags = rng.permutation(frags).tolist()
        elif canonical:
            frags = sorted(
                frags,
                key=lambda x: x.GetNumAtoms(),
                reverse=True,
            )

        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [
                atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0
            ]
            frags_str.append(
                Chem.MolToSmiles(
                    frag,
                    isomericSmiles=True,
                    canonical=True,  # needs to always be true
                    rootedAtAtom=non_map_atom_idxs[0],
                )
            )

        scaffold_str = ".".join(frags_str)
        # EN: fix for https://github.com/datamol-io/safe/issues/37
        # we were using the wrong branch number count which did not take into account
        # possible change in digit utilization after bond slicing
        scf_branch_num = self._find_branch_number(scaffold_str) + branch_numbers

        # don't capture atom mapping in the scaffold
        attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", scaffold_str))
        if canonical:
            attach_pos = sorted(attach_pos)
        starting_num = 1 if len(scf_branch_num) == 0 else max(scf_branch_num) + 1
        for attach in attach_pos:
            val = str(starting_num) if starting_num < 10 else f"%{starting_num}"
            # we cannot have anything of the form "\([@=-#-$/\]*\d+\)"
            attach_regexp = re.compile(r"(" + re.escape(attach) + r")")
            scaffold_str = attach_regexp.sub(val, scaffold_str)
            starting_num += 1

        # now we need to remove all the parenthesis around digit only number
        wrong_attach = re.compile(r"\(([\%\d]*)\)")
        scaffold_str = wrong_attach.sub(r"\g<1>", scaffold_str)
        # furthermore, we autoapply rdkit-compatible digit standardization.
        if rdkit_safe:
            pattern = r"\(([=-@#\/\\]{0,2})(%?\d{1,2})\)"
            replacement = r"\g<1>\g<2>"
            scaffold_str = re.sub(pattern, replacement, scaffold_str)

        return scaffold_str
