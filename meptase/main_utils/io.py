import os
import shutil
import logging

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import ase

from ..exceptions import DeserializationException

logger = logging.getLogger(__name__)


@dataclass
class IOControl:

    output_dir: str

    mol_from_smiles: str | None = None
    mol_from_mol_file: str | None = None
    mol_file_atom_map_num_indices: list[int] | None = None
    mmff_optimize: bool = False
    overwrite_output_dir: bool = False
    etkdvg3_seed: int = 0

    def __post_init__(self):

        # Cannot generate molecule from SMILES and from a file at the same time
        if self.mol_from_smiles is not None and self.mol_from_mol_file is not None:

            error = DeserializationException(
                "Parameters \"mol_from_smiles\" and \"mol_from_mol_file\" cannot be "
                "set at the same time!"
            )
            logger.error(str(error))
            raise error

        # At least one molecule generation method must be set
        elif self.mol_from_smiles is None and self.mol_from_mol_file is None:

            error = DeserializationException(
                "Either the \"mol_from_smiles\" or the \"mol_from_mol_file\" "
                "parameter must be set!"
            )
            logger.error(str(error))
            raise error

        # If the molecule is read from a file, valid atom map numbers must be provided
        if self.mol_from_mol_file is not None and self.mol_file_atom_map_num_indices is None:

            error = DeserializationException(
                "If the molecule is loaded from a mol file, the atom map numbers must be set "
                "by a list called \"mol_file_atom_map_num_indices\"!"
            )
            logger.error(str(error))
            raise error

        elif self.mol_from_mol_file is not None:
            if len(self.mol_file_atom_map_num_indices) != len(set(self.mol_file_atom_map_num_indices)):

                error = DeserializationException(
                    "Cannot set multiple atom map numbers to the same atom!"
                )
                logger.error(str(error))
                raise error


def create_molecule(
    io_control: IOControl
) -> tuple[Chem.Mol, ase.Atoms, dict[int, int]]:

    output_dir_path = Path(io_control.output_dir)
    map_num_to_idx = dict()

    if io_control.mol_from_smiles is not None:

        rdkit_mol = Chem.MolFromSmiles(io_control.mol_from_smiles)

        if rdkit_mol is None:

            error = DeserializationException(
                f"Cannot create molecule from SMILES \"{io_control.mol_from_smiles}\"! "
                f"Maybe it is not a valid SMILES code?"
            )
            logger.error(str(error))
            raise error

        logger.info("RDKit molecule created from SMILES.")

        rdkit_mol = Chem.AddHs(rdkit_mol)
        logger.info("Hydrogen atoms added.")

        # 2D embedding
        AllChem.Compute2DCoords(rdkit_mol)
        Draw.MolToFile(rdkit_mol, str(output_dir_path / "structure_2d.png"))
        logger.info("2D embedding created.")

        # 3D embedding
        etkdg = AllChem.ETKDGv3()
        etkdg.randomSeed = io_control.etkdvg3_seed
        AllChem.EmbedMolecule(rdkit_mol, etkdg)
        Chem.MolToMolFile(rdkit_mol, str(output_dir_path / "ETKDGv3_embedded.mol"))
        logger.info("3D ETKDGv3 embedding created.")

    elif io_control.mol_from_mol_file is not None:

        rdkit_mol = Chem.MolFromMolFile(io_control.mol_from_mol_file)

        if rdkit_mol is None:

            error = DeserializationException(
                f"Error in parsing the the molecule from the mol file!"
            )
            logger.error(str(error))
            raise error

        logger.info("RDKit molecule created from mol file.")

        # Set atom map numbers, because they are not set in the mol file
        for atom_map_num, atom_idx in enumerate(io_control.mol_file_atom_map_num_indices):
            rdkit_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(atom_map_num + 1)

        logger.info("Atom map numbers set.")

    else:
        raise Exception("Unreachable!")

    # AtomMapNum to index mapping
    for atom_idx, rdkit_atom in enumerate(rdkit_mol.GetAtoms()):

        atom_map_num = rdkit_atom.GetAtomMapNum()
        if atom_map_num != 0:
            map_num_to_idx[atom_map_num] = atom_idx
            logger.info(f"Mapped atom map number %d to index %d.", atom_map_num, atom_idx)

    # MMFF optimization, if necessary
    if io_control.mmff_optimize:
        AllChem.MMFFOptimizeMolecule(rdkit_mol)
        Chem.MolToMolFile(rdkit_mol, str(output_dir_path / "MMFF_optimized.mol"))
        logger.info("MMFF optimization performed.")

    # Coordinate extraction for ASE system creation
    rdkit_atoms = rdkit_mol.GetAtoms()
    rdkit_positions = rdkit_mol.GetConformer().GetPositions()
    ase_atom_list = [
        ase.Atom(
            symbol=rdkit_atom.GetSymbol(),
            position=rdkit_posi,
            mass=rdkit_atom.GetMass(),
            charge=rdkit_atom.GetFormalCharge()
        ) for rdkit_atom, rdkit_posi in zip(rdkit_atoms, rdkit_positions)
    ]
    ase_mol = ase.Atoms(ase_atom_list)
    logger.info("ASE molecule successfully created.")

    return rdkit_mol, ase_mol, map_num_to_idx


def io_main(io_control: IOControl):

    # Create the project directory, if needed
    if os.path.exists(io_control.output_dir) and not io_control.overwrite_output_dir:

        error = IOError("Target path already exists and overwriting it is not allowed!")
        logger.error(str(error))
        raise error

    elif os.path.exists(io_control.output_dir):

        if os.path.isfile(io_control.output_dir):

            error = IOError("Target path points to a file, not a directory!")
            logger.error(str(error))
            raise error

        shutil.rmtree(io_control.output_dir)
        logger.info(
            "Project directory already existed, but overwriting was allowed. "
            "Original project deleted."
        )

    os.mkdir(io_control.output_dir)
    logger.info(f"Project directory created at \"%s\".", io_control.output_dir)

    # Create molecule
    rdkit_mol, ase_mol, selected_atom_id_to_idx = create_molecule(io_control)

    return rdkit_mol, ase_mol, selected_atom_id_to_idx