import json
import os

from pathlib import Path
from argparse import ArgumentParser, Namespace

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import ase

import torch

from .collective_variables import (
    MergeCV, CV_REGISTRY, DeserializableCV
)
from .additional_potentials import (
    PEF_REGISTRY, DeserializablePEF, PotentialEnergyFunction, MergedPEF
)
from .exceptions import InvalidTypeSelectionException


def parse_arguments() -> Namespace:

    parser = ArgumentParser()
    parser.add_argument(
        "-in", "--input_file",
        type=Path, required=True, help="The input json file."
    )
    return parser.parse_args()


def rdkit_mol_log(rdkit_mol: Chem.Mol) -> None:

    print("RDKit Molecule object successfully created!")
    print(
        "[ ATOMS ]\n",
        "Idx".rjust(6),
        "Symbol".rjust(10),
        "Charge".rjust(10),
        "Hybrid".rjust(12),
        "IsArom".rjust(10)
    )

    for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
        print(
            f"{atom_idx:>6d}",
            f"{atom.GetSymbol():>10s}",
            f"{atom.GetFormalCharge():>10d}",
            f"{str(atom.GetHybridization()):>12s}",
            f"{str(atom.GetIsAromatic()):>10s}"
        )

    print(
        "[ BONDS ]\n",
        "Idx1".rjust(6),
        "Idx2".rjust(6),
        "Type".rjust(15)
    )

    for bond in rdkit_mol.GetBonds():
        print(
            f"{bond.GetBeginAtomIdx():>6d}",
            f"{bond.GetEndAtomIdx():>6d}",
            f"{str(bond.GetBondType()):>15s}"
        )


def create_molecule(smiles: str, output_dir: Path) -> tuple[Chem.Mol, ase.Atoms, dict[int, int]]:

    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)

    rdkit_mol_log(rdkit_mol)

    AllChem.Compute2DCoords(rdkit_mol)
    Draw.MolToFile(rdkit_mol, str(output_dir / "structure_2d.png"))

    AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDGv3())
    Chem.MolToMolFile(rdkit_mol, str(output_dir / "ETKDGv3_embedded.mol"))

    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    Chem.MolToMolFile(rdkit_mol, str(output_dir / "MMFF_optimized.mol"))

    positions = rdkit_mol.GetConformer().GetPositions()
    ase_atom_list = list()
    selected_atom_id_to_idx = dict()
    for atom_idx, (rdkit_atom, rdkit_posi) in enumerate(zip(rdkit_mol.GetAtoms(), positions)):

        ase_atom_list.append(ase.Atom(
            symbol=rdkit_atom.GetSymbol(),
            position=rdkit_posi,
            mass=rdkit_atom.GetMass(),
            charge=rdkit_atom.GetFormalCharge()
        ))

        atom_id = rdkit_atom.GetAtomMapNum()
        if atom_id != 0:
            selected_atom_id_to_idx[atom_id] = atom_idx

    ase_mol = ase.Atoms(ase_atom_list)

    print("ASE Atoms object successfully created!")

    return rdkit_mol, ase_mol, selected_atom_id_to_idx


def test_cv_mappers(ase_mol: ase.Atoms, cv_mappers: list[DeserializableCV]) -> None:

    print("[ TEST CVS ]")

    start_coordinates = torch.tensor(
        ase_mol.get_positions(),
        device="cpu", dtype=torch.float32, requires_grad=False
    )
    for mapper in cv_mappers:

        with torch.no_grad():
            current_cvs = mapper(start_coordinates).tolist()

        mapper_type = str(type(mapper))
        print(f"- Mapper type: {mapper_type}, name: {mapper.name}")

        for idx_list, cv in zip(mapper.indices.tolist(), current_cvs):
            idx_list_str = ", ".join(map(str, idx_list))
            print(f"Value({idx_list_str}) = {cv}")


def test_additional_potentials(
    ase_mol: ase.Atoms,
    additional_potentials: list[DeserializablePEF],
    merged_cv_mapper: MergeCV
) -> None:

    print("[ TEST ADDITIONAL POTENTIALS ]")

    start_coordinates = torch.tensor(
        ase_mol.get_positions(),
        device="cpu", dtype=torch.float32, requires_grad=False
    )
    with torch.no_grad():
        merged_cv = merged_cv_mapper(start_coordinates)

    for pef in additional_potentials:
        with torch.no_grad():
            energy = pef(merged_cv).item()
        pef_type = str(type(pef))
        print(f"- PEF type: {pef_type}\nEnergy = {energy}")


def main():

    # Parse the CLI arguments and load the JSON dictionary
    arguments = parse_arguments()
    with open(arguments.input_file, "r") as f:
        input_file_content = json.load(f)

    # Create the project directory
    output_dir = Path(input_file_content["output_dir"])
    if os.path.exists(output_dir):
        raise Exception("Project directory already exists!")
    os.mkdir(output_dir)

    print(f"Project directory created at \"{output_dir}\"")

    # Create the molecule from the given SMILES code
    mol_smiles = input_file_content["molecule_smiles"]
    rdkit_mol, ase_mol, selected_atom_id_to_idx = create_molecule(mol_smiles, output_dir)

    # Deserialize and collect the CVs from the JSON file
    cv_mappers = list()
    for cv_dict in input_file_content["collective_variables"]:

        cv_type = cv_dict["type"]
        if cv_type not in CV_REGISTRY:
            raise InvalidTypeSelectionException(
                f"The collective variable type {cv_type} is not registered!"
            )

        cv_mappers.append(CV_REGISTRY[cv_type].from_config(
            index_mapper=selected_atom_id_to_idx,
            **cv_dict["parameters"]
        ))

    # Run the CVs as a test on the current coordinates
    test_cv_mappers(ase_mol, cv_mappers)

    # Merge the CV collection to a single CV
    merged_cvs = MergeCV("cv_merger", cv_mappers)

    # Deserialize and collect the PEFs from the JSON file
    all_additional_potentials = list()
    for pef_dict in input_file_content["additional_potentials"]:

        pef_type = pef_dict["type"]
        if pef_type not in PEF_REGISTRY:
            raise InvalidTypeSelectionException(
                f"The potential energy function type {pef_type} is not registered!"
            )

        all_additional_potentials.append(PEF_REGISTRY[pef_type].from_config(
            names_to_idx=merged_cvs.names_to_idx,
            cv_names=pef_dict["target_cvs"],
            **pef_dict["parameters"]
        ))

    # Run the PEFs as a test on the current CVs
    test_additional_potentials(ase_mol, all_additional_potentials, merged_cvs)

    # Merge the PEFs to a single PEF
    merged_pef = MergedPEF(all_additional_potentials)

    return


if __name__ == "__main__":
    main()
