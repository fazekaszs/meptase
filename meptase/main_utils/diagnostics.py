import ase
import torch

from rdkit import Chem

from ..collective_variables import DeserializableCV, MergeCV
from ..additional_potentials import DeserializablePEF


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

