import ase
import logging
import torch

from rdkit import Chem

from ..collective_variables import DeserializableCV, MergeCV
from ..additional_potentials import DeserializablePEF

logger = logging.getLogger(__name__)


def rdkit_mol_log(rdkit_mol: Chem.Mol) -> None:

    rdkit_mol_log_out = "-- Atoms --\n"
    rdkit_mol_log_out += "Idx".rjust(6) + " "
    rdkit_mol_log_out += "AtomMapNum".rjust(10) + " "
    rdkit_mol_log_out += "Symbol".rjust(10) + " "
    rdkit_mol_log_out += "Charge".rjust(10) + " "
    rdkit_mol_log_out += "Hybrid".rjust(12) + " "
    rdkit_mol_log_out += "IsAromatic".rjust(10) + " \n"

    for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
        rdkit_mol_log_out += f"{atom_idx:>6d} "
        rdkit_mol_log_out += f"{atom.GetAtomMapNum():>10d}"
        rdkit_mol_log_out += f"{atom.GetSymbol():>10s} "
        rdkit_mol_log_out += f"{atom.GetFormalCharge():>10d} "
        rdkit_mol_log_out += f"{str(atom.GetHybridization()):>12s} "
        rdkit_mol_log_out += f"{str(atom.GetIsAromatic()):>10s} \n"

    rdkit_mol_log_out += "-- Bonds --\n"
    rdkit_mol_log_out += "Idx1".rjust(6) + " "
    rdkit_mol_log_out += "Idx2".rjust(6) + " "
    rdkit_mol_log_out += "Type".rjust(15) + " \n"

    for bond in rdkit_mol.GetBonds():
        rdkit_mol_log_out += f"{bond.GetBeginAtomIdx():>6d} "
        rdkit_mol_log_out += f"{bond.GetEndAtomIdx():>6d} "
        rdkit_mol_log_out += f"{str(bond.GetBondType()):>15s} \n"

    logger.info(rdkit_mol_log_out)


def cv_mappers_log(ase_mol: ase.Atoms, cv_mappers: list[DeserializableCV]) -> None:

    cv_mappers_log_out = "-- Test CV mappers --\n"

    start_coordinates = torch.tensor(
        ase_mol.get_positions(),
        device="cpu", dtype=torch.float32, requires_grad=False
    )
    for mapper in cv_mappers:

        with torch.no_grad():
            current_cvs = mapper(start_coordinates).tolist()

        mapper_type = str(type(mapper))
        cv_mappers_log_out += f"> Mapper type: {mapper_type}, name: {mapper.name}\n"

        for idx_list, cv in zip(mapper.indices.tolist(), current_cvs):
            idx_list_str = ", ".join(map(str, idx_list))
            cv_mappers_log_out += f"Value({idx_list_str}) = {cv}\n"

    logger.info(cv_mappers_log_out)


def additional_potentials_log(
    ase_mol: ase.Atoms,
    additional_potentials: list[DeserializablePEF],
    merged_cv_mapper: MergeCV
) -> None:

    additional_potentials_log_out = "-- Test additional potentials --\n"

    if len(additional_potentials) == 0:
        additional_potentials_log_out += "> No additional potentials were set..."
        logger.info(additional_potentials_log_out)
        return

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
        additional_potentials_log_out += f"> PEF type: {pef_type}\nEnergy = {energy}\n"

    logger.info(additional_potentials_log_out)
