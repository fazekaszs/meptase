import json
import os
import pickle

from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import ase
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.io import read as ase_read
from ase.io import write as ase_write

import torch

from .main_utils.io import IOControl, io_main
from .collective_variables import MergeCV, DeserializableCV
from .additional_potentials import DeserializablePEF, MergedPEF
from .metadynamics import MetaDynamicsEngine, MetaDynamicsCalculator
from .main_utils.deserialize import (
    deserialize_cvs, deserialize_additional_potentials,
    deserialize_kernels, deserialize_calculator,
    deserialize_run_control
)


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


def create_xh_bond_constraint(ase_mol: ase.Atoms, rdkit_mol: Chem.Mol) -> FixBondLengths:

    positions = ase_mol.get_positions()
    element_symbols = ase_mol.get_chemical_symbols()

    idx_pairs = list()
    bond_lengths = list()
    for bond in rdkit_mol.GetBonds():

        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        if "H" not in {element_symbols[atom1_idx], element_symbols[atom2_idx]}:
            continue

        distance = np.sqrt(np.sum((positions[atom1_idx] - positions[atom2_idx]) ** 2))

        idx_pairs.append((atom1_idx, atom2_idx))
        bond_lengths.append(distance)

    return FixBondLengths(idx_pairs, bondlengths=bond_lengths)


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

    # Perform input operations: read in IO config, read in the molecule and its attributes
    io_control = IOControl(**input_file_content["io_control"])
    rdkit_mol, ase_mol, selected_atom_id_to_idx = io_main(io_control)

    # Deserialize and collect the CVs from the JSON file
    cv_mappers = deserialize_cvs(
        input_file_content["collective_variables"],
        selected_atom_id_to_idx
    )

    # Run the CVs as a test on the current coordinates
    test_cv_mappers(ase_mol, cv_mappers)

    # Merge the CV collection to a single CV
    merged_cvs = MergeCV("cv_merger", cv_mappers)

    # Deserialize and collect the PEFs from the JSON file
    all_additional_potentials = deserialize_additional_potentials(
        input_file_content["additional_potentials"],
        merged_cvs
    )

    # Run the PEFs as a test on the current CVs
    test_additional_potentials(ase_mol, all_additional_potentials, merged_cvs)

    # Merge the PEFs to a single PEF
    merged_pef = MergedPEF(all_additional_potentials)

    # Deserialize and collect the kernels from the JSON file
    all_kernels, kernel_target_cv_indices = deserialize_kernels(
        input_file_content["kernels"],
        merged_cvs
    )

    # Deserialize the unbiased calculator from the JSON file
    unbiased_calculator = deserialize_calculator(input_file_content["unbiased_calculator"])

    # Parse the run control parameters from the JSON file
    run_control = deserialize_run_control(input_file_content["run_control"])

    # Create the metadynamics run objects
    engine = MetaDynamicsEngine(
        mapper=merged_cvs,
        additional_potential=merged_pef,
        kernels=all_kernels,
        kernel_indices=torch.tensor(kernel_target_cv_indices, dtype=torch.int),
        kernel_height=run_control.kernel_height
    )
    ase_mol.calc = MetaDynamicsCalculator(
        unbiased_calculator=unbiased_calculator,
        engine=engine
    )

    # Constrain the X-H bonds and initialize the velocities
    ase_mol.set_constraint(create_xh_bond_constraint(ase_mol, rdkit_mol))
    MaxwellBoltzmannDistribution(ase_mol, temperature_K=run_control.temperature)

    # Set up the trajectory writer and the Langevin dynamics
    trajectory_path = Path(io_control.output_dir) / "output.traj"
    ase_trajectory = Trajectory(str(trajectory_path), "w", ase_mol)
    ase_dynamics = Langevin(
        atoms=ase_mol,
        timestep=run_control.timestep * units.fs,
        temperature_K=run_control.temperature,
        friction=run_control.friction
    )
    ase_dynamics.attach(ase_trajectory, interval=run_control.trajectory_write_interval)

    # Run the metadynamics simulation
    for _ in range(run_control.n_hills):
        ase_dynamics.run(run_control.steps_between_hills)
        ase_mol.calc.deposit_hill()

        n_calc_calls = ase_mol.calc.performance_statistics["n_observations"]
        avg_t_unbiased = ase_mol.calc.performance_statistics["total_unbiased_runtime"] / n_calc_calls
        avg_t_biasing = ase_mol.calc.performance_statistics["total_biasing_runtime"] / n_calc_calls

        print(
            f"-- Profiler: avg. t unbiased = {avg_t_unbiased:.5f} s, "
            f"avg. t biasing = {avg_t_biasing:.5f} s"
        )

        fes_domain, fes = ase_mol.calc.get_fes()

        with open(Path(io_control.output_dir) / "hills.pickle", "wb") as f:
            pickle.dump((fes_domain, fes), f)

    ase_trajectory.close()

    # Export the trajectory to XYZ
    traj_buffer = ase_read(str(trajectory_path), index=":")
    ase_write(str(Path(io_control.output_dir) / "output.xyz"), traj_buffer)


if __name__ == "__main__":
    main()
