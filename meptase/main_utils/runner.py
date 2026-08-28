import pickle
import logging

from typing import ClassVar
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

import ase
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.io import read as ase_read
from ase.io import write as ase_write

import torch
import numpy as np

from .io import IOControl
from .registries.calculators import Calculator
from .registries.kernels import DeserializableKernel

from ..additional_potentials import MergedPEF
from ..collective_variables import MergeCV
from ..exceptions import DeserializationException
from ..metadynamics import MetaDynamicsEngine, MetaDynamicsCalculator

logger = logging.getLogger(__name__)


@dataclass
class RunControl:

    # Fields annotated as ClassVar will be automatically recognized by the dataclasses
    # decorator and will be created as a class-level variable.
    _positive_fields: ClassVar[tuple[str, ...]] = (
        "temperature", "timestep", "friction", "kernel_height",
        "steps_between_hills", "n_hills", "trajectory_write_interval",
    )

    _optional_positive_fields: ClassVar[tuple[str, ...]] = (
        "well_tempered_temperature",
    )

    temperature: float
    timestep: float
    friction: float
    kernel_height: float

    steps_between_hills: int
    n_hills: int
    trajectory_write_interval: int

    well_tempered_temperature: float | None = None

    def __post_init__(self):

        for field_name in self._positive_fields:
            field_value = getattr(self, field_name)
            if field_value <= 0:

                error = DeserializationException(
                    f"The field \"{field_name}\" in a RunControl object "
                    f"must be positive! Instead, it was set to be {field_value}."
                )
                logger.error(str(error))
                raise error

        for field_name in self._optional_positive_fields:
            field_value = getattr(self, field_name)
            if field_value is not None and field_value <= 0:

                error = DeserializationException(
                    f"The field \"{field_name}\" in a RunControl object "
                    f"must be either positive or not set! "
                    f"Instead, it was set to be {field_value}."
                )
                logger.error(str(error))
                raise error


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


def runner_main(
    ase_mol: ase.Atoms,
    rdkit_mol: Chem.Mol,
    merged_cvs: MergeCV,
    merged_pef: MergedPEF,
    all_kernels: list[DeserializableKernel],
    unbiased_calculator: Calculator,
    kernel_target_cv_indices: list[int],
    run_control: RunControl,
    io_control: IOControl
):
    # Create the metadynamics run objects
    engine = MetaDynamicsEngine(
        mapper=merged_cvs,
        additional_potential=merged_pef,
        kernels=all_kernels,
        kernel_indices=torch.tensor(kernel_target_cv_indices, dtype=torch.int),
        kernel_height=run_control.kernel_height,
        well_tempered_temperature=run_control.well_tempered_temperature
    )
    ase_mol.calc = MetaDynamicsCalculator(
        unbiased_calculator=unbiased_calculator,
        engine=engine
    )
    logger.info("Successful metadynamics engine and calculator setup.")

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
    logger.info(
        "Trajectory will be written to \"%s\". "
        "Langevin dynamics at %.3f K will be performed.",
        str(trajectory_path), run_control.temperature
    )

    # Run the metadynamics simulation
    for _ in range(run_control.n_hills):
        ase_dynamics.run(run_control.steps_between_hills)
        ase_mol.calc.deposit_hill()

        n_calc_calls = ase_mol.calc.performance_statistics["n_observations"]
        avg_t_unbiased = ase_mol.calc.performance_statistics["total_unbiased_runtime"] / n_calc_calls
        avg_t_biasing = ase_mol.calc.performance_statistics["total_biasing_runtime"] / n_calc_calls

        logger.info(
            "Profiler: avg. t unbiased = %.5f s, avg. t biasing = %.5f s",
            avg_t_unbiased, avg_t_biasing
        )

        fes_domain, fes = ase_mol.calc.get_fes()

        with open(Path(io_control.output_dir) / "hills.pickle", "wb") as f:
            pickle.dump((fes_domain, fes), f)

    ase_trajectory.close()
    logger.info("End of simulation, trajectory closed.")

    # Export the trajectory to XYZ
    traj_buffer = ase_read(str(trajectory_path), index=":")
    ase_write(str(Path(io_control.output_dir) / "output.xyz"), traj_buffer)
    logger.info("Trajectory successfully converted to xyz.")
