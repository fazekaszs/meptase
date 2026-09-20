import matplotlib.pyplot as plt

import torch

from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.build import bulk
from ase.io.trajectory import Trajectory
from ase.io import read as ase_read
from ase.io import write as ase_write

from meptase.metadynamics import MetaDynamicsEngine, MetaDynamicsCalculator
from meptase.kernels import GaussianKernel
from meptase.collective_variables import DistanceCV
from meptase.additional_potentials import FlatBottomedHarmonic

from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator


def main():

    ase_mol = bulk("Cu", "fcc", a=3.58, cubic=True)

    device = "cuda"
    orb_ff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
        device=device, precision="float32-high"
    )
    unbiased_calculator = ORBCalculator(
        model=orb_ff, atoms_adapter=atoms_adapter, device=device
    )
    selected_cv = DistanceCV(
        name="d",
        indices=torch.tensor([[0, 1], ], dtype=torch.int)
    )
    additional_potential = FlatBottomedHarmonic(
        indices=torch.tensor([0, ], dtype=torch.int),
        cv_min=2.1,
        cv_max=2.9,
        force_constant=10.  # eV / A^2
    )
    engine = MetaDynamicsEngine(
        mapper=selected_cv,
        additional_potential=additional_potential,
        kernels=[GaussianKernel(0.1), ],
        kernel_indices=torch.tensor([0, ], dtype=torch.int),
        kernel_height=0.05
    )
    ase_mol.calc = MetaDynamicsCalculator(
        unbiased_calculator=unbiased_calculator,
        engine=engine
    )

    MaxwellBoltzmannDistribution(ase_mol, temperature_K=310)
    ase_trajectory = Trajectory("../output.traj", "w", ase_mol)

    ase_dynamics = Langevin(ase_mol, 0.5 * units.fs, temperature_K=310, friction=0.2)
    ase_dynamics.attach(ase_trajectory, interval=100)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)
    fig.subplots_adjust(wspace=0.3)

    energies = list()
    for idx in range(4000):

        ase_dynamics.run(50)
        ase_mol.calc.deposit_hill()
        energies.append(ase_mol.get_total_energy())

        ax[0].cla()
        ax[0].plot(energies)
        ax[0].set_xlabel("frame number")
        ax[0].set_ylabel("energy / eV")

        ax[1].cla()
        ax[1].plot(engine.cv_history.numpy())
        ax[1].set_xlabel("frame number")
        ax[1].set_ylabel("collective variable / Angstrom")

        fes_domain, fes_values = ase_mol.calc.get_fes()

        ax[2].cla()
        ax[2].scatter(fes_domain[:, 0], fes_values)
        ax[2].set_xlabel("collective variable / Angstrom")
        ax[2].set_ylabel("bias potential / eV")

        traj_buffer = ase_read("../output.traj", index=":")
        ase_write("../output.xyz", traj_buffer)

        fig.savefig(f"img_trajectory/frame_{idx}.png")


if __name__ == "__main__":
    main()