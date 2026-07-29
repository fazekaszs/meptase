import matplotlib.pyplot as plt

import numpy as np
import torch

import ase
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.io import read as ase_read
from ase.io import write as ase_write

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from tblite.ase import TBLite

from sklearn.decomposition import PCA

from meptase.metadynamics import MetaDynamicsEngine, MetaDynamicsCalculator
from meptase.kernels import GaussianKernel, BetaKernel
from meptase.collective_variables import DistanceCV, AngleCV
from meptase.additional_potentials import FlatBottomedHarmonic


SYMBOL_TO_COLOR = {
    "C": "grey", "O": "red", "N": "blue", "H": "wheat", "Cl": "green"
}


def draw_molecule(ax: plt.Axes, ase_mol: ase.Atoms, rdkit_mol: Chem.Mol) -> None:

    positions = PCA(n_components=3).fit_transform(ase_mol.get_positions())
    element_symbols = ase_mol.get_chemical_symbols()

    for atom_idx, atom in enumerate(element_symbols):

        p = positions[atom_idx]
        ax.text(p[0], p[1], f"{element_symbols[atom_idx]}:{atom_idx}", zorder=p[2])

    for bond in rdkit_mol.GetBonds():

        atom1_idx, atom2_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        p1 = positions[atom1_idx]
        p2 = positions[atom2_idx]
        z_order = (p1[2] + p2[2]) / 2.
        midpoint = (p1[:2] + p2[:2]) / 2.

        ax.plot(
            [p1[0], midpoint[0]], [p1[1], midpoint[1]],
            color=SYMBOL_TO_COLOR[element_symbols[atom1_idx]],
            zorder=z_order
        )
        ax.plot(
            [midpoint[0], p2[0]], [midpoint[1], p2[1]],
            color=SYMBOL_TO_COLOR[element_symbols[atom2_idx]],
            zorder=z_order
        )


def create_molecule() -> tuple[Chem.Mol, ase.Atoms, list[int]]:

    # rdkit_mol = Chem.MolFromSmiles("C1C(=[O:1])C(C(Cl)(Cl)[OH:2])C(=O)C1")
    rdkit_mol = Chem.MolFromSmiles("[CH3:1]CC[CH2:2]CC[CH2:3]C")
    AllChem.Compute2DCoords(rdkit_mol)
    Draw.MolToFile(rdkit_mol, "output.png")

    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    Chem.MolToMolFile(rdkit_mol, "output.mol")

    positions = rdkit_mol.GetConformer().GetPositions()
    ase_atom_list = list()
    selected_atoms = list()
    for atom_idx, (rdkit_atom, rdkit_posi) in enumerate(zip(rdkit_mol.GetAtoms(), positions)):

        ase_atom_list.append(ase.Atom(
            symbol=rdkit_atom.GetSymbol(),
            position=rdkit_posi,
            mass=rdkit_atom.GetMass(),
            charge=rdkit_atom.GetFormalCharge()
        ))

        if rdkit_atom.GetAtomMapNum() != 0:
            selected_atoms.append(atom_idx)

    ase_mol = ase.Atoms(ase_atom_list)

    return rdkit_mol, ase_mol, selected_atoms


def constraint_xh_bonds(ase_mol: ase.Atoms, rdkit_mol: Chem.Mol) -> FixBondLengths:

    positions = ase_mol.get_positions()
    element_symbols = ase_mol.get_chemical_symbols()

    idx_pairs = list()
    bond_lengths = list()
    for bond in rdkit_mol.GetBonds():

        atom1_idx, atom2_idx = bond.GetBeginAtomIdx(), bond.GetBeginAtomIdx()

        if "H" not in {element_symbols[atom1_idx], element_symbols[atom2_idx]}:
            continue

        distance = np.sqrt(np.sum((positions[atom1_idx] - positions[atom2_idx]) ** 2))

        idx_pairs.append((atom1_idx, atom2_idx))
        bond_lengths.append(distance)

    return FixBondLengths(idx_pairs, bondlengths=bond_lengths)


def main():

    rdkit_mol, ase_mol, (atom1_idx, atom2_idx, atom3_idx) = create_molecule()

    unbiased_calculator = TBLite(
        max_iterations=1000,
        accuracy=1.0,
        verbosity=0
    )
    selected_cv = AngleCV(
        name="theta",
        indices=torch.tensor([[atom1_idx, atom2_idx, atom3_idx], ], dtype=torch.int)
    )
    additional_potential = FlatBottomedHarmonic(
        indices=torch.tensor([0, ], dtype=torch.int),
        cv_min=10. * torch.pi / 180.,
        cv_max=170. * torch.pi / 180.,
        force_constant=500.  # eV / rad^2
    )
    engine = MetaDynamicsEngine(
        mapper=selected_cv,
        additional_potential=additional_potential,
        kernels=[BetaKernel(5. * torch.pi / 180.), ],
        kernel_indices=torch.tensor([0, ], dtype=torch.int),
        kernel_height=0.05
    )
    ase_mol.calc = MetaDynamicsCalculator(
        unbiased_calculator=unbiased_calculator,
        engine=engine
    )

    ase_mol.set_constraint(constraint_xh_bonds(ase_mol, rdkit_mol))

    MaxwellBoltzmannDistribution(ase_mol, temperature_K=310)
    ase_trajectory = Trajectory("output.traj", "w", ase_mol)

    ase_dynamics = Langevin(ase_mol, 0.5 * units.fs, temperature_K=310, friction=0.2)
    ase_dynamics.attach(ase_trajectory, interval=100)

    fig, ax = plt.subplots(1, 4)
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
        ax[1].plot(engine.history.numpy())
        ax[1].set_xlabel("frame number")
        ax[1].set_ylabel("collective variable / Angstrom")

        ax[2].cla()
        draw_molecule(ax[2], ase_mol, rdkit_mol)
        ax[2].axis("off")

        fes_domain, fes_values = ase_mol.calc.get_fes()

        ax[3].cla()
        ax[3].scatter(fes_domain[:, 0], fes_values)
        ax[3].set_xlabel("collective variable / Angstrom")
        ax[3].set_ylabel("bias potential / eV")

        traj_buffer = ase_read("output.traj", index=":")
        ase_write("output.xyz", traj_buffer)

        fig.savefig(f"img_trajectory/frame_{idx}.png")


if __name__ == "__main__":
    main()