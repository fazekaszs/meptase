from typing import Callable

import torch

import ase
from ase.calculators.calculator import Calculator, all_changes

from .kernels import KernelBase
from .exceptions import InvalidShapeException, UnusedKernelException, InvalidParameterCombinationException


type CVMapper = Callable[[torch.Tensor, ], torch.Tensor]


class MetaDynamicsEngine:

    def __init__(
        self,
        mapper: CVMapper,
        additional_potential: CVMapper,
        kernels: list[KernelBase],
        kernel_indices: torch.Tensor,
        kernel_height: float
    ):

        self.mapper = mapper
        self.additional_potential = additional_potential
        self.kernels = kernels
        self.kernel_indices = kernel_indices
        self.kernel_height = kernel_height

        self.history: None | torch.Tensor = None

    def deposit_hill(self, coordinates: torch.Tensor) -> None:

        current_cv = self.mapper(coordinates)

        if self.history is None:
            self.history = current_cv[None, :]
        else:
            self.history = torch.concat([self.history, current_cv[None, :]], dim=0)

    def get_energies_and_forces(
        self,
        coordinates: torch.Tensor | None,
        current_cv: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        # Shape checks and argument control flow...
        # The coordinates argument is given, while current_cv is not set:
        if coordinates is not None and current_cv is None:

            if len(coordinates.shape) == 2:
                coordinates = coordinates[None]  # add batch dimension
            elif len(coordinates.shape) == 3:
                pass
            else:
                raise InvalidShapeException(
                    "The coordinates argument should have a shape of (N_atoms, 3) or "
                    f"(N_batches, N_atoms, 3)! Got a shape of {coordinates.shape}."
                )

            current_cv = torch.vmap(self.mapper)(coordinates)

        # The current_cv argument is given, while coordinates is not set:
        elif coordinates is None and current_cv is not None:

            if len(current_cv.shape) == 1:
                current_cv = current_cv[None]  # add batch dimension
            elif len(current_cv.shape) == 2:
                pass
            else:
                raise InvalidShapeException(
                    "The current_cv argument should have a shape of (N_CVs) or "
                    f"(N_batches, N_CVs)! Got a shape of {current_cv.shape}."
                )

        # Any other combination is invalid
        else:
            raise InvalidParameterCombinationException(
                "One and only one of the coordinates argument or the current_cv argument should be None!"
            )

        # After the control blocks current_cv is guaranteed to be not None
        # and has to have a shape of (N_batches, N_CVs)!

        # The additional potential is calculated for all batches, resulting in a shape of (N_batches, ).
        additional_potential = torch.vmap(self.additional_potential)(current_cv)

        # History existence check.
        if self.history is not None:

            densities = torch.zeros((current_cv.shape[0], *self.history.shape))  # (N_batches, N_timesteps, N_CVs)
            for kernel_idx, kernel in enumerate(self.kernels):

                mask = kernel_idx == self.kernel_indices
                if not torch.any(mask):
                    raise UnusedKernelException(
                        f"The kernel named {type(self.kernels[kernel_idx])} at index {kernel_idx} is not "
                        f"used for any of the CVs!"
                    )

                densities[:, :, mask] = densities[:, :, mask] + kernel(self.history[:, mask], current_cv[:, mask])

            # The final density is the product of all densities along the axis with size N_CVs,
            # and then the sum of these along the axis with size N_timesteps.
            # It will have a shape of (N_batches, ).
            metadynamics_potentials = self.kernel_height * torch.sum(torch.prod(densities, dim=2), dim=1)

        else:
            metadynamics_potentials = torch.zeros(size=(current_cv.shape[0], ))

        total_potential = additional_potential + metadynamics_potentials

        # If the coordinates were not given, we cannot calculate forces!
        if coordinates is None:
            forces = None
        else:
            forces = -1. * torch.autograd.grad(
                outputs=total_potential,
                inputs=coordinates,
                grad_outputs=torch.ones_like(total_potential)
            )[0]

        return total_potential, forces


class MetaDynamicsCalculator(Calculator):

    implemented_properties = [
        "energy", "bias_potential", "forces", "bias_forces"
    ]

    def __init__(
        self,
        unbiased_calculator: Calculator,
        engine: MetaDynamicsEngine,
        atoms: ase.Atoms | None = None,
        **kwargs
    ) -> None:

        super().__init__(atoms=atoms, **kwargs)

        self.unbiased_calculator = unbiased_calculator
        self.engine = engine

    def set_atoms(self, atoms: ase.Atoms) -> None:
        self.unbiased_calculator.atoms = atoms

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes
    ) -> None:

        if atoms is None:
            atoms = self.unbiased_calculator.atoms

        self.unbiased_calculator.calculate(atoms, properties, system_changes)
        self.results = self.unbiased_calculator.results

        atom_positions = torch.tensor(
            atoms.get_positions(),
            requires_grad=True,
            dtype=torch.float32,
            device="cpu"
        )

        bias_potential, bias_forces = self.engine.get_energies_and_forces(atom_positions)

        self.results["bias_potential"] = float(bias_potential[0].detach().item())
        self.results["energy"] += self.results["bias_potential"]

        self.results["bias_forces"] = bias_forces[0].detach().numpy()
        self.results["forces"] += self.results["bias_forces"]

    def deposit_hill(self, atoms: ase.Atoms | None = None) -> None:

        if atoms is None:
            atoms = self.unbiased_calculator.atoms

        positions_np = atoms.get_positions()
        atom_positions = torch.tensor(positions_np, dtype=torch.float32)
        self.engine.deposit_hill(atom_positions)

        current_cv = self.engine.history[-1].numpy()
        print(f"Deposited Gaussian hill at CV = {current_cv}. Total hills: {len(self.engine.history)}")

    def get_fes(self):

        fes_domain = self.engine.history
        fes_values, _ = self.engine.get_energies_and_forces(
            coordinates=None, current_cv=fes_domain
        )

        return fes_domain.numpy(), fes_values.detach().cpu().numpy()
