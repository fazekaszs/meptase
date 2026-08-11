import time
import logging

from typing import Callable

import numpy as np
import torch

import ase
from ase.calculators.calculator import Calculator, all_changes

from .kernels import KernelBase
from .exceptions import InvalidShapeException, UnusedKernelException, InvalidParameterCombinationException


type CVMapper = Callable[[torch.Tensor, ], torch.Tensor]


BOLTZMANN_CONSTANT = 8.61733326E-05  # eV/K

logger = logging.getLogger(__name__)


class MetaDynamicsEngine:

    def __init__(
        self,
        mapper: CVMapper,
        additional_potential: CVMapper,
        kernels: list[KernelBase],
        kernel_indices: torch.Tensor,
        kernel_height: float,
        well_tempered_temperature: float | None = None
    ):
        """
        Creates a metadynamics engine, suitable for an ASE metadynamics calculator.
        This manages the calculation of collective variables, kernel densities, additional potentials,
            bias forces, etc., as well as hill deposition in its cv_history field.

        :param mapper: A callable that maps 2D tensors of shape (N_atoms, 3) to 1D tensors of shape (N_CVs, ).
            This callable is responsible for the collective variable calculation.
            It is possible to achieve this using subclasses of CVBase or by writing your own callable.
            The callable must be differentiable (use PyTorch).
            Use the MergeCV subclass, if the collection of many different CV types are needed.
        :param additional_potential: Another callable, mapping CV vectors to potential energies.
            Use instances of PotentialEnergyFunction subclasses or write your own function.
            Again, the callable must be differentiable (use PyTorch).
            The potential energy is measured in eVs.
        :param kernels: A list of KernelBase subclass instances.
            These map pairs of CV collections (batches of CVs and a CV history tensor)
            to a 3D tensor of CV densities.
            Namely, if the batches of CVs tensor has shape of (N_batches, N_CVs) and the cv_history tensor
            (N_timesteps, N_CVs), then the resulting density tensor should have a shape of
            (N_batches, N_timesteps, N_CVs).
            The final density of a batch is calculated as the product along the third dimension and then
            as the sum along the second dimension.
            Different types of CVs may need different kernels, that's why a list of them can be given.
        :param kernel_indices: An indexing tensor of shape (N_CVs, ), mapping every CV to a kernel index
            given in the kernels parameter.
        :param kernel_height: The maximum height of the kernel product. Given in eVs.
            If the well_tempered_temperature parameter is set (not None), then it is the energy multiplier
            before the exponential in the sample weight calculation.
        :param well_tempered_temperature: Whether to perform well tempered metadynamics and what biasing
            temperature to use, if so. In well tempered metadynamics the sample weights in the kernel density
            estimator biasing potential is not constant, but rather dependent on the bias potential already
            placed. Larger temperature factors result in more exploration and slower convergence. Given in K.
            Leave it as None if there is no need for well tempered metadynamics.
        """

        self.mapper = mapper
        self.additional_potential = additional_potential
        self.kernels = kernels
        self.kernel_indices = kernel_indices
        self.kernel_height = kernel_height
        self.well_tempered_temperature = well_tempered_temperature

        self.cv_history: None | torch.Tensor = None

        # This is only needed in case of well tempered metadynamics.
        # This is the bias potential-dependent kernel weight.
        self.wt_metad_weight_history: None | torch.Tensor = None

    def _construct_coordinates_and_cv(
        self,
        coordinates: torch.Tensor | None,
        current_cv: torch.Tensor | None = None
    ):

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

        return coordinates, current_cv

    def _calculate_metadynamics_potential(self, current_cv: torch.Tensor) -> torch.Tensor:

        # History existence check.
        if self.cv_history is not None:

            # The shape of the densities tensor: (N_batches, N_timesteps, N_CVs)
            densities = torch.zeros((current_cv.shape[0], *self.cv_history.shape))
            for kernel_idx, kernel in enumerate(self.kernels):

                mask = torch.eq(self.kernel_indices, kernel_idx)
                if not torch.any(mask):
                    raise UnusedKernelException(
                        f"The kernel named {type(self.kernels[kernel_idx])} "
                        f"at index {kernel_idx} is not used for any of the CVs!"
                    )

                kernel_evaluation = kernel(self.cv_history[:, mask], current_cv[:, mask])
                densities[:, :, mask] = densities[:, :, mask] + kernel_evaluation

            # The final density is the product of all densities along the axis with size N_CVs,
            # and then the sum of these along the axis with size N_timesteps.
            # The sum is weighted in the case of well tempered metadynamics.
            # It will have a shape of (N_batches, ).
            if self.wt_metad_weight_history is None:
                metadynamics_potentials = self.kernel_height * torch.sum(torch.prod(densities, dim=2), dim=1)
            else:
                metadynamics_potentials = torch.sum(
                    self.wt_metad_weight_history * torch.prod(densities, dim=2),
                    dim=1
                )

        else:
            metadynamics_potentials = torch.zeros(size=(current_cv.shape[0],), requires_grad=True)

        return metadynamics_potentials

    def _deposit_hill(
        self,
        current_cv: torch.Tensor,
        bias_potential: torch.Tensor
    ) -> None:
        """
        Deposits a hill, i.e. adds a CV vector to the CV history.

        :param current_cv: The CV vector to be added to the CV history.
            Note, that it should have a batched shape of (1, N_CVs)!
        :param bias_potential: The bias potential at the current CV.
            Only used if well tempered metadynamics is performed.
            Note, that it should have a batched shape of (1, )!
        """

        simulation_start = self.cv_history is None
        wt_metad_on = self.wt_metad_weight_history is not None

        if simulation_start:
            self.cv_history = current_cv.clone()
        else:
            self.cv_history = torch.concat([self.cv_history, current_cv], dim=0)

        if wt_metad_on:

            current_weight = self.kernel_height * torch.exp(
                - bias_potential / (self.well_tempered_temperature * BOLTZMANN_CONSTANT)
            )

            if simulation_start:
                self.wt_metad_weight_history = current_weight.clone()
            else:
                self.wt_metad_weight_history = torch.concat(
                    [self.wt_metad_weight_history, current_weight],
                    dim=0
                )


    def get_observables(
        self,
        coordinates: torch.Tensor | None,
        current_cv: torch.Tensor | None = None,
        deposit_hill: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns the current CVs, energies and forces.

        :param coordinates: The coordinates at which the current CVs, energy and forces should be calculated.
            Can have shapes of (N_atoms, 3) or (N_batches, N_atoms, 3).
            If it is not given (i.e. None), then the current_cv arguments must not be None.
        :param current_cv: The current CVs at which the energy and forces should be calculated.
            Can have shapes of (N_CVs, ) or (N_batches, N_CVs, ).
            If it is not given (i.e. None), then the coordinates arguments must not be None.
        :param deposit_hill: Whether to place a hill in the CV history at the current CV vector.
            Can only be true if a single batch of coordinates or single batch of CVs are given.
        :return: The current CVs, the energies and the forces.
            Forces can only be returned if the coordinates argument was set.
            Otherwise, None is returned as forces.
        """

        coordinates, current_cv = self._construct_coordinates_and_cv(coordinates, current_cv)

        # The additional potential is calculated for all batches, resulting in a shape of (N_batches, ).
        additional_potential = torch.vmap(self.additional_potential)(current_cv)

        metadynamics_potentials = self._calculate_metadynamics_potential(current_cv)

        total_potential = additional_potential + metadynamics_potentials

        # Update the CV history by adding a hill, if requested.
        # Note: detach the current_cv and metadynamics_potentials tensors
        # from the computational graph, since we want the history to be gradientless!
        if deposit_hill and current_cv.shape[0] == 1:
            self._deposit_hill(
                current_cv.detach(),
                metadynamics_potentials.detach()
            )
        elif deposit_hill and current_cv.shape[0] != 1:
            raise InvalidParameterCombinationException(
                f"A hill deposition is requested, but a batch of current CVs is given! "
                f"A CV vector can only be registered in the CV history if the batch size is 1. "
                f"Instead, a batch size of {current_cv.shape[0]} was detected!"
            )

        # If the coordinates were not given, we cannot calculate forces!
        if coordinates is None:
            forces = None
        else:
            forces = -1. * torch.autograd.grad(
                outputs=total_potential,
                inputs=coordinates,
                grad_outputs=torch.ones_like(total_potential),
                materialize_grads=True
            )[0]

        return current_cv, total_potential, forces


class MetaDynamicsCalculator(Calculator):

    implemented_properties = [
        "collective_variables", "energy", "bias_potential", "forces", "bias_forces"
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

        self.performance_statistics = {
            "total_unbiased_runtime": 0.0,
            "total_biasing_runtime": 0.0,
            "n_observations": 0,
        }

    def set_atoms(self, atoms: ase.Atoms) -> None:
        self.unbiased_calculator.atoms = atoms

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
        deposit_hill: bool = False
    ) -> None:

        if atoms is None:
            atoms = self.unbiased_calculator.atoms

        unbiased_start = time.perf_counter()
        self.unbiased_calculator.calculate(atoms, properties, system_changes)
        self.results = self.unbiased_calculator.results
        self.performance_statistics["total_unbiased_runtime"] += time.perf_counter() - unbiased_start

        atom_positions = torch.tensor(
            atoms.get_positions(),
            requires_grad=True,
            dtype=torch.float32,
            device="cpu"
        )

        biasing_start = time.perf_counter()
        current_cv, bias_potential, bias_forces = self.engine.get_observables(
            coordinates=atom_positions,
            deposit_hill=deposit_hill
        )
        self.performance_statistics["total_biasing_runtime"] += time.perf_counter() - biasing_start

        self.performance_statistics["n_observations"] += 1

        self.results["collective_variables"] = current_cv[0].detach().numpy()

        self.results["bias_potential"] = float(bias_potential[0].detach().item())
        self.results["energy"] += self.results["bias_potential"]

        self.results["bias_forces"] = bias_forces[0].detach().numpy()
        self.results["forces"] += self.results["bias_forces"]

    def deposit_hill(self, atoms: ase.Atoms | None = None) -> None:

        if atoms is None:
            atoms = self.unbiased_calculator.atoms

        self.calculate(deposit_hill=True)
        current_cv = self.results["collective_variables"]

        # Create statistics about bias forces
        bias_forces_size = np.sqrt(np.sum(self.results["bias_forces"] ** 2, axis=1))
        mean_bias_force = np.mean(bias_forces_size)
        max_bias_force_idx = np.argmax(bias_forces_size)
        max_bias_force_atom = atoms[max_bias_force_idx].symbol + str(max_bias_force_idx)
        max_bias_force_size = bias_forces_size[max_bias_force_idx]

        # Create statistics about forces
        forces_size = np.sqrt(np.sum(self.results["forces"] ** 2, axis=1))
        mean_force = np.mean(forces_size)
        max_force_idx = np.argmax(forces_size)
        max_force_atom = atoms[max_force_idx].symbol + str(max_force_idx)
        max_force_size = forces_size[max_force_idx]

        logger.info(
            f"Deposited Gaussian hill at CV = {current_cv}.\n"
            f"    - Total hills: {len(self.engine.cv_history)}.\n"
            f"    - Bias potential: {self.results['bias_potential']:.5} eV.\n"
            f"    - Total potential: {self.results['energy']:.5} eV.\n"
            f"    - Mean bias force: {mean_bias_force:.5} eV/A.\n"
            f"    - Max bias force: {max_bias_force_size:.5} eV/A (at atom {max_bias_force_atom}).\n"
            f"    - Mean force: {mean_force:.5} eV/A.\n"
            f"    - Max force: {max_force_size:.5} eV/A (at atom {max_force_atom}).",
        )

    def get_fes(self):

        fes_domain = self.engine.cv_history
        _, fes_values, _ = self.engine.get_observables(coordinates=None, current_cv=fes_domain)

        return fes_domain.numpy(), -1. * fes_values.detach().cpu().numpy()
