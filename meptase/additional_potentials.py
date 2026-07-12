from typing import Callable
from abc import ABC, abstractmethod

import torch

class PotentialEnergyFunction(ABC):

    @abstractmethod
    def __call__(self, current_cv: torch.Tensor) -> torch.Tensor:
        """
        Calculates (history-independent) potential energies from collective variables.

        :param current_cv: Must have a shape of (N_batches, N_CVs)
        :return: A vector of potential energies shaped (N_batches, )-
        """
        pass


PEF_REGISTRY: dict[str, type[PotentialEnergyFunction]] = dict()


def _register_potential[T: PotentialEnergyFunction](name: str) -> Callable[[type[T], ], type[T]]:
    """
    Created a decorator that registers a PotentialEnergyFunction (PEF) class in the PEF_REGISTRY dictionary.
    This will be later used for the deserialization of PEFs from the JSON config.

    :param name: The serialized name of the PEF.
    :return: The decorator that registers the PEF.
    """

    def decorator(cls: type[T]) -> type[T]:
        PEF_REGISTRY[name] = cls
        return cls

    return decorator


@_register_potential("lower_harmonic_wall")
class LowerHarmonicWall(PotentialEnergyFunction):

    def __init__(
        self,
        indices: torch.Tensor,
        cv_min: float,
        force_constant: float
    ):
        super().__init__()
        self.indices = indices
        self.cv_min = cv_min
        self.force_constant = force_constant

    def __call__(self, current_cv: torch.Tensor):

        selected_cv = current_cv[:, self.indices]
        zero_level = torch.zeros_like(selected_cv)

        potential_left = 0.5 * self.force_constant * (selected_cv - self.cv_min) ** 2
        potential_left = torch.where(selected_cv < self.cv_min, potential_left, zero_level)

        return potential_left


@_register_potential("upper_harmonic_wall")
class UpperHarmonicWall(PotentialEnergyFunction):

    def __init__(
        self,
        indices: torch.Tensor,
        cv_max: float,
        force_constant: float
    ):
        super().__init__()
        self.indices = indices
        self.cv_max = cv_max
        self.force_constant = force_constant

    def __call__(self, current_cv: torch.Tensor):

        selected_cv = current_cv[:, self.indices]
        zero_level = torch.zeros_like(selected_cv)

        potential_right = 0.5 * self.force_constant * (selected_cv - self.cv_max) ** 2
        potential_right = torch.where(selected_cv > self.cv_max, potential_right, zero_level)

        return potential_right


@_register_potential("flat_bottomed_harmonic_wall")
class FlatBottomedHarmonic(PotentialEnergyFunction):

    def __init__(
        self,
        indices: torch.Tensor,
        cv_min: float,
        cv_max: float,
        force_constant: float
    ):

        super().__init__()
        self.lower_wall = LowerHarmonicWall(indices, cv_min, force_constant)
        self.upper_wall = UpperHarmonicWall(indices, cv_max, force_constant)

    def __call__(self, current_cv: torch.Tensor) -> torch.Tensor:

        return self.lower_wall(current_cv) + self.upper_wall(current_cv)
