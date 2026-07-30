from typing import Callable
from abc import ABC, abstractmethod

import torch

from .exceptions import InvalidShapeException


class PotentialEnergyFunction(ABC):

    @abstractmethod
    def run(self, current_cv: torch.Tensor) -> torch.Tensor:
        """
        Calculates (history-independent) potential energies from collective variables.

        :param current_cv: Must have a shape of (N_batches, N_CVs)
        :return: A vector of potential energies shaped (N_batches, )-
        """
        pass

    @staticmethod
    def _check_current_cv_shape(current_cv: torch.Tensor) -> None:
        if len(current_cv.shape) != 1:
            raise InvalidShapeException(
                f"The current_cv tensor should have a shape of (N_CVs, )! "
                f"Instead, it has a shape of {current_cv.shape}!"
            )

    @staticmethod
    def _check_energy_shape(energy: torch.Tensor) -> None:
        if len(energy.shape) != 0:
            raise InvalidShapeException(
                f"The run method of the potential energy function should return a single energy value "
                f"with a shape of an empty tuple! "
                f"Instead, it has a shape of {energy.shape}!"
            )

    def __call__(self, current_cv: torch.Tensor) -> torch.Tensor:
        self._check_current_cv_shape(current_cv)
        potential_energy = self.run(current_cv)
        self._check_energy_shape(potential_energy)
        return potential_energy


class DeserializablePEF(PotentialEnergyFunction, ABC):

    @classmethod
    def from_config[T: PotentialEnergyFunction](
        cls: type[T],
        names_to_idx: dict[str, int] | None = None,
        cv_names: list[str] | None = None,
        **kwargs
    ) -> T:
        if names_to_idx is not None and cv_names is not None:
            kwargs["indices"] = torch.tensor(
                [names_to_idx[name] for name in cv_names],
                dtype=torch.long
            )
        elif "indices" in kwargs:
            kwargs["indices"] = torch.tensor(kwargs["indices"], dtype=torch.long)
        return cls(**kwargs)


PEF_REGISTRY: dict[str, type[DeserializablePEF]] = dict()


def _register_potential[T: DeserializablePEF](name: str) -> Callable[[type[T], ], type[T]]:
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


# We do not register this as a PEF, since it should be unavailable in the JSON runfile.
class MergedPEF(PotentialEnergyFunction):
    """
    Adds the results of multiple potential energy functions.
    """

    def __init__(self, potential_energy_functions: list[PotentialEnergyFunction]):
        super().__init__()
        self.potential_energy_functions = potential_energy_functions

    def run(self, current_cv: torch.Tensor) -> torch.Tensor:
        return sum(
            (pef(current_cv) for pef in self.potential_energy_functions),
            start=torch.zeros(tuple())
        )


@_register_potential("lower_harmonic_wall")
class LowerHarmonicWall(DeserializablePEF):

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

    def run(self, current_cv: torch.Tensor):

        selected_cv = current_cv[self.indices]
        zero_level = torch.zeros_like(selected_cv)

        potential_left = 0.5 * self.force_constant * (selected_cv - self.cv_min) ** 2
        potential_left = torch.where(selected_cv < self.cv_min, potential_left, zero_level)

        return torch.sum(potential_left)


@_register_potential("upper_harmonic_wall")
class UpperHarmonicWall(DeserializablePEF):

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

    def run(self, current_cv: torch.Tensor):

        selected_cv = current_cv[self.indices]
        zero_level = torch.zeros_like(selected_cv)

        potential_right = 0.5 * self.force_constant * (selected_cv - self.cv_max) ** 2
        potential_right = torch.where(selected_cv > self.cv_max, potential_right, zero_level)

        return torch.sum(potential_right)


@_register_potential("flat_bottomed_harmonic_wall")
class FlatBottomedHarmonic(DeserializablePEF):

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

    def run(self, current_cv: torch.Tensor) -> torch.Tensor:

        return self.lower_wall(current_cv) + self.upper_wall(current_cv)
