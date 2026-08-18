from typing import Callable
from abc import ABC

import torch

from ...additional_potentials import (
    PotentialEnergyFunction,
    LowerHarmonicWall, UpperHarmonicWall, FlatBottomedHarmonic
)


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


@_register_potential("lower_harmonic_wall")
class LowerHarmonicWallS(LowerHarmonicWall, DeserializablePEF):
    pass


@_register_potential("upper_harmonic_wall")
class UpperHarmonicWallS(UpperHarmonicWall, DeserializablePEF):
    pass


@_register_potential("flat_bottomed_harmonic_wall")
class FlatBottomedHarmonicS(FlatBottomedHarmonic, DeserializablePEF):
    pass
