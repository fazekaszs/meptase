from typing import Callable
from abc import ABC

import torch

from ...exceptions import DeserializationException
from ...collective_variables import (
    CVBase,
    DistanceCV, AngleCV, DihedralCV
)


class DeserializableCV(CVBase, ABC):

    @staticmethod
    def _transform_index_mapper(
        index_mapper: dict[int, int]
    ) -> torch.Tensor:
        """
        Transforms an index mapper dictionary to an index mapper tensor M.
        Then, if X is a tensor containing the indices to be mapped, then the
        mapping can be performed as M[X], i.e. with advanced indexing.

        :param index_mapper: The dictionary to be transformed.
        :return: The tensor, with which the mapping can be carried out.
        """

        source_max = max(index_mapper.keys()) + 1
        mapper_tensor = torch.tensor([
            index_mapper.get(source_idx, -1)
            for source_idx in range(source_max)
        ], dtype=torch.long)

        return mapper_tensor

    @classmethod
    def from_config[T: CVBase](
        cls: type[T],
        index_mapper: dict[int, int] | None = None,
        **kwargs
    ) -> T:
        """
        Preprocesses the deserialized arguments to comply with the collective variable's
        constructor signature and with other requirements.

        :param index_mapper: Preprocesses the "indices" argument through index mapping. This is usually
            needed to convert AtomMapNum values of an RDKit Atom object to the index of that same atom
            in an ASE Atoms object. For example, the SMILES code "C[C:2]C(=[O:4])C[N:1]" must have an
            index_mapper of {2: 1, 4: 3, 1: 5}, because the marked atoms are at indices 1, 3 and 5.
        :param kwargs: The keyword arguments needed to be preprocessed.
        :return: A deserialized collective variable instance.
        """

        if index_mapper is not None:

            mapper_tensor = DeserializableCV._transform_index_mapper(index_mapper)

            if "indices" not in kwargs:
                raise DeserializationException(
                    f"The indices argument is necessary if the index_mapper is given!"
                )

            kwargs["indices"] = mapper_tensor[torch.tensor(kwargs["indices"], dtype=torch.long)]

            if torch.any(kwargs["indices"] < 0):
                raise Exception("Unreachable!")

        return cls(**kwargs)


CV_REGISTRY: dict[str, type[DeserializableCV]] = dict()


def _register_cv[T: DeserializableCV](name: str) -> Callable[[type[T], ], type[T]]:
    """
    Created a decorator that registers a CV class in the CV_REGISTRY dictionary.
    This will be later used for the deserialization of CVs from the JSON config.

    :param name: The serialized name of the CV.
    :return: The decorator that registers the CV.
    """

    def decorator(cls: type[T]) -> type[T]:
        CV_REGISTRY[name] = cls
        return cls

    return decorator


@_register_cv("distance")
class DistanceCVS(DistanceCV, DeserializableCV):
    pass


@_register_cv("angle")
class AngleCVS(AngleCV, DeserializableCV):
    pass


@_register_cv("dihedral")
class DihedralCVS(DihedralCV, DeserializableCV):
    pass

