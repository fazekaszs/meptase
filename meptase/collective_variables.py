from typing import Callable, Self
from abc import ABC, abstractmethod

import torch

from .exceptions import InvalidShapeException, DeserializationException


class CVBase(ABC):

    @abstractmethod
    def run(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Maps the Cartesian coordinates of atoms to a vector of collective variables.

        :param positions: Positions of the atoms. Must have a shape of (N_atoms, 3).
        :return: A vector of collective variables shaped (N_CVs, ). If a single CV is
            needed, then the shape should be (1, ).
        """
        pass

    @staticmethod
    def _check_positions_shape(positions: torch.Tensor) -> None:
        if len(positions.shape) != 2 or positions.shape[-1] != 3:
            raise InvalidShapeException(
                f"The positions should have a shape of (N_positions, 3)! "
                f"Instead, it has a shape of {positions.shape}!"
            )

    @staticmethod
    def _check_current_cv_shape(current_cv: torch.Tensor) -> None:
        if len(current_cv.shape) != 1:
            raise InvalidShapeException(
                f"The run method of the collective variable should return a current_cv tensor "
                f"with a shape of (N_CVs, )! "
                f"Instead, it has a shape of {current_cv.shape}!"
            )

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        self._check_positions_shape(positions)
        current_cv = self.run(positions)
        self._check_current_cv_shape(current_cv)
        return current_cv


class DeserializableCV(CVBase, ABC):

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

            if "indices" not in kwargs:
                raise DeserializationException(
                    f"The indices argument is necessary if the index_mapper is given!"
                )

            kwargs["indices"] = torch.tensor([
                [index_mapper[x] for x in idx_line]
                for idx_line in kwargs["indices"]
            ], dtype=torch.long)

        return cls(**kwargs)


EPSILON = 1E-8
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


# We do not register this as a CV, since it should be unavailable in the JSON runfile.
class MergeCV(CVBase):
    """
    Calculates and merges multiple collective variable vectors through simple concatenation.
    """

    def __init__(self, cv_mappers: list[CVBase]):
        super().__init__()
        self.cv_mappers = cv_mappers

    def run(self, positions: torch.Tensor) -> torch.Tensor:
        cv_list = [mapper(positions) for mapper in self.cv_mappers]
        return torch.concat(cv_list, dim=0)


@_register_cv("distance")
class DistanceCV(DeserializableCV):
    """
    Calculates the distance between coordinate pairs given by their indices in the coordinate matrix.
    """

    def __init__(self, indices: torch.Tensor):
        """
        Class constructor.

        :param indices: An indexing tensor containing integers that has a shape of (M, 2), where
            M is the number of distances to be calculated.
        """
        super().__init__()
        self.indices = indices

    def run(self, positions: torch.Tensor) -> torch.Tensor:

        # - positions is (N, 3), where N denotes the number of atoms
        # - self.indices is (M, 2), where M denotes the number of distances to calculate
        # - positions[self.indices] is a tensor of shape (M, 2, 3), which contains the
        #     coordinate triplet pairs: [p1_m p2_m] = [[p1_xm p1_ym p1_zm] [p2_xm p2_ym p2_zm]],
        #     where m: 0...M-1.

        vector_pairs = positions[self.indices]
        deltas = vector_pairs[:, 0, :] - vector_pairs[:, 1, :]  # (M, 3)
        distances = torch.sqrt(torch.sum(deltas ** 2, dim=1) + EPSILON)  # (M, )

        return distances


@_register_cv("angle")
class AngleCV(DeserializableCV):
    """
    Calculates the angle between coordinate triplets given by their indices in the coordinate matrix.
    """

    def __init__(self, indices: torch.Tensor):
        """
        Class constructor.

        :param indices: An indexing tensor containing integers that has a shape of (M, 3), where
            M is the number of angles to be calculated.
        """
        super().__init__()
        self.indices = indices

    def run(self, positions: torch.Tensor) -> torch.Tensor:

        # - positions is (N, 3), where N denotes the number of atoms
        # - self.indices is (M, 3), where M denotes the number of angles to calculate
        # - positions[self.indices] is a tensor of shape (M, 3, 3), which contains the
        #     coordinate triplet triplets: [p1_m p2_m p3_m] = [
        #       [p1_xm p1_ym p1_zm] [p2_xm p2_ym p2_zm] [p3_xm p3_ym p3_zm]
        #     ],
        #     where m: 0...M-1.

        vector_triplets = positions[self.indices]

        # Vectorize displacement vectors from the central vertex (index 1 in dim 1)
        # Both v_ba and v_bc will have a shape of (M, 3)
        v_ba = vector_triplets[:, 0, :] - vector_triplets[:, 1, :]
        v_bc = vector_triplets[:, 2, :] - vector_triplets[:, 1, :]

        # Batch reduce norms along the spatial axis (dim=1) -> shape (M,)
        norm_ba = torch.sqrt(torch.sum(v_ba ** 2, dim=1) + EPSILON)
        norm_bc = torch.sqrt(torch.sum(v_bc ** 2, dim=1) + EPSILON)

        # Batch dot product via element-wise multiplication and summation -> shape (M,)
        dot_product = torch.sum(v_ba * v_bc, dim=1)
        cos_theta = dot_product / (norm_ba * norm_bc)

        # Keep gradients completely safe from NaN anomalies at boundaries
        cos_theta = torch.clamp(cos_theta, -1.0 + EPSILON, 1.0 - EPSILON)

        return torch.acos(cos_theta)


@_register_cv("dihedral")
class DihedralCV(DeserializableCV):
    """
    Calculates the dihedral angle between coordinate quadruplets given by their indices in the coordinate matrix.
    """

    def __init__(self, indices: torch.Tensor):
        """
        Class constructor.

        :param indices: An indexing tensor containing integers that has a shape of (M, 4), where
            M is the number of dihedral angles to be calculated.
        """
        super().__init__()
        self.indices = indices

    def run(self, positions: torch.Tensor) -> torch.Tensor:

        # - positions is (N, 3), where N denotes the number of atoms
        # - self.indices is (M, 4), where M denotes the number of angles to calculate
        # - positions[self.indices] is a tensor of shape (M, 4, 3), which contains the
        #     coordinate triplet quadruplets: [p1_m p2_m p3_m p4_m] = [
        #       [p1_xm p1_ym p1_zm] [p2_xm p2_ym p2_zm] [p3_xm p3_ym p3_zm] [p4_xm p4_ym p4_zm]
        #     ],
        #     where m: 0...M-1.

        vector_quadruplets = positions[self.indices]

        # Batch extract bond displacement vectors -> shape (M, 3)
        b1 = vector_quadruplets[:, 1, :] - vector_quadruplets[:, 0, :]
        b2 = vector_quadruplets[:, 2, :] - vector_quadruplets[:, 1, :]
        b3 = vector_quadruplets[:, 3, :] - vector_quadruplets[:, 2, :]

        # Batch cross product to find plane normals
        # torch.linalg.cross automatically evaluates over the last, spatial dimension
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)

        # Batch normalize the vectors
        n1 = n1 / torch.sqrt(torch.sum(n1 ** 2, dim=1, keepdim=True) + EPSILON)
        n2 = n2 / torch.sqrt(torch.sum(n2 ** 2, dim=1, keepdim=True) + EPSILON)

        b2_norm = b2 / torch.sqrt(torch.sum(b2 ** 2, dim=1, keepdim=True) + EPSILON)

        # Project orthogonal components
        x = torch.sum(n1 * n2, dim=1)

        m1 = torch.linalg.cross(n1, b2_norm)
        y = torch.sum(m1 * n2, dim=1)

        # Batch evaluation of atan2 returns (M, ) array containing full periodic values
        return torch.atan2(y, x)
