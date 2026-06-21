from abc import ABC, abstractmethod

import torch

from .exceptions import InvalidShapeException

class KernelBase(ABC):

    @abstractmethod
    def __call__(self, cv_history: torch.Tensor, current_cv: torch.Tensor) -> torch.Tensor:
        """

        :param cv_history: Must have a shape of (N_timesteps, N_CVs)
        :param current_cv: Must have a shape of (N_batches, N_CVs)
        :return: The density tensor of shape (N_batches, N_timesteps, N_CVs)
        """
        pass


class GaussianKernel(KernelBase):

    def __init__(self, gaussian_width: float | torch.Tensor) -> None:
        self.gaussian_width = gaussian_width

    def __call__(self, cv_history: torch.Tensor, current_cv: torch.Tensor) -> torch.Tensor:

        if cv_history.shape[1] != current_cv.shape[1]:
            raise InvalidShapeException(
                "The parameter cv_history should be a series of CV vectors, "
                "but there is a shape mismatch!"
            )

        density = torch.exp(
            - 0.5 * ((cv_history[None, :, :] - current_cv[:, None, :]) / self.gaussian_width) ** 2
        )
        return density


class VonMisesKernel(KernelBase):

    def __init__(
        self,
        concentration: float,
        period: float = 2 * torch.pi
    ) -> None:

        super().__init__()

        self.concentration = concentration
        self.period = period

    def __call__(self, cv_history: torch.Tensor, current_cv: torch.Tensor) -> torch.Tensor:

        if cv_history.shape[1] != current_cv.shape[1]:
            raise InvalidShapeException(
                "The parameter cv_history should be a series of CV vectors, "
                "but there is a shape mismatch!"
            )

        diff = torch.abs(cv_history[None, :, :] - current_cv[:, None, :])
        diff = torch.where(diff > self.period / 2, self.period - diff, diff)
        scaled_diff = diff * 2 * torch.pi / self.period

        density = torch.exp(self.concentration * (torch.cos(scaled_diff) - 1))
        return density
