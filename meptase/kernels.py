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


def validate_shapes(cv_history: torch.Tensor, current_cv: torch.Tensor) -> None:

    if cv_history.shape[1] != current_cv.shape[1]:
        raise InvalidShapeException(
            "The parameter cv_history should be have a shape of (n_timesteps, n_CVs), "
            "and current_cv should be of (n_samples, n_CVs), "
            f"i.e. the second axis sizes should match! "
            f"I got shapes {cv_history.shape} and {current_cv.shape}, respectively."
        )


class GaussianKernel(KernelBase):

    def __init__(self, gaussian_width: float | torch.Tensor) -> None:
        self.gaussian_width = gaussian_width

    def __call__(self, cv_history: torch.Tensor, current_cv: torch.Tensor) -> torch.Tensor:

        validate_shapes(cv_history, current_cv)

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

        validate_shapes(cv_history, current_cv)

        diff = torch.abs(cv_history[None, :, :] - current_cv[:, None, :])
        diff = torch.where(diff > self.period / 2, self.period - diff, diff)
        scaled_diff = diff * 2 * torch.pi / self.period

        density = torch.exp(self.concentration * (torch.cos(scaled_diff) - 1))
        return density


class BetaKernel(KernelBase):

    def __init__(
        self,
        width: float,
        domain: float = torch.pi
    ) -> None:

        super().__init__()

        self.width = width
        self.domain = domain

        self._reduced_width = self.width / self.domain

    def __call__(self, cv_history: torch.Tensor, current_cv: torch.Tensor) -> torch.Tensor:

        validate_shapes(cv_history, current_cv)

        # reduced variables:
        # since cv_history and current_cv contains values between [0, self.domain],
        # we scale them to the interval [0, 1], which is the natural domain of the beta distribution.

        reduced_cv_history = cv_history / self.domain
        reduced_current_cv = current_cv / self.domain

        # The beta PDF is parametrized by two parameters:
        # - the mode m of the PDF, which has to be placed at the reduced_cv_history values,
        # - the width of the PDF, which (instead of the variance) is the distance
        #   between the two extrema of the beta PDF derivative (w = self._reduced_width).
        # Using this parametrization, the proxy parameters are;
        # - common_param = 4 * m * (1 - m) / w^2 + 1
        # - alpha = m * common_param + 1
        # - beta = (1 - m) * common_param + 1

        common_param = 4 * reduced_cv_history * (1 - reduced_cv_history) / self._reduced_width ** 2 + 1
        alpha = reduced_cv_history * common_param + 1
        beta = (1 - reduced_cv_history) * common_param + 1

        # The unnormalized density:
        # f(x) = x^a * (1 - x)^b
        # The normalization fixes the function at the mode (reduced_cv_history) to 1:
        # g(x) = f(x) / f(m) = x^a * (1 - x)^b / (m^a * (1 - m)^b)
        # In log space, the calculation is more stable:
        # log(g(x)) = log(x) * a + log(1 - x) * b - log(m) * a - log(1-m) * b

        log_d1 = torch.log(reduced_current_cv[:, None, :]) * (alpha[None, :, :] - 1)
        log_d2 = torch.log(1 - reduced_current_cv[:, None, :]) * (beta[None, :, :] - 1)
        log_norm_d1 = torch.log(reduced_cv_history) * (alpha - 1)
        log_norm_d2 =  torch.log(1 - reduced_cv_history) * (beta - 1)
        log_density = log_d1 + log_d2 - log_norm_d1[None, :, :] - log_norm_d2[None, :, :]
        density = torch.exp(log_density)

        return density
