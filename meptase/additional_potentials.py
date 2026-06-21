import torch
import torch.nn as nn

class FlatBottomedHarmonic(nn.Module):

    def __init__(
        self,
        cv_idx: int | torch.Tensor,
        cv_min: float,
        cv_max: float,
        force_constant: float
    ):

        super().__init__()
        self.cv_idx = cv_idx
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.force_constant = force_constant

    def forward(self, current_cv: torch.Tensor) -> torch.Tensor:

        selected_cv = current_cv[self.cv_idx]
        zero_level = torch.zeros_like(selected_cv)

        potential_right = 0.5 * self.force_constant * (selected_cv - self.cv_max) ** 2
        potential_right = torch.where(selected_cv > self.cv_max, potential_right, zero_level)

        potential_left = 0.5 * self.force_constant * (selected_cv - self.cv_min) ** 2
        potential_left = torch.where(selected_cv < self.cv_min, potential_left, zero_level)

        return potential_right + potential_left
