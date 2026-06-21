import torch
import torch.nn as nn

EPSILON = 1E-8

class DistanceCV(nn.Module):

    def __init__(self, indices: torch.Tensor):
        super().__init__()
        self.indices = indices

    def forward(self, positions: torch.Tensor) -> torch.Tensor:

        # - positions is (N, 3), where N denotes the number of atoms
        # - self.indices is (M, 2), where M denotes the number of distances to calculate
        # - positions[self.indices] is a tensor of shape (M, 2, 3), which contains the
        #     coordinate triplet pairs: [p1_m p2_m] = [[p1_xm p1_ym p1_zm] [p2_xm p2_ym p2_zm]],
        #     where m: 0...M-1.

        triplet_pairs = positions[self.indices]
        deltas = triplet_pairs[:, 0, :] - triplet_pairs[:, 1, :]  # (M, 3)
        distances = torch.sqrt(torch.sum(deltas ** 2, dim=1) + EPSILON)  # (M, )

        return distances


class AngleCV(nn.Module):
    def __init__(self, idx_a: int, idx_b: int, idx_c: int):
        super().__init__()
        self.idx_a = idx_a
        self.idx_b = idx_b
        self.idx_c = idx_c

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        v_ba = positions[self.idx_a] - positions[self.idx_b]
        v_bc = positions[self.idx_c] - positions[self.idx_b]

        norm_ba = torch.sqrt(torch.sum(v_ba ** 2) + EPSILON)
        norm_bc = torch.sqrt(torch.sum(v_bc ** 2) + EPSILON)

        cos_theta = torch.sum(v_ba * v_bc) / (norm_ba * norm_bc)
        cos_theta = torch.clamp(cos_theta, -1.0 + EPSILON, 1.0 - EPSILON)

        return torch.acos(cos_theta)


class DihedralCV(nn.Module):

    def __init__(self, idx_a: int, idx_b: int, idx_c: int, idx_d: int):
        super().__init__()
        self.idx_a = idx_a
        self.idx_b = idx_b
        self.idx_c = idx_c
        self.idx_d = idx_d

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        b1 = positions[self.idx_b] - positions[self.idx_a]
        b2 = positions[self.idx_c] - positions[self.idx_b]
        b3 = positions[self.idx_d] - positions[self.idx_c]

        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)

        n1 = n1 / torch.sqrt(torch.sum(n1 ** 2) + EPSILON)
        n2 = n2 / torch.sqrt(torch.sum(n2 ** 2) + EPSILON)
        b2_norm = b2 / torch.sqrt(torch.sum(b2 ** 2) + EPSILON)

        x = torch.sum(n1 * n2)
        m1 = torch.linalg.cross(n1, b2_norm)
        y = torch.sum(m1 * n2)

        return torch.atan2(y, x)
