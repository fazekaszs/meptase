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

    def __init__(self, indices: torch.Tensor):
        super().__init__()
        self.indices = indices

    def forward(self, positions: torch.Tensor) -> torch.Tensor:

        # - positions is (N, 3), where N denotes the number of atoms
        # - self.indices is (M, 3), where M denotes the number of angles to calculate
        # - positions[self.indices] is a tensor of shape (M, 3, 3), which contains the
        #     coordinate triplet triplets: [p1_m p2_m p3_m] = [
        #       [p1_xm p1_ym p1_zm] [p2_xm p2_ym p2_zm] [p3_xm p3_ym p3_zm]
        #     ],
        #     where m: 0...M-1.

        triplets = positions[self.indices]

        # Vectorize displacement vectors from the central vertex (index 1 in dim 1)
        # Both v_ba and v_bc will have a shape of (M, 3)
        v_ba = triplets[:, 0, :] - triplets[:, 1, :]
        v_bc = triplets[:, 2, :] - triplets[:, 1, :]

        # Batch reduce norms along the spatial axis (dim=1) -> shape (M,)
        norm_ba = torch.sqrt(torch.sum(v_ba ** 2, dim=1) + EPSILON)
        norm_bc = torch.sqrt(torch.sum(v_bc ** 2, dim=1) + EPSILON)

        # Batch dot product via element-wise multiplication and summation -> shape (M,)
        dot_product = torch.sum(v_ba * v_bc, dim=1)
        cos_theta = dot_product / (norm_ba * norm_bc)

        # Keep gradients completely safe from NaN anomalies at boundaries
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
