import math
from typing import Tuple, Any, Optional
import torch

from .base import CollocationSampler

class BoundaryLayerSampler(CollocationSampler):
    """
    Stretched / Shishkin / Geometrically clustered collocation point sampler
    focusing high density points towards boundary layer singularities (e.g. y -> 1.0).
    Builds discrete variable-step negative Laplacian matrix G for non-uniform grids.
    """

    def __init__(self, stretch_gamma: float = 3.0):
        self.stretch_gamma = stretch_gamma

    def sample_points(
        self,
        n_points_x: int,
        n_points_t: int,
        length: float = 1.0,
        total_time: float = 1.0,
        device: Any = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_raw = torch.linspace(0.0, length, steps=n_points_x, requires_grad=True)

        hat_t = torch.linspace(0.0, 1.0, steps=n_points_t)
        gamma = self.stretch_gamma
        t_stretched = total_time * (1.0 - (torch.tanh(gamma * (1.0 - hat_t)) / math.tanh(gamma)))
        t_raw = t_stretched.detach().clone().requires_grad_(True)

        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
        x_grid = grids[0].flatten().reshape(-1, 1).to(device)
        t_grid = grids[1].flatten().reshape(-1, 1).to(device)

        return x_grid, t_grid

    def build_gram_matrix(
        self,
        n_points_x: int,
        n_points_t: int,
        length: float = 1.0,
        total_time: float = 1.0,
        device: Any = "cpu"
    ) -> Optional[Any]:
        x_raw = torch.linspace(0.0, length, steps=n_points_x)
        hat_t = torch.linspace(0.0, 1.0, steps=n_points_t)
        gamma = self.stretch_gamma
        t_raw = total_time * (1.0 - (torch.tanh(gamma * (1.0 - hat_t)) / math.tanh(gamma)))

        N_total = n_points_x * n_points_t
        G = torch.eye(N_total, dtype=torch.float32)

        def linearized(ix, iy):
            return ix * n_points_t + iy

        for ix in range(1, n_points_x - 1):
            hx_prev = float(x_raw[ix] - x_raw[ix - 1])
            hx_next = float(x_raw[ix + 1] - x_raw[ix])

            for iy in range(1, n_points_t - 1):
                ht_prev = float(t_raw[iy] - t_raw[iy - 1])
                ht_next = float(t_raw[iy + 1] - t_raw[iy])

                i = linearized(ix, iy)

                # Variable-step 5-point discrete negative Laplacian stencil
                G[i, i] = 2.0 / (hx_prev * hx_next) + 2.0 / (ht_prev * ht_next)
                G[i, linearized(ix - 1, iy)] = -2.0 / (hx_prev * (hx_prev + hx_next))
                G[i, linearized(ix + 1, iy)] = -2.0 / (hx_next * (hx_prev + hx_next))
                G[i, linearized(ix, iy - 1)] = -2.0 / (ht_prev * (ht_prev + ht_next))
                G[i, linearized(ix, iy + 1)] = -2.0 / (ht_next * (ht_prev + ht_next))

                # Weighting factor cell area
                w_i = 0.25 * (hx_prev + hx_next) * (ht_prev + ht_next)
                G[i, :] = G[i, :] / w_i

        G = G.to(device)
        return torch.linalg.lu_factor(G)
