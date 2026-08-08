from typing import Tuple, Any, Optional
import torch

from .base import CollocationSampler

class UniformGridSampler(CollocationSampler):
    """
    Uniform Cartesian tensor-product meshgrid sampler.
    Also builds standard 5-point discrete negative Laplacian matrix G for RPINN.
    """

    def sample_points(self, n_points_x: int, n_points_t: int, length: float = 1.0, total_time: float = 1.0, device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        x_domain = [0.0, length]
        t_domain = [0.0, total_time]

        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x, requires_grad=True)
        t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_t, requires_grad=True)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        x_grid = grids[0].flatten().reshape(-1, 1).to(device)
        t_grid = grids[1].flatten().reshape(-1, 1).to(device)

        return x_grid, t_grid

    def build_gram_matrix(self, n_points_x: int, n_points_t: int, device: Any = "cpu") -> Optional[Any]:
        G = torch.eye(n_points_x * n_points_t)

        def linearized(ix, iy):
            return ix * n_points_t + iy

        def nearby(ix, iy):
            return [(ix + 1, iy), (ix - 1, iy), (ix, iy - 1), (ix, iy + 1)]

        for ix in range(n_points_x):
            for iy in range(n_points_t):
                i = linearized(ix, iy)
                G[i, i] = 1

        for ix in range(1, n_points_x - 1):
            for iy in range(1, n_points_t - 1):
                i = linearized(ix, iy)
                G[i, i] = 4
                for jx, jy in nearby(ix, iy):
                    j = linearized(jx, jy)
                    G[i, j] = -1

        hx = 1.0 / n_points_x
        hy = 1.0 / n_points_t
        G = G / (hx * hy)
        G = G.to(device)
        return torch.linalg.lu_factor(G)
