from typing import Any

import torch

from .base import CollocationSampler


class UniformGridSampler(CollocationSampler):
    """
    Uniform Cartesian tensor-product meshgrid sampler.
    Also builds standard 5-point discrete negative Laplacian matrix G for RPINN.
    """

    def sample_points(self, n_points_x: int, n_points_t: int, length: float = 1.0, total_time: float = 1.0, device: Any = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        x_domain = [0.0, length]
        t_domain = [0.0, total_time]

        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x, requires_grad=True)
        t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_t, requires_grad=True)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        x_grid = grids[0].flatten().reshape(-1, 1).to(device)
        t_grid = grids[1].flatten().reshape(-1, 1).to(device)

        return x_grid, t_grid

    def build_gram_matrix(self, n_points_x: int, n_points_t: int, length: float = 1.0, total_time: float = 1.0, device: Any = "cpu") -> Any | None:
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

    def build_gram_solver(self, n_points_x: int, n_points_t: int, length: float = 1.0, total_time: float = 1.0, device: Any = "cpu", optimized: bool = False) -> Any:
        if not optimized:
            G_LU = self.build_gram_matrix(n_points_x, n_points_t, length=length, total_time=total_time, device=device)
            return lambda loss: torch.linalg.lu_solve(*G_LU, loss.reshape(-1, 1))

        # Fast GPU/CPU Spectral Kronecker/DST Poisson solver for 5-point discrete Laplacian
        mx = n_points_x - 2
        mt = n_points_t - 2
        hx = 1.0 / n_points_x
        hy = 1.0 / n_points_t
        scale = hx * hy

        # Compute 1D eigenvectors and eigenvalues
        ix = torch.arange(1, mx + 1, device=device, dtype=torch.float32).unsqueeze(1)
        kx = torch.arange(1, mx + 1, device=device, dtype=torch.float32).unsqueeze(0)
        Qx = torch.sqrt(torch.tensor(2.0 / (mx + 1), device=device)) * torch.sin((ix * kx * torch.pi) / (mx + 1))
        lambdax = 4.0 * torch.sin((kx * torch.pi) / (2.0 * (mx + 1))) ** 2

        it = torch.arange(1, mt + 1, device=device, dtype=torch.float32).unsqueeze(1)
        kt = torch.arange(1, mt + 1, device=device, dtype=torch.float32).unsqueeze(0)
        Qt = torch.sqrt(torch.tensor(2.0 / (mt + 1), device=device)) * torch.sin((it * kt * torch.pi) / (mt + 1))
        lambdat = 4.0 * torch.sin((kt * torch.pi) / (2.0 * (mt + 1))) ** 2

        Lambda = lambdax.t() + lambdat  # (mx, mt)

        def fast_uniform_gram_solve(loss: torch.Tensor) -> torch.Tensor:
            dtype = loss.dtype
            loss_2d = loss.reshape(n_points_x, n_points_t).to(torch.float32)
            sol_2d = loss_2d * scale

            R_int = loss_2d[1:-1, 1:-1]
            R_hat = Qx @ R_int @ Qt
            V_hat = R_hat / Lambda
            V_int = Qx @ V_hat @ Qt

            sol_2d[1:-1, 1:-1] = V_int * scale
            return sol_2d.reshape(-1, 1).to(dtype)

        return fast_uniform_gram_solve
