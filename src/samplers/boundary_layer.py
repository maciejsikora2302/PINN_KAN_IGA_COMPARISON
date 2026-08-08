import math
from typing import Tuple, Any, Optional
import torch

from .base import CollocationSampler

class BoundaryLayerSampler(CollocationSampler):
    """
    Stretched / Shishkin / Geometrically clustered collocation point sampler
    focusing high density points towards boundary layer singularities (e.g. y -> 1.0).
    """

    def __init__(self, stretch_gamma: float = 3.0):
        self.stretch_gamma = stretch_gamma

    def sample_points(self, n_points_x: int, n_points_t: int, length: float = 1.0, total_time: float = 1.0, device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        # Uniform x distribution
        x_raw = torch.linspace(0.0, length, steps=n_points_x, requires_grad=True)

        # Stretched t/y distribution towards 1.0 using tanh transformation
        hat_t = torch.linspace(0.0, 1.0, steps=n_points_t)
        gamma = self.stretch_gamma
        t_stretched = total_time * (1.0 - torch.tanh(gamma * (1.0 - hat_t)) / math.tanh(gamma))
        t_raw = t_stretched.detach().clone().requires_grad_(True)

        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
        x_grid = grids[0].flatten().reshape(-1, 1).to(device)
        t_grid = grids[1].flatten().reshape(-1, 1).to(device)

        return x_grid, t_grid

    def build_gram_matrix(self, n_points_x: int, n_points_t: int, device: Any = "cpu") -> Optional[Any]:
        # Non-uniform Gram matrix builder (placeholder for non-uniform stencils)
        # Defaults to uniform grid Gram matrix approximation until full metric tensor is built
        from .uniform_grid import UniformGridSampler
        return UniformGridSampler().build_gram_matrix(n_points_x, n_points_t, device=device)
