from typing import Any

import numpy as np
import torch

from .base import BasePDEProblem
from .poisson_sine import _df


class PoissonExpProblem(BasePDEProblem):
    """
    Example 2: 2D Poisson Equation with Exponential-Sine exact solution.
    Exact solution: u(x, y) = -exp(pi*(x - 2y)) * sin(2*pi*x) * sin(pi*y)
    """

    def __init__(self, epsilon: float = 1.0):
        super().__init__(epsilon=1.0)
        self.advection_velocity = (0.0, 0.0)

    def exact_solution(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        return sin(2.0 * pi * x) * sin(2.0 * pi * y) * exp(pi * (x - 2.0 * y))

    def exact_dx(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        E = exp(pi * (x - 2.0 * y))
        Sx = sin(2.0 * pi * x)
        Cx = cos(2.0 * pi * x)
        Sy = sin(2.0 * pi * y)

        return pi * E * Sy * (2.0 * Cx + Sx)

    def exact_dy(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        E = exp(pi * (x - 2.0 * y))
        Sx = sin(2.0 * pi * x)
        Sy = sin(2.0 * pi * y)
        Cy = cos(2.0 * pi * y)

        return 2.0 * pi * E * Sx * (Cy - Sy)

    def exact_dx2(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        E = exp(pi * (x - 2.0 * y))
        Sx = sin(2.0 * pi * x)
        Cx = cos(2.0 * pi * x)
        Sy = sin(2.0 * pi * y)

        return pi * pi * E * Sy * (4.0 * Cx - 3.0 * Sx)

    def exact_dy2(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        E = exp(pi * (x - 2.0 * y))
        Sx = sin(2.0 * pi * x)
        Cy = cos(2.0 * pi * y)

        return -8.0 * pi * pi * E * Sx * Cy

    def rhs(self, x: Any, y: Any) -> Any:
        # f(x,y) = pi^2 * E * [3*Sx*Sy - 4*Cx*Sy + 8*Sx*Cy]
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        E = exp(pi * (x - 2.0 * y))
        Sx = sin(2.0 * pi * x)
        Cx = cos(2.0 * pi * x)
        Sy = sin(2.0 * pi * y)
        Cy = cos(2.0 * pi * y)

        return pi * pi * E * (3.0 * Sx * Sy - 4.0 * Cx * Sy + 8.0 * Sx * Cy)

    def compute_strong_residual(self, model: Any, x: torch.Tensor, y: torch.Tensor, optimized: bool = False) -> torch.Tensor:
        u_pred = model(x, y)
        if optimized:
            grads = torch.autograd.grad(
                u_pred,
                (x, y),
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True,
                retain_graph=True,
            )
            u_x, u_y = grads[0], grads[1]
            u_xx = torch.autograd.grad(
                u_x,
                x,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
                retain_graph=True,
            )[0]
            u_yy = torch.autograd.grad(
                u_y,
                y,
                grad_outputs=torch.ones_like(u_y),
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            u_yy = _df(u_pred, y, order=2)
            u_xx = _df(u_pred, x, order=2)
        rhs_val = self.rhs(x, y)
        return u_yy + u_xx + rhs_val
