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

        return -exp(pi * (x - 2.0 * y)) * sin(2.0 * pi * x) * sin(pi * y)

    def exact_dx(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        exp1 = -pi * exp(pi * (x - 2.0 * y)) * sin(pi * y)
        sin1 = sin(2.0 * pi * x) + 2.0 * cos(2.0 * pi * x)
        return exp1 * sin1

    def exact_dy(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        exp1 = -pi * exp(pi * (x - 2.0 * y)) * sin(2.0 * pi * x)
        sin1 = cos(pi * y) - 2.0 * sin(pi * y)
        return exp1 * sin1

    def exact_dx2(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        exp1 = -pi * pi * exp(pi * (x - 2.0 * y)) * sin(pi * y)
        sin1 = 4.0 * cos(2.0 * pi * x) - 3.0 * sin(2.0 * pi * x)
        return exp1 * sin1

    def exact_dy2(self, x: Any, y: Any) -> Any:
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        exp1 = pi * pi * exp(pi * (x - 2.0 * y)) * sin(2.0 * pi * x)
        sin1 = 4.0 * cos(pi * y) - 3.0 * sin(pi * y)
        return exp1 * sin1

    def rhs(self, x: Any, y: Any) -> Any:
        f1 = self.exact_dx2(x, y)
        f2 = self.exact_dy2(x, y)
        return -(f1 + f2)

    def compute_strong_residual(self, model: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u_pred = model(x, y)
        u_yy = _df(u_pred, y, order=2)
        u_xx = _df(u_pred, x, order=2)
        rhs_val = self.rhs(x, y)
        return u_yy + u_xx + rhs_val
