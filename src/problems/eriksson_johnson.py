import math
from typing import Any
import numpy as np
import torch

from .base import BasePDEProblem
from .poisson_sine import _df

class ErikssonJohnsonProblem(BasePDEProblem):
    """
    Example 3: 2D Eriksson-Johnson Advection-Diffusion Problem with Boundary Layer
    Governing PDE: u_t - epsilon * (u_xx + u_yy) = 0
    Employs a hard boundary shift S(x, t) = sin(pi * x) * (1 - t)
    """

    def __init__(self, epsilon: float = 0.01):
        super().__init__(epsilon)
        self.advection_velocity = (0.0, 1.0)

    def _get_roots(self):
        eps = self.epsilon
        r1 = (1.0 + math.sqrt(1.0 + 4.0 * eps * eps * math.pi * math.pi)) / (2.0 * eps)
        r2 = (1.0 - math.sqrt(1.0 + 4.0 * eps * eps * math.pi * math.pi)) / (2.0 * eps)
        denom = math.exp(-r1) - math.exp(-r2)
        return r1, r2, denom

    def exact_solution(self, x: Any, y: Any) -> Any:
        r1, r2, denom = self._get_roots()
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return (exp(r1 * (y - 1.0)) - exp(r2 * (y - 1.0))) / denom * sin(pi * x)

    def exact_dx(self, x: Any, y: Any) -> Any:
        r1, r2, denom = self._get_roots()
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return (exp(r1 * (y - 1.0)) - exp(r2 * (y - 1.0))) / denom * pi * cos(pi * x)

    def exact_dy(self, x: Any, y: Any) -> Any:
        r1, r2, denom = self._get_roots()
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return (r1 * exp(r1 * (y - 1.0)) - r2 * exp(r2 * (y - 1.0))) / denom * sin(pi * x)

    def exact_dx2(self, x: Any, y: Any) -> Any:
        r1, r2, denom = self._get_roots()
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return (exp(r1 * (y - 1.0)) - exp(r2 * (y - 1.0))) / denom * (-pi * pi * sin(pi * x))

    def exact_dy2(self, x: Any, y: Any) -> Any:
        r1, r2, denom = self._get_roots()
        exp = torch.exp if isinstance(x, torch.Tensor) else np.exp
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return (r1 * r1 * exp(r1 * (y - 1.0)) - r2 * r2 * exp(r2 * (y - 1.0))) / denom * sin(pi * x)

    def shift_function(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return sin(pi * x) * (1.0 - y)

    def shift_dx(self, x: Any, y: Any) -> Any:
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return pi * cos(pi * x) * (1.0 - y)

    def shift_dy(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return -sin(pi * x)

    def shift_dx2(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi
        return -pi * pi * sin(pi * x) * (1.0 - y)

    def shift_dy2(self, x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        else:
            return np.zeros_like(x)

    def rhs(self, x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        else:
            return np.zeros_like(x)

    def compute_strong_residual(self, model: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u_pred = model(x, y)
        u_y = _df(u_pred, y, order=1)
        u_yy = _df(u_y, y, order=1)
        u_xx = _df(u_pred, x, order=2)
        
        # Add shift terms
        shift_y = self.shift_dy(x, y)
        shift_yy = self.shift_dy2(x, y)
        shift_xx = self.shift_dx2(x, y)

        res = (u_y - self.epsilon * u_yy - self.epsilon * u_xx) + (shift_y - self.epsilon * shift_yy - self.epsilon * shift_xx)
        return res
