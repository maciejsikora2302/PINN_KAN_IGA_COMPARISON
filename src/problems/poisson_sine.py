from typing import Any
import numpy as np
import torch

from .base import BasePDEProblem

def _df(output: torch.Tensor, input_tensor: torch.Tensor, order: int = 1) -> torch.Tensor:
    df_val = output
    for _ in range(order):
        df_val = torch.autograd.grad(
            df_val,
            input_tensor,
            grad_outputs=torch.ones_like(input_tensor),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_val

class PoissonSineProblem(BasePDEProblem):
    """
    Example 1: 2D Poisson Equation on [0, 1]^2
    Exact solution: u(x, y) = sin(2*pi*x) * sin(2*pi*y)
    PDE: -Delta u = f  =>  u_xx + u_yy + f = 0
    """

    def __init__(self, epsilon: float = 1.0):
        super().__init__(epsilon=1.0)
        self.advection_velocity = (0.0, 0.0)

    def exact_solution(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        return sin(2.0 * pi * x) * sin(2.0 * pi * y)

    def exact_dx(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        return 2.0 * pi * cos(2.0 * pi * x) * sin(2.0 * pi * y)

    def exact_dy(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        cos = torch.cos if isinstance(x, torch.Tensor) else np.cos
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        return 2.0 * pi * sin(2.0 * pi * x) * cos(2.0 * pi * y)

    def exact_dx2(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        return -4.0 * pi * pi * sin(2.0 * pi * x) * sin(2.0 * pi * y)

    def exact_dy2(self, x: Any, y: Any) -> Any:
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        return -4.0 * pi * pi * sin(2.0 * pi * x) * sin(2.0 * pi * y)

    def rhs(self, x: Any, y: Any) -> Any:
        """f = - (u_xx + u_yy) = 8 * pi^2 * sin(2*pi*x) * sin(2*pi*y)"""
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        pi = torch.pi if isinstance(x, torch.Tensor) else np.pi

        return 8.0 * pi * pi * sin(2.0 * pi * x) * sin(2.0 * pi * y)

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

        return u_yy + u_xx + self.rhs(x, y)
