from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np
import torch

class BasePDEProblem(ABC):
    """
    Abstract Base Class for Partial Differential Equation (PDE) benchmark problems.
    Encapsulates exact analytical solutions, derivatives, RHS forcing functions,
    shift functions for boundary layer transformations, and PDE residuals.
    """

    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        self.advection_velocity: Tuple[float, float] = (0.0, 1.0)

    @abstractmethod
    def exact_solution(self, x: Any, y: Any) -> Any:
        """Returns the exact analytical solution u(x, y)."""
        pass

    @abstractmethod
    def exact_dx(self, x: Any, y: Any) -> Any:
        """Returns first spatial derivative du/dx."""
        pass

    @abstractmethod
    def exact_dy(self, x: Any, y: Any) -> Any:
        """Returns first temporal/y derivative du/dy."""
        pass

    @abstractmethod
    def exact_dx2(self, x: Any, y: Any) -> Any:
        """Returns second spatial derivative d2u/dx2."""
        pass

    @abstractmethod
    def exact_dy2(self, x: Any, y: Any) -> Any:
        """Returns second temporal/y derivative d2u/dy2."""
        pass

    def shift_function(self, x: Any, y: Any) -> Any:
        """Returns shift function S(x, y) if boundary conditions use hard boundary shift."""
        zeros_like = torch.zeros_like if isinstance(x, torch.Tensor) else np.zeros_like
        return zeros_like(x)

    def shift_dx(self, x: Any, y: Any) -> Any:
        zeros_like = torch.zeros_like if isinstance(x, torch.Tensor) else np.zeros_like
        return zeros_like(x)

    def shift_dy(self, x: Any, y: Any) -> Any:
        zeros_like = torch.zeros_like if isinstance(x, torch.Tensor) else np.zeros_like
        return zeros_like(x)
        
    def shift_dx2(self, x: Any, y: Any) -> Any:
        zeros_like = torch.zeros_like if isinstance(x, torch.Tensor) else np.zeros_like
        return zeros_like(x)

    def shift_dy2(self, x: Any, y: Any) -> Any:
        zeros_like = torch.zeros_like if isinstance(x, torch.Tensor) else np.zeros_like
        return zeros_like(x)

    @abstractmethod
    def rhs(self, x: Any, y: Any) -> Any:
        """Returns RHS source term f(x, y)."""
        pass

    @abstractmethod
    def compute_strong_residual(self, model: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the strong form PDE residual L(x, y) for a neural network model.
        """
        pass
