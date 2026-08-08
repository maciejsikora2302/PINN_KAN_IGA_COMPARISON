from .base import BasePDEProblem
from .poisson_sine import PoissonSineProblem
from .poisson_exp import PoissonExpProblem
from .eriksson_johnson import ErikssonJohnsonProblem
from .factory import get_problem

__all__ = [
    "BasePDEProblem",
    "PoissonSineProblem",
    "PoissonExpProblem",
    "ErikssonJohnsonProblem",
    "get_problem",
]
