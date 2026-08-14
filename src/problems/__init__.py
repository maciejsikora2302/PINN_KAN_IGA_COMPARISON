from .base import BasePDEProblem
from .eriksson_johnson import ErikssonJohnsonProblem
from .factory import get_problem
from .poisson_exp import PoissonExpProblem
from .poisson_sine import PoissonSineProblem

__all__ = [
    "BasePDEProblem",
    "ErikssonJohnsonProblem",
    "PoissonExpProblem",
    "PoissonSineProblem",
    "get_problem",
]
