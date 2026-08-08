from .base import BasePDEProblem
from .poisson_sine import PoissonSineProblem
from .poisson_exp import PoissonExpProblem
from .eriksson_johnson import ErikssonJohnsonProblem

def get_problem(example_id: int, epsilon: float = 0.01) -> BasePDEProblem:
    """
    Factory function to retrieve the appropriate PDEProblem instance by example ID.
    """
    if example_id == 1:
        return PoissonSineProblem(epsilon=epsilon)
    elif example_id == 2:
        return PoissonExpProblem(epsilon=epsilon)
    elif example_id == 3:
        return ErikssonJohnsonProblem(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown PDE Problem Example index: {example_id}")
