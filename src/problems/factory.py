from typing import Union
from .base import BasePDEProblem
from .poisson_sine import PoissonSineProblem
from .poisson_exp import PoissonExpProblem
from .eriksson_johnson import ErikssonJohnsonProblem

def get_problem(problem_spec: Union[str, int], epsilon: float = 0.01) -> BasePDEProblem:
    """
    Factory function to retrieve the appropriate PDEProblem instance by name or integer ID.
    Supported names: 'poisson_sine', 'poisson_exp', 'eriksson_johnson'
    """
    if isinstance(problem_spec, str):
        key = problem_spec.lower().strip()
        if "poisson_sin" in key or key == "1":
            return PoissonSineProblem(epsilon=epsilon)
        elif "poisson_exp" in key or key == "2":
            return PoissonExpProblem(epsilon=epsilon)
        elif "eriksson" in key or "johnson" in key or key == "3":
            return ErikssonJohnsonProblem(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown PDE Problem name: '{problem_spec}'. Supported: 'poisson_sine', 'poisson_exp', 'eriksson_johnson'.")
    elif isinstance(problem_spec, int):
        if problem_spec == 1:
            return PoissonSineProblem(epsilon=epsilon)
        elif problem_spec == 2:
            return PoissonExpProblem(epsilon=epsilon)
        elif problem_spec == 3:
            return ErikssonJohnsonProblem(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown PDE Problem index: {problem_spec}")
    else:
        raise TypeError(f"problem_spec must be str or int, got {type(problem_spec).__name__}")
