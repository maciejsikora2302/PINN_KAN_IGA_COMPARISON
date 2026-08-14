"""
Visualization module for PINN, KAN, and IGA benchmarks.
Provides publication-quality diagnostic plotting routines and multi-problem comparison utilities.
"""

from .global_comparison import GlobalComparator
from .problem_comparison import ProblemComparator
from .run_visualizer import (
    plot_1d_slices,
    plot_2d_error,
    plot_2d_solution,
    plot_h1_convergence,
    plot_loss_curve,
)

__all__ = [
    "GlobalComparator",
    "ProblemComparator",
    "plot_1d_slices",
    "plot_2d_error",
    "plot_2d_solution",
    "plot_h1_convergence",
    "plot_loss_curve",
]
