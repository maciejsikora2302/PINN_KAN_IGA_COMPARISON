"""
Visualization module for PINN, KAN, and IGA benchmarks.
Provides publication-quality diagnostic plotting routines and multi-problem comparison utilities.
"""

from .run_visualizer import (
    plot_loss_curve,
    plot_h1_convergence,
    plot_2d_solution,
    plot_2d_error,
    plot_1d_slices,
)
from .problem_comparison import ProblemComparator
from .global_comparison import GlobalComparator

__all__ = [
    "plot_loss_curve",
    "plot_h1_convergence",
    "plot_2d_solution",
    "plot_2d_error",
    "plot_1d_slices",
    "ProblemComparator",
    "GlobalComparator",
]
