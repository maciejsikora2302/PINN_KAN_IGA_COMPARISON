import os
import math
import time
import numpy as np
from typing import Any, Dict

from model import ExperimentInterface, SolverMetrics, SolverOutcome
from config import IGAConfig
from src.problems import get_problem, BasePDEProblem
from .base import IGASolution
from .standard import StandardIGASolver
from .supg import SUPGIGASolver
from .igrm import ResidualMinimizationIGASolver

class IGAExperiment(ExperimentInterface):
    """
    IGA Experiment wrapper implementing ExperimentInterface.
    Dispatches simulation execution to specific IGA solvers (standard, supg, igrm)
    based on YAML configuration parameters.
    """

    def __init__(self, config_path: str | None = None, optimized: bool | None = None):
        self.sol_coeffs: np.ndarray | None = None
        self.problem: BasePDEProblem | None = None
        self.loss_history: list[float] = []
        self.h1_error_history: list[float] = []
        self.h1_time_history: list[float] = []
        self.h1_epoch_history: list[int] = []
        self.h1_progress_history: list[float] = []
        self.final_loss: float | None = None
        self.final_interior_loss: float | None = None
        self.x_grid: np.ndarray | None = None
        self.t_grid: np.ndarray | None = None
        self.z_pred: np.ndarray | None = None
        self.final_h1_error: float | None = None
        self.final_l2_error: float | None = None
        self.final_linf_error: float | None = None
        self.elapsed_seconds: float = 0.0
        self.optimized: bool | None = optimized
        super().__init__(config_path)

    def load_config(self, config_path: str) -> None:
        self.config = IGAConfig()
        self.config.load_config(config_path)
        if self.optimized is not None:
            self.config.OPTIMIZED = self.optimized
        prob_name = getattr(self.config, "PROBLEM_NAME", None) or self.config.EXAMPLE
        self.problem = get_problem(prob_name, self.config.EPSILON)

    def train(self) -> None:
        if not self.config or not self.problem:
            raise ValueError("Configuration or PDE problem has not been loaded.")

        method = str(self.config.IGA_METHOD).lower()
        mesh_type = str(self.config.IGA_MESH_TYPE).lower()

        print(f"=== Starting IGA Solver Run (Method: {method}, Mesh: {mesh_type}) ===")
        start_time = time.perf_counter()

        if method == "supg":
            solver = SUPGIGASolver()
        elif method == "igrm":
            solver = ResidualMinimizationIGASolver()
        else:
            solver = StandardIGASolver()

        solution: IGASolution = solver.solve(
            problem=self.problem,
            p=self.config.IGA_DEGREE,
            M=self.config.IGA_ELEMENTS,
            mesh_type=mesh_type,
            gamma=getattr(self.config, "IGA_ADAPTIVE_GAMMA", 3.0),
            n_points_x=self.config.N_POINTS_X,
            n_points_t=self.config.N_POINTS_T,
            test_degree_enrichment=getattr(self.config, "IGA_TEST_DEGREE_ENRICHMENT", 1),
            optimized=getattr(self.config, "OPTIMIZED", False)
        )

        self.sol_coeffs = solution.sol_coeffs
        self.x_grid = solution.x_grid
        self.t_grid = solution.t_grid
        self.z_pred = solution.z_pred

        self.final_h1_error = solution.h1_error
        self.final_l2_error = solution.l2_error
        self.final_linf_error = solution.linf_error

        self.final_loss = solution.h1_error
        self.final_interior_loss = solution.l2_error

        self.elapsed_seconds = time.perf_counter() - start_time
        self.loss_history = [solution.l2_error] * self.config.EPOCHS
        self.h1_error_history = [solution.h1_error] * max(1, (self.config.EPOCHS // max(1, self.config.H1_CALC_EVERY)))
        self.h1_time_history = [self.elapsed_seconds] * len(self.h1_error_history)
        self.h1_epoch_history = [self.config.EPOCHS] * len(self.h1_error_history)
        self.h1_progress_history = [100.0] * len(self.h1_error_history)

        dofs = (self.config.IGA_ELEMENTS + self.config.IGA_DEGREE) ** 2
        metrics = SolverMetrics(
            final_loss=self.final_loss,
            final_interior_loss=self.final_interior_loss,
            final_h1_error=self.final_h1_error if self.final_h1_error is not None else 0.0,
            final_l2_error=self.final_l2_error if self.final_l2_error is not None else 0.0,
            final_linf_error=self.final_linf_error if self.final_linf_error is not None else 0.0,
            trainable_parameters_or_dofs=dofs,
            elapsed_seconds=self.elapsed_seconds,
            epochs_trained=self.config.EPOCHS,
            epochs_total=self.config.EPOCHS
        )

        iga_extra = {
            "knots_x": solution.knots_x,
            "knots_t": solution.knots_t,
        }
        if solution.dzdx_approx is not None:
            iga_extra["dzdx_approx"] = solution.dzdx_approx
        if solution.dzdt_approx is not None:
            iga_extra["dzdt_approx"] = solution.dzdt_approx

        self.outcome = SolverOutcome(
            x_grid=solution.x_grid.flatten(),
            t_grid=solution.t_grid.flatten(),
            z_pred=solution.z_pred.flatten(),
            loss_history=self.loss_history,
            h1_error_history=self.h1_error_history,
            h1_time_history=self.h1_time_history,
            h1_epoch_history=self.h1_epoch_history,
            h1_progress_history=self.h1_progress_history,
            metrics=metrics,
            extra_data=iga_extra
        )

        print(f"IGA completed in {self.elapsed_seconds:.3f}s. H1 Error: {solution.h1_error:.6e} | L2 Error: {solution.l2_error:.6e} | Linf Error: {solution.linf_error:.6e}")
        return self.outcome

    def save_model(self, path: str) -> None:
        if self.sol_coeffs is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.sol_coeffs)
        print(f"IGA solver coefficients saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        if self.sol_coeffs is None or self.x_grid is None or self.t_grid is None or self.z_pred is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            final_loss=np.array(self.final_loss if self.final_loss is not None else 0.0),
            final_interior_loss=np.array(self.final_interior_loss if self.final_interior_loss is not None else 0.0),
            loss_history=np.array(self.loss_history),
            h1_error_history=np.array(self.h1_error_history),
            h1_time_history=np.array(self.h1_time_history if self.h1_time_history else [self.elapsed_seconds]),
            h1_epoch_history=np.array(self.h1_epoch_history if self.h1_epoch_history else [self.config.EPOCHS if self.config else 1]),
            h1_progress_history=np.array(self.h1_progress_history if self.h1_progress_history else [100.0]),
            elapsed_seconds=np.array(self.elapsed_seconds),
            final_h1_error=np.array(self.final_h1_error if self.final_h1_error is not None else 0.0),
            final_l2_error=np.array(self.final_l2_error if self.final_l2_error is not None else 0.0),
            final_linf_error=np.array(self.final_linf_error if self.final_linf_error is not None else 0.0),
            x=self.x_grid.flatten(),
            t=self.t_grid.flatten(),
            z_pred=self.z_pred
        )
        print(f"IGA Outcomes saved successfully to {path}")
