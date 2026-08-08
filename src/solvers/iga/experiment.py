import os
import math
import numpy as np
from typing import Any, Dict

from model import ExperimentInterface
from config import IGAConfig
from src.problems import get_problem, BasePDEProblem
from .standard import StandardIGASolver
from .supg import SUPGIGASolver
from .igrm import ResidualMinimizationIGASolver

class IGAExperiment(ExperimentInterface):
    """
    IGA Experiment wrapper implementing ExperimentInterface.
    Dispatches simulation execution to specific IGA solvers (standard, supg, igrm)
    based on YAML configuration parameters.
    """

    def __init__(self, config_path: str | None = None):
        self.sol_coeffs = None
        self.problem: BasePDEProblem = None
        self.loss_history = []
        self.h1_error_history = []
        self.final_loss = None
        self.final_interior_loss = None
        self.x_grid = None
        self.t_grid = None
        self.z_pred = None
        self.final_h1_error = None
        self.final_l2_error = None
        self.final_linf_error = None
        super().__init__(config_path)

    def load_config(self, config_path: str) -> None:
        self.config = IGAConfig()
        self.config.load_config(config_path)
        self.config.validate_config()
        self.problem = get_problem(self.config.EXAMPLE, self.config.EPSILON)

    def train(self) -> None:
        if not self.config or not self.problem:
            raise ValueError("Configuration or PDE problem has not been loaded.")

        method = str(self.config.IGA_METHOD).lower()
        mesh_type = str(self.config.IGA_MESH_TYPE).lower()

        print(f"=== Starting IGA Solver Run (Method: {method}, Mesh: {mesh_type}) ===")

        if method == "supg":
            solver = SUPGIGASolver()
        elif method == "igrm":
            solver = ResidualMinimizationIGASolver()
        else:
            solver = StandardIGASolver()

        sol_coeffs, knots_x, knots_t, x_grid, t_grid, z_pred, metrics = solver.solve(
            problem=self.problem,
            p=self.config.IGA_DEGREE,
            M=self.config.IGA_ELEMENTS,
            mesh_type=mesh_type,
            gamma=getattr(self.config, "IGA_ADAPTIVE_GAMMA", 3.0),
            n_points_x=self.config.N_POINTS_X,
            n_points_t=self.config.N_POINTS_T,
            test_degree_enrichment=getattr(self.config, "IGA_TEST_DEGREE_ENRICHMENT", 1)
        )

        self.sol_coeffs = sol_coeffs
        self.x_grid = x_grid
        self.t_grid = t_grid
        self.z_pred = z_pred

        self.final_h1_error = metrics["h1_error"]
        self.final_l2_error = metrics["l2_error"]
        self.final_linf_error = metrics["linf_error"]

        self.final_loss = metrics["h1_error"]
        self.final_interior_loss = metrics["l2_error"]

        self.loss_history = [metrics["l2_error"]] * self.config.EPOCHS
        self.h1_error_history = [metrics["h1_error"]] * (self.config.EPOCHS // max(1, self.config.H1_CALC_EVERY))

        print(f"IGA completed. H1 Error: {metrics['h1_error']:.6e} | L2 Error: {metrics['l2_error']:.6e} | Linf Error: {metrics['linf_error']:.6e}")

    def save_model(self, path: str) -> None:
        if self.sol_coeffs is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.sol_coeffs)
        print(f"IGA solver coefficients saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        if self.sol_coeffs is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            final_loss=np.array(self.final_loss),
            final_interior_loss=np.array(self.final_interior_loss),
            loss_history=np.array(self.loss_history),
            h1_error_history=np.array(self.h1_error_history),
            final_h1_error=np.array(self.final_h1_error if self.final_h1_error is not None else 0.0),
            final_l2_error=np.array(self.final_l2_error if self.final_l2_error is not None else 0.0),
            final_linf_error=np.array(self.final_linf_error if self.final_linf_error is not None else 0.0),
            x=self.x_grid.flatten(),
            t=self.t_grid.flatten(),
            z_pred=self.z_pred
        )
        print(f"IGA Outcomes saved successfully to {path}")
