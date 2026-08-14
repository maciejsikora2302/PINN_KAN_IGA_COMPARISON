import os
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Any, Optional

from model import ExperimentInterface, SolverMetrics, SolverOutcome
from config import IGAConfig
from src.problems import get_problem, BasePDEProblem

from functools import lru_cache

@lru_cache(maxsize=10240)
def bspline_basis_single_cached(i: int, p: int, knots: tuple, x: float) -> float:
    if p == 0:
        if i == len(knots) - 2:
            return 1.0 if (knots[i] <= x <= knots[i+1]) else 0.0
        else:
            return 1.0 if (knots[i] <= x < knots[i+1]) else 0.0

    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]

    val1 = 0.0
    if denom1 > 0:
        val1 = (x - knots[i]) / denom1 * bspline_basis_single_cached(i, p-1, knots, x)

    val2 = 0.0
    if denom2 > 0:
        val2 = (knots[i+p+1] - x) / denom2 * bspline_basis_single_cached(i+1, p-1, knots, x)

    return val1 + val2

def bspline_basis_single(i: int, p: int, knots: np.ndarray, x: float) -> float:
    return bspline_basis_single_cached(i, p, tuple(knots), x)

@lru_cache(maxsize=10240)
def bspline_basis_deriv_single_cached(i: int, p: int, knots: tuple, x: float) -> float:
    if p == 0:
        return 0.0

    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]

    val1 = 0.0
    if denom1 > 0:
        val1 = p / denom1 * bspline_basis_single_cached(i, p-1, knots, x)

    val2 = 0.0
    if denom2 > 0:
        val2 = -p / denom2 * bspline_basis_single_cached(i+1, p-1, knots, x)

    return val1 + val2

def bspline_basis_deriv_single(i: int, p: int, knots: np.ndarray, x: float) -> float:
    return bspline_basis_deriv_single_cached(i, p, tuple(knots), x)

def bspline_basis(i: int, p: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda val: bspline_basis_single(i, p, knots, val))(x)

def bspline_basis_deriv(i: int, p: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda val: bspline_basis_deriv_single(i, p, knots, val))(x)

def open_knot_vector(p: int, M: int) -> np.ndarray:
    internal_knots = np.linspace(0, 1, M + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), internal_knots, np.ones(p + 1)])


class IGAExperiment(ExperimentInterface):
    """
    Refactored Isogeometric Analysis (IGA) Experiment solver.
    Decoupled from problem definitions via the PDEProblem strategy pattern.
    """

    def __init__(self, config_path: str | None = None):
        self.sol_coeffs = None
        self.problem: Optional[BasePDEProblem] = None
        self.loss_history = []
        self.h1_error_history = []
        self.final_loss = None
        self.final_interior_loss = None
        self.x_grid = None
        self.t_grid = None
        self.z_pred = None
        self.final_h1_error = None
        super().__init__(config_path)

    def load_config(self, config_path: str) -> None:
        self.config = IGAConfig()
        self.config.load_config(config_path)
        self.config.validate_config()
        example = self.config.EXAMPLE if self.config.EXAMPLE is not None else 1
        self.problem = get_problem(example, self.config.EPSILON)

    def train(self) -> SolverOutcome:
        if not self.config or not self.problem:
            raise ValueError("Configuration or PDE problem has not been loaded.")

        print("=== Starting training/solving for IGA ===")

        p = self.config.IGA_DEGREE
        M = self.config.IGA_ELEMENTS
        n = M + p

        knots_x = open_knot_vector(p, M)
        knots_t = open_knot_vector(p, M)

        gp_local, gw_local = np.polynomial.legendre.leggauss(p + 1)

        element_spans_x = []
        for i in range(len(knots_x) - 1):
            if knots_x[i] < knots_x[i+1]:
                element_spans_x.append((knots_x[i], knots_x[i+1], i))

        element_spans_t = []
        for j in range(len(knots_t) - 1):
            if knots_t[j] < knots_t[j+1]:
                element_spans_t.append((knots_t[j], knots_t[j+1], j))

        N_dofs = n * n
        K_data = []
        K_row = []
        K_col = []
        F = np.zeros(N_dofs)

        # Assemble Stiffness Matrix K and Load Vector F
        for ex_left, ex_right, idx_x in element_spans_x:
            det_J_x = (ex_right - ex_left) / 2.0
            gp_x = det_J_x * gp_local + (ex_right + ex_left) / 2.0
            gw_x = gw_local * det_J_x

            basis_x = np.zeros((p + 1, len(gp_x)))
            deriv_x = np.zeros((p + 1, len(gp_x)))
            for a in range(p + 1):
                basis_x[a, :] = bspline_basis(idx_x - p + a, p, knots_x, gp_x)
                deriv_x[a, :] = bspline_basis_deriv(idx_x - p + a, p, knots_x, gp_x)

            for et_left, et_right, idx_t in element_spans_t:
                det_J_t = (et_right - et_left) / 2.0
                gp_t = det_J_t * gp_local + (et_right + et_left) / 2.0
                gw_t = gw_local * det_J_t

                basis_t = np.zeros((p + 1, len(gp_t)))
                deriv_t = np.zeros((p + 1, len(gp_t)))
                for b in range(p + 1):
                    basis_t[b, :] = bspline_basis(idx_t - p + b, p, knots_t, gp_t)
                    deriv_t[b, :] = bspline_basis_deriv(idx_t - p + b, p, knots_t, gp_t)

                for ax in range(p + 1):
                    for ay in range(p + 1):
                        row_global = (idx_x - p + ax) * n + (idx_t - p + ay)
                        f_integral = 0.0

                        for gx in range(len(gp_x)):
                            for gt in range(len(gp_t)):
                                x_val = gp_x[gx]
                                t_val = gp_t[gt]
                                w = gw_x[gx] * gw_t[gt]
                                N_val = basis_x[ax, gx] * basis_t[ay, gt]
                                dN_dx = deriv_x[ax, gx] * basis_t[ay, gt]
                                dN_dt = basis_x[ax, gx] * deriv_t[ay, gt]

                                rhs_val = self.problem.rhs(x_val, t_val)
                                if self.config.EXAMPLE in [1, 2]:
                                    f_integral += rhs_val * N_val * w
                                elif self.config.EXAMPLE == 3:
                                    eps = self.config.EPSILON
                                    S_t = self.problem.shift_dy(x_val, t_val)
                                    S_x = self.problem.shift_dx(x_val, t_val)
                                    f_integral -= (S_t * N_val + eps * (S_x * dN_dx + S_t * dN_dt)) * w

                        F[row_global] += f_integral

                        for bx in range(p + 1):
                            for by in range(p + 1):
                                col_global = (idx_x - p + bx) * n + (idx_t - p + by)
                                k_val = 0.0
                                for gx in range(len(gp_x)):
                                    for gt in range(len(gp_t)):
                                        w = gw_x[gx] * gw_t[gt]
                                        dNa_dx = deriv_x[ax, gx] * basis_t[ay, gt]
                                        dNa_dt = basis_x[ax, gx] * deriv_t[ay, gt]
                                        Na_val = basis_x[ax, gx] * basis_t[ay, gt]

                                        dNb_dx = deriv_x[bx, gx] * basis_t[by, gt]
                                        dNb_dt = basis_x[bx, gx] * deriv_t[by, gt]
                                        Nb_val = basis_x[bx, gx] * basis_t[by, gt]

                                        if self.config.EXAMPLE in [1, 2]:
                                            k_val += (dNa_dx * dNb_dx + dNa_dt * dNb_dt) * w
                                        elif self.config.EXAMPLE == 3:
                                            eps = self.config.EPSILON
                                            k_val += (dNb_dt * Na_val + eps * (dNa_dx * dNb_dx + dNa_dt * dNb_dt)) * w

                                K_data.append(k_val)
                                K_row.append(row_global)
                                K_col.append(col_global)

        K_sparse = sp.coo_matrix((K_data, (K_row, K_col)), shape=(N_dofs, N_dofs)).tocsr()

        # Enforce Boundary Conditions
        is_boundary = np.zeros(N_dofs, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    is_boundary[i * n + j] = True

        K_bc = K_sparse.tolil()
        for i in range(N_dofs):
            if is_boundary[i]:
                K_bc.rows[i] = [i]
                K_bc.data[i] = [1.0]
                F[i] = 0.0

        K_bc = K_bc.tocsr()
        self.sol_coeffs = spla.spsolve(K_bc, F)

        # Evaluate at grid
        x_domain = [0.0, self.config.LENGTH]
        t_domain = [0.0, self.config.TOTAL_TIME]
        x_raw = np.linspace(x_domain[0], x_domain[1], num=self.config.N_POINTS_X)
        t_raw = np.linspace(t_domain[0], t_domain[1], num=self.config.N_POINTS_T)
        grid_x, grid_t = np.meshgrid(x_raw, t_raw, indexing="ij")
        self.x_grid = grid_x.flatten().reshape(-1, 1)
        self.t_grid = grid_t.flatten().reshape(-1, 1)

        self.z_pred = np.zeros_like(self.x_grid).flatten()
        for idx in range(len(self.x_grid)):
            x_coord = self.x_grid[idx, 0]
            t_coord = self.t_grid[idx, 0]

            span_idx_x = np.searchsorted(knots_x, x_coord) - 1
            span_idx_x = min(max(span_idx_x, p), len(knots_x) - p - 2)

            span_idx_t = np.searchsorted(knots_t, t_coord) - 1
            span_idx_t = min(max(span_idx_t, p), len(knots_t) - p - 2)

            val = 0.0
            for ax in range(p + 1):
                idx_basis_x = span_idx_x - p + ax
                Bx = bspline_basis_single(idx_basis_x, p, knots_x, x_coord)
                for ay in range(p + 1):
                    idx_basis_t = span_idx_t - p + ay
                    Bt = bspline_basis_single(idx_basis_t, p, knots_t, t_coord)
                    coeff = self.sol_coeffs[idx_basis_x * n + idx_basis_t]
                    val += coeff * Bx * Bt
            self.z_pred[idx] = val

        # Add shift function from PDEProblem
        z_shift = self.problem.shift_function(self.x_grid, self.t_grid).flatten()
        self.z_pred = self.z_pred + z_shift

        # Compute derivatives and H1 error
        dzdx_approx = np.zeros_like(self.z_pred)
        dzdt_approx = np.zeros_like(self.z_pred)
        for idx in range(len(self.x_grid)):
            x_coord = self.x_grid[idx, 0]
            t_coord = self.t_grid[idx, 0]

            span_idx_x = np.searchsorted(knots_x, x_coord) - 1
            span_idx_x = min(max(span_idx_x, p), len(knots_x) - p - 2)
            span_idx_t = np.searchsorted(knots_t, t_coord) - 1
            span_idx_t = min(max(span_idx_t, p), len(knots_t) - p - 2)

            val_dx = 0.0
            val_dt = 0.0
            for ax in range(p + 1):
                idx_basis_x = span_idx_x - p + ax
                Bx = bspline_basis_single(idx_basis_x, p, knots_x, x_coord)
                dBx = bspline_basis_deriv_single(idx_basis_x, p, knots_x, x_coord)
                for ay in range(p + 1):
                    idx_basis_t = span_idx_t - p + ay
                    Bt = bspline_basis_single(idx_basis_t, p, knots_t, t_coord)
                    dBt = bspline_basis_deriv_single(idx_basis_t, p, knots_t, t_coord)
                    coeff = self.sol_coeffs[idx_basis_x * n + idx_basis_t]
                    val_dx += coeff * dBx * Bt
                    val_dt += coeff * Bx * dBt
            dzdx_approx[idx] = val_dx
            dzdt_approx[idx] = val_dt

        dzdx_exact = self.problem.exact_dx(self.x_grid, self.t_grid).flatten()
        dzdt_exact = self.problem.exact_dy(self.x_grid, self.t_grid).flatten()

        dzdx = dzdx_exact - dzdx_approx - self.problem.shift_dx(self.x_grid, self.t_grid).flatten()
        dzdt = dzdt_exact - dzdt_approx - self.problem.shift_dy(self.x_grid, self.t_grid).flatten()

        N = self.config.N_POINTS_X * self.config.N_POINTS_T
        h1_error = math.sqrt((np.sum(dzdx ** 2) + np.sum(dzdt ** 2)) / N)

        system_res = K_bc.dot(self.sol_coeffs) - F
        final_l2_residual = float(np.sum(system_res ** 2))

        self.final_loss = final_l2_residual
        self.final_interior_loss = final_l2_residual
        self.final_h1_error = h1_error
        self.loss_history = [self.final_loss] * self.config.EPOCHS
        self.h1_error_history = [h1_error] * (self.config.EPOCHS // self.config.H1_CALC_EVERY)

        print(f"IGA completed solver. Final H1 Error: {h1_error:.6e}")

        metrics = SolverMetrics(
            final_loss=self.final_loss,
            final_interior_loss=self.final_interior_loss,
            final_h1_error=self.final_h1_error,
            final_l2_error=0.0,
            final_linf_error=0.0,
            trainable_parameters_or_dofs=n * n,
            elapsed_seconds=0.0,
            epochs_trained=self.config.EPOCHS,
            epochs_total=self.config.EPOCHS
        )
        self.outcome = SolverOutcome(
            x_grid=self.x_grid,
            t_grid=self.t_grid,
            z_pred=self.z_pred,
            loss_history=self.loss_history,
            h1_error_history=self.h1_error_history,
            h1_time_history=[0.0] * len(self.h1_error_history),
            h1_epoch_history=[i * self.config.H1_CALC_EVERY for i in range(len(self.h1_error_history))],
            h1_progress_history=[0.0] * len(self.h1_error_history),
            metrics=metrics
        )
        return self.outcome

    def save_model(self, path: str) -> None:
        if self.sol_coeffs is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.sol_coeffs)
        print(f"IGA solver coefficients saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        if self.sol_coeffs is None:
            raise ValueError("Model has not been trained yet.")
        if self.x_grid is None or self.t_grid is None or self.z_pred is None:
            raise ValueError("Evaluation outcomes have not been computed yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            final_loss=np.array(self.final_loss),
            final_interior_loss=np.array(self.final_interior_loss),
            loss_history=np.array(self.loss_history),
            h1_error_history=np.array(self.h1_error_history),
            x=self.x_grid.flatten(),
            t=self.t_grid.flatten(),
            z_pred=self.z_pred
        )
        print(f"IGA Outcomes saved successfully to {path}")
