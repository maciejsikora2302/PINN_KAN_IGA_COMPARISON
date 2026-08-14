import math
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import scipy.sparse as sp

from dataclasses import dataclass
from src.problems.base import BasePDEProblem

@dataclass
class IGAMetrics:
    l2_error: float
    h1_error: float
    linf_error: float

@dataclass
class IGASolution:
    sol_coeffs: np.ndarray
    knots_x: np.ndarray
    knots_t: np.ndarray
    x_grid: np.ndarray
    t_grid: np.ndarray
    z_pred: np.ndarray
    metrics: IGAMetrics
    dzdx_approx: Optional[np.ndarray] = None
    dzdt_approx: Optional[np.ndarray] = None

    @property
    def h1_error(self) -> float:
        return self.metrics.h1_error

    @property
    def l2_error(self) -> float:
        return self.metrics.l2_error

    @property
    def linf_error(self) -> float:
        return self.metrics.linf_error

def bspline_basis_single(i: int, p: int, knots: np.ndarray, x: float) -> float:
    """Evaluates 1D B-spline basis function N_{i,p}(x) via Cox-de Boor recursion."""
    if p == 0:
        if i == len(knots) - 2:
            return 1.0 if (knots[i] <= x <= knots[i+1]) else 0.0
        else:
            return 1.0 if (knots[i] <= x < knots[i+1]) else 0.0

    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]

    val1 = 0.0
    if denom1 > 0:
        val1 = (x - knots[i]) / denom1 * bspline_basis_single(i, p-1, knots, x)

    val2 = 0.0
    if denom2 > 0:
        val2 = (knots[i+p+1] - x) / denom2 * bspline_basis_single(i+1, p-1, knots, x)

    return val1 + val2


def bspline_basis_deriv_single(i: int, p: int, knots: np.ndarray, x: float) -> float:
    """Evaluates 1st derivative of 1D B-spline basis function N'_{i,p}(x)."""
    if p == 0:
        return 0.0

    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]

    val1 = 0.0
    if denom1 > 0:
        val1 = p / denom1 * bspline_basis_single(i, p-1, knots, x)

    val2 = 0.0
    if denom2 > 0:
        val2 = -p / denom2 * bspline_basis_single(i+1, p-1, knots, x)

    return val1 + val2


def bspline_basis_deriv2_single(i: int, p: int, knots: np.ndarray, x: float) -> float:
    """Evaluates 2nd derivative of 1D B-spline basis function N''_{i,p}(x)."""
    if p <= 1:
        return 0.0

    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]

    val1 = 0.0
    if denom1 > 0:
        val1 = p / denom1 * bspline_basis_deriv_single(i, p-1, knots, x)

    val2 = 0.0
    if denom2 > 0:
        val2 = -p / denom2 * bspline_basis_deriv_single(i+1, p-1, knots, x)

    return val1 + val2


def bspline_basis(i: int, p: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda val: bspline_basis_single(i, p, knots, val))(x)


def bspline_basis_deriv(i: int, p: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda val: bspline_basis_deriv_single(i, p, knots, val))(x)


def bspline_basis_deriv2(i: int, p: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda val: bspline_basis_deriv2_single(i, p, knots, val))(x)


class BaseIGASolver(ABC):
    """
    Abstract Base Class for Isogeometric Analysis (IGA) 2D finite element solvers.
    Encapsulates knot vector creation, B-spline evaluations, Gauss quadrature,
    boundary condition enforcement, and error norm computations.
    """

    @staticmethod
    def open_uniform_knots(p: int, M: int) -> np.ndarray:
        """Generates open uniform knot vector on [0, 1]."""
        internal_knots = np.linspace(0, 1, M + 1)[1:-1]
        return np.concatenate([np.zeros(p + 1), internal_knots, np.ones(p + 1)])

    @staticmethod
    def open_graded_knots(p: int, M: int, gamma: float = 3.0) -> np.ndarray:
        """
        Generates open graded knot vector clustered towards 1.0 (boundary layer singularity).
        Formula: xi_j = 1 - (1 - hat_xi_j)^gamma
        """
        hat_knots = np.linspace(0, 1, M + 1)[1:-1]
        internal_knots = 1.0 - (1.0 - hat_knots) ** gamma
        return np.concatenate([np.zeros(p + 1), internal_knots, np.ones(p + 1)])

    @staticmethod
    def get_quadrature(p: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns p+1 Legendre-Gauss quadrature points and weights on [-1, 1]."""
        return np.polynomial.legendre.leggauss(p + 1)

    @staticmethod
    def get_element_spans(knots: np.ndarray) -> List[Tuple[float, float, int]]:
        """Extracts active non-empty element spans (left, right, knot_index)."""
        spans = []
        for i in range(len(knots) - 1):
            if knots[i] < knots[i+1]:
                spans.append((knots[i], knots[i+1], i))
        return spans

    def apply_dirichlet_bc(self, K_sparse: Any, F: np.ndarray, n_x: int, n_t: int) -> Tuple[Any, np.ndarray]:
        """Applies Dirichlet boundary conditions by setting boundary control points to 0."""
        N_dofs = n_x * n_t
        is_boundary = np.zeros(N_dofs, dtype=bool)

        for i in range(n_x):
            for j in range(n_t):
                if i == 0 or i == n_x - 1 or j == 0 or j == n_t - 1:
                    is_boundary[i * n_t + j] = True

        K_bc = K_sparse.tolil()
        for i in range(N_dofs):
            if is_boundary[i]:
                K_bc.rows[i] = [i]
                K_bc.data[i] = [1.0]
                F[i] = 0.0

        return K_bc.tocsr(), F

    def evaluate_solution_fast(
        self,
        sol_coeffs: np.ndarray,
        x_raw: np.ndarray,
        t_raw: np.ndarray,
        knots_x: np.ndarray,
        knots_t: np.ndarray,
        p_x: int,
        p_t: int,
        n_x: int,
        n_t: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast tensor-product B-spline evaluation: U = B_X C B_T^T.
        Computes 2D solution and derivatives in sub-millisecond vectorized matrix multiplications.
        """
        N_x = len(x_raw)
        N_t = len(t_raw)

        Bx = np.zeros((N_x, n_x))
        dBx = np.zeros((N_x, n_x))
        for i in range(n_x):
            Bx[:, i] = bspline_basis(i, p_x, knots_x, x_raw)
            dBx[:, i] = bspline_basis_deriv(i, p_x, knots_x, x_raw)

        Bt = np.zeros((N_t, n_t))
        dBt = np.zeros((N_t, n_t))
        for j in range(n_t):
            Bt[:, j] = bspline_basis(j, p_t, knots_t, t_raw)
            dBt[:, j] = bspline_basis_deriv(j, p_t, knots_t, t_raw)

        C = sol_coeffs.reshape(n_x, n_t)

        U_2d = Bx @ C @ Bt.T
        Ux_2d = dBx @ C @ Bt.T
        Ut_2d = Bx @ C @ dBt.T

        return U_2d.flatten(), Ux_2d.flatten(), Ut_2d.flatten()

    def evaluate_solution(
        self,
        sol_coeffs: np.ndarray,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        knots_x: np.ndarray,
        knots_t: np.ndarray,
        p_x: int,
        p_t: int,
        n_t: int,
        optimized: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates u_h(x, y), du_h/dx, du_h/dy at grid points.
        Returns (z_pred, dzdx_approx, dzdt_approx).
        """
        if optimized and x_grid.ndim == 2:
            n_x = len(knots_x) - p_x - 1
            x_unique = np.unique(x_grid[:, 0])
            t_unique = np.unique(t_grid[:, 0])
            if len(x_unique) * len(t_unique) == len(x_grid):
                return self.evaluate_solution_fast(
                    sol_coeffs, x_unique, t_unique, knots_x, knots_t, p_x, p_t, n_x, n_t
                )

        N_pts = len(x_grid)
        z_pred = np.zeros(N_pts)
        dzdx_approx = np.zeros(N_pts)
        dzdt_approx = np.zeros(N_pts)

        for idx in range(N_pts):
            x_coord = x_grid[idx, 0] if x_grid.ndim > 1 else x_grid[idx]
            t_coord = t_grid[idx, 0] if t_grid.ndim > 1 else t_grid[idx]

            span_idx_x = np.searchsorted(knots_x, x_coord, side='right') - 1
            span_idx_x = min(max(span_idx_x, p_x), len(knots_x) - p_x - 2)

            span_idx_t = np.searchsorted(knots_t, t_coord, side='right') - 1
            span_idx_t = min(max(span_idx_t, p_t), len(knots_t) - p_t - 2)

            val = 0.0
            val_dx = 0.0
            val_dt = 0.0

            for ax in range(p_x + 1):
                idx_basis_x = span_idx_x - p_x + ax
                Bx = bspline_basis_single(idx_basis_x, p_x, knots_x, x_coord)
                dBx = bspline_basis_deriv_single(idx_basis_x, p_x, knots_x, x_coord)

                for ay in range(p_t + 1):
                    idx_basis_t = span_idx_t - p_t + ay
                    Bt = bspline_basis_single(idx_basis_t, p_t, knots_t, t_coord)
                    dBt = bspline_basis_deriv_single(idx_basis_t, p_t, knots_t, t_coord)

                    coeff = sol_coeffs[idx_basis_x * n_t + idx_basis_t]
                    val += coeff * Bx * Bt
                    val_dx += coeff * dBx * Bt
                    val_dt += coeff * Bx * dBt

            z_pred[idx] = val
            dzdx_approx[idx] = val_dx
            dzdt_approx[idx] = val_dt

        return z_pred, dzdx_approx, dzdt_approx

    def compute_error_norms(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        problem: BasePDEProblem,
        z_pred: np.ndarray,
        dzdx_approx: np.ndarray,
        dzdt_approx: np.ndarray
    ) -> IGAMetrics:
        """
        Computes L2 error, H1 semi-norm error, and L_infty maximum error.
        """
        # Exact solution and exact derivatives
        z_exact = problem.exact_solution(x_grid, t_grid).flatten()
        dzdx_exact = problem.exact_dx(x_grid, t_grid).flatten()
        dzdt_exact = problem.exact_dy(x_grid, t_grid).flatten()

        # Add shift function to prediction for total numerical solution
        shift_val = problem.shift_function(x_grid, t_grid).flatten()
        shift_dx = problem.shift_dx(x_grid, t_grid).flatten()
        shift_dt = problem.shift_dy(x_grid, t_grid).flatten()

        total_z_pred = z_pred + shift_val
        err = z_exact - total_z_pred

        dzdx_err = dzdx_exact - (dzdx_approx + shift_dx)
        dzdt_err = dzdt_exact - (dzdt_approx + shift_dt)

        N = len(x_grid)
        l2_error = math.sqrt(np.sum(err ** 2) / N)
        h1_error = math.sqrt((np.sum(dzdx_err ** 2) + np.sum(dzdt_err ** 2)) / N)
        linf_error = float(np.max(np.abs(err)))

        return IGAMetrics(
            l2_error=l2_error,
            h1_error=h1_error,
            linf_error=linf_error
        )

    @abstractmethod
    def solve(
        self,
        problem: BasePDEProblem,
        p: int,
        M: int,
        mesh_type: str = "uniform",
        gamma: float = 3.0,
        n_points_x: int = 100,
        n_points_t: int = 100,
        **kwargs
    ) -> IGASolution:
        """
        Solves the PDE problem.
        Returns an IGASolution dataclass instance containing all solution grids, coefficients, and metrics.
        """
        pass
