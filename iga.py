import os
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Any, List

from model import ExperimentInterface
from config import IGAConfig
from pinn import exact_solution_dx, exact_solution_dt, exact_solution, shift_EJ, shift_EJ_dx, shift_EJ_dt

# 1D B-spline Basis Functions and Derivatives via Cox-de Boor recursion
def bspline_basis_single(i: int, p: int, knots: np.ndarray, x: float) -> float:
    """Evaluates N_{i,p}(x) at a single coordinate x."""
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
    """Evaluates N'_{i,p}(x) at a single coordinate x."""
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


# Vectorized versions of basis evaluation
def bspline_basis(i: int, p: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda val: bspline_basis_single(i, p, knots, val))(x)


def bspline_basis_deriv(i: int, p: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda val: bspline_basis_deriv_single(i, p, knots, val))(x)


# 1D open knot vector generation
def open_knot_vector(p: int, M: int) -> np.ndarray:
    internal_knots = np.linspace(0, 1, M + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), internal_knots, np.ones(p + 1)])


class IGAExperiment(ExperimentInterface):
    """
    Isogeometric Analysis (IGA) Experiment solver for the 2D Poisson equation.
    Uses 2D tensor-product B-spline finite elements.
    """
    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)
        self.sol_coeffs = None
        self.loss_history = []
        self.h1_error_history = []
        self.final_loss = None
        self.final_interior_loss = None
        self.x_grid = None
        self.t_grid = None
        self.z_pred = None
        self.final_h1_error = None

    def load_config(self, config_path: str) -> None:
        self.config = IGAConfig()
        self.config.load_config(config_path)
        self.config.validate_config()

    def train(self) -> None:
        """Solves the 2D Poisson/Convection-Diffusion problem using B-spline IGA."""
        if not self.config:
            raise ValueError("Configuration has not been loaded.")

        print("=== Starting training/solving for IGA ===")

        # 1. Load solver parameters
        p = self.config.IGA_DEGREE
        M = self.config.IGA_ELEMENTS
        n = M + p  # number of basis functions in each direction

        # Open knot vectors
        knots_x = open_knot_vector(p, M)
        knots_t = open_knot_vector(p, M)

        # 2. Quadrature setup (p+1 Gauss points per element)
        gp_local, gw_local = np.polynomial.legendre.leggauss(p + 1)

        # Element intervals (active spans)
        element_spans_x = []
        for i in range(len(knots_x) - 1):
            if knots_x[i] < knots_x[i+1]:
                element_spans_x.append((knots_x[i], knots_x[i+1], i))

        element_spans_t = []
        for j in range(len(knots_t) - 1):
            if knots_t[j] < knots_t[j+1]:
                element_spans_t.append((knots_t[j], knots_t[j+1], j))

        # Size of global system: n * n equations
        N_dofs = n * n
        K_data = []
        K_row = []
        K_col = []
        F = np.zeros(N_dofs)

        # 3. Assemble Stiffness Matrix K and Load Vector F
        for ex_left, ex_right, idx_x in element_spans_x:
            # Map Gauss points to element
            det_J_x = (ex_right - ex_left) / 2.0
            gp_x = det_J_x * gp_local + (ex_right + ex_left) / 2.0
            gw_x = gw_local * det_J_x

            # Evaluate B-spline basis on element spans in x direction
            # active basis functions are idx_x - p to idx_x
            basis_x = np.zeros((p + 1, len(gp_x)))
            deriv_x = np.zeros((p + 1, len(gp_x)))
            for a in range(p + 1):
                basis_x[a, :] = bspline_basis(idx_x - p + a, p, knots_x, gp_x)
                deriv_x[a, :] = bspline_basis_deriv(idx_x - p + a, p, knots_x, gp_x)

            for et_left, et_right, idx_t in element_spans_t:
                det_J_t = (et_right - et_left) / 2.0
                gp_t = det_J_t * gp_local + (et_right + et_left) / 2.0
                gw_t = gw_local * det_J_t

                # Evaluate B-spline basis on element spans in t direction
                basis_t = np.zeros((p + 1, len(gp_t)))
                deriv_t = np.zeros((p + 1, len(gp_t)))
                for b in range(p + 1):
                    basis_t[b, :] = bspline_basis(idx_t - p + b, p, knots_t, gp_t)
                    deriv_t[b, :] = bspline_basis_deriv(idx_t - p + b, p, knots_t, gp_t)

                # Compute local stiffness and force values via 2D Gauss integration
                for ax in range(p + 1):
                    for ay in range(p + 1):
                        row_global = (idx_x - p + ax) * n + (idx_t - p + ay)

                        # Load vector integration
                        f_integral = 0.0
                        for gx in range(len(gp_x)):
                            for gt in range(len(gp_t)):
                                x_val = gp_x[gx]
                                t_val = gp_t[gt]
                                w = gw_x[gx] * gw_t[gt]
                                N_val = basis_x[ax, gx] * basis_t[ay, gt]
                                dN_dx = deriv_x[ax, gx] * basis_t[ay, gt]
                                dN_dt = basis_x[ax, gx] * deriv_t[ay, gt]

                                if self.config.EXAMPLE in [1, 2]:
                                    # Poisson RHS: f = -Delta u
                                    if self.config.EXAMPLE == 1:
                                        rhs_val = 8.0 * np.pi * np.pi * np.sin(2.0 * np.pi * x_val) * np.sin(2.0 * np.pi * t_val)
                                    else: # Example 2
                                        f1 = -np.pi * np.pi * np.exp(np.pi * (x_val - 2.0 * t_val)) * np.sin(np.pi * t_val) * (4.0 * np.cos(2.0 * np.pi * x_val) - 3.0 * np.sin(2.0 * np.pi * x_val))
                                        f2 = np.pi * np.pi * np.exp(np.pi * (x_val - 2.0 * t_val)) * np.sin(2.0 * np.pi * x_val) * (4.0 * np.cos(np.pi * t_val) - 3.0 * np.sin(np.pi * t_val))
                                        rhs_val = - (f1 + f2)
                                    f_integral += rhs_val * N_val * w
                                elif self.config.EXAMPLE == 3:
                                    # Convection-Diffusion RHS: - (S_t v + epsilon * (S_x v_x + S_t v_t))
                                    eps = self.config.EPSILON
                                    S_t = -np.sin(np.pi * x_val)
                                    S_x = np.pi * np.cos(np.pi * x_val) * (1.0 - t_val)
                                    f_integral -= (S_t * N_val + eps * (S_x * dN_dx + S_t * dN_dt)) * w
                        F[row_global] += f_integral

                        # Stiffness matrix integration
                        for bx in range(p + 1):
                            for by in range(p + 1):
                                col_global = (idx_x - p + bx) * n + (idx_t - p + by)

                                k_val = 0.0
                                for gx in range(len(gp_x)):
                                    for gt in range(len(gp_t)):
                                        w = gw_x[gx] * gw_t[gt]
                                        # Derivatives of test basis (ax, ay)
                                        dNa_dx = deriv_x[ax, gx] * basis_t[ay, gt]
                                        dNa_dt = basis_x[ax, gx] * deriv_t[ay, gt]
                                        Na_val = basis_x[ax, gx] * basis_t[ay, gt]

                                        # Derivatives of trial basis (bx, by)
                                        dNb_dx = deriv_x[bx, gx] * basis_t[by, gt]
                                        dNb_dt = basis_x[bx, gx] * deriv_t[by, gt]
                                        Nb_val = basis_x[bx, gx] * basis_t[by, gt]

                                        if self.config.EXAMPLE in [1, 2]:
                                            # Poisson stiffness: grad(u) . grad(v)
                                            k_val += (dNa_dx * dNb_dx + dNa_dt * dNb_dt) * w
                                        elif self.config.EXAMPLE == 3:
                                            # Convection-diffusion: u_t v + eps * grad(u) . grad(v)
                                            eps = self.config.EPSILON
                                            k_val += (dNb_dt * Na_val + eps * (dNa_dx * dNb_dx + dNa_dt * dNb_dt)) * w

                                K_data.append(k_val)
                                K_row.append(row_global)
                                K_col.append(col_global)

        # Build Sparse Matrix
        K_sparse = sp.coo_matrix((K_data, (K_row, K_col)), shape=(N_dofs, N_dofs)).tocsr()

        # 4. Enforce Boundary Conditions (Dirichlet: c_{i,j} = 0 on boundaries)
        is_boundary = np.zeros(N_dofs, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    is_boundary[i * n + j] = True

        # Apply Dirichlet by setting boundary rows to identity, corresponding right-hand side to 0
        K_bc = K_sparse.tolil()
        for i in range(N_dofs):
            if is_boundary[i]:
                K_bc.rows[i] = [i]
                K_bc.data[i] = [1.0]
                F[i] = 0.0

        K_bc = K_bc.tocsr()

        # 5. Solve linear system
        self.sol_coeffs = spla.spsolve(K_bc, F)

        # 6. Evaluate solution at Collocation Points Grid to compute outcomes
        x_domain = [0.0, self.config.LENGTH]
        t_domain = [0.0, self.config.TOTAL_TIME]
        x_raw = np.linspace(x_domain[0], x_domain[1], num=self.config.N_POINTS_X)
        t_raw = np.linspace(t_domain[0], t_domain[1], num=self.config.N_POINTS_T)
        grid_x, grid_t = np.meshgrid(x_raw, t_raw, indexing="ij")
        self.x_grid = grid_x.flatten().reshape(-1, 1)
        self.t_grid = grid_t.flatten().reshape(-1, 1)

        # Evaluate B-spline expansion at collocation points
        self.z_pred = np.zeros_like(self.x_grid).flatten()
        for idx in range(len(self.x_grid)):
            x_coord = self.x_grid[idx, 0]
            t_coord = self.t_grid[idx, 0]

            # Find active intervals
            span_idx_x = np.searchsorted(knots_x, x_coord) - 1
            span_idx_x = min(max(span_idx_x, p), len(knots_x) - p - 2)

            span_idx_t = np.searchsorted(knots_t, t_coord) - 1
            span_idx_t = min(max(span_idx_t, p), len(knots_t) - p - 2)

            # Sum over local active basis functions
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

        # For Example 3, add the shift function EJ to prediction
        if self.config.EXAMPLE == 3:
            # We can convert numpy matrices into tensors to reuse pinn.py's functions
            import torch
            with torch.no_grad():
                z_shift = shift_EJ(torch.tensor(self.x_grid), torch.tensor(self.t_grid)).flatten().numpy()
                self.z_pred = self.z_pred + z_shift

        # Calculate final metric: H1 semi-norm error
        import torch
        with torch.no_grad():
            tx = torch.tensor(self.x_grid)
            tt = torch.tensor(self.t_grid)
            
            # Predict derivatives by finite differences or exact derivatives.
            # For H1 norms on IGA grid, we can evaluate analytical derivatives of the B-splines.
            # Let's compute B-spline derivatives analytically at all collocation points:
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

            # Subtract from exact derivatives
            dzdx_exact = exact_solution_dx(tx, tt, self.config).flatten().numpy()
            dzdt_exact = exact_solution_dt(tx, tt, self.config).flatten().numpy()
            
            dzdx = dzdx_exact - dzdx_approx
            dzdt = dzdt_exact - dzdt_approx

            if self.config.EXAMPLE == 3:
                z_shift_dx = shift_EJ_dx(tx, tt).flatten().numpy()
                z_shift_dt = shift_EJ_dt(tx, tt).flatten().numpy()
                dzdx = dzdx - z_shift_dx
                dzdt = dzdt - z_shift_dt

            N = self.config.N_POINTS_X * self.config.N_POINTS_T
            h1_error = math.sqrt((np.sum(dzdx ** 2) + np.sum(dzdt ** 2)) / N)

        # Residual loss: we can compute the L2 norm of the strong form PDE residuals
        # over the collocation grid to get a comparable "loss" value
        residuals = np.zeros_like(self.z_pred)
        # For simplicity, since IGA solves the weak form directly, the residual of the strong form
        # is very small or can be computed directly.
        # Let's approximate it by evaluating the PDE: -Delta u - f
        # (or for Example 3: u_t - epsilon Delta u).
        # To make it comparable, we can just use the final L2 norm of the weak form residual
        # or a value proportional to the system residual ||K U - F||.
        # Let's define it as the norm of the system residual:
        system_res = K_bc.dot(self.sol_coeffs) - F
        final_l2_residual = float(np.sum(system_res ** 2))

        self.final_loss = final_l2_residual
        self.final_interior_loss = final_l2_residual
        self.final_h1_error = h1_error

        # Create history (replicate across epochs to show flat IGA baseline)
        self.loss_history = [self.final_loss] * self.config.EPOCHS
        self.h1_error_history = [h1_error] * (self.config.EPOCHS // self.config.H1_CALC_EVERY)

        print(f"IGA completed solver. Final H1 Error: {h1_error:.6e}")

    def save_model(self, path: str) -> None:
        """Saves solver coefficients to path."""
        if self.sol_coeffs is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.sol_coeffs)
        print(f"IGA solver coefficients saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        """Saves final loss values and logs to a NumPy .npz file."""
        if self.sol_coeffs is None:
            raise ValueError("Model has not been trained yet.")
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
