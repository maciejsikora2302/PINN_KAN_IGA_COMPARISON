import math
from typing import Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import BaseIGASolver, bspline_basis, bspline_basis_deriv, bspline_basis_deriv2
from src.problems.base import BasePDEProblem

class SUPGIGASolver(BaseIGASolver):
    """
    Streamline-Upwind Petrov-Galerkin (SUPG) Stabilized IGA-FEM Solver.
    Combines classical Galerkin weak form with residual-based upwind stabilization.
    """

    @staticmethod
    def compute_tau_e(h_e: float, b_norm: float, eps: float) -> float:
        """Computes element stabilization parameter tau_e using coth(Pe_e) - 1/Pe_e formula."""
        if b_norm < 1e-12:
            return 0.0
        Pe_e = (b_norm * h_e) / (2.0 * eps)
        if Pe_e < 1e-3:
            # Taylor series approximation near zero: coth(x) - 1/x ~ x/3
            return (h_e / (2.0 * b_norm)) * (Pe_e / 3.0)
        elif Pe_e > 30.0:
            # Asymptotic limit: coth(x) -> 1
            return (h_e / (2.0 * b_norm)) * (1.0 - 1.0 / Pe_e)
        else:
            coth_val = 1.0 / math.tanh(Pe_e)
            return (h_e / (2.0 * b_norm)) * (coth_val - 1.0 / Pe_e)

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:

        # 1. Generate knot vectors
        if mesh_type == "adaptive":
            knots_x = self.open_uniform_knots(p, M)
            knots_t = self.open_graded_knots(p, M, gamma=gamma)
        else:
            knots_x = self.open_uniform_knots(p, M)
            knots_t = self.open_uniform_knots(p, M)

        n_x = M + p
        n_t = M + p
        N_dofs = n_x * n_t

        gp_local, gw_local = self.get_quadrature(p)
        element_spans_x = self.get_element_spans(knots_x)
        element_spans_t = self.get_element_spans(knots_t)

        K_data = []
        K_row = []
        K_col = []
        F = np.zeros(N_dofs)

        b_x, b_y = problem.advection_velocity
        b_norm = math.sqrt(b_x ** 2 + b_y ** 2)
        eps = problem.epsilon

        is_optimized = kwargs.get("optimized", False)
        if is_optimized:
            for ex_left, ex_right, idx_x in element_spans_x:
                hx = ex_right - ex_left
                det_J_x = hx / 2.0
                gp_x = det_J_x * gp_local + (ex_right + ex_left) / 2.0
                gw_x = gw_local * det_J_x

                basis_x = np.zeros((p + 1, len(gp_x)))
                deriv_x = np.zeros((p + 1, len(gp_x)))
                deriv2_x = np.zeros((p + 1, len(gp_x)))
                for a in range(p + 1):
                    basis_x[a, :] = bspline_basis(idx_x - p + a, p, knots_x, gp_x)
                    deriv_x[a, :] = bspline_basis_deriv(idx_x - p + a, p, knots_x, gp_x)
                    deriv2_x[a, :] = bspline_basis_deriv2(idx_x - p + a, p, knots_x, gp_x)

                for et_left, et_right, idx_t in element_spans_t:
                    hy = et_right - et_left
                    det_J_t = hy / 2.0
                    gp_t = det_J_t * gp_local + (et_right + et_left) / 2.0
                    gw_t = gw_local * det_J_t

                    basis_t = np.zeros((p + 1, len(gp_t)))
                    deriv_t = np.zeros((p + 1, len(gp_t)))
                    deriv2_t = np.zeros((p + 1, len(gp_t)))
                    for b_idx in range(p + 1):
                        basis_t[b_idx, :] = bspline_basis(idx_t - p + b_idx, p, knots_t, gp_t)
                        deriv_t[b_idx, :] = bspline_basis_deriv(idx_t - p + b_idx, p, knots_t, gp_t)
                        deriv2_t[b_idx, :] = bspline_basis_deriv2(idx_t - p + b_idx, p, knots_t, gp_t)

                    h_e = math.sqrt(hx ** 2 + hy ** 2)
                    tau_e = self.compute_tau_e(h_e, b_norm, eps)

                    # 2D tensor product basis evaluations at flattened Gauss points
                    # Shape ((p+1)^2, n_gp_x * n_gp_t)
                    N_2d = np.kron(basis_x, basis_t)
                    dN_dx = np.kron(deriv_x, basis_t)
                    dN_dt = np.kron(basis_x, deriv_t)
                    lap_N = np.kron(deriv2_x, basis_t) + np.kron(basis_x, deriv2_t)

                    b_grad = b_x * dN_dx + b_y * dN_dt
                    w_2d = np.outer(gw_x, gw_t).flatten()
                    W_diag = np.diag(w_2d)

                    # Stiffness matrix components
                    diff_term = eps * (dN_dx @ W_diag @ dN_dx.T + dN_dt @ W_diag @ dN_dt.T)
                    adv_term = N_2d @ W_diag @ b_grad.T
                    supg_term = tau_e * (b_grad @ W_diag @ (b_grad - eps * lap_N).T)
                    Ke = diff_term + adv_term + supg_term

                    # Load vector
                    GX, GT = np.meshgrid(gp_x, gp_t, indexing="ij")
                    rhs_val = problem.rhs(GX, GT).flatten()
                    Sx = problem.shift_dx(GX, GT).flatten()
                    St = problem.shift_dy(GX, GT).flatten()
                    Sxx = problem.shift_dx2(GX, GT).flatten()
                    Stt = problem.shift_dy2(GX, GT).flatten()

                    f_eff = rhs_val - ((b_x * Sx + b_y * St) - eps * (Sxx + Stt))
                    v_supg = N_2d + tau_e * b_grad
                    Fe = v_supg @ (f_eff * w_2d)

                    # Global DOF index mapping
                    rows_x = idx_x - p + np.arange(p + 1)
                    cols_t = idx_t - p + np.arange(p + 1)
                    global_indices = np.array([rx * n_t + ct for rx in rows_x for ct in cols_t])

                    F[global_indices] += Fe
                    for i_loc, r_glob in enumerate(global_indices):
                        for j_loc, c_glob in enumerate(global_indices):
                            K_data.append(Ke[i_loc, j_loc])
                            K_row.append(r_glob)
                            K_col.append(c_glob)
        else:
            # 2. Element Assembly Loop
            for ex_left, ex_right, idx_x in element_spans_x:
                hx = ex_right - ex_left
                det_J_x = hx / 2.0
                gp_x = det_J_x * gp_local + (ex_right + ex_left) / 2.0
                gw_x = gw_local * det_J_x

                basis_x = np.zeros((p + 1, len(gp_x)))
                deriv_x = np.zeros((p + 1, len(gp_x)))
                deriv2_x = np.zeros((p + 1, len(gp_x)))
                for a in range(p + 1):
                    basis_x[a, :] = bspline_basis(idx_x - p + a, p, knots_x, gp_x)
                    deriv_x[a, :] = bspline_basis_deriv(idx_x - p + a, p, knots_x, gp_x)
                    deriv2_x[a, :] = bspline_basis_deriv2(idx_x - p + a, p, knots_x, gp_x)

                for et_left, et_right, idx_t in element_spans_t:
                    hy = et_right - et_left
                    det_J_t = hy / 2.0
                    gp_t = det_J_t * gp_local + (et_right + et_left) / 2.0
                    gw_t = gw_local * det_J_t

                    basis_t = np.zeros((p + 1, len(gp_t)))
                    deriv_t = np.zeros((p + 1, len(gp_t)))
                    deriv2_t = np.zeros((p + 1, len(gp_t)))
                    for b_idx in range(p + 1):
                        basis_t[b_idx, :] = bspline_basis(idx_t - p + b_idx, p, knots_t, gp_t)
                        deriv_t[b_idx, :] = bspline_basis_deriv(idx_t - p + b_idx, p, knots_t, gp_t)
                        deriv2_t[b_idx, :] = bspline_basis_deriv2(idx_t - p + b_idx, p, knots_t, gp_t)

                    # Compute local element length h_e along advection stream
                    h_e = math.sqrt(hx ** 2 + hy ** 2)
                    tau_e = self.compute_tau_e(h_e, b_norm, eps)

                    for ax in range(p + 1):
                        for ay in range(p + 1):
                            row_global = (idx_x - p + ax) * n_t + (idx_t - p + ay)
                            f_integral = 0.0

                            for gx in range(len(gp_x)):
                                for gt in range(len(gp_t)):
                                    x_val = gp_x[gx]
                                    t_val = gp_t[gt]
                                    w = gw_x[gx] * gw_t[gt]

                                    N_val = basis_x[ax, gx] * basis_t[ay, gt]
                                    dN_dx = deriv_x[ax, gx] * basis_t[ay, gt]
                                    dN_dt = basis_x[ax, gx] * deriv_t[ay, gt]

                                    # Streamline derivative b . grad v
                                    b_grad_v = b_x * dN_dx + b_y * dN_dt

                                    rhs_val = problem.rhs(x_val, t_val)
                                    S_x = problem.shift_dx(x_val, t_val)
                                    S_t = problem.shift_dy(x_val, t_val)
                                    S_xx = problem.shift_dx2(x_val, t_val)
                                    S_tt = problem.shift_dy2(x_val, t_val)

                                    f_eff = rhs_val - ((b_x * S_x + b_y * S_t) - eps * (S_xx + S_tt))

                                    # SUPG force integrand: f_eff * (v + tau_e * b.grad v)
                                    v_supg = N_val + tau_e * b_grad_v
                                    f_integral += f_eff * v_supg * w

                            F[row_global] += f_integral

                            for bx in range(p + 1):
                                for by in range(p + 1):
                                    col_global = (idx_x - p + bx) * n_t + (idx_t - p + by)
                                    k_val = 0.0

                                    for gx in range(len(gp_x)):
                                        for gt in range(len(gp_t)):
                                            w = gw_x[gx] * gw_t[gt]

                                            dNa_dx = deriv_x[ax, gx] * basis_t[ay, gt]
                                            dNa_dt = basis_x[ax, gx] * deriv_t[ay, gt]
                                            Na_val = basis_x[ax, gx] * basis_t[ay, gt]

                                            dNb_dx = deriv_x[bx, gx] * basis_t[by, gt]
                                            dNb_dt = basis_x[bx, gx] * deriv_t[by, gt]
                                            d2Nb_dx2 = deriv2_x[bx, gx] * basis_t[by, gt]
                                            d2Nb_dt2 = basis_x[bx, gx] * deriv2_t[by, gt]

                                            b_grad_v = b_x * dNa_dx + b_y * dNa_dt
                                            b_grad_u = b_x * dNb_dx + b_y * dNb_dt
                                            laplacian_u = d2Nb_dx2 + d2Nb_dt2

                                            # Standard Galerkin terms
                                            diff_term = eps * (dNa_dx * dNb_dx + dNa_dt * dNb_dt)
                                            adv_term = b_grad_u * Na_val

                                            # SUPG stabilization term: tau_e * (b.grad u - eps * laplacian u) * (b.grad v)
                                            supg_term = tau_e * (b_grad_u - eps * laplacian_u) * b_grad_v

                                            k_val += (diff_term + adv_term + supg_term) * w

                                    K_data.append(k_val)
                                    K_row.append(row_global)
                                    K_col.append(col_global)

        K_sparse = sp.coo_matrix((K_data, (K_row, K_col)), shape=(N_dofs, N_dofs)).tocsr()

        # 3. Enforce BCs and Solve
        K_bc, F_bc = self.apply_dirichlet_bc(K_sparse, F, n_x, n_t)
        sol_coeffs = spla.spsolve(K_bc, F_bc)

        # 4. Evaluate Solution & Metrics
        x_raw = np.linspace(0.0, 1.0, num=n_points_x)
        t_raw = np.linspace(0.0, 1.0, num=n_points_t)
        grid_x, grid_t = np.meshgrid(x_raw, t_raw, indexing="ij")
        x_grid = grid_x.flatten().reshape(-1, 1)
        t_grid = grid_t.flatten().reshape(-1, 1)

        z_pred, dzdx_approx, dzdt_approx = self.evaluate_solution(
            sol_coeffs, x_grid, t_grid, knots_x, knots_t, p, p, n_t, optimized=is_optimized
        )

        metrics = self.compute_error_norms(x_grid, t_grid, problem, z_pred, dzdx_approx, dzdt_approx)

        return sol_coeffs, knots_x, knots_t, x_grid, t_grid, z_pred, metrics
