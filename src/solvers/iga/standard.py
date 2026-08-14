from typing import Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import BaseIGASolver, bspline_basis, bspline_basis_deriv
from src.problems.base import BasePDEProblem

class StandardIGASolver(BaseIGASolver):
    """
    Standard Galerkin Isogeometric Analysis (IGA-FEM) Solver.
    """

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

        # 2. Quadrature setup
        gp_local, gw_local = self.get_quadrature(p)
        element_spans_x = self.get_element_spans(knots_x)
        element_spans_t = self.get_element_spans(knots_t)

        K_data = []
        K_row = []
        K_col = []
        F = np.zeros(N_dofs)

        b_x, b_y = problem.advection_velocity
        eps = problem.epsilon

        is_optimized = kwargs.get("optimized", False)
        if is_optimized:
            for ex_left, ex_right, idx_x in element_spans_x:
                det_J_x = (ex_right - ex_left) / 2.0
                gp_x = det_J_x * gp_local + (ex_right + ex_left) / 2.0
                gw_x = gw_local * det_J_x

                basis_x = np.zeros((p + 1, len(gp_x)))
                deriv_x = np.zeros((p + 1, len(gp_x)))
                for a in range(p + 1):
                    basis_x[a, :] = bspline_basis(idx_x - p + a, p, knots_x, gp_x)
                    deriv_x[a, :] = bspline_basis_deriv(idx_x - p + a, p, knots_x, gp_x)

                Mx = basis_x @ np.diag(gw_x) @ basis_x.T
                Kx = deriv_x @ np.diag(gw_x) @ deriv_x.T
                Cx = basis_x @ np.diag(gw_x) @ deriv_x.T

                for et_left, et_right, idx_t in element_spans_t:
                    det_J_t = (et_right - et_left) / 2.0
                    gp_t = det_J_t * gp_local + (et_right + et_left) / 2.0
                    gw_t = gw_local * det_J_t

                    basis_t = np.zeros((p + 1, len(gp_t)))
                    deriv_t = np.zeros((p + 1, len(gp_t)))
                    for b_idx in range(p + 1):
                        basis_t[b_idx, :] = bspline_basis(idx_t - p + b_idx, p, knots_t, gp_t)
                        deriv_t[b_idx, :] = bspline_basis_deriv(idx_t - p + b_idx, p, knots_t, gp_t)

                    Mt = basis_t @ np.diag(gw_t) @ basis_t.T
                    Kt = deriv_t @ np.diag(gw_t) @ deriv_t.T
                    Ct = basis_t @ np.diag(gw_t) @ deriv_t.T

                    # Element stiffness matrix via Kronecker sum factorization
                    K_diff_e = eps * (np.kron(Kx, Mt) + np.kron(Mx, Kt))
                    K_adv_e = b_x * np.kron(Cx, Mt) + b_y * np.kron(Mx, Ct)
                    Ke = K_diff_e + K_adv_e

                    # Element load vector via 2D tensor contraction
                    GX, GT = np.meshgrid(gp_x, gp_t, indexing="ij")
                    rhs_val = problem.rhs(GX, GT)
                    Sx = problem.shift_dx(GX, GT)
                    St = problem.shift_dy(GX, GT)
                    Sxx = problem.shift_dx2(GX, GT)
                    Stt = problem.shift_dy2(GX, GT)
                    f_eff = rhs_val - ((b_x * Sx + b_y * St) - eps * (Sxx + Stt))
                    W = f_eff * (np.outer(gw_x, gw_t))
                    Fe = (basis_x @ W @ basis_t.T).flatten()

                    # Global DOF index mapping
                    rows_x = idx_x - p + np.arange(p + 1)
                    cols_t = idx_t - p + np.arange(p + 1)
                    global_indices = np.array([rx * n_t + ct for rx in rows_x for ct in cols_t])

                    # Scatter into global system
                    F[global_indices] += Fe
                    for i_loc, r_glob in enumerate(global_indices):
                        for j_loc, c_glob in enumerate(global_indices):
                            K_data.append(Ke[i_loc, j_loc])
                            K_row.append(r_glob)
                            K_col.append(c_glob)
        else:
            # 3. Standard Element Loop
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
                    for b_idx in range(p + 1):
                        basis_t[b_idx, :] = bspline_basis(idx_t - p + b_idx, p, knots_t, gp_t)
                        deriv_t[b_idx, :] = bspline_basis_deriv(idx_t - p + b_idx, p, knots_t, gp_t)

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

                                    rhs_val = problem.rhs(x_val, t_val)
                                    S_x = problem.shift_dx(x_val, t_val)
                                    S_t = problem.shift_dy(x_val, t_val)
                                    S_xx = problem.shift_dx2(x_val, t_val)
                                    S_tt = problem.shift_dy2(x_val, t_val)

                                    # f_eff = f - (b . grad S - eps * Delta S)
                                    f_eff = rhs_val - ((b_x * S_x + b_y * S_t) - eps * (S_xx + S_tt))

                                    f_integral += f_eff * N_val * w

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

                                            # a(u, v) = eps * grad(u) . grad(v) + (b . grad u) v
                                            diff_term = eps * (dNa_dx * dNb_dx + dNa_dt * dNb_dt)
                                            adv_term = (b_x * dNb_dx + b_y * dNb_dt) * Na_val
                                            k_val += (diff_term + adv_term) * w

                                    K_data.append(k_val)
                                    K_row.append(row_global)
                                    K_col.append(col_global)

        K_sparse = sp.coo_matrix((K_data, (K_row, K_col)), shape=(N_dofs, N_dofs)).tocsr()

        # 4. Enforce Boundary Conditions & Solve
        K_bc, F_bc = self.apply_dirichlet_bc(K_sparse, F, n_x, n_t)
        sol_coeffs = spla.spsolve(K_bc, F_bc)

        # 5. Evaluate Solution & Metrics
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
