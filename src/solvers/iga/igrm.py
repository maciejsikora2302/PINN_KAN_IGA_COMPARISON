from typing import Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import BaseIGASolver, bspline_basis, bspline_basis_deriv
from src.problems.base import BasePDEProblem

class ResidualMinimizationIGASolver(BaseIGASolver):
    """
    Isogeometric Residual Minimization (iGRM / IGA-RM) Solver (Calo et al., 2021).
    Solves (B^T G_V^-1 B) U = B^T G_V^-1 F where V_h is an enriched test space (degree p + Delta p).
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
        test_degree_enrichment: int = 1,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:

        # 1. Trial space U_h (degree p) and Enriched Test space V_h (degree p_v = p + Delta p)
        p_u = p
        p_v = p + test_degree_enrichment

        if mesh_type == "adaptive":
            knots_x_u = self.open_uniform_knots(p_u, M)
            knots_t_u = self.open_graded_knots(p_u, M, gamma=gamma)
            knots_x_v = self.open_uniform_knots(p_v, M)
            knots_t_v = self.open_graded_knots(p_v, M, gamma=gamma)
        else:
            knots_x_u = self.open_uniform_knots(p_u, M)
            knots_t_u = self.open_uniform_knots(p_u, M)
            knots_x_v = self.open_uniform_knots(p_v, M)
            knots_t_v = self.open_uniform_knots(p_v, M)

        n_x_u = M + p_u
        n_t_u = M + p_u
        N_u = n_x_u * n_t_u

        n_x_v = M + p_v
        n_t_v = M + p_v
        N_v = n_x_v * n_t_v

        # 2. Quadrature setup (use p_v + 1 Gauss points to integrate enriched space accurately)
        gp_local, gw_local = self.get_quadrature(p_v)
        element_spans_x = self.get_element_spans(knots_x_u)
        element_spans_t = self.get_element_spans(knots_t_u)

        b_x, b_y = problem.advection_velocity
        eps = problem.epsilon

        # Assembly arrays for B (N_v x N_u), G_v (N_v x N_v), F (N_v)
        B_data, B_row, B_col = [], [], []
        G_data, G_row, G_col = [], [], []
        F = np.zeros(N_v)

        for ex_left, ex_right, idx_x in element_spans_x:
            det_J_x = (ex_right - ex_left) / 2.0
            gp_x = det_J_x * gp_local + (ex_right + ex_left) / 2.0
            gw_x = gw_local * det_J_x

            # Evaluate U_h trial basis
            basis_x_u = np.zeros((p_u + 1, len(gp_x)))
            deriv_x_u = np.zeros((p_u + 1, len(gp_x)))
            for a in range(p_u + 1):
                basis_x_u[a, :] = bspline_basis(idx_x - p_u + a, p_u, knots_x_u, gp_x)
                deriv_x_u[a, :] = bspline_basis_deriv(idx_x - p_u + a, p_u, knots_x_u, gp_x)

            # Evaluate V_h test basis
            idx_x_v = np.searchsorted(knots_x_v, gp_x[len(gp_x)//2]) - 1
            idx_x_v = min(max(idx_x_v, p_v), len(knots_x_v) - p_v - 2)

            basis_x_v = np.zeros((p_v + 1, len(gp_x)))
            deriv_x_v = np.zeros((p_v + 1, len(gp_x)))
            for a in range(p_v + 1):
                basis_x_v[a, :] = bspline_basis(idx_x_v - p_v + a, p_v, knots_x_v, gp_x)
                deriv_x_v[a, :] = bspline_basis_deriv(idx_x_v - p_v + a, p_v, knots_x_v, gp_x)

            for et_left, et_right, idx_t in element_spans_t:
                det_J_t = (et_right - et_left) / 2.0
                gp_t = det_J_t * gp_local + (et_right + et_left) / 2.0
                gw_t = gw_local * det_J_t

                basis_t_u = np.zeros((p_u + 1, len(gp_t)))
                deriv_t_u = np.zeros((p_u + 1, len(gp_t)))
                for b_idx in range(p_u + 1):
                    basis_t_u[b_idx, :] = bspline_basis(idx_t - p_u + b_idx, p_u, knots_t_u, gp_t)
                    deriv_t_u[b_idx, :] = bspline_basis_deriv(idx_t - p_u + b_idx, p_u, knots_t_u, gp_t)

                idx_t_v = np.searchsorted(knots_t_v, gp_t[len(gp_t)//2]) - 1
                idx_t_v = min(max(idx_t_v, p_v), len(knots_t_v) - p_v - 2)

                basis_t_v = np.zeros((p_v + 1, len(gp_t)))
                deriv_t_v = np.zeros((p_v + 1, len(gp_t)))
                for b_idx in range(p_v + 1):
                    basis_t_v[b_idx, :] = bspline_basis(idx_t_v - p_v + b_idx, p_v, knots_t_v, gp_t)
                    deriv_t_v[b_idx, :] = bspline_basis_deriv(idx_t_v - p_v + b_idx, p_v, knots_t_v, gp_t)

                # Assemble test space Gram matrix G_v (N_v x N_v)
                for avx in range(p_v + 1):
                    for avy in range(p_v + 1):
                        row_v = (idx_x_v - p_v + avx) * n_t_v + (idx_t_v - p_v + avy)

                        # Linear form F
                        f_integral = 0.0
                        for gx in range(len(gp_x)):
                            for gt in range(len(gp_t)):
                                x_val = gp_x[gx]
                                t_val = gp_t[gt]
                                w = gw_x[gx] * gw_t[gt]

                                N_v_val = basis_x_v[avx, gx] * basis_t_v[avy, gt]
                                dN_v_dx = deriv_x_v[avx, gx] * basis_t_v[avy, gt]
                                dN_v_dt = basis_x_v[avx, gx] * deriv_t_v[avy, gt]

                                rhs_val = problem.rhs(x_val, t_val)
                                S_x = problem.shift_dx(x_val, t_val)
                                S_t = problem.shift_dy(x_val, t_val)
                                S_xx = problem.shift_dx2(x_val, t_val)
                                S_tt = problem.shift_dy2(x_val, t_val)
                                f_eff = rhs_val - ((b_x * S_x + b_y * S_t) - eps * (S_xx + S_tt))

                                f_integral += f_eff * N_v_val * w
                        F[row_v] += f_integral

                        for bvx in range(p_v + 1):
                            for bvy in range(p_v + 1):
                                col_v = (idx_x_v - p_v + bvx) * n_t_v + (idx_t_v - p_v + bvy)
                                g_val = 0.0

                                for gx in range(len(gp_x)):
                                    for gt in range(len(gp_t)):
                                        w = gw_x[gx] * gw_t[gt]
                                        dNav_dx = deriv_x_v[avx, gx] * basis_t_v[avy, gt]
                                        dNav_dt = basis_x_v[avx, gx] * deriv_t_v[avy, gt]
                                        Nav_val = basis_x_v[avx, gx] * basis_t_v[avy, gt]

                                        dNbv_dx = deriv_x_v[bvx, gx] * basis_t_v[bvy, gt]
                                        dNbv_dt = basis_x_v[bvx, gx] * deriv_t_v[bvy, gt]
                                        Nbv_val = basis_x_v[bvx, gx] * basis_t_v[bvy, gt]

                                        # (G_v)_ij = integral (eps * grad(v_i).grad(v_j) + v_i * v_j)
                                        g_val += (eps * (dNav_dx * dNbv_dx + dNav_dt * dNb_dt if 'dNb_dt' in locals() else dNav_dt * dNbv_dt) + Nav_val * Nbv_val) * w

                                G_data.append(g_val)
                                G_row.append(row_v)
                                G_col.append(col_v)

                        # Assemble rectangular matrix B (N_v x N_u)
                        for aux in range(p_u + 1):
                            for auy in range(p_u + 1):
                                col_u = (idx_x - p_u + aux) * n_t_u + (idx_t - p_u + auy)
                                b_val = 0.0

                                for gx in range(len(gp_x)):
                                    for gt in range(len(gp_t)):
                                        w = gw_x[gx] * gw_t[gt]

                                        dNav_dx = deriv_x_v[avx, gx] * basis_t_v[avy, gt]
                                        dNav_dt = basis_x_v[avx, gx] * deriv_t_v[avy, gt]
                                        Nav_val = basis_x_v[avx, gx] * basis_t_v[avy, gt]

                                        dNau_dx = deriv_x_u[aux, gx] * basis_t_u[auy, gt]
                                        dNau_dt = basis_x_u[aux, gx] * deriv_t_u[auy, gt]

                                        diff_term = eps * (dNav_dx * dNau_dx + dNav_dt * dNau_dt)
                                        adv_term = (b_x * dNau_dx + b_y * dNau_dt) * Nav_val

                                        b_val += (diff_term + adv_term) * w

                                B_data.append(b_val)
                                B_row.append(row_v)
                                B_col.append(col_u)

        G_v = sp.coo_matrix((G_data, (G_row, G_col)), shape=(N_v, N_v)).tocsr()
        B_mat = sp.coo_matrix((B_data, (B_row, B_col)), shape=(N_v, N_u)).tocsr()

        # Enforce Boundary conditions on Trial space U_h
        is_boundary_u = np.zeros(N_u, dtype=bool)
        for i in range(n_x_u):
            for j in range(n_t_u):
                if i == 0 or i == n_x_u - 1 or j == 0 or j == n_t_u - 1:
                    is_boundary_u[i * n_t_u + j] = True

        # Apply Dirichlet BCs directly to the rectangular system
        B_lil = B_mat.tolil()
        for col_u in range(N_u):
            if is_boundary_u[col_u]:
                # Zero out trial column except diagonal indicator in B^T B
                pass

        B_bc = B_lil.tocsr()

        # Solve (B^T G_V^-1 B) U = B^T G_V^-1 F
        solve_Gv = spla.factorized(G_v)
        Ginv_B = np.zeros((N_v, N_u))
        for col in range(N_u):
            Ginv_B[:, col] = solve_Gv(B_bc[:, col].toarray().flatten())

        Ginv_F = solve_Gv(F)

        # Form normal equations (N_u x N_u)
        K_igrm = B_bc.T.dot(Ginv_B)
        F_igrm = B_bc.T.dot(Ginv_F)

        # Apply BC identity to K_igrm
        K_igrm_lil = sp.lil_matrix(K_igrm)
        for i in range(N_u):
            if is_boundary_u[i]:
                K_igrm_lil.rows[i] = [i]
                K_igrm_lil.data[i] = [1.0]
                F_igrm[i] = 0.0

        K_igrm_bc = K_igrm_lil.tocsr()
        sol_coeffs = spla.spsolve(K_igrm_bc, F_igrm)

        # Evaluate solution
        x_raw = np.linspace(0.0, 1.0, num=n_points_x)
        t_raw = np.linspace(0.0, 1.0, num=n_points_t)
        grid_x, grid_t = np.meshgrid(x_raw, t_raw, indexing="ij")
        x_grid = grid_x.flatten().reshape(-1, 1)
        t_grid = grid_t.flatten().reshape(-1, 1)

        z_pred, dzdx_approx, dzdt_approx = self.evaluate_solution(
            sol_coeffs, x_grid, t_grid, knots_x_u, knots_t_u, p_u, p_u, n_t_u
        )

        metrics = self.compute_error_norms(x_grid, t_grid, problem, z_pred, dzdx_approx, dzdt_approx)

        return sol_coeffs, knots_x_u, knots_t_u, x_grid, t_grid, z_pred, metrics
