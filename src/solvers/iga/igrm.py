from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.problems.base import BasePDEProblem

from .base import BaseIGASolver, IGASolution, bspline_basis, bspline_basis_deriv


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
    ) -> IGASolution:

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

        # Pre-compute quadrature weights and basis matrices
        gw_2d_local = np.outer(gw_local, gw_local).flatten()

        for ex_left, ex_right, idx_x in element_spans_x:
            det_J_x = (ex_right - ex_left) / 2.0
            gp_x = det_J_x * gp_local + (ex_right + ex_left) / 2.0
            gw_x = gw_local * det_J_x

            # Evaluate U_h trial basis on x
            basis_x_u = np.zeros((p_u + 1, len(gp_x)))
            deriv_x_u = np.zeros((p_u + 1, len(gp_x)))
            for a in range(p_u + 1):
                basis_x_u[a, :] = bspline_basis(idx_x - p_u + a, p_u, knots_x_u, gp_x)
                deriv_x_u[a, :] = bspline_basis_deriv(idx_x - p_u + a, p_u, knots_x_u, gp_x)

            # Evaluate V_h test basis on x
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

                w_2d = np.outer(gw_x, gw_t).flatten()
                W_diag = np.diag(w_2d)

                # 2D tensor product basis for V (dim: (p_v+1)^2 x n_gp)
                N_v_2d = np.kron(basis_x_v, basis_t_v)
                dN_v_dx = np.kron(deriv_x_v, basis_t_v)
                dN_v_dt = np.kron(basis_x_v, deriv_t_v)

                # 2D tensor product basis for U (dim: (p_u+1)^2 x n_gp)
                N_u_2d = np.kron(basis_x_u, basis_t_u)
                dN_u_dx = np.kron(deriv_x_u, basis_t_u)
                dN_u_dt = np.kron(basis_x_u, deriv_t_u)

                # Vectorized Gram matrix Ge: (p_v+1)^2 x (p_v+1)^2
                diff_G = eps * (dN_v_dx @ W_diag @ dN_v_dx.T + dN_v_dt @ W_diag @ dN_v_dt.T)
                mass_G = N_v_2d @ W_diag @ N_v_2d.T
                Ge = diff_G + mass_G

                # Vectorized Rectangular matrix Be: (p_v+1)^2 x (p_u+1)^2
                # diff = eps * (grad(v) . grad(u))
                diff_B = eps * (dN_v_dx @ W_diag @ dN_u_dx.T + dN_v_dt @ W_diag @ dN_u_dt.T)
                # adv = (b . grad u) * v = v * (b_x * du/dx + b_y * du/dt)
                adv_u = b_x * dN_u_dx + b_y * dN_u_dt
                adv_B = N_v_2d @ W_diag @ adv_u.T
                Be = diff_B + adv_B

                # Load vector Fe: (p_v+1)^2
                GX, GT = np.meshgrid(gp_x, gp_t, indexing="ij")
                rhs_val = problem.rhs(GX, GT).flatten()
                Sx = problem.shift_dx(GX, GT).flatten()
                St = problem.shift_dy(GX, GT).flatten()
                Sxx = problem.shift_dx2(GX, GT).flatten()
                Stt = problem.shift_dy2(GX, GT).flatten()
                f_eff = rhs_val - ((b_x * Sx + b_y * St) - eps * (Sxx + Stt))
                Fe = N_v_2d @ (f_eff * w_2d)

                # Index mappings
                rows_v = idx_x_v - p_v + np.arange(p_v + 1)
                cols_v = idx_t_v - p_v + np.arange(p_v + 1)
                global_idx_v = np.array([rx * n_t_v + ct for rx in rows_v for ct in cols_v])

                rows_u = idx_x - p_u + np.arange(p_u + 1)
                cols_u = idx_t - p_u + np.arange(p_u + 1)
                global_idx_u = np.array([rx * n_t_u + ct for rx in rows_u for ct in cols_u])

                F[global_idx_v] += Fe

                for iv, r_glob in enumerate(global_idx_v):
                    for jv, c_glob in enumerate(global_idx_v):
                        G_data.append(Ge[iv, jv])
                        G_row.append(r_glob)
                        G_col.append(c_glob)

                for iv, r_glob in enumerate(global_idx_v):
                    for ju, c_glob in enumerate(global_idx_u):
                        B_data.append(Be[iv, ju])
                        B_row.append(r_glob)
                        B_col.append(c_glob)

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

        is_optimized = kwargs.get("optimized", False)

        # Solve (B^T G_V^-1 B) U = B^T G_V^-1 F
        G_v_csc = G_v.tocsc()
        solve_Gv = spla.factorized(G_v_csc)
        if is_optimized:
            # Multi-RHS vectorized SuperLU solve
            Ginv_B = spla.spsolve(G_v_csc, B_bc.tocsc())
            if sp.issparse(Ginv_B):
                Ginv_B = Ginv_B.toarray()
        else:
            Ginv_B = np.zeros((N_v, N_u))
            for col in range(N_u):
                col_sparse: Any = B_bc[:, col]
                Ginv_B[:, col] = solve_Gv(col_sparse.toarray().flatten())

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
            sol_coeffs, x_grid, t_grid, knots_x_u, knots_t_u, p_u, p_u, n_t_u, optimized=is_optimized
        )

        metrics = self.compute_error_norms(x_grid, t_grid, problem, z_pred, dzdx_approx, dzdt_approx)
        shift_val = problem.shift_function(x_grid, t_grid).flatten()
        total_z_pred = z_pred + shift_val

        return IGASolution(
            sol_coeffs=sol_coeffs,
            knots_x=knots_x_u,
            knots_t=knots_t_u,
            x_grid=x_grid,
            t_grid=t_grid,
            z_pred=total_z_pred,
            metrics=metrics,
            dzdx_approx=dzdx_approx,
            dzdt_approx=dzdt_approx
        )

