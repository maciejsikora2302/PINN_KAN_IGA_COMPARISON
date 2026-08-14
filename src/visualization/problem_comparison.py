import csv
import glob
import math
import os
from typing import Any

import matplotlib
import numpy as np
import yaml
from scipy.interpolate import griddata

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.problems import get_problem

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Family Color Palette (Blues for PINN, Greens for KAN, Purples/Reds for IGA)
METHOD_COLORS = {
    # PINN Family
    "pinn_uniform": "#1f77b4",          # Classic Blue
    "pinn_boundary": "#4ba3e3",         # Light Blue
    "r_pinn_uniform": "#004c6d",        # Dark Navy Blue
    "r_pinn_boundary": "#257d9d",       # Medium Navy Blue
    # KAN Family
    "nurbs_kan_uniform": "#2ca02c",     # Medium Green
    "nurbs_kan_boundary": "#66c2a5",    # Mint Green
    "r_nurbs_kan_uniform": "#005a24",   # Deep Forest Green
    "r_nurbs_kan_boundary": "#238b45",  # Emerald Green
    # IGA Family
    "iga_standard_uniform": "#9467bd",  # Purple
    "iga_supg_uniform": "#8c564b",      # Brown/Rust
    "iga_supg_adaptive": "#d62728",     # Crimson Red
    "iga_igrm_adaptive": "#e377c2",     # Magenta Pink
}

METHOD_LINESTYLES = {
    "pinn_uniform": "-",
    "pinn_boundary": "--",
    "r_pinn_uniform": "-.",
    "r_pinn_boundary": ":",
    "nurbs_kan_uniform": "-",
    "nurbs_kan_boundary": "--",
    "r_nurbs_kan_uniform": "-.",
    "r_nurbs_kan_boundary": ":",
    "iga_standard_uniform": "-",
    "iga_supg_uniform": "--",
    "iga_supg_adaptive": "-.",
    "iga_igrm_adaptive": ":",
}

METHOD_MARKERS = {
    "pinn_uniform": "o",
    "pinn_boundary": "s",
    "r_pinn_uniform": "D",
    "r_pinn_boundary": "v",
    "nurbs_kan_uniform": "^",
    "nurbs_kan_boundary": "P",
    "r_nurbs_kan_uniform": "X",
    "r_nurbs_kan_boundary": "*",
    "iga_standard_uniform": "o",
    "iga_supg_uniform": "s",
    "iga_supg_adaptive": "^",
    "iga_igrm_adaptive": "D",
}

METHOD_LABELS = {
    "pinn_uniform": "PINN (Uniform)",
    "pinn_boundary": "PINN (Boundary Layer)",
    "r_pinn_uniform": "R-PINN (Uniform)",
    "r_pinn_boundary": "R-PINN (Boundary Layer)",
    "nurbs_kan_uniform": "NURBS-KAN (Uniform)",
    "nurbs_kan_boundary": "NURBS-KAN (Boundary Layer)",
    "r_nurbs_kan_uniform": "R-NURBS-KAN (Uniform)",
    "r_nurbs_kan_boundary": "R-NURBS-KAN (Boundary Layer)",
    "iga_standard_uniform": "Standard IGA (Uniform)",
    "iga_supg_uniform": "SUPG IGA (Uniform)",
    "iga_supg_adaptive": "SUPG IGA (Adaptive)",
    "iga_igrm_adaptive": "iGRM IGA (Adaptive)",
}

CANONICAL_METHOD_ORDER = [
    "pinn_uniform",
    "pinn_boundary",
    "r_pinn_uniform",
    "r_pinn_boundary",
    "nurbs_kan_uniform",
    "nurbs_kan_boundary",
    "r_nurbs_kan_uniform",
    "r_nurbs_kan_boundary",
    "iga_standard_uniform",
    "iga_supg_uniform",
    "iga_supg_adaptive",
    "iga_igrm_adaptive",
]


def calculate_ces(h1_error: float, elapsed_seconds: float, dofs_or_params: int) -> float:
    """
    Calculates the Computational Efficiency Score (CES):
    CES = -log10( H1_error * sqrt(Time) * (DoFs)^(1/4) )
    Higher score indicates superior accuracy per unit of compute and memory resource.
    """
    try:
        h1 = max(float(h1_error), 1e-15)
        t = max(float(elapsed_seconds), 1e-4)
        d = max(int(dofs_or_params), 1)
        cost = h1 * math.sqrt(t) * (d ** 0.25)
        return -math.log10(cost)
    except Exception:
        return float('nan')


class ProblemComparator:
    """
    Scans a problem output directory, loads outcomes and metadata for all methods,
    and produces comprehensive cross-method diagnostic plots and summary tables.
    """

    def __init__(self, problem_dir: str):
        self.problem_dir = os.path.abspath(problem_dir)
        self.comparisons_dir = os.path.join(self.problem_dir, "comparisons")
        os.makedirs(self.comparisons_dir, exist_ok=True)

        self.problem_id = os.path.basename(self.problem_dir)
        self.methods_data: dict[str, dict[str, Any]] = {}
        self.problem_name = "poisson_sine"
        self.epsilon = 0.01

        self._load_all_methods()

    def _load_all_methods(self):
        """Discovers and loads all method directories inside problem_dir in canonical order."""
        subdirs = [d for d in glob.glob(os.path.join(self.problem_dir, "*")) if os.path.isdir(d)]
        dir_dict = {os.path.basename(d): d for d in subdirs if os.path.basename(d) != "comparisons"}

        # Sort according to CANONICAL_METHOD_ORDER
        ordered_names = [m for m in CANONICAL_METHOD_ORDER if m in dir_dict]
        remaining = [m for m in dir_dict if m not in CANONICAL_METHOD_ORDER]
        all_ordered = ordered_names + sorted(remaining)

        for name in all_ordered:
            d = dir_dict[name]
            outcomes_path = os.path.join(d, "outcomes.npz")
            metadata_path = os.path.join(d, "metadata.yaml")

            if not os.path.exists(outcomes_path):
                continue

            try:
                npz_data = dict(np.load(outcomes_path, allow_pickle=True))
            except Exception as e:
                print(f"Warning: could not load {outcomes_path}: {e}")
                continue

            meta = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        meta = yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"Warning: could not load {metadata_path}: {e}")

            cfg = meta.get("config", {})
            self.problem_name = cfg.get("PROBLEM_NAME", cfg.get("problem_id", self.problem_name))
            self.epsilon = float(cfg.get("EPSILON", self.epsilon))

            self.methods_data[name] = {
                "dir": d,
                "outcomes": npz_data,
                "metadata": meta,
            }

        self.problem = get_problem(self.problem_name, self.epsilon)

    def _get_color(self, name: str, idx: int = 0) -> str:
        return METHOD_COLORS.get(name, plt.cm.tab10(idx % 10))

    def _get_linestyle(self, name: str) -> str:
        return METHOD_LINESTYLES.get(name, "-")

    def _get_marker(self, name: str) -> str:
        return METHOD_MARKERS.get(name, "o")

    def _get_label(self, name: str) -> str:
        return METHOD_LABELS.get(name, name.replace("_", " ").title())

    def _interpolate_to_100x100(self, x: np.ndarray, t: np.ndarray, z: np.ndarray):
        """Standardizes any 2D scattered or tensor point cloud onto a clean 100x100 grid."""
        x_f = np.asarray(x).flatten()
        t_f = np.asarray(t).flatten()
        z_f = np.asarray(z).flatten()

        xi = np.linspace(0.0, 1.0, 100)
        ti = np.linspace(0.0, 1.0, 100)
        XI, TI = np.meshgrid(xi, ti)

        points = np.column_stack((x_f, t_f))
        ZI = griddata(points, z_f, (XI, TI), method="cubic")
        if np.any(np.isnan(ZI)):
            ZI = griddata(points, z_f, (XI, TI), method="nearest")
        return XI, TI, ZI

    def plot_convergence_overlay_epochs(self, save_name: str = "convergence_overlay_epochs.png"):
        """1. Overlaid H^1 error vs. training epochs for iterative neural solvers."""
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        has_plots = False

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            h1 = outcomes.get("h1_error_history", np.array([])).flatten()
            if len(h1) <= 1:
                continue

            ep_hist = outcomes.get("h1_epoch_history", None)
            if ep_hist is not None and len(ep_hist) == len(h1):
                epochs = np.asarray(ep_hist).flatten()
            else:
                epochs = np.linspace(1, outcomes.get("epochs_trained", len(h1)), len(h1))

            valid = np.isfinite(h1) & (h1 > 0)
            if np.any(valid):
                ax.semilogy(
                    epochs[valid], h1[valid],
                    color=self._get_color(m_name, i),
                    linestyle=self._get_linestyle(m_name),
                    label=self._get_label(m_name),
                    lw=2.2,
                    alpha=0.9
                )
                has_plots = True

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("$H^1$ Semi-Norm Error (Log Scale)", fontweight="bold")
        ax.set_title(f"$H^1$ Convergence vs Epochs — {self.problem_name}", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        if has_plots:
            ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_convergence_overlay_time(self, save_name: str = "convergence_overlay_time.png"):
        """2. Overlaid H^1 error vs. wall-clock time across neural + IGA reference markers."""
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            h1 = outcomes.get("h1_error_history", np.array([])).flatten()
            time_hist = outcomes.get("h1_time_history", np.array([])).flatten()
            color = self._get_color(m_name, i)
            ls = self._get_linestyle(m_name)
            label = self._get_label(m_name)

            if len(h1) > 1 and len(time_hist) == len(h1):
                # Neural iterative method
                valid = np.isfinite(h1) & (h1 > 0)
                if np.any(valid):
                    ax.semilogy(time_hist[valid], h1[valid], color=color, linestyle=ls, label=label, lw=2.2)
            else:
                # Direct / IGA solver: instant solve marked with star & horizontal reference line
                final_h1 = float(outcomes.get("final_h1_error", np.nan))
                final_time = float(outcomes.get("elapsed_seconds", 0.0))
                if np.isfinite(final_h1) and final_h1 > 0:
                    ax.axhline(final_h1, color=color, linestyle="--", alpha=0.6, label=f"{label} (Final $H^1$)")
                    ax.plot(final_time, final_h1, marker="*", markersize=12, color=color)

        ax.set_xlabel("Wall-Clock Time (seconds)", fontweight="bold")
        ax.set_ylabel("$H^1$ Semi-Norm Error (Log Scale)", fontweight="bold")
        ax.set_title(f"$H^1$ Convergence vs Wall-Clock Time — {self.problem_name}", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_convergence_overlay_progress(self, save_name: str = "convergence_overlay_progress.png"):
        """3. Overlaid H^1 error vs. training progress percentage (0–100%)."""
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        has_plots = False

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            h1 = outcomes.get("h1_error_history", np.array([])).flatten()
            if len(h1) <= 1:
                continue

            prog = outcomes.get("h1_progress_history", np.linspace(0, 100, len(h1))).flatten()
            valid = np.isfinite(h1) & (h1 > 0)
            if np.any(valid):
                ax.semilogy(
                    prog[valid], h1[valid],
                    color=self._get_color(m_name, i),
                    linestyle=self._get_linestyle(m_name),
                    label=self._get_label(m_name),
                    lw=2.2
                )
                has_plots = True

        ax.set_xlabel("Training Progress (%)", fontweight="bold")
        ax.set_ylabel("$H^1$ Semi-Norm Error (Log Scale)", fontweight="bold")
        ax.set_title(f"$H^1$ Error vs Normalized Training Progress — {self.problem_name}", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        if has_plots:
            ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _render_multi_panel_solution(self, methods_subset: dict[str, dict[str, Any]], title_suffix: str, save_name: str):
        """Helper to render side-by-side solution grids with Exact solution on 100x100 resolution."""
        if not methods_subset:
            return

        total_plots = len(methods_subset) + 1  # Exact + methods
        cols = min(3, total_plots)
        rows = int(np.ceil(total_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        # Panel 0: Exact Solution on 100x100 grid
        x_raw = np.linspace(0, 1, 100)
        t_raw = np.linspace(0, 1, 100)
        X_ex, T_ex = np.meshgrid(x_raw, t_raw)
        Z_ex = self.problem.exact_solution(X_ex, T_ex)

        ax0 = axes_flat[0]
        c0 = ax0.contourf(X_ex, T_ex, Z_ex, levels=50, cmap="viridis")
        fig.colorbar(c0, ax=ax0, label="$u(x, y)$")
        ax0.set_title("Exact Analytical Solution", fontweight="bold", pad=10)
        ax0.set_xlabel("$x$")
        ax0.set_ylabel("$y$ (or $t$)")

        # Panels 1..N: Methods
        for i, (m_name, m_info) in enumerate(methods_subset.items(), start=1):
            ax = axes_flat[i]
            x_m = m_info["outcomes"]["x"]
            t_m = m_info["outcomes"]["t"]
            z_m = m_info["outcomes"]["z_pred"]

            XI, TI, ZI = self._interpolate_to_100x100(x_m, t_m, z_m)
            c = ax.contourf(XI, TI, ZI, levels=50, cmap="viridis")
            fig.colorbar(c, ax=ax, label="$u_h(x, y)$")
            ax.set_title(self._get_label(m_name), fontweight="bold", pad=10)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$ (or $t$)")

        # Hide unused subplots
        for j in range(total_plots, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"Solution Fields Comparison ({title_suffix}) — {self.problem_name}", fontsize=14, fontweight="bold", y=1.01)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _render_multi_panel_error(self, methods_subset: dict[str, dict[str, Any]], title_suffix: str, save_name: str):
        """Helper to render side-by-side synchronized log10 error heatmaps on 100x100 resolution."""
        if not methods_subset:
            return

        total_plots = len(methods_subset)
        cols = min(3, total_plots)
        rows = int(np.ceil(total_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        # Compute global vmin/vmax for exact synchronization
        all_errors = []
        method_grids = {}
        for m_name, m_info in methods_subset.items():
            x_m = m_info["outcomes"]["x"]
            t_m = m_info["outcomes"]["t"]
            z_m = m_info["outcomes"]["z_pred"]
            z_ex = self.problem.exact_solution(x_m, t_m)
            err = np.abs(z_ex - z_m)
            err_log10 = np.log10(np.clip(err, 1e-16, None))

            XI, TI, ZI_err = self._interpolate_to_100x100(x_m, t_m, err_log10)
            method_grids[m_name] = (XI, TI, ZI_err)
            all_errors.extend(ZI_err.flatten())

        valid_errs = [e for e in all_errors if np.isfinite(e)]
        vmin = float(np.percentile(valid_errs, 1)) if valid_errs else -8.0
        vmax = float(np.percentile(valid_errs, 99)) if valid_errs else 0.0

        for i, (m_name, (XI, TI, ZI_err)) in enumerate(method_grids.items()):
            ax = axes_flat[i]
            levels = np.linspace(vmin, vmax, 50)
            c = ax.contourf(XI, TI, ZI_err, levels=levels, cmap="inferno", vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(c, ax=ax)
            cbar.set_label("$\\log_{10}|u - u_h|$")
            ax.set_title(self._get_label(m_name), fontweight="bold", pad=10)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$ (or $t$)")

        for j in range(total_plots, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"Pointwise Error Heatmaps ({title_suffix}) — {self.problem_name}", fontsize=14, fontweight="bold", y=1.01)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_side_by_side_solution_grids(self):
        """Generates separated Solution Grids (2D and 3D) for Uniform and Boundary Layer sampling."""
        uniform_methods = {k: v for k, v in self.methods_data.items() if "boundary" not in k and "adaptive" not in k}
        boundary_methods = {k: v for k, v in self.methods_data.items() if "boundary" in k or "adaptive" in k}

        # 1. 2D Solution Grids
        if uniform_methods:
            self._render_multi_panel_solution(uniform_methods, "Uniform Sampling / Meshing", "side_by_side_solution_uniform.png")
        if boundary_methods:
            self._render_multi_panel_solution(boundary_methods, "Boundary Layer & Adaptive Sampling", "side_by_side_solution_boundary.png")
        self._render_multi_panel_solution(self.methods_data, "All Methods", "side_by_side_solution_grid.png")

        # 2. 3D Solution Surface Grids
        if uniform_methods:
            self._render_multi_panel_3d_solution(uniform_methods, "Uniform Sampling / Meshing", "side_by_side_3d_solution_uniform.png")
        if boundary_methods:
            self._render_multi_panel_3d_solution(boundary_methods, "Boundary Layer & Adaptive Sampling", "side_by_side_3d_solution_boundary.png")
        self._render_multi_panel_3d_solution(self.methods_data, "All Methods", "side_by_side_3d_solution_grid.png")

    def _render_multi_panel_3d_solution(self, methods_subset: dict[str, dict[str, Any]], title_suffix: str, save_name: str):
        """Helper to render side-by-side 3D elevation solution surfaces with exact solution wireframes."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if not methods_subset:
            return

        total_plots = len(methods_subset) + 1  # Exact + methods
        cols = min(3, total_plots)
        rows = int(np.ceil(total_plots / cols))

        fig = plt.figure(figsize=(6.0 * cols, 5.2 * rows), dpi=300)

        # Panel 0: Exact Solution on 100x100 grid
        x_raw = np.linspace(0, 1, 100)
        t_raw = np.linspace(0, 1, 100)
        X_ex, T_ex = np.meshgrid(x_raw, t_raw)
        Z_ex = self.problem.exact_solution(X_ex, T_ex)

        ax0 = fig.add_subplot(rows, cols, 1, projection="3d")
        surf0 = ax0.plot_surface(X_ex, T_ex, Z_ex, cmap="viridis", alpha=0.9, edgecolor="none", antialiased=True)
        ax0.set_title("Exact Solution $u(x, y)$", fontweight="bold", pad=12)
        ax0.set_xlabel("$x$", labelpad=6)
        ax0.set_ylabel("$y$ (or $t$)", labelpad=6)
        ax0.set_zlabel("$u$", labelpad=6)
        ax0.view_init(elev=28, azim=-125)

        # Panels 1..N: Methods
        for i, (m_name, m_info) in enumerate(methods_subset.items(), start=1):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            x_m = m_info["outcomes"]["x"]
            t_m = m_info["outcomes"]["t"]
            z_m = m_info["outcomes"]["z_pred"]

            XI, TI, ZI = self._interpolate_to_100x100(x_m, t_m, z_m)
            ax.plot_surface(XI, TI, ZI, cmap="viridis", alpha=0.85, edgecolor="none", antialiased=True)
            # Overlay exact solution wireframe for direct 3D visual comparison
            ax.plot_wireframe(X_ex, T_ex, Z_ex, color="black", rstride=10, cstride=10, alpha=0.35, linewidth=0.7)

            ax.set_title(self._get_label(m_name), fontweight="bold", pad=12)
            ax.set_xlabel("$x$", labelpad=6)
            ax.set_ylabel("$y$ (or $t$)", labelpad=6)
            ax.set_zlabel("$u_h$", labelpad=6)
            ax.view_init(elev=28, azim=-125)

        fig.suptitle(f"3D Elevation Solutions Comparison ({title_suffix}) — {self.problem_name}", fontsize=14, fontweight="bold", y=1.02)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    def plot_side_by_side_error_grids(self):
        """Generates separated Error Heatmaps (2D and 3D) for Uniform and Boundary Layer sampling."""
        uniform_methods = {k: v for k, v in self.methods_data.items() if "boundary" not in k and "adaptive" not in k}
        boundary_methods = {k: v for k, v in self.methods_data.items() if "boundary" in k or "adaptive" in k}

        # 1. 2D Error Heatmaps
        if uniform_methods:
            self._render_multi_panel_error(uniform_methods, "Uniform Sampling / Meshing", "side_by_side_error_uniform.png")
        if boundary_methods:
            self._render_multi_panel_error(boundary_methods, "Boundary Layer & Adaptive Sampling", "side_by_side_error_boundary.png")
        self._render_multi_panel_error(self.methods_data, "All Methods", "side_by_side_error_grid.png")

        # 2. 3D Error Elevation Surfaces
        if uniform_methods:
            self._render_multi_panel_3d_error(uniform_methods, "Uniform Sampling / Meshing", "side_by_side_3d_error_uniform.png")
        if boundary_methods:
            self._render_multi_panel_3d_error(boundary_methods, "Boundary Layer & Adaptive Sampling", "side_by_side_3d_error_boundary.png")
        self._render_multi_panel_3d_error(self.methods_data, "All Methods", "side_by_side_3d_error_grid.png")

    def _render_multi_panel_3d_error(self, methods_subset: dict[str, dict[str, Any]], title_suffix: str, save_name: str):
        """Helper to render side-by-side 3D elevation error surfaces with synchronized scales on 100x100 resolution."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if not methods_subset:
            return

        total_plots = len(methods_subset)
        cols = min(3, total_plots)
        rows = int(np.ceil(total_plots / cols))

        fig = plt.figure(figsize=(6.0 * cols, 5.2 * rows), dpi=300)

        # Precompute 3D error surfaces and find global vmax for unified color scale
        method_grids = {}
        max_err = 1e-12
        for m_name, m_info in methods_subset.items():
            x_m = m_info["outcomes"]["x"]
            t_m = m_info["outcomes"]["t"]
            z_m = m_info["outcomes"]["z_pred"]
            z_ex = self.problem.exact_solution(x_m, t_m)
            err = np.abs(z_ex - z_m)

            XI, TI, ZI_err = self._interpolate_to_100x100(x_m, t_m, err)
            method_grids[m_name] = (XI, TI, ZI_err)
            valid_err = ZI_err[np.isfinite(ZI_err)]
            if len(valid_err) > 0:
                max_err = max(max_err, float(np.percentile(valid_err, 99.5)))

        for i, (m_name, (XI, TI, ZI_err)) in enumerate(method_grids.items()):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            ax.plot_surface(
                XI, TI, ZI_err,
                cmap="inferno",
                alpha=0.9,
                edgecolor="none",
                antialiased=True,
                vmin=0.0,
                vmax=max_err
            )
            ax.set_title(self._get_label(m_name), fontweight="bold", pad=12)
            ax.set_xlabel("$x$", labelpad=6)
            ax.set_ylabel("$y$ (or $t$)", labelpad=6)
            ax.set_zlabel("Error $|u - u_h|$", labelpad=6)
            ax.set_zlim(0, max_err * 1.05)
            ax.view_init(elev=28, azim=-125)

        fig.suptitle(f"3D Pointwise Error Elevation ({title_suffix}) — {self.problem_name}", fontsize=14, fontweight="bold", y=1.02)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_boundary_layer_slice_comparison(self, save_name: str = "boundary_layer_slice_comparison.png"):
        """6. Multi-method 1D cross-sectional cut with highlighted Exact Solution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        ax1 = axes[0]
        ax2 = axes[1]

        # Exact lines on ultra-fine grid (thick solid black)
        t_fine = np.linspace(0, 1, 400)
        x_mid = np.full_like(t_fine, 0.5)
        z_exact_mid = self.problem.exact_solution(x_mid, t_fine)
        ax1.plot(t_fine, z_exact_mid, color="black", linestyle="-", lw=3.0, label="Exact $u(x=0.5, y)$", zorder=10)

        x_fine = np.linspace(0, 1, 400)
        y_near = 0.95
        t_near = np.full_like(x_fine, y_near)
        z_exact_near = self.problem.exact_solution(x_fine, t_near)
        ax2.plot(x_fine, z_exact_near, color="black", linestyle="-", lw=3.0, label=f"Exact $u(x, y={y_near})$", zorder=10)

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            x_arr = outcomes["x"]
            t_arr = outcomes["t"]
            z_p = outcomes["z_pred"]

            color = self._get_color(m_name, i)
            ls = self._get_linestyle(m_name)
            marker = self._get_marker(m_name)
            label = self._get_label(m_name)

            unique_x = np.unique(np.round(x_arr, 5))
            mid_x = unique_x[np.argmin(np.abs(unique_x - 0.5))]
            mask_x = np.isclose(x_arr, mid_x, atol=1e-2)

            if np.any(mask_x):
                t_sub = t_arr[mask_x]
                sort_idx = np.argsort(t_sub)
                mark_every = max(1, len(sort_idx) // 10)
                ax1.plot(
                    t_sub[sort_idx], z_p[mask_x][sort_idx],
                    color=color, linestyle=ls, lw=2.0, alpha=0.85,
                    marker=marker, markevery=mark_every, markersize=5,
                    label=label
                )

            unique_t = np.unique(np.round(t_arr, 5))
            target_t = unique_t[np.argmin(np.abs(unique_t - y_near))]
            mask_t = np.isclose(t_arr, target_t, atol=1e-2)

            if np.any(mask_t):
                x_sub = x_arr[mask_t]
                sort_idx = np.argsort(x_sub)
                mark_every = max(1, len(sort_idx) // 10)
                ax2.plot(
                    x_sub[sort_idx], z_p[mask_t][sort_idx],
                    color=color, linestyle=ls, lw=2.0, alpha=0.85,
                    marker=marker, markevery=mark_every, markersize=5,
                    label=label
                )

        ax1.set_xlabel("$y$ (or $t$)", fontweight="bold")
        ax1.set_ylabel("Solution $u$", fontweight="bold")
        ax1.set_title("Cross-Section along $x = 0.5$", fontweight="bold", pad=10)
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2.set_xlabel("$x$", fontweight="bold")
        ax2.set_ylabel("Solution $u$", fontweight="bold")
        ax2.set_title(f"Boundary Layer Cross-Section at $y \\approx {y_near}$", fontweight="bold", pad=10)
        ax2.grid(True, linestyle="--", alpha=0.4)

        # Unified outside legend below plots to completely eliminate overlap
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(5, len(labels)),
            frameon=True,
            fontsize=8.5
        )

        fig.suptitle(f"1D Cross-Sectional Profiles Comparison — {self.problem_name}", fontsize=14, fontweight="bold", y=1.02)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def generate_summary_tables(self, md_name: str = "summary_table.md", csv_name: str = "summary_table.csv"):
        """7. Comprehensive set of formatted Markdown and CSV tables comparing accuracy, efficiency, and families."""
        fieldnames = [
            "Method",
            "Method ID",
            "Family",
            "Sampling / Mesh",
            "DoFs / Params",
            "Elapsed Time (s)",
            "H1 Semi-Norm Error",
            "L2 Error",
            "L_inf Error",
            "CES Score",
        ]

        rows = []
        for m_name, m_info in self.methods_data.items():
            meta = m_info["metadata"]
            outcomes = m_info["outcomes"]
            res = meta.get("results", {})
            sol_key = list(res.keys())[0] if res else ""
            sol_meta = res.get(sol_key, {})

            dofs = (
                sol_meta.get("trainable_parameters_or_dofs")
                or sol_meta.get("trainable_parameters")
                or sol_meta.get("degrees_of_freedom")
                or 0
            )
            elapsed = float(outcomes.get("elapsed_seconds", sol_meta.get("elapsed_seconds", 0.0)))
            h1 = float(outcomes.get("final_h1_error", sol_meta.get("final_h1_error", np.nan)))
            l2 = float(outcomes.get("final_l2_error", sol_meta.get("final_l2_error", np.nan)))
            linf = float(outcomes.get("final_linf_error", sol_meta.get("final_linf_error", np.nan)))
            ces = calculate_ces(h1, elapsed, dofs)

            # Infer Family and Sampling
            fam = "PINN" if "pinn" in m_name else ("KAN" if "kan" in m_name else "IGA")
            samp = "Boundary / Adaptive" if ("boundary" in m_name or "adaptive" in m_name) else "Uniform"

            rows.append({
                "Method": self._get_label(m_name),
                "Method ID": m_name,
                "Family": fam,
                "Sampling / Mesh": samp,
                "DoFs / Params": dofs,
                "Elapsed Time (s)": f"{elapsed:.3f}",
                "H1 Semi-Norm Error": f"{h1:.6e}" if np.isfinite(h1) else "N/A",
                "L2 Error": f"{l2:.6e}" if np.isfinite(l2) else "N/A",
                "L_inf Error": f"{linf:.6e}" if np.isfinite(linf) else "N/A",
                "CES Score": f"{ces:.2f}" if np.isfinite(ces) else "N/A",
            })

        # Save Main CSV
        csv_path = os.path.join(self.comparisons_dir, csv_name)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Save Rich Multi-Section Markdown
        md_path = os.path.join(self.comparisons_dir, md_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Comprehensive Benchmark Summary: {self.problem_name} ({self.problem_id})\n\n")
            f.write(f"- **Problem Name**: `{self.problem_name}`\n")
            f.write(f"- **Perturbation Parameter ($\epsilon$)**: `{self.epsilon}`\n")
            f.write(f"- **Evaluated Methods**: {len(rows)}\n\n")

            # Table 1: Full Overview
            f.write("## 1. Full Method Comparison Table\n\n")
            f.write("| " + " | ".join(fieldnames) + " |\n")
            f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(str(r[k]) for k in fieldnames) + " |\n")
            f.write("\n")

            # Table 2: Ranked by Accuracy (H1 Error)
            valid_h1_rows = [r for r in rows if r["H1 Semi-Norm Error"] != "N/A"]
            valid_h1_rows.sort(key=lambda r: float(r["H1 Semi-Norm Error"]))
            f.write("## 2. Methods Ranked by Accuracy ($H^1$ Semi-Norm Error)\n\n")
            f.write("| Rank | Method | Family | $H^1$ Semi-Norm Error | $L_2$ Error | $L_\infty$ Error |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for rank, r in enumerate(valid_h1_rows, start=1):
                f.write(f"| {rank} | **{r['Method']}** | {r['Family']} | `{r['H1 Semi-Norm Error']}` | `{r['L2 Error']}` | `{r['L_inf Error']}` |\n")
            f.write("\n")

            # Table 3: Ranked by Computational Efficiency Score (CES)
            valid_ces_rows = [r for r in rows if r["CES Score"] != "N/A"]
            valid_ces_rows.sort(key=lambda r: float(r["CES Score"]), reverse=True)
            f.write("## 3. Methods Ranked by Computational Efficiency Score (CES)\n\n")
            f.write("> **CES Formula**: $\\text{CES} = -\\log_{10}(H^1_{\\text{error}} \\cdot \\sqrt{\\text{Time}} \\cdot \\sqrt[4]{\\text{DoFs}})$\n\n")
            f.write("| Rank | Method | Family | CES Score | Elapsed Time (s) | DoFs / Params | $H^1$ Error |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for rank, r in enumerate(valid_ces_rows, start=1):
                f.write(f"| {rank} | **{r['Method']}** | {r['Family']} | **{r['CES Score']}** | {r['Elapsed Time (s)']}s | {r['DoFs / Params']} | `{r['H1 Semi-Norm Error']}` |\n")
            f.write("\n")

            # Table 4: Family-Level Aggregate Summary
            f.write("## 4. Family-Level Aggregate Performance\n\n")
            f.write("| Method Family | Methods Tested | Best $H^1$ Error | Avg Elapsed Time (s) | Best CES Score |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for fam in ["PINN", "KAN", "IGA"]:
                fam_rows = [r for r in rows if r["Family"] == fam and r["H1 Semi-Norm Error"] != "N/A"]
                if fam_rows:
                    best_h1 = min(float(r["H1 Semi-Norm Error"]) for r in fam_rows)
                    avg_time = sum(float(r["Elapsed Time (s)"]) for r in fam_rows) / len(fam_rows)
                    best_ces = max(float(r["CES Score"]) for r in fam_rows if r["CES Score"] != "N/A")
                    f.write(f"| **{fam} Family** | {len(fam_rows)} | `{best_h1:.6e}` | {avg_time:.2f}s | **{best_ces:.2f}** |\n")
            f.write("\n")

        print(f"  [+] Saved: {csv_name}")
        print(f"  [+] Saved: {md_name}")

    def generate_all(self):
        """Alias for generate_all_problem_comparisons."""
        self.generate_all_problem_comparisons()

    def generate_all_problem_comparisons(self):
        """Generates all 7 problem-level comparison artifacts."""
        if not self.methods_data:
            print(f"No method runs found in {self.problem_dir}")
            return

        print(f"=== Generating Problem Comparisons for [{self.problem_name}] ({len(self.methods_data)} methods) ===")
        self.plot_convergence_overlay_epochs()
        self.plot_convergence_overlay_time()
        self.plot_convergence_overlay_progress()
        self.plot_side_by_side_solution_grids()
        self.plot_side_by_side_error_grids()
        self.plot_boundary_layer_slice_comparison()
        self.generate_summary_tables()
        print(f"Comparisons generated in: {self.comparisons_dir}\n")
