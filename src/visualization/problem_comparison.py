import os
import glob
import csv
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from src.problems import get_problem


METHOD_COLORS = {
    "pinn_uniform": "#1f77b4",
    "pinn_boundary": "#aec7e8",
    "r_pinn_uniform": "#ff7f0e",
    "r_pinn_boundary": "#ffbb78",
    "nurbs_kan_uniform": "#2ca02c",
    "nurbs_kan_boundary": "#98df8a",
    "r_nurbs_kan_uniform": "#d62728",
    "r_nurbs_kan_boundary": "#ff9896",
    "iga_standard_uniform": "#9467bd",
    "iga_supg_uniform": "#8c564b",
    "iga_supg_adaptive": "#e377c2",
    "iga_igrm_adaptive": "#7f7f7f",
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
        self.methods_data: Dict[str, Dict[str, Any]] = {}
        self.problem_name = "poisson_sine"
        self.epsilon = 0.01

        self._load_all_methods()

    def _load_all_methods(self):
        """Discovers and loads all method directories inside problem_dir."""
        subdirs = [d for d in glob.glob(os.path.join(self.problem_dir, "*")) if os.path.isdir(d)]
        for d in sorted(subdirs):
            name = os.path.basename(d)
            if name == "comparisons":
                continue

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
                "config": cfg,
            }

        self.problem = get_problem(self.problem_name, epsilon=self.epsilon)

    def _get_color(self, method_name: str, idx: int = 0) -> str:
        if method_name in METHOD_COLORS:
            return METHOD_COLORS[method_name]
        colors = plt.cm.tab10.colors
        return colors[idx % len(colors)]

    def _get_label(self, method_name: str) -> str:
        return METHOD_LABELS.get(method_name, method_name.replace("_", " ").title())

    def generate_all(self):
        """Runs the complete suite of 7 comparison artifacts."""
        if not self.methods_data:
            print(f"No valid method runs found in {self.problem_dir}")
            return

        print(f"=== Generating Problem Comparisons for [{self.problem_id}] ({len(self.methods_data)} methods) ===")
        self.plot_convergence_overlay_epochs()
        self.plot_convergence_overlay_time()
        self.plot_convergence_overlay_progress()
        self.plot_side_by_side_solution_grid()
        self.plot_side_by_side_error_grid()
        self.plot_boundary_layer_slice_comparison()
        self.generate_summary_tables()
        print(f"Comparisons generated in: {self.comparisons_dir}\n")

    def plot_convergence_overlay_epochs(self, save_name: str = "convergence_overlay_epochs.png"):
        """1. Overlaid H^1 error vs. epochs for iterative neural methods."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        has_plots = False

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            h1 = outcomes.get("h1_error_history", np.array([])).flatten()
            if len(h1) == 0:
                continue

            epochs = outcomes.get("h1_epoch_history", np.arange(1, len(h1) + 1)).flatten()
            valid = np.isfinite(h1) & (h1 > 0)
            if np.any(valid):
                ax.semilogy(
                    epochs[valid], h1[valid],
                    color=self._get_color(m_name, i),
                    label=self._get_label(m_name),
                    lw=2.0,
                    alpha=0.9
                )
                has_plots = True

        ax.set_xlabel("Epoch")
        ax.set_ylabel("$H^1$ Semi-Norm Error")
        ax.set_title(f"$H^1$ Convergence vs Epochs — {self.problem_name}", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        if has_plots:
            ax.legend(frameon=True, loc="upper right")

        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_convergence_overlay_time(self, save_name: str = "convergence_overlay_time.png"):
        """2. Overlaid H^1 error vs. wall-clock time across neural + IGA reference lines."""
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            h1 = outcomes.get("h1_error_history", np.array([])).flatten()
            time_hist = outcomes.get("h1_time_history", np.array([])).flatten()
            color = self._get_color(m_name, i)
            label = self._get_label(m_name)

            if len(h1) > 1 and len(time_hist) == len(h1):
                # Neural iterative method
                valid = np.isfinite(h1) & (h1 > 0)
                if np.any(valid):
                    ax.semilogy(time_hist[valid], h1[valid], color=color, label=label, lw=2.0)
            else:
                # Direct / IGA solver: show final point and horizontal reference line
                final_h1 = float(outcomes.get("final_h1_error", np.nan))
                final_time = float(outcomes.get("elapsed_seconds", 0.0))
                if np.isfinite(final_h1) and final_h1 > 0:
                    ax.axhline(final_h1, color=color, linestyle="--", alpha=0.7, label=f"{label} (Final $H^1$)")
                    ax.plot(final_time, final_h1, marker="*", markersize=10, color=color)

        ax.set_xlabel("Wall-Clock Time (seconds)")
        ax.set_ylabel("$H^1$ Semi-Norm Error")
        ax.set_title(f"$H^1$ Convergence vs Wall-Clock Time — {self.problem_name}", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, loc="upper right", bbox_to_anchor=(1.35, 1.0))

        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_convergence_overlay_progress(self, save_name: str = "convergence_overlay_progress.png"):
        """3. Overlaid H^1 error vs. training progress percentage (0–100%)."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        has_plots = False

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            h1 = outcomes.get("h1_error_history", np.array([])).flatten()
            if len(h1) == 0:
                continue

            prog = outcomes.get("h1_progress_history", np.linspace(0, 100, len(h1))).flatten()
            valid = np.isfinite(h1) & (h1 > 0)
            if np.any(valid):
                ax.semilogy(
                    prog[valid], h1[valid],
                    color=self._get_color(m_name, i),
                    label=self._get_label(m_name),
                    lw=2.0
                )
                has_plots = True

        ax.set_xlabel("Training Progress (%)")
        ax.set_ylabel("$H^1$ Semi-Norm Error")
        ax.set_title(f"$H^1$ Error vs Normalized Training Progress — {self.problem_name}", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        if has_plots:
            ax.legend(frameon=True, loc="upper right")

        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _get_structured_2d(self, x, t, z):
        x_f = np.asarray(x).flatten()
        t_f = np.asarray(t).flatten()
        z_f = np.asarray(z).flatten()
        nx = len(np.unique(np.round(x_f, 7)))
        nt = len(np.unique(np.round(t_f, 7)))
        if nx * nt == len(z_f):
            return x_f.reshape(nx, nt), t_f.reshape(nx, nt), z_f.reshape(nx, nt)
        dim = int(np.sqrt(len(z_f)))
        if dim * dim == len(z_f):
            return x_f.reshape(dim, dim), t_f.reshape(dim, dim), z_f.reshape(dim, dim)
        return None, None, None

    def plot_side_by_side_solution_grid(self, save_name: str = "side_by_side_solution_grid.png"):
        """4. Multi-panel figure comparing Exact Solution against each numerical method."""
        num_methods = len(self.methods_data)
        total_plots = num_methods + 1  # 1 for Exact
        cols = min(3, total_plots)
        rows = int(np.ceil(total_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        # Plot 1: Exact Solution
        first_method = list(self.methods_data.values())[0]
        x_base = first_method["outcomes"]["x"]
        t_base = first_method["outcomes"]["t"]
        z_exact = self.problem.exact_solution(x_base, t_base)

        X, T, Z = self._get_structured_2d(x_base, t_base, z_exact)
        ax0 = axes_flat[0]
        if X is not None:
            c0 = ax0.contourf(X, T, Z, levels=50, cmap="viridis")
            fig.colorbar(c0, ax=ax0)
        else:
            s0 = ax0.scatter(x_base, t_base, c=z_exact, cmap="viridis", s=15)
            fig.colorbar(s0, ax=ax0)
        ax0.set_title("Exact Analytical Solution", fontweight="bold")
        ax0.set_xlabel("$x$")
        ax0.set_ylabel("$y$ (or $t$)")

        # Plot numerical methods
        for i, (m_name, m_info) in enumerate(self.methods_data.items(), start=1):
            ax = axes_flat[i]
            x_m = m_info["outcomes"]["x"]
            t_m = m_info["outcomes"]["t"]
            z_m = m_info["outcomes"]["z_pred"]
            Xm, Tm, Zm = self._get_structured_2d(x_m, t_m, z_m)

            if Xm is not None:
                c = ax.contourf(Xm, Tm, Zm, levels=50, cmap="viridis")
                fig.colorbar(c, ax=ax)
            else:
                s = ax.scatter(x_m, t_m, c=z_m, cmap="viridis", s=15)
                fig.colorbar(s, ax=ax)
            ax.set_title(self._get_label(m_name), fontweight="bold")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$ (or $t$)")

        # Hide unused subplots
        for j in range(total_plots, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"Side-by-Side Solution Field Comparison — {self.problem_name}", fontsize=14, fontweight="bold", y=1.01)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_side_by_side_error_grid(self, save_name: str = "side_by_side_error_grid.png"):
        """5. Side-by-side log10 pointwise error heatmaps with synchronized color scale."""
        num_methods = len(self.methods_data)
        cols = min(3, num_methods)
        rows = int(np.ceil(num_methods / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), dpi=300, squeeze=False)
        axes_flat = axes.flatten()

        # Compute global min and max log error for synchronization
        all_log_errors = []
        method_errors = {}
        for m_name, m_info in self.methods_data.items():
            x_m = m_info["outcomes"]["x"]
            t_m = m_info["outcomes"]["t"]
            z_m = m_info["outcomes"]["z_pred"]
            z_ex = self.problem.exact_solution(x_m, t_m)
            err = np.abs(z_ex - z_m)
            err_log10 = np.log10(np.clip(err, 1e-16, None))
            all_log_errors.extend(err_log10)
            method_errors[m_name] = (x_m, t_m, err_log10)

        vmin = float(np.percentile(all_log_errors, 1)) if all_log_errors else -8.0
        vmax = float(np.percentile(all_log_errors, 99)) if all_log_errors else 0.0

        for i, (m_name, (x_m, t_m, err_log10)) in enumerate(method_errors.items()):
            ax = axes_flat[i]
            Xm, Tm, Zm = self._get_structured_2d(x_m, t_m, err_log10)
            levels = np.linspace(vmin, vmax, 50)

            if Xm is not None:
                c = ax.contourf(Xm, Tm, Zm, levels=levels, cmap="inferno", vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(c, ax=ax)
                cbar.set_label("$\\log_{10}|u - u_h|$")
            else:
                s = ax.scatter(x_m, t_m, c=err_log10, cmap="inferno", s=15, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(s, ax=ax)
                cbar.set_label("$\\log_{10}|u - u_h|$")

            ax.set_title(self._get_label(m_name), fontweight="bold")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$ (or $t$)")

        for j in range(num_methods, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"Pointwise Absolute Error $\\log_{10}|u - u_h|$ — {self.problem_name}", fontsize=14, fontweight="bold", y=1.01)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_boundary_layer_slice_comparison(self, save_name: str = "boundary_layer_slice_comparison.png"):
        """6. Multi-method 1D cross-sectional cut across singular boundary layer."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Slice 1: x = 0.5
        ax1 = axes[0]
        # Slice 2: y near 0.95 or 1.0
        ax2 = axes[1]

        # Exact line on fine grid
        t_fine = np.linspace(0, 1, 300)
        x_mid = np.full_like(t_fine, 0.5)
        z_exact_mid = self.problem.exact_solution(x_mid, t_fine)
        ax1.plot(t_fine, z_exact_mid, "k-", lw=2.5, label="Exact $u(x=0.5, y)$", zorder=5)

        x_fine = np.linspace(0, 1, 300)
        y_near = 0.95
        t_near = np.full_like(x_fine, y_near)
        z_exact_near = self.problem.exact_solution(x_fine, t_near)
        ax2.plot(x_fine, z_exact_near, "k-", lw=2.5, label=f"Exact $u(x, y={y_near})$", zorder=5)

        for i, (m_name, m_info) in enumerate(self.methods_data.items()):
            outcomes = m_info["outcomes"]
            x_arr = outcomes["x"]
            t_arr = outcomes["t"]
            z_p = outcomes["z_pred"]

            unique_x = np.unique(np.round(x_arr, 5))
            mid_x = unique_x[np.argmin(np.abs(unique_x - 0.5))]
            mask_x = np.isclose(x_arr, mid_x, atol=1e-2)

            color = self._get_color(m_name, i)
            label = self._get_label(m_name)

            if np.any(mask_x):
                t_sub = t_arr[mask_x]
                sort_idx = np.argsort(t_sub)
                ax1.plot(t_sub[sort_idx], z_p[mask_x][sort_idx], "--", color=color, lw=1.8, label=label)

            unique_t = np.unique(np.round(t_arr, 5))
            target_t = unique_t[np.argmin(np.abs(unique_t - y_near))]
            mask_t = np.isclose(t_arr, target_t, atol=1e-2)

            if np.any(mask_t):
                x_sub = x_arr[mask_t]
                sort_idx = np.argsort(x_sub)
                ax2.plot(x_sub[sort_idx], z_p[mask_t][sort_idx], "--", color=color, lw=1.8, label=label)

        ax1.set_xlabel("$y$ (or $t$)")
        ax1.set_ylabel("Solution $u$")
        ax1.set_title("Cross-Section along $x = 0.5$", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(frameon=True, fontsize=9)

        ax2.set_xlabel("$x$")
        ax2.set_ylabel("Solution $u$")
        ax2.set_title(f"Boundary Layer Cross-Section at $y \\approx {y_near}$", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(frameon=True, fontsize=9)

        fig.suptitle(f"Cross-Method Boundary Layer Resolution — {self.problem_name}", fontsize=14, fontweight="bold", y=1.02)
        save_path = os.path.join(self.comparisons_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def generate_summary_tables(self, md_name: str = "summary_table.md", csv_name: str = "summary_table.csv"):
        """7. Formatted Markdown and CSV tables summarizing L2, H1, Linf errors, DoFs, and runtimes."""
        fieldnames = [
            "Method",
            "Method ID",
            "DoFs / Params",
            "Elapsed Time (s)",
            "H1 Semi-Norm Error",
            "L2 Error",
            "L_inf Error",
        ]
        rows = []
        for m_name, m_info in self.methods_data.items():
            meta = m_info["metadata"]
            outcomes = m_info["outcomes"]
            res = meta.get("results", {})

            # Extract solver-specific result dict
            solver_key = list(res.keys())[0] if res else ""
            s_dict = res.get(solver_key, {})

            h1 = float(outcomes.get("final_h1_error", s_dict.get("final_h1_error", np.nan)))
            l2 = float(outcomes.get("final_l2_error", s_dict.get("final_l2_error", np.nan)))
            linf = float(outcomes.get("final_linf_error", s_dict.get("final_linf_error", np.nan)))
            time_sec = float(outcomes.get("elapsed_seconds", s_dict.get("elapsed_seconds", np.nan)))
            dofs = int(s_dict.get("trainable_parameters_or_dofs", 0))

            rows.append({
                "Method": self._get_label(m_name),
                "Method ID": m_name,
                "DoFs / Params": str(dofs),
                "Elapsed Time (s)": f"{time_sec:.3f}" if np.isfinite(time_sec) else "N/A",
                "H1 Semi-Norm Error": f"{h1:.6e}" if np.isfinite(h1) else "N/A",
                "L2 Error": f"{l2:.6e}" if np.isfinite(l2) else "N/A",
                "L_inf Error": f"{linf:.6e}" if np.isfinite(linf) else "N/A",
            })

        # Save CSV using built-in csv module
        csv_path = os.path.join(self.comparisons_dir, csv_name)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Save Markdown table
        md_path = os.path.join(self.comparisons_dir, md_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Benchmark Summary: {self.problem_name} ({self.problem_id})\n\n")
            f.write(f"- **Problem**: `{self.problem_name}`\n")
            f.write(f"- **Epsilon**: `{self.epsilon}`\n")
            f.write(f"- **Evaluated Methods**: {len(rows)}\n\n")
            
            # Format markdown table headers
            f.write("| " + " | ".join(fieldnames) + " |\n")
            f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(str(r[k]) for k in fieldnames) + " |\n")
            f.write("\n")

        print(f"  [+] Saved: {csv_name}")
        print(f"  [+] Saved: {md_name}")
