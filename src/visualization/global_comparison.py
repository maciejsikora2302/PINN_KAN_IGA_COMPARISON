import csv
import glob
import math
import os
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Distinct Family Color Palette
METHOD_COLORS = {
    # PINN Family (Blues)
    "pinn_uniform": "#1f77b4",
    "pinn_boundary": "#4ba3e3",
    "r_pinn_uniform": "#004c6d",
    "r_pinn_boundary": "#257d9d",
    # KAN Family (Greens)
    "nurbs_kan_uniform": "#2ca02c",
    "nurbs_kan_boundary": "#66c2a5",
    "r_nurbs_kan_uniform": "#005a24",
    "r_nurbs_kan_boundary": "#238b45",
    # IGA Family (Purples/Reds)
    "iga_standard_uniform": "#9467bd",
    "iga_supg_uniform": "#8c564b",
    "iga_supg_adaptive": "#d62728",
    "iga_igrm_adaptive": "#e377c2",
}

PROBLEM_MARKERS = {
    "poisson_sine": "o",
    "poisson_exp": "s",
    "eriksson_johnson": "^",
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


def safe_float(val: Any, default: float = float('nan')) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class GlobalComparator:
    """
    Scans the entire output/ directory across all problem folders and generates
    global multi-problem benchmark synthesis artifacts in output/global_comparisons/.
    """

    def __init__(self, root_output_dir: str = "output"):
        self.root_dir = os.path.abspath(root_output_dir)
        self.global_dir = os.path.join(self.root_dir, "global_comparisons")
        os.makedirs(self.global_dir, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self._scan_all_runs()

    def _scan_all_runs(self):
        """Scans all subdirectories of output/ for metadata.yaml and outcomes.npz."""
        metadata_paths = glob.glob(os.path.join(self.root_dir, "*", "*", "metadata.yaml"))
        for meta_path in metadata_paths:
            run_dir = os.path.dirname(meta_path)
            problem_dir = os.path.dirname(run_dir)
            problem_id = os.path.basename(problem_dir)
            method_name = os.path.basename(run_dir)

            if problem_id in ("global_comparisons", "test_runs") or method_name == "comparisons":
                continue

            outcomes_path = os.path.join(run_dir, "outcomes.npz")
            if not os.path.exists(outcomes_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                npz_data = dict(np.load(outcomes_path, allow_pickle=True))
            except Exception as e:
                print(f"Warning: Failed to load {meta_path}: {e}")
                continue

            cfg = meta.get("config", {})
            prob_name = cfg.get("PROBLEM_NAME", cfg.get("problem_id", problem_id))
            eps = safe_float(cfg.get("EPSILON", 0.01))

            res = meta.get("results", {})
            sol_key = list(res.keys())[0] if res else ""
            sol_meta = res.get(sol_key, {})

            dofs = (
                sol_meta.get("trainable_parameters_or_dofs")
                or sol_meta.get("trainable_parameters")
                or sol_meta.get("degrees_of_freedom")
                or 0
            )

            h1 = safe_float(npz_data.get("final_h1_error", sol_meta.get("final_h1_error")))
            l2 = safe_float(npz_data.get("final_l2_error", sol_meta.get("final_l2_error")))
            linf = safe_float(npz_data.get("final_linf_error", sol_meta.get("final_linf_error")))
            time_s = safe_float(npz_data.get("elapsed_seconds", sol_meta.get("elapsed_seconds")))
            ces = calculate_ces(h1, time_s, dofs)

            self.records.append({
                "problem_id": problem_id,
                "problem_name": prob_name,
                "epsilon": eps,
                "method_name": method_name,
                "solver": cfg.get("SOLVER", sol_meta.get("solver", "unknown")),
                "rpinn": int(cfg.get("RPINN", 0)),
                "sampler_type": cfg.get("SAMPLER_TYPE", cfg.get("IGA_MESH_TYPE", "uniform")),
                "h1_error": h1,
                "l2_error": l2,
                "linf_error": linf,
                "elapsed_seconds": time_s,
                "dofs": dofs,
                "ces_score": ces,
                "run_dir": run_dir,
            })

    def _get_color(self, method_name: str, idx: int = 0) -> str:
        return METHOD_COLORS.get(method_name, plt.cm.tab10(idx % 10))

    def _get_marker(self, problem_name: str) -> str:
        for k, m in PROBLEM_MARKERS.items():
            if k in problem_name.lower():
                return m
        return "o"

    def _get_label(self, method_name: str) -> str:
        return METHOD_LABELS.get(method_name, method_name.replace("_", " ").title())

    def generate_all(self):
        """Generates all 7 global benchmark synthesis artifacts."""
        if not self.records:
            print(f"No completed method runs found in {self.root_dir}.")
            return

        print(f"=== Generating Global Synthesis Benchmark Suite ({len(self.records)} runs across all problems) ===")
        self.generate_global_metrics_table()
        self.plot_pareto_h1_vs_time()
        self.plot_pareto_h1_vs_dofs()
        self.plot_computational_efficiency_ranking()
        self.plot_formulation_performance_analysis()
        self.plot_uniform_vs_adaptive_sampling_impact()
        self.plot_abstract_summary_overview()
        print(f"Global synthesis suite saved to: {self.global_dir}\n")

    def generate_global_metrics_table(self, md_name: str = "global_metrics_table.md", csv_name: str = "global_metrics_table.csv"):
        """1. Complete markdown and CSV metrics table spanning all problems and methods."""
        fieldnames = [
            "Problem",
            "Epsilon",
            "Method",
            "Method ID",
            "DoFs / Params",
            "Elapsed Time (s)",
            "H1 Semi-Norm Error",
            "L2 Error",
            "L_inf Error",
            "CES Score",
        ]
        rows = []
        for r in self.records:
            h1 = r["h1_error"]
            l2 = r["l2_error"]
            linf = r["linf_error"]
            time_s = r["elapsed_seconds"]
            ces = r["ces_score"]

            rows.append({
                "Problem": r["problem_name"],
                "Epsilon": str(r["epsilon"]),
                "Method": self._get_label(r["method_name"]),
                "Method ID": r["method_name"],
                "DoFs / Params": str(r["dofs"]),
                "Elapsed Time (s)": f"{time_s:.3f}" if np.isfinite(time_s) else "N/A",
                "H1 Semi-Norm Error": f"{h1:.6e}" if np.isfinite(h1) else "N/A",
                "L2 Error": f"{l2:.6e}" if np.isfinite(l2) else "N/A",
                "L_inf Error": f"{linf:.6e}" if np.isfinite(linf) else "N/A",
                "CES Score": f"{ces:.2f}" if np.isfinite(ces) else "N/A",
            })

        # Sort by Problem, Epsilon, Method
        rows.sort(key=lambda x: (x["Problem"], float(x["Epsilon"]), x["Method"]))

        csv_path = os.path.join(self.global_dir, csv_name)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        md_path = os.path.join(self.global_dir, md_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Global Multi-Problem Benchmark Synthesis Table\n\n")
            f.write(f"- **Total Method Runs Analyzed**: {len(rows)}\n\n")
            f.write("| " + " | ".join(fieldnames) + " |\n")
            f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(str(r[k]) for k in fieldnames) + " |\n")
            f.write("\n")

        print(f"  [+] Saved: {csv_name}")
        print(f"  [+] Saved: {md_name}")

    def plot_pareto_h1_vs_time(self, save_name: str = "pareto_efficiency_h1_vs_time.png"):
        """2. Clean Pareto efficiency frontier (H1 error vs. wall-clock time) with collision-free legends."""
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

        plotted_methods = set()
        for i, r in enumerate(self.records):
            time_s = r["elapsed_seconds"]
            h1 = r["h1_error"]
            if not (np.isfinite(time_s) and np.isfinite(h1) and time_s > 0 and h1 > 0):
                continue

            color = self._get_color(r["method_name"], i)
            marker = self._get_marker(r["problem_name"])
            label = self._get_label(r["method_name"])
            legend_label = label if label not in plotted_methods else ""
            if legend_label:
                plotted_methods.add(label)

            ax.scatter(
                time_s, h1,
                color=color, marker=marker, s=110, alpha=0.9,
                edgecolors="black", linewidths=0.8,
                label=legend_label
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall-Clock Execution Time (seconds, Log Scale)", fontweight="bold")
        ax.set_ylabel("$H^1$ Semi-Norm Error (Log Scale)", fontweight="bold")
        ax.set_title("Pareto Efficiency: $H^1$ Accuracy vs. Compute Time", fontweight="bold", pad=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

        # Place clean legend outside to prevent any text overlaps
        ax.legend(
            title="Methods & Solvers",
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8.5,
            title_fontsize=9.5
        )

        save_path = os.path.join(self.global_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_pareto_h1_vs_dofs(self, save_name: str = "pareto_efficiency_h1_vs_dofs.png"):
        """3. Clean Pareto efficiency frontier (H1 error vs. DoFs / trainable parameters)."""
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

        plotted_methods = set()
        for i, r in enumerate(self.records):
            dofs = r["dofs"]
            h1 = r["h1_error"]
            if not (np.isfinite(dofs) and np.isfinite(h1) and dofs > 0 and h1 > 0):
                continue

            color = self._get_color(r["method_name"], i)
            marker = self._get_marker(r["problem_name"])
            label = self._get_label(r["method_name"])
            legend_label = label if label not in plotted_methods else ""
            if legend_label:
                plotted_methods.add(label)

            ax.scatter(
                dofs, h1,
                color=color, marker=marker, s=110, alpha=0.9,
                edgecolors="black", linewidths=0.8,
                label=legend_label
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degrees of Freedom / Trainable Parameters (Log Scale)", fontweight="bold")
        ax.set_ylabel("$H^1$ Semi-Norm Error (Log Scale)", fontweight="bold")
        ax.set_title("Pareto Frontier: $H^1$ Accuracy vs. Model Representation Size", fontweight="bold", pad=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

        ax.legend(
            title="Methods & Solvers",
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8.5,
            title_fontsize=9.5
        )

        save_path = os.path.join(self.global_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_computational_efficiency_ranking(self, save_name: str = "computational_efficiency_ranking.png"):
        """4. Bar chart of Computational Efficiency Score (CES) across all methods."""
        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        # Average CES per method across problems
        method_ces: dict[str, list[float]] = {}
        for r in self.records:
            ces = r["ces_score"]
            if np.isfinite(ces):
                method_ces.setdefault(r["method_name"], []).append(ces)

        if not method_ces:
            plt.close(fig)
            return

        sorted_methods = sorted(method_ces.keys(), key=lambda m: np.mean(method_ces[m]), reverse=True)
        means = [np.mean(method_ces[m]) for m in sorted_methods]
        stds = [np.std(method_ces[m]) if len(method_ces[m]) > 1 else 0.0 for m in sorted_methods]
        labels = [self._get_label(m) for m in sorted_methods]
        colors = [self._get_color(m) for m in sorted_methods]

        x_pos = np.arange(len(sorted_methods))
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, edgecolor="black", alpha=0.85, capsize=4)

        ax.set_ylabel("Computational Efficiency Score (CES)", fontweight="bold")
        ax.set_title("Overall Computational Efficiency Score (Higher is Better)", fontweight="bold", pad=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            offset = 0.1 if height >= 0 else -0.3
            ax.annotate(
                f"{mean_val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height + offset),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=8.5, fontweight="bold"
            )

        save_path = os.path.join(self.global_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_formulation_performance_analysis(self, save_name: str = "formulation_performance_analysis.png"):
        """5. Formulation Performance Analysis (Standard vs. Robust) across all models."""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        pairs = [
            ("pinn_uniform", "r_pinn_uniform", "PINN\n(Uniform)"),
            ("pinn_boundary", "r_pinn_boundary", "PINN\n(Boundary Layer)"),
            ("nurbs_kan_uniform", "r_nurbs_kan_uniform", "NURBS-KAN\n(Uniform)"),
            ("nurbs_kan_boundary", "r_nurbs_kan_boundary", "NURBS-KAN\n(Boundary Layer)"),
            ("iga_standard_uniform", "iga_supg_uniform", "IGA Galerkin vs\nSUPG (Uniform)"),
            ("iga_standard_uniform", "iga_supg_adaptive", "IGA Galerkin vs\nSUPG (Adaptive)"),
            ("iga_standard_uniform", "iga_igrm_adaptive", "IGA Galerkin vs\niGRM (Adaptive)"),
        ]

        def get_avg_h1(m_name: str) -> float:
            vals = [r["h1_error"] for r in self.records if r["method_name"] == m_name and np.isfinite(r["h1_error"])]
            return float(np.mean(vals)) if vals else float('nan')

        valid_pairs = []
        std_vals, rob_vals, labels = [], [], []
        for std_m, rob_m, lbl in pairs:
            v_std = get_avg_h1(std_m)
            v_rob = get_avg_h1(rob_m)
            if np.isfinite(v_std) or np.isfinite(v_rob):
                valid_pairs.append((std_m, rob_m, lbl))
                std_vals.append(v_std if np.isfinite(v_std) else 1e-10)
                rob_vals.append(v_rob if np.isfinite(v_rob) else 1e-10)
                labels.append(lbl)

        if not valid_pairs:
            plt.close(fig)
            return

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, std_vals, width, label="Standard Formulation", color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.bar(x + width/2, rob_vals, width, label="Robust / Variational Formulation", color="#d62728", edgecolor="black", alpha=0.85)

        ax.set_yscale("log")
        ax.set_ylabel("$H^1$ Semi-Norm Error (Log Scale)", fontweight="bold")
        ax.set_title("Formulation Performance Analysis (Standard vs. Robust Variational)", fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8.5)
        ax.legend(frameon=True, loc="upper right")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

        save_path = os.path.join(self.global_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_robust_loss_ablation(self, save_name: str = "robust_loss_ablation.png"):
        """Backward compatibility alias for plot_formulation_performance_analysis."""
        self.plot_formulation_performance_analysis(save_name=save_name)

    def plot_uniform_vs_adaptive_sampling_impact(self, save_name: str = "uniform_vs_adaptive_sampling_impact.png"):
        """6. Compares Uniform vs. Boundary Layer / Adaptive point collocation across all methods."""
        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        pairs = [
            ("pinn_uniform", "pinn_boundary", "Standard PINN"),
            ("r_pinn_uniform", "r_pinn_boundary", "Robust PINN"),
            ("nurbs_kan_uniform", "nurbs_kan_boundary", "Standard NURBS-KAN"),
            ("r_nurbs_kan_uniform", "r_nurbs_kan_boundary", "Robust NURBS-KAN"),
            ("iga_standard_uniform", "iga_supg_adaptive", "Standard vs Adaptive IGA"),
            ("iga_supg_uniform", "iga_supg_adaptive", "SUPG Uniform vs Adaptive"),
        ]

        def get_avg_h1(m_name: str) -> float:
            vals = [r["h1_error"] for r in self.records if r["method_name"] == m_name and np.isfinite(r["h1_error"])]
            return float(np.mean(vals)) if vals else float('nan')

        valid_labels, unif_vals, adapt_vals = [], [], []
        for u_m, a_m, lbl in pairs:
            v_u = get_avg_h1(u_m)
            v_a = get_avg_h1(a_m)
            if np.isfinite(v_u) or np.isfinite(v_a):
                valid_labels.append(lbl)
                unif_vals.append(v_u if np.isfinite(v_u) else 1e-10)
                adapt_vals.append(v_a if np.isfinite(v_a) else 1e-10)

        if not valid_labels:
            plt.close(fig)
            return

        x = np.arange(len(valid_labels))
        width = 0.35

        ax.bar(x - width/2, unif_vals, width, label="Uniform Collocation / Mesh", color="#2ca02c", edgecolor="black", alpha=0.85)
        ax.bar(x + width/2, adapt_vals, width, label="Boundary-Layer / Adaptive Mesh", color="#9467bd", edgecolor="black", alpha=0.85)

        ax.set_yscale("log")
        ax.set_ylabel("$H^1$ Semi-Norm Error (Log Scale)", fontweight="bold")
        ax.set_title("Collocation Sampling Impact: Uniform vs. Boundary Layer Enriched", fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, rotation=30, ha="right", fontsize=9)
        ax.legend(frameon=True)
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

        save_path = os.path.join(self.global_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_abstract_summary_overview(self, save_name: str = "abstract_summary_overview.png"):
        """7. High-level comparative overview spanning PINN, KAN, and IGA families."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        families = ["pinn", "kan", "iga"]
        family_names = ["PINN Family", "KAN Family", "IGA Family"]

        h1_by_fam = []
        time_by_fam = []

        for fam in families:
            matching = [r for r in self.records if fam in r["solver"].lower() or fam in r["method_name"].lower()]
            h1s = [r["h1_error"] for r in matching if np.isfinite(r["h1_error"]) and r["h1_error"] > 0]
            times = [r["elapsed_seconds"] for r in matching if np.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > 0]
            h1_by_fam.append(float(np.mean(h1s)) if h1s else 1.0)
            time_by_fam.append(float(np.mean(times)) if times else 1.0)

        # Subplot 1: Average H1 Error
        ax1 = axes[0]
        bars1 = ax1.bar(family_names, h1_by_fam, color=["#1f77b4", "#2ca02c", "#9467bd"], edgecolor="black", alpha=0.85)
        ax1.set_yscale("log")
        ax1.set_ylabel("Mean $H^1$ Semi-Norm Error", fontweight="bold")
        ax1.set_title("Method Family Accuracy ($H^1$ Error)", fontweight="bold", pad=10)
        ax1.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

        # Subplot 2: Average Wall-Clock Time
        ax2 = axes[1]
        bars2 = ax2.bar(family_names, time_by_fam, color=["#1f77b4", "#2ca02c", "#9467bd"], edgecolor="black", alpha=0.85)
        ax2.set_yscale("log")
        ax2.set_ylabel("Mean Runtime (seconds)", fontweight="bold")
        ax2.set_title("Method Family Compute Time", fontweight="bold", pad=10)
        ax2.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

        fig.suptitle("High-Level Benchmark Synthesis: PINN vs. KAN vs. IGA", fontsize=14, fontweight="bold", y=1.02)
        save_path = os.path.join(self.global_dir, save_name)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
