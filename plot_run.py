#!/usr/bin/env python
"""
Decoupled Parallel Per-Run Visualizer CLI.
Generates all 5 publication-quality diagnostic plots for a single method run concurrently.

Usage:
    python plot_run.py --output-dir output/poisson_sine/nurbs_kan_uniform
    python plot_run.py --config training_config/test/test_test_poisson_sine_nurbs_kan_uniform.yaml
"""

import os
import argparse
import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.problems import get_problem
from src.visualization.run_visualizer import (
    plot_loss_curve,
    plot_h1_convergence,
    plot_2d_solution,
    plot_2d_error,
    plot_1d_slices,
)


def _task_loss_curve(args):
    loss_hist, save_path, title = args
    return plot_loss_curve(loss_hist, save_path, title)


def _task_h1_convergence(args):
    h1_hist, ep_hist, time_hist, save_path, title = args
    return plot_h1_convergence(h1_hist, ep_hist, time_hist, save_path, title)


def _task_2d_solution(args):
    x, t, z_pred, title, save_path = args
    return plot_2d_solution(x, t, z_pred, title, save_path)


def _task_2d_error(args):
    x, t, z_pred, z_exact, title, save_path = args
    return plot_2d_error(x, t, z_pred, z_exact, title, save_path)


def _task_1d_slices(args):
    x, t, z_pred, z_exact, prob_name, save_path = args
    return plot_1d_slices(x, t, z_pred, z_exact, prob_name, save_path)


def plot_single_run(output_dir: str, max_workers: int = 5) -> None:
    """
    Loads outcomes.npz and metadata.yaml from output_dir, computes exact solution,
    and runs all 5 plot tasks in parallel using ProcessPoolExecutor.
    """
    output_dir = os.path.abspath(output_dir)
    outcomes_path = os.path.join(output_dir, "outcomes.npz")
    metadata_path = os.path.join(output_dir, "metadata.yaml")

    if not os.path.exists(outcomes_path):
        raise FileNotFoundError(f"Missing outcomes file: {outcomes_path}")

    # Load outcomes data
    data = np.load(outcomes_path, allow_pickle=True)
    x = data["x"]
    t = data["t"]
    z_pred = data["z_pred"]

    loss_history = data["loss_history"] if "loss_history" in data else np.array([])
    h1_error_history = data["h1_error_history"] if "h1_error_history" in data else np.array([])
    h1_epoch_history = data["h1_epoch_history"] if "h1_epoch_history" in data else None
    h1_time_history = data["h1_time_history"] if "h1_time_history" in data else None

    # Load metadata / problem configuration
    problem_name = "poisson_sine"
    epsilon = 0.01
    method_name = os.path.basename(output_dir)
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
            cfg = meta.get("config", {})
            problem_name = cfg.get("PROBLEM_NAME", cfg.get("problem_id", problem_name))
            epsilon = cfg.get("EPSILON", 0.01)
            method_name = cfg.get("METHOD_NAME", method_name)

    # Instantiate problem to compute exact analytical solution
    problem = get_problem(problem_name, epsilon=epsilon)
    z_exact = problem.exact_solution(x, t)

    print(f"=== Generating 5 Diagnostic Plots for [{problem_name} / {method_name}] ===")
    print(f"  Target Output Dir: {output_dir}")

    # Plot paths
    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    h1_plot_path = os.path.join(output_dir, "h1_error_curve.png")
    sol_plot_path = os.path.join(output_dir, "prediction_contour.png")
    err_plot_path = os.path.join(output_dir, "error_contour.png")
    slice_plot_path = os.path.join(output_dir, "solution_slices.png")

    tasks = [
        (_task_loss_curve, (loss_history, loss_plot_path, f"Loss Curve - {method_name}")),
        (_task_h1_convergence, (h1_error_history, h1_epoch_history, h1_time_history, h1_plot_path, f"$H^1$ Convergence - {method_name}")),
        (_task_2d_solution, (x, t, z_pred, f"Prediction $u_h(x, y)$ - {method_name}", sol_plot_path)),
        (_task_2d_error, (x, t, z_pred, z_exact, f"Pointwise Error - {method_name}", err_plot_path)),
        (_task_1d_slices, (x, t, z_pred, z_exact, problem_name, slice_plot_path)),
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, arg): func.__name__ for func, arg in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                res_path = future.result()
                print(f"  [+] Saved: {os.path.basename(res_path)}")
            except Exception as e:
                print(f"  [-] Error in {name}: {e}")

    print("All diagnostic plots generated successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Parallel Per-Run Diagnostic Visualizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=str, help="Path to run output directory containing outcomes.npz")
    group.add_argument("--config", type=str, help="Path to run configuration YAML file")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker processes")

    args = parser.parse_args()

    if args.output_dir:
        target_dir = args.output_dir
    else:
        # Load output_dir from config file
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            target_dir = cfg.get("output_dir")
            if not target_dir:
                prob = cfg.get("PROBLEM_NAME", "poisson_sine")
                meth = cfg.get("METHOD_NAME", "method")
                target_dir = os.path.join("output", prob, meth)

    plot_single_run(target_dir, max_workers=args.workers)


if __name__ == "__main__":
    main()
