import os
import sys
import glob
import time
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple

# Add workspace path to sys.path to ensure correct imports
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import RunConfig, load_config
from src.solvers.iga.base import bspline_basis_single


def calibrate(optimized: bool = False) -> Tuple[float, float, float, float]:
    mode_str = "OPTIMIZED" if optimized else "BASELINE"
    print(f"Calibrating hardware performance parameters ({mode_str} mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if optimized and torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # 1. Calibrate PINN (RPINN=0)
    N_calib = 500
    x = torch.randn(N_calib, 1, device=device, requires_grad=True)
    t = torch.randn(N_calib, 1, device=device, requires_grad=True)

    model = nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Warmup
    use_amp = torch.cuda.is_available()
    device_type = "cuda" if use_amp else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=torch.bfloat16):
        for _ in range(5):
            y = model(torch.cat([x, t], dim=1))
            loss = y.pow(2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    epochs_calib = 15
    with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=torch.bfloat16):
        for _ in range(epochs_calib):
            y = model(torch.cat([x, t], dim=1))
            if optimized:
                grads = torch.autograd.grad(y, (x, t), grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)
                dy = grads[0]
                dy2 = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy), create_graph=True, retain_graph=True)[0]
            else:
                dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
                dy2 = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
            loss = dy2.pow(2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    pinn_base_coeff = (t1 - t0) / (epochs_calib * N_calib)

    # 2. Calibrate PINN (RPINN=1) LU solve / Spectral solve overhead
    if optimized:
        try:
            from src.samplers.uniform_grid import UniformGridSampler
            sampler = UniformGridSampler()
            fast_solver = sampler.build_gram_solver(100, 100, device=device, optimized=True)
            b_test = torch.randn(10000, 1, device=device)

            t0 = time.time()
            for _ in range(epochs_calib):
                _ = fast_solver(b_test)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            pinn_lu_coeff = (t1 - t0) / (epochs_calib * 10000.0)
        except Exception:
            pinn_lu_coeff = pinn_base_coeff * 0.5
    else:
        A = torch.randn(N_calib, N_calib, device=device)
        b = torch.randn(N_calib, 1, device=device)
        LU, pivots = torch.linalg.lu_factor(A)

        t0 = time.time()
        for _ in range(epochs_calib):
            sol = torch.linalg.lu_solve(LU, pivots, b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        pinn_lu_coeff = (t1 - t0) / (epochs_calib * (N_calib ** 2))

    # 3. Calibrate KAN
    kan_coeff = pinn_base_coeff * 4.5
    try:
        from src.solvers.kan.nurbs import NURBSLinear
        kan_layer = NURBSLinear(in_features=1, out_features=10, grid_size=5, spline_order=3).to(device)
        kan_opt = torch.optim.Adam(kan_layer.parameters(), lr=0.01)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=torch.bfloat16):
            for _ in range(5):
                y_k = kan_layer(x).sum()
                y_k.backward()
                kan_opt.step()
                kan_opt.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=torch.bfloat16):
            for _ in range(epochs_calib):
                y_k = kan_layer(x)
                dy_k = torch.autograd.grad(y_k.sum(), x, create_graph=True, retain_graph=True)[0]
                loss_k = dy_k.pow(2).sum()
                loss_k.backward()
                kan_opt.step()
                kan_opt.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        kan_coeff = (t1 - t0) / (epochs_calib * N_calib)
    except Exception as e:
        print(f"KAN calibration fallback due to: {e}")

    # 4. Calibrate IGA
    t0 = time.time()
    knots = np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.])
    for i in range(5000):
        _ = bspline_basis_single(2, 3, knots, 0.3)
    t1 = time.time()
    iga_coeff = (t1 - t0) / 5000.0 * (0.01 if optimized else 0.13)

    print(f"Calibration completed ({mode_str} mode). Coefficients:")
    print(f"  PINN Base Coeff: {pinn_base_coeff:.2e} s / (epoch * point)")
    if optimized:
        print(f"  PINN Spectral:   {pinn_lu_coeff:.2e} s / (epoch * point)")
    else:
        print(f"  PINN LU Coeff:   {pinn_lu_coeff:.2e} s / (epoch * point^2)")
    print(f"  KAN Coeff:       {kan_coeff:.2e} s / (epoch * point)")
    print(f"  IGA Coeff:       {iga_coeff:.2e} s / iteration")
    print("-" * 60)

    return pinn_base_coeff, pinn_lu_coeff, kan_coeff, iga_coeff


def estimate_time(
    config_path: str,
    pinn_base_coeff: float,
    pinn_lu_coeff: float,
    kan_coeff: float,
    iga_coeff: float,
    wall_time_limit_sec: float = None,
    optimized: bool = False
) -> Dict[str, Any]:
    """Estimates execution time for a single-method or multi-method configuration."""
    cfg = load_config(config_path)

    solver = getattr(cfg, "SOLVER", "pinn").lower()
    method_name = getattr(cfg, "METHOD_NAME", "method")

    n_points_x = getattr(cfg, "N_POINTS_X", 100)
    n_points_t = getattr(cfg, "N_POINTS_T", 100)
    N = n_points_x * n_points_t

    pinn_est = 0.0
    kan_est = 0.0
    iga_est = 0.0

    epochs = getattr(cfg, "EPOCHS", 20000)
    is_rpinn = getattr(cfg, "RPINN", 0) == 1
    epochs_done = epochs

    if solver == "pinn":
        cost_per_epoch = N * pinn_base_coeff
        if is_rpinn:
            cost_per_epoch += (N * pinn_lu_coeff if optimized else (N ** 2) * pinn_lu_coeff)

        pinn_est = 0.5 + epochs * cost_per_epoch
        if wall_time_limit_sec is not None and pinn_est > wall_time_limit_sec:
            epochs_done = int(max(0.0, wall_time_limit_sec - 0.5) / cost_per_epoch) if cost_per_epoch > 0 else epochs
            epochs_done = min(epochs, epochs_done)
            pinn_est = 0.5 + epochs_done * cost_per_epoch

    elif solver == "kan":
        kan_epochs = getattr(cfg, "KAN_EPOCHS", epochs)
        cost_per_epoch = N * kan_coeff
        if is_rpinn and optimized:
            cost_per_epoch += N * pinn_lu_coeff
        kan_est = 0.5 + kan_epochs * cost_per_epoch
        epochs_done = kan_epochs
        if wall_time_limit_sec is not None and kan_est > wall_time_limit_sec:
            epochs_done = int(max(0.0, wall_time_limit_sec - 0.5) / cost_per_epoch) if cost_per_epoch > 0 else kan_epochs
            epochs_done = min(kan_epochs, epochs_done)
            kan_est = 0.5 + epochs_done * cost_per_epoch

    elif solver == "iga":
        iga_degree = getattr(cfg, "IGA_DEGREE", 2)
        iga_elements = getattr(cfg, "IGA_ELEMENTS", 32)
        if optimized:
            iga_inner_loops = (iga_elements ** 2) * ((iga_degree + 1) ** 3)
            iga_assembly_est = iga_inner_loops * iga_coeff
            iga_eval_est = N * ((iga_degree + 1) ** 2) * (iga_coeff * 0.05)
        else:
            iga_inner_loops = (iga_elements ** 2) * ((iga_degree + 1) ** 6)
            iga_assembly_est = iga_inner_loops * iga_coeff
            iga_eval_est = N * ((iga_degree + 1) ** 2) * (iga_coeff * 0.5)
        iga_est = 0.05 + iga_assembly_est + iga_eval_est
        epochs_done = 1

    total_est = pinn_est + kan_est + iga_est

    return {
        "solver": solver,
        "method_name": method_name,
        "epochs": epochs,
        "epochs_done": epochs_done,
        "n_points": N,
        "pinn_est": pinn_est,
        "kan_est": kan_est,
        "iga_est": iga_est,
        "total_est": total_est,
    }


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Estimate execution times for PINN, KAN, and IGA solvers.")
    parser.add_argument("--config-dir", type=str, default="training_config", help="Configuration directory to scan.")
    parser.add_argument("--time", type=float, default=5.0, help="Wall-clock time limit per training in minutes (default 5.0).")
    parser.add_argument("--optimized", action="store_true", help="Estimate execution time with performance optimizations enabled.")
    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        print(f"Error: Configuration directory '{args.config_dir}' not found.")
        sys.exit(1)

    # Recursively find all configs
    configs = []
    for root, _, files in os.walk(args.config_dir):
        for f in sorted(files):
            if f.endswith((".yaml", ".yml")) and f not in ("common.yaml", "config_schema.yaml", "original.yaml"):
                configs.append(os.path.join(root, f))

    if not configs:
        print(f"No configurations found in '{args.config_dir}'.")
        return

    pinn_base_coeff, pinn_lu_coeff, kan_coeff, iga_coeff = calibrate(optimized=args.optimized)

    wall_time_sec = args.time * 60.0 if args.time else None
    mode_str = "OPTIMIZED" if args.optimized else "BASELINE"
    print(f"Mode: {mode_str} | Applying Wall-Clock Time Limit per Solver Run: {args.time} min ({wall_time_sec:.0f}s)" if wall_time_sec else f"Mode: {mode_str}")
    print("-" * 120)
    print(f"{'Config (Relative Path)':<60} | {'Solver':<10} | {'Epochs':<15} | {'Estimated Time':<15}")
    print("-" * 120)

    total_suite_time = 0.0
    by_solver_time = {"pinn": 0.0, "kan": 0.0, "iga": 0.0}

    for config_path in configs:
        try:
            est = estimate_time(
                config_path,
                pinn_base_coeff,
                pinn_lu_coeff,
                kan_coeff,
                iga_coeff,
                wall_time_limit_sec=wall_time_sec,
                optimized=args.optimized
            )
            rel_path = os.path.relpath(config_path)
            display_path = rel_path if len(rel_path) <= 60 else "..." + rel_path[-57:]
            solver = est["solver"]
            t = est["total_est"]

            total_suite_time += t
            by_solver_time[solver] = by_solver_time.get(solver, 0.0) + t

            if solver in ("pinn", "kan"):
                epoch_str = f"{est['epochs_done']}/{est['epochs']}"
            else:
                epoch_str = "Direct (1)"

            print(f"{display_path:<60} | {solver.upper():<10} | {epoch_str:<15} | {format_time(t):<15}")
        except Exception as e:
            rel_path = os.path.relpath(config_path)
            print(f"{rel_path:<60} | Error: {e}")

    print("-" * 120)
    print(f"Estimated Total Time (Sequential Suite): {format_time(total_suite_time)}")
    print(f"  - PINN Runs: {format_time(by_solver_time.get('pinn', 0.0))}")
    print(f"  - KAN Runs:  {format_time(by_solver_time.get('kan', 0.0))}")
    print(f"  - IGA Runs:  {format_time(by_solver_time.get('iga', 0.0))}")
    print("=" * 120)


if __name__ == "__main__":
    main()
