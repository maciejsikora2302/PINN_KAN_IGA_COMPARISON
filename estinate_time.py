import os
import sys
import time
import re
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np

# Add workspace path to sys.path to ensure correct imports
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import SharedConfig
def natural_sort_key(filename):
    if "test" in filename.lower():
        is_test_config = 0 if filename == "test_config.yaml" else 1
        return (3, is_test_config, filename)
    match = re.search(r'exp(\d+)', filename)
    if match:
        return (1, int(match.group(1)), filename)
    return (2, 0, filename)

def natural_sort(files):
    return sorted(files, key=natural_sort_key)

def calibrate():
    print("Calibrating hardware performance parameters...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    for _ in range(epochs_calib):
        y = model(torch.cat([x, t], dim=1))
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
    
    # 2. Calibrate PINN (RPINN=1) LU solve overhead
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
        from src.solvers.kan.kan_solver import KANLinear
        kan_layer = KANLinear(in_features=1, out_features=10, grid_size=5, spline_order=3).to(device)
        kan_opt = torch.optim.Adam(kan_layer.parameters(), lr=0.01)
        # Warmup
        for _ in range(5):
            y_k = kan_layer(x).sum()
            y_k.backward()
            kan_opt.step()
            kan_opt.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.time()
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
        
    # 4. Calibrate IGA (runs on CPU)
    t0 = time.time()
    from src.solvers.iga.iga_solver import bspline_basis_single
    knots = np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.])
    for i in range(5000):
        _ = bspline_basis_single(2, 3, knots, 0.3)
    t1 = time.time()
    iga_coeff = (t1 - t0) / 5000.0 * 0.13
    
    print(f"Calibration completed. Coefficients:")
    print(f"  PINN Base Coeff: {pinn_base_coeff:.2e} s / (epoch * point)")
    print(f"  PINN LU Coeff:   {pinn_lu_coeff:.2e} s / (epoch * point^2)")
    print(f"  KAN Coeff:       {kan_coeff:.2e} s / (epoch * point)")
    print(f"  IGA Coeff:       {iga_coeff:.2e} s / iteration")
    print("-" * 60)
    
    return pinn_base_coeff, pinn_lu_coeff, kan_coeff, iga_coeff

def estimate_time(config_path, pinn_base_coeff, pinn_lu_coeff, kan_coeff, iga_coeff, wall_time_limit_sec=None):
    # Load config
    config = SharedConfig()
    config.load_config(config_path)
    
    # Calculate N_POINTS
    n_points_x = config.N_POINTS_X or 100
    n_points_t = config.N_POINTS_T or 100
    N = n_points_x * n_points_t
    
    # 1. PINN time estimation
    is_rpinn = getattr(config, 'RPINN', 0) == 1
    if is_rpinn and getattr(config, 'RPINN_EPOCHS', None) is not None:
        pinn_epochs = config.RPINN_EPOCHS
    else:
        pinn_epochs = config.EPOCHS or 20000

    cost_per_epoch_pinn = N * pinn_base_coeff
    if is_rpinn:
        cost_per_epoch_pinn += (N ** 2) * pinn_lu_coeff
        
    pinn_est = 1.0 + pinn_epochs * cost_per_epoch_pinn
    pinn_epochs_done = pinn_epochs
    
    if wall_time_limit_sec is not None:
        if pinn_est > wall_time_limit_sec:
            pinn_epochs_done = int(max(0.0, wall_time_limit_sec - 1.0) / cost_per_epoch_pinn) if cost_per_epoch_pinn > 0 else pinn_epochs
            pinn_epochs_done = min(pinn_epochs, pinn_epochs_done)
            pinn_est = 1.0 + pinn_epochs_done * cost_per_epoch_pinn
            
    # 2. KAN time estimation
    if is_rpinn and getattr(config, 'KAN_RPINN_EPOCHS', None) is not None:
        kan_epochs = config.KAN_RPINN_EPOCHS
    elif getattr(config, 'KAN_EPOCHS', None) is not None:
        kan_epochs = config.KAN_EPOCHS
    elif is_rpinn and getattr(config, 'RPINN_EPOCHS', None) is not None:
        kan_epochs = config.RPINN_EPOCHS
    else:
        kan_epochs = config.EPOCHS or 20000

    cost_per_epoch_kan = N * kan_coeff
    kan_est = 0.5 + kan_epochs * cost_per_epoch_kan
    kan_epochs_done = kan_epochs
    
    if wall_time_limit_sec is not None:
        if kan_est > wall_time_limit_sec:
            kan_epochs_done = int(max(0.0, wall_time_limit_sec - 0.5) / cost_per_epoch_kan) if cost_per_epoch_kan > 0 else kan_epochs
            kan_epochs_done = min(kan_epochs, kan_epochs_done)
            kan_est = 0.5 + kan_epochs_done * cost_per_epoch_kan

    # 3. IGA time estimation
    iga_degree = config.IGA_DEGREE or 3
    iga_elements = config.IGA_ELEMENTS or 32
    # Stiffness matrix assembly complexity: M^2 * (p+1)^6
    iga_inner_loops = (iga_elements ** 2) * ((iga_degree + 1) ** 6)
    iga_assembly_est = iga_inner_loops * iga_coeff
    # Evaluate at grid: N * (p+1)^2
    iga_eval_est = N * ((iga_degree + 1) ** 2) * (iga_coeff * 0.5)
    iga_est = 0.2 + iga_assembly_est + iga_eval_est
    
    total_est = pinn_est + kan_est + iga_est
    
    return {
        'pinn_epochs': pinn_epochs,
        'pinn_epochs_done': pinn_epochs_done,
        'kan_epochs': kan_epochs,
        'kan_epochs_done': kan_epochs_done,
        'n_points': N,
        'rpinn': getattr(config, 'RPINN', 1),
        'iga_elements': iga_elements,
        'iga_degree': iga_degree,
        'pinn_est': pinn_est,
        'kan_est': kan_est,
        'iga_est': iga_est,
        'total_est': total_est
    }

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

def main():
    parser = argparse.ArgumentParser(description="Estimate execution times for PINN, KAN, and IGA solvers.")
    parser.add_argument("--time", type=float, default=5.0, help="Wall-clock time limit per training in minutes (default 5.0).")
    args = parser.parse_args()
    
    config_dir = "training_config"
    if not os.path.exists(config_dir):
        print(f"Error: Configuration directory '{config_dir}' not found.")
        sys.exit(1)
        
    configs = []
    for f in os.listdir(config_dir):
        if (f.endswith(".yaml") or f.endswith(".yml")) and f not in ("common.yaml", "original.yaml"):
            configs.append(f)
            
    configs = natural_sort(configs)
    
    if not configs:
        print("No configurations found.")
        return
        
    pinn_base_coeff, pinn_lu_coeff, kan_coeff, iga_coeff = calibrate()
    
    wall_time_sec = args.time * 60.0
    print(f"Applying Wall-Clock Time Limit per Solver Run: {args.time} minutes ({wall_time_sec:.0f}s)")
    print("-" * 145)
    
    print(f"{'Config Name':<35} | {'PINN Est (Epochs)':<25} | {'KAN Est (Epochs)':<25} | {'IGA Est':<10} | {'Total Est':<10} | {'Longest':<8} | {'Diff (L-S)':<12}")
    print("-" * 145)
    
    total_suite_time = 0.0
    
    for c in configs:
        config_path = os.path.join(config_dir, c)
        try:
            est = estimate_time(config_path, pinn_base_coeff, pinn_lu_coeff, kan_coeff, iga_coeff, wall_time_limit_sec=wall_time_sec)
            total_suite_time += est['total_est']
            
            pinn_t = est['pinn_est']
            kan_t = est['kan_est']
            iga_t = est['iga_est']
            times = {'PINN': pinn_t, 'KAN': kan_t, 'IGA': iga_t}
            longest_name = max(times, key=times.get)
            shortest_name = min(times, key=times.get)
            diff_val = times[longest_name] - times[shortest_name]
            
            pinn_str = f"{format_time(pinn_t)} ({est['pinn_epochs_done']}/{est['pinn_epochs']})"
            kan_str = f"{format_time(kan_t)} ({est['kan_epochs_done']}/{est['kan_epochs']})"
            
            display_name = c if len(c) <= 35 else c[:32] + "..."
            print(f"{display_name:<35} | {pinn_str:<25} | {kan_str:<25} | {format_time(iga_t):<10} | {format_time(est['total_est']):<10} | {longest_name:<8} | {format_time(diff_val):<12}")
        except Exception as e:
            display_name = c if len(c) <= 35 else c[:32] + "..."
            print(f"{display_name:<35} | Error: {e}")
            
    print("-" * 145)
    print(f"Estimated Total Time to run all configurations sequentially: {format_time(total_suite_time)}")

if __name__ == "__main__":
    main()
