import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.problems import get_problem
from src.solvers.iga.standard import StandardIGASolver
from src.solvers.iga.supg import SUPGIGASolver
from src.solvers.iga.igrm import ResidualMinimizationIGASolver

def load_data(output_dir="output"):
    """Scans for all metadata and .npz outputs in the output directory."""
    runs = {}
    yaml_paths = glob.glob(os.path.join(output_dir, "*", "metadata.yaml"))
    for yaml_path in yaml_paths:
        try:
            with open(yaml_path, "r") as f:
                meta = yaml.safe_load(f)
            folder = os.path.dirname(yaml_path)
            run_name = os.path.basename(folder)
            
            # Load npz files if they exist
            pinn_data = None
            kan_data = None
            iga_data = None
            
            if os.path.exists(os.path.join(folder, "pinn.npz")):
                pinn_data = np.load(os.path.join(folder, "pinn.npz"), allow_pickle=True)
            if os.path.exists(os.path.join(folder, "kan.npz")):
                kan_data = np.load(os.path.join(folder, "kan.npz"), allow_pickle=True)
            if os.path.exists(os.path.join(folder, "iga.npz")):
                iga_data = np.load(os.path.join(folder, "iga.npz"), allow_pickle=True)
                
            runs[run_name] = {
                "meta": meta,
                "pinn": pinn_data,
                "kan": kan_data,
                "iga": iga_data,
                "folder": folder
            }
        except Exception as e:
            print(f"Warning: Failed to load data from {yaml_path}: {e}")
    return runs

def plot_trajectory_overlays(runs, save_dir):
    """
    1. Convergence Trajectory Overlays:
       For each problem/scenario, overlay PINN vs. R-PINN vs. IGA-KANN vs. R-IGA-KANN on:
       - Mode A: Error vs. Epochs
       - Mode B: Error vs. Wall-Clock Time (seconds)
       - Mode C: Error vs. Training Progress (%)
       Plus a consolidated 3-panel figure combining all 3 axes side-by-side.
    """
    print("Generating Figure 1: Convergence Trajectory Overlays (Epochs, Time, Progress %)...")
    os.makedirs(save_dir, exist_ok=True)
    
    examples = {
        1: "Poisson Sine (Ex 1)",
        2: "Poisson Exp (Ex 2)",
        3: "Eriksson-Johnson (Ex 3)"
    }
    
    colors = {
        "PINN": "#1f77b4",
        "R-PINN": "#17becf",
        "IGA-KANN": "#2ca02c",
        "R-IGA-KANN": "#8c564b"
    }
    
    for ex_id, title in examples.items():
        ex_runs = []
        for name, run in runs.items():
            if run["meta"].get("config", {}).get("EXAMPLE") == ex_id:
                ex_runs.append(run)
                
        if not ex_runs:
            print(f"No runs found for {title}, skipping trajectory plot.")
            continue
            
        h1_calc_every = 100
        
        # We will create a 3-panel figure: [H1 vs Epochs, H1 vs Time (s), H1 vs Progress %]
        fig_multi, axes_multi = plt.subplots(1, 3, figsize=(20, 5.5))
        
        # We also create a 2-panel figure: [Loss vs Epochs, H1 vs Epochs]
        fig_epochs, axes_epochs = plt.subplots(1, 2, figsize=(14, 5))
        
        has_plots = False
        iga_refs = {}
        
        for run in ex_runs:
            config = run["meta"].get("config", {})
            rpinn = config.get("RPINN", 0)
            prefix = "R-" if rpinn else ""
            
            pinn = run["pinn"]
            kan = run["kan"]
            iga = run["iga"]
            
            # --- PINN ---
            if pinn is not None and "loss_history" in pinn.files:
                loss_hist = pinn["loss_history"]
                h1_hist = pinn["h1_error_history"]
                epochs_loss = np.arange(1, len(loss_hist) + 1)
                
                # Epochs
                if "h1_epoch_history" in pinn.files and len(pinn["h1_epoch_history"]) > 0:
                    epochs_h1 = pinn["h1_epoch_history"]
                else:
                    epochs_h1 = np.arange(len(h1_hist)) * h1_calc_every + 1
                    
                # Time
                if "h1_time_history" in pinn.files and len(pinn["h1_time_history"]) > 0:
                    time_h1 = pinn["h1_time_history"]
                else:
                    elapsed = run["meta"].get("results", {}).get("PINN", {}).get("elapsed_seconds", 1.0)
                    time_h1 = np.linspace(0, elapsed, len(h1_hist))
                    
                # Progress %
                if "h1_progress_history" in pinn.files and len(pinn["h1_progress_history"]) > 0:
                    prog_h1 = pinn["h1_progress_history"]
                else:
                    prog_h1 = np.linspace(0, 100.0, len(h1_hist))
                
                label_name = f"{prefix}PINN"
                c = colors.get(label_name, "#1f77b4")
                
                axes_epochs[0].semilogy(epochs_loss, loss_hist, label=label_name, color=c, alpha=0.7)
                if len(h1_hist) > 0:
                    axes_epochs[1].semilogy(epochs_h1, h1_hist, label=label_name, color=c, linewidth=2)
                    axes_multi[0].semilogy(epochs_h1, h1_hist, label=label_name, color=c, linewidth=2)
                    axes_multi[1].semilogy(time_h1, h1_hist, label=label_name, color=c, linewidth=2)
                    axes_multi[2].semilogy(prog_h1, h1_hist, label=label_name, color=c, linewidth=2)
                has_plots = True
                
            # --- KAN ---
            if kan is not None and "loss_history" in kan.files:
                loss_hist = kan["loss_history"]
                h1_hist = kan["h1_error_history"]
                epochs_loss = np.arange(1, len(loss_hist) + 1)
                
                # Epochs
                if "h1_epoch_history" in kan.files and len(kan["h1_epoch_history"]) > 0:
                    epochs_h1 = kan["h1_epoch_history"]
                else:
                    epochs_h1 = np.arange(len(h1_hist)) * h1_calc_every + 1
                    
                # Time
                if "h1_time_history" in kan.files and len(kan["h1_time_history"]) > 0:
                    time_h1 = kan["h1_time_history"]
                else:
                    elapsed = run["meta"].get("results", {}).get("KAN", {}).get("elapsed_seconds", 1.0)
                    time_h1 = np.linspace(0, elapsed, len(h1_hist))
                    
                # Progress %
                if "h1_progress_history" in kan.files and len(kan["h1_progress_history"]) > 0:
                    prog_h1 = kan["h1_progress_history"]
                else:
                    prog_h1 = np.linspace(0, 100.0, len(h1_hist))
                
                label_name = f"{prefix}IGA-KANN"
                c = colors.get(label_name, "#2ca02c")
                
                axes_epochs[0].semilogy(epochs_loss, loss_hist, label=label_name, color=c, alpha=0.7)
                if len(h1_hist) > 0:
                    axes_epochs[1].semilogy(epochs_h1, h1_hist, label=label_name, color=c, linewidth=2)
                    axes_multi[0].semilogy(epochs_h1, h1_hist, label=label_name, color=c, linewidth=2)
                    axes_multi[1].semilogy(time_h1, h1_hist, label=label_name, color=c, linewidth=2)
                    axes_multi[2].semilogy(prog_h1, h1_hist, label=label_name, color=c, linewidth=2)
                has_plots = True
                
            # Collect IGA References
            results = run["meta"].get("results", {})
            if "IGA" in results:
                info = results["IGA"]
                method = info.get("method", "standard").upper()
                h1_err = info.get("final_h1_error")
                if h1_err is not None:
                    iga_refs[f"IGA-{method}"] = h1_err

        # Draw IGA horizontal dashed reference lines
        for label, ref_val in iga_refs.items():
            axes_epochs[1].axhline(y=ref_val, linestyle="--", alpha=0.7, color="crimson", label=f"{label} Ref")
            for ax in axes_multi:
                ax.axhline(y=ref_val, linestyle="--", alpha=0.7, color="crimson", label=f"{label} Ref")

        if has_plots:
            # 1. Save standard 2-panel figure
            axes_epochs[0].set_title("Training Loss vs. Epochs")
            axes_epochs[0].set_xlabel("Epochs")
            axes_epochs[0].set_ylabel("Loss")
            axes_epochs[0].grid(True, which="both", linestyle="--", alpha=0.5)
            axes_epochs[0].legend()
            
            axes_epochs[1].set_title("H1 Error vs. Epochs")
            axes_epochs[1].set_xlabel("Epochs")
            axes_epochs[1].set_ylabel("H1 Error")
            axes_epochs[1].grid(True, which="both", linestyle="--", alpha=0.5)
            axes_epochs[1].legend()
            
            fig_epochs.suptitle(f"Convergence Trajectories - {title}", fontsize=14)
            fig_epochs.tight_layout()
            fig_epochs.savefig(os.path.join(save_dir, f"trajectory_ex{ex_id}.png"), dpi=300, bbox_inches='tight')
            
            # 2. Save 3-panel multi-axis figure
            axes_multi[0].set_title("H1 Error vs. Epochs (Iteration Budget)")
            axes_multi[0].set_xlabel("Epochs")
            axes_multi[0].set_ylabel("H1 Error Semi-Norm")
            axes_multi[0].grid(True, which="both", linestyle="--", alpha=0.5)
            axes_multi[0].legend()
            
            axes_multi[1].set_title("H1 Error vs. Wall-Clock Time (Seconds)")
            axes_multi[1].set_xlabel("Wall-Clock Time [s]")
            axes_multi[1].set_ylabel("H1 Error Semi-Norm")
            axes_multi[1].grid(True, which="both", linestyle="--", alpha=0.5)
            axes_multi[1].legend()
            
            axes_multi[2].set_title("H1 Error vs. Normalized Training Progress (%)")
            axes_multi[2].set_xlabel("Training Progress [%]")
            axes_multi[2].set_ylabel("H1 Error Semi-Norm")
            axes_multi[2].grid(True, which="both", linestyle="--", alpha=0.5)
            axes_multi[2].legend()
            
            fig_multi.suptitle(f"Multi-Metric Convergence Comparison - {title}", fontsize=15, y=1.02)
            fig_multi.tight_layout()
            fig_multi.savefig(os.path.join(save_dir, f"trajectory_multi_axis_ex{ex_id}.png"), dpi=300, bbox_inches='tight')
            
        plt.close(fig_epochs)
        plt.close(fig_multi)

def plot_epsilon_curves(runs, save_dir):
    """
    2. Error vs. Diffusion Parameter (eps) Curves:
       For the Eriksson–Johnson problem, plot H1 and L2 error vs. eps in {1.0, 0.1, 0.01, 0.001}.
       Compare uniform vs. non-uniform/adaptive point distribution across all solvers.
    """
    print("Generating Figure 2: Error vs. Diffusion Parameter Curves...")
    
    eps_list = [1.0, 0.1, 0.01, 0.001]
    
    # We want to collect L2 and H1 errors for Uniform vs. Adaptive meshes/samplers
    # Uniform: SAMPLER_TYPE = "uniform", IGA_MESH_TYPE = "uniform"
    # Adaptive: SAMPLER_TYPE = "boundary_layer", IGA_MESH_TYPE = "adaptive"
    
    solvers = ["PINN", "IGA-KANN", "IGA-SUPG", "IGA-RM"]
    
    data_uniform = {s: {"eps": [], "h1": [], "l2": []} for s in solvers}
    data_adaptive = {s: {"eps": [], "h1": [], "l2": []} for s in solvers}
    
    for name, run in runs.items():
        config = run["meta"].get("config", {})
        if config.get("EXAMPLE") != 3:
            continue
            
        eps = config.get("EPSILON")
        if eps not in eps_list:
            continue
            
        sampler = config.get("SAMPLER_TYPE", "uniform")
        is_adaptive = (sampler == "boundary_layer") or (config.get("IGA_MESH_TYPE") == "adaptive")
        
        results = run["meta"].get("results", {})
        
        for s in solvers:
            d = data_adaptive[s] if is_adaptive else data_uniform[s]
            
            if s == "PINN" and "PINN" in results:
                info = results["PINN"]
                d["eps"].append(eps)
                d["h1"].append(info.get("final_h1_error", np.nan))
                d["l2"].append(info.get("final_l2_error", np.nan))
            elif s == "IGA-KANN" and "KAN" in results:
                info = results["KAN"]
                d["eps"].append(eps)
                d["h1"].append(info.get("final_h1_error", np.nan))
                d["l2"].append(info.get("final_l2_error", np.nan))
            elif s == "IGA-SUPG" and "IGA" in results:
                info = results["IGA"]
                if info.get("method") == "supg":
                    d["eps"].append(eps)
                    d["h1"].append(info.get("final_h1_error", np.nan))
                    d["l2"].append(info.get("final_l2_error", np.nan))
            elif s == "IGA-RM" and "IGA" in results:
                info = results["IGA"]
                if info.get("method") == "igrm":
                    d["eps"].append(eps)
                    d["h1"].append(info.get("final_h1_error", np.nan))
                    d["l2"].append(info.get("final_l2_error", np.nan))
                    
    # Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"PINN": "blue", "IGA-KANN": "forestgreen", "IGA-SUPG": "orange", "IGA-RM": "red"}
    
    for s in solvers:
        # Uniform (Solid lines)
        du = data_uniform[s]
        if du["eps"]:
            # sort by eps
            sorted_idx = np.argsort(du["eps"])
            eps_s = np.array(du["eps"])[sorted_idx]
            h1_s = np.array(du["h1"])[sorted_idx]
            axes[0].loglog(eps_s, h1_s, marker='o', linestyle='-', color=colors[s], label=f"{s} (Uniform)")
            
        # Adaptive (Dashed lines)
        da = data_adaptive[s]
        if da["eps"]:
            sorted_idx = np.argsort(da["eps"])
            eps_s = np.array(da["eps"])[sorted_idx]
            h1_s = np.array(da["h1"])[sorted_idx]
            axes[0].loglog(eps_s, h1_s, marker='s', linestyle='--', color=colors[s], label=f"{s} (Adaptive)")
            
    axes[0].set_title("H1 Error vs. Diffusion Parameter (eps)")
    axes[0].set_xlabel("Diffusion Parameter (eps)")
    axes[0].set_ylabel("H1 Error")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend()
    
    # Same for L2
    for s in solvers:
        du = data_uniform[s]
        if du["eps"]:
            sorted_idx = np.argsort(du["eps"])
            eps_s = np.array(du["eps"])[sorted_idx]
            l2_s = np.array(du["l2"])[sorted_idx]
            axes[1].loglog(eps_s, l2_s, marker='o', linestyle='-', color=colors[s], label=f"{s} (Uniform)")
            
        da = data_adaptive[s]
        if da["eps"]:
            sorted_idx = np.argsort(da["eps"])
            eps_s = np.array(da["eps"])[sorted_idx]
            l2_s = np.array(da["l2"])[sorted_idx]
            axes[1].loglog(eps_s, l2_s, marker='s', linestyle='--', color=colors[s], label=f"{s} (Adaptive)")
            
    axes[1].set_title("L2 Error vs. Diffusion Parameter (eps)")
    axes[1].set_xlabel("Diffusion Parameter (eps)")
    axes[1].set_ylabel("L2 Error")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_vs_epsilon.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_iga_degree_convergence(save_dir):
    """
    3. Multi-Degree IGA Convergence (p=2, 3, 4):
       Bar chart or line plot showing H1 and L2 error convergence of IGA-FEM (Standard, SUPG, iGRM)
       across polynomial orders p in {2, 3, 4}.
    """
    print("Generating Figure 3: Multi-Degree IGA Convergence...")
    
    degrees = [2, 3, 4]
    results = {
        "Standard": {"h1": [], "l2": []},
        "SUPG": {"h1": [], "l2": []},
        "iGRM": {"h1": [], "l2": []}
    }
    
    cache_path = os.path.join(save_dir, "cache_iga_degree_convergence.npz")
    loaded_cache = False
    
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            results["Standard"]["h1"] = list(data["standard_h1"])
            results["Standard"]["l2"] = list(data["standard_l2"])
            results["SUPG"]["h1"] = list(data["supg_h1"])
            results["SUPG"]["l2"] = list(data["supg_l2"])
            results["iGRM"]["h1"] = list(data["igrm_h1"])
            results["iGRM"]["l2"] = list(data["igrm_l2"])
            loaded_cache = True
            print("Loaded Figure 3 IGA degree convergence from cache.")
        except Exception as e:
            print(f"Warning: Failed to load IGA degree convergence cache: {e}")
            
    if not loaded_cache:
        # We solve this dynamically to ensure perfect, complete convergence plots!
        problem = get_problem(3, epsilon=0.01) # Eriksson-Johnson
        
        for p in degrees:
            # Standard
            try:
                solver = StandardIGASolver()
                _, _, _, _, _, _, m = solver.solve(problem, p, M=8, mesh_type="uniform")
                results["Standard"]["h1"].append(m["h1_error"])
                results["Standard"]["l2"].append(m["l2_error"])
            except Exception as e:
                print(f"Failed to run Standard IGA solver for p={p}: {e}")
                results["Standard"]["h1"].append(np.nan)
                results["Standard"]["l2"].append(np.nan)
                
            # SUPG
            try:
                solver = SUPGIGASolver()
                _, _, _, _, _, _, m = solver.solve(problem, p, M=8, mesh_type="uniform")
                results["SUPG"]["h1"].append(m["h1_error"])
                results["SUPG"]["l2"].append(m["l2_error"])
            except Exception as e:
                print(f"Failed to run SUPG IGA solver for p={p}: {e}")
                results["SUPG"]["h1"].append(np.nan)
                results["SUPG"]["l2"].append(np.nan)
                
            # iGRM
            try:
                solver = ResidualMinimizationIGASolver()
                _, _, _, _, _, _, m = solver.solve(problem, p, M=8, mesh_type="uniform", test_degree_enrichment=1)
                results["iGRM"]["h1"].append(m["h1_error"])
                results["iGRM"]["l2"].append(m["l2_error"])
            except Exception as e:
                print(f"Failed to run iGRM solver for p={p}: {e}")
                results["iGRM"]["h1"].append(np.nan)
                results["iGRM"]["l2"].append(np.nan)
                
        # Save to cache
        try:
            np.savez(
                cache_path,
                standard_h1=np.array(results["Standard"]["h1"]),
                standard_l2=np.array(results["Standard"]["l2"]),
                supg_h1=np.array(results["SUPG"]["h1"]),
                supg_l2=np.array(results["SUPG"]["l2"]),
                igrm_h1=np.array(results["iGRM"]["h1"]),
                igrm_l2=np.array(results["iGRM"]["l2"])
            )
            print(f"Saved IGA degree convergence cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save IGA degree convergence cache: {e}")
            
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(degrees))
    width = 0.25
    
    # H1 bar plot
    axes[0].bar(x - width, results["Standard"]["h1"], width, label="Standard", color="royalblue")
    axes[0].bar(x, results["SUPG"]["h1"], width, label="SUPG", color="orange")
    axes[0].bar(x + width, results["iGRM"]["h1"], width, label="iGRM", color="crimson")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"p={p}" for p in degrees])
    axes[0].set_ylabel("H1 Error")
    axes[0].set_yscale("log")
    axes[0].set_title("H1 Error vs. Polynomial Order p")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend()
    
    # L2 bar plot
    axes[1].bar(x - width, results["Standard"]["l2"], width, label="Standard", color="royalblue")
    axes[1].bar(x, results["SUPG"]["l2"], width, label="SUPG", color="orange")
    axes[1].bar(x + width, results["iGRM"]["l2"], width, label="iGRM", color="crimson")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"p={p}" for p in degrees])
    axes[1].set_ylabel("L2 Error")
    axes[1].set_yscale("log")
    axes[1].set_title("L2 Error vs. Polynomial Order p")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iga_degree_convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_solution_error_maps(runs, save_dir):
    """
    4. Side-by-Side 2D Solution & Pointwise Absolute Error Comparison:
       A multi-panel subplot grid comparing Exact, PINN, R-PINN, IGA-KANN, R-IGA-KANN, and IGA-FEM
       solutions and error maps for the sharp boundary layer (eps=0.001).
    """
    print("Generating Figure 4: Side-by-Side Solution & Pointwise Absolute Error Comparison...")
    
    # Find any run for Example 3 and eps=0.001
    target_run = None
    for name, run in runs.items():
        config = run["meta"].get("config", {})
        if config.get("EXAMPLE") == 3 and config.get("EPSILON") == 0.001:
            target_run = run
            break
            
    # Fallback to eps=0.01 if 0.001 is not fully run yet
    if target_run is None:
        print("Warning: eps=0.001 data not found. Falling back to eps=0.01 for visualization.")
        for name, run in runs.items():
            config = run["meta"].get("config", {})
            if config.get("EXAMPLE") == 3 and config.get("EPSILON") == 0.01:
                target_run = run
                break
                
    if target_run is None:
        print("No Eriksson-Johnson data found at all. Skipping Figure 4.")
        return
        
    pinn = target_run["pinn"]
    kan = target_run["kan"]
    iga = target_run["iga"]
    
    if pinn is None or kan is None or iga is None:
        print("Missing solver files in the selected run. Skipping Figure 4.")
        return
        
    x = pinn["x"]
    t = pinn["t"]
    z_pinn = pinn["z_pred"]
    z_kan = kan["z_pred"]
    z_iga = iga["z_pred"]
    
    problem = get_problem(3, epsilon=target_run["meta"]["config"].get("EPSILON", 0.01))
    z_exact = problem.exact_solution(x, t)
    
    # Pointwise absolute errors
    err_pinn = np.abs(z_exact - z_pinn)
    err_kan = np.abs(z_exact - z_kan)
    err_iga = np.abs(z_exact - z_iga)
    
    n_x = target_run["meta"]["config"].get("N_POINTS_X", 100)
    n_t = target_run["meta"]["config"].get("N_POINTS_T", 100)
    
    X = x.reshape(n_t, n_x)
    T = t.reshape(n_t, n_x)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # Solutions
    im0 = axes[0, 0].pcolor(T, X, z_exact.reshape(n_t, n_x), shading='auto', cmap='viridis')
    axes[0, 0].set_title("Exact Solution")
    fig.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].pcolor(T, X, z_pinn.reshape(n_t, n_x), shading='auto', cmap='viridis')
    axes[0, 1].set_title("PINN Solution")
    fig.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].pcolor(T, X, z_kan.reshape(n_t, n_x), shading='auto', cmap='viridis')
    axes[0, 2].set_title("IGA-KANN Solution")
    fig.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[0, 3].pcolor(T, X, z_iga.reshape(n_t, n_x), shading='auto', cmap='viridis')
    axes[0, 3].set_title("IGA-FEM Solution")
    fig.colorbar(im3, ax=axes[0, 3])
    
    # Error maps
    axes[1, 0].axis('off') # empty panel
    
    im4 = axes[1, 1].pcolor(T, X, err_pinn.reshape(n_t, n_x), shading='auto', cmap='inferno')
    axes[1, 1].set_title("PINN Absolute Error")
    fig.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].pcolor(T, X, err_kan.reshape(n_t, n_x), shading='auto', cmap='inferno')
    axes[1, 2].set_title("IGA-KANN Absolute Error")
    fig.colorbar(im5, ax=axes[1, 2])
    
    im6 = axes[1, 3].pcolor(T, X, err_iga.reshape(n_t, n_x), shading='auto', cmap='inferno')
    axes[1, 3].set_title("IGA-FEM Absolute Error")
    fig.colorbar(im6, ax=axes[1, 3])
    
    for ax in axes.flat:
        if ax != axes[1, 0]:
            ax.set_xlabel("Time/y")
            ax.set_ylabel("Space/x")
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "solution_error_maps_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_edge_activations(runs, save_dir):
    """
    5. Edge Activation Summary Grid:
       High-resolution summary plot illustrating learned NURBS activation functions phi_{i,j}(x)
       across layers for IGA-KANN.
    """
    print("Generating Figure 5: Edge Activation Summary Grid...")
    
    # Find a run with KAN output that contains activations
    target_run = None
    for name, run in runs.items():
        if run["kan"] is not None:
            # check if there's at least one layer phi in the files
            if any(f.startswith("kan_layer_") and f.endswith("_phi") for f in run["kan"].files):
                target_run = run
                break
                
    if target_run is None:
        print("No KAN run with saved edge activations found. Skipping Figure 5.")
        return
        
    kan = target_run["kan"]
    x_eval = kan["kan_x_eval"]
    
    # Let's find all layers
    phi_keys = sorted([f for f in kan.files if f.startswith("kan_layer_") and f.endswith("_phi")])
    
    if not phi_keys:
        print("No KAN edge activation files found in kan.npz. Skipping Figure 5.")
        return
        
    for key in phi_keys:
        phi = kan[key] # shape (out_features, in_features, N_eval)
        out_features, in_features, _ = phi.shape
        
        # Plot up to 4x4
        max_out = min(4, out_features)
        max_in = min(4, in_features)
        
        fig, axes = plt.subplots(max_out, max_in, figsize=(3 * max_in, 2.5 * max_out), sharex=True, sharey=False)
        fig.suptitle(f"Learned NURBS Activation Functions - {key.replace('_', ' ').upper()}", fontsize=14, y=1.02)
        
        # Handle single subplot cases
        if max_out == 1 and max_in == 1:
            axes = np.array([[axes]])
        elif max_out == 1:
            axes = axes[np.newaxis, :]
        elif max_in == 1:
            axes = axes[:, np.newaxis]
            
        for i in range(max_out):
            for j in range(max_in):
                ax = axes[i, j]
                ax.plot(x_eval, phi[i, j], color='forestgreen', linewidth=2)
                ax.plot(x_eval, np.zeros_like(x_eval), '--', color='gray', alpha=0.3, linewidth=0.8)
                ax.grid(True, which="both", ls="--", alpha=0.3)
                if i == 0:
                    ax.set_title(f"Input {j}")
                if j == 0:
                    ax.set_ylabel(f"Output {i}")
                    
        plt.tight_layout()
        layer_name = key.replace("_phi", "")
        plt.savefig(os.path.join(save_dir, f"nurbs_activations_{layer_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_sampling_comparison(runs, save_dir):
    print("Generating Figure 6: Sampling Convergence Comparison (Uniform vs Adaptive)...")
    eps_values = [0.001, 0.01, 0.1, 1.0]
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    
    for idx, eps in enumerate(eps_values):
        # Filter runs for this epsilon and EXAMPLE == 3
        eps_runs = []
        for name, run in runs.items():
            config = run["meta"].get("config", {})
            if config.get("EXAMPLE") == 3 and abs(config.get("EPSILON", 0.0) - eps) < 1e-7:
                eps_runs.append(run)
                
        ax_pinn = axes[idx, 0]
        ax_kan = axes[idx, 1]
        
        for run in eps_runs:
            config = run["meta"].get("config", {})
            sampler = config.get("SAMPLER_TYPE", "uniform")
            rpinn = config.get("RPINN", 0)
            label_suffix = f" (R-PINN)" if rpinn else ""
            
            color = "blue" if sampler == "uniform" else "red"
            linestyle = "-" if rpinn else "--"
            
            # PINN
            pinn = run["pinn"]
            if pinn is not None and "loss_history" in pinn.files:
                loss_hist = pinn["loss_history"]
                ax_pinn.semilogy(loss_hist, label=f"{sampler.capitalize()} PINN{label_suffix}", color=color, linestyle=linestyle, alpha=0.7)
                
            # KAN
            kan = run["kan"]
            if kan is not None and "loss_history" in kan.files:
                loss_hist = kan["loss_history"]
                ax_kan.semilogy(loss_hist, label=f"{sampler.capitalize()} KAN{label_suffix}", color=color, linestyle=linestyle, alpha=0.7)
                
        ax_pinn.set_title(f"PINN Convergence (Epsilon = {eps})")
        ax_pinn.set_xlabel("Epochs")
        ax_pinn.set_ylabel("Loss")
        ax_pinn.grid(True, which="both", ls="--", alpha=0.5)
        ax_pinn.legend(fontsize=8)
        
        ax_kan.set_title(f"KAN Convergence (Epsilon = {eps})")
        ax_kan.set_xlabel("Epochs")
        ax_kan.set_ylabel("Loss")
        ax_kan.grid(True, which="both", ls="--", alpha=0.5)
        ax_kan.legend(fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sampling_convergence_comparison.png"), dpi=300)
    plt.close()

def main():
    save_dir = "output/comparison_suite"
    os.makedirs(save_dir, exist_ok=True)
    
    runs = load_data()
    
    plot_trajectory_overlays(runs, save_dir)
    plot_epsilon_curves(runs, save_dir)
    plot_iga_degree_convergence(save_dir)
    plot_solution_error_maps(runs, save_dir)
    plot_edge_activations(runs, save_dir)
    plot_sampling_comparison(runs, save_dir)
    
    print(f"\nAll comparison suite figures generated successfully and saved in '{save_dir}/'.")

if __name__ == "__main__":
    main()
