import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml


def numpy_exact_solution(x, t, config_data):
    example = config_data.get("EXAMPLE", 1)
    epsilon = config_data.get("EPSILON", 0.01)
    if example == 1:
        return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * t)
    elif example == 2:
        return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * t) * np.exp(np.pi * (x - 2.0 * t))
    elif example == 3:
        r1 = (1.0 + np.sqrt(1.0 + 4.0 * epsilon * epsilon * np.pi * np.pi)) / (2.0 * epsilon)
        r2 = (1.0 - np.sqrt(1.0 + 4.0 * epsilon * epsilon * np.pi * np.pi)) / (2.0 * epsilon)
        # Note: t is used as y in the formula
        res_t = (np.exp(r1 * (t - 1.0)) - np.exp(r2 * (t - 1.0))) / (np.exp(-r1) - np.exp(-r2))
        res_x_dx = np.sin(np.pi * x)
        return res_t * res_x_dx
    else:
        raise ValueError(f"Unknown Example index: {example}")

def run_plot_task(task):
    import matplotlib
    matplotlib.use('Agg')
    from visualizer import Visualizer

    method_name = task["method"]
    kwargs = task["kwargs"]

    viz = Visualizer()
    method = getattr(viz, method_name)
    try:
        method(**kwargs)
        return True, None
    except Exception as e:
        import traceback
        err_msg = f"Error: {e}\n{traceback.format_exc()}"
        return False, err_msg

def main():
    parser = argparse.ArgumentParser(description="Read saved solver outputs and generate Colab-style plots in parallel.")
    parser.add_argument("--config", type=str, help="Configuration filename or path.")
    parser.add_argument("config_pos", type=str, nargs="?", help="Configuration filename or path (positional).")
    args = parser.parse_args()

    config_filename = args.config if args.config else args.config_pos
    if not config_filename:
        parser.print_help()
        return

    # Resolve configuration path
    if os.path.exists(config_filename):
        config_path = config_filename
    else:
        config_path = os.path.join("training_config", config_filename)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_filename} or {config_path}")

    # Load YAML config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join("output", config_name)

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}. Run train.py first.")

    print(f"=== Loading Raw Output Data from: {output_dir} ===")

    # Initialize data dictionaries
    losses = {}
    h1_errors = {}
    predictions = {}

    algorithms = ["pinn", "kan", "iga"]
    for algo in algorithms:
        file_path = os.path.join(output_dir, f"{algo}.npz")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                if "loss_history" in data.files:
                    losses[algo.upper()] = data["loss_history"]
                if "h1_error_history" in data.files:
                    h1_errors[algo.upper()] = data["h1_error_history"]
                if "x" in data.files and "t" in data.files and "z_pred" in data.files:
                    predictions[algo.upper()] = {
                        "x": data["x"],
                        "t": data["t"],
                        "z": data["z_pred"]
                    }
                print(f"Loaded {algo.upper()} data successfully.")
            except Exception as e:
                print(f"Error loading {algo} data: {e}")
        else:
            print(f"No output file found for {algo.upper()} at {file_path}")

    # Get config variables
    example = config_data.get("EXAMPLE", 1)
    epsilon = config_data.get("EPSILON", 0.01)
    rpinn = config_data.get("RPINN", 1)
    h1_calc_every = config_data.get("H1_CALC_EVERY", 100)
    n_points_x = config_data.get("N_POINTS_X", 100)
    n_points_t = config_data.get("N_POINTS_T", 100)

    # Gather tasks
    tasks = []

    # 1. Plot running average and raw loss for each method
    for algo, loss_history in losses.items():
        if len(loss_history) > 0:
            avg_plot_path = os.path.join(output_dir, f"{algo.lower()}_loss_running_average.png")
            tasks.append({
                "method": "plot_running_average",
                "kwargs": {
                    "y": loss_history,
                    "window": min(100, len(loss_history)),
                    "title": f"Loss function (running average) - {algo} - {config_name}",
                    "save_path": avg_plot_path
                }
            })

    # 1.5. Plot Multi-Method Comparison (Loss and H1 Error)
    if losses:
        comparison_losses = {algo: {"y": history, "x": None} for algo, history in losses.items()}
        comp_loss_path = os.path.join(output_dir, "comparison_loss.png")
        tasks.append({
            "method": "plot_comparison",
            "kwargs": {
                "curves": comparison_losses,
                "title": f"Multi-Solver Loss Comparison - {config_name}",
                "xlabel": "Epochs",
                "ylabel": "Loss",
                "log_scale": True,
                "save_path": comp_loss_path
            }
        })

    if h1_errors:
        comparison_h1 = {}
        for algo, h1_vals in h1_errors.items():
            if len(h1_vals) > 0:
                epochs_h1 = np.arange(0, len(h1_vals)) * h1_calc_every
                comparison_h1[algo] = {"y": h1_vals, "x": epochs_h1}
        
        if comparison_h1:
            comp_h1_path = os.path.join(output_dir, "comparison_h1_error.png")
            tasks.append({
                "method": "plot_comparison",
                "kwargs": {
                    "curves": comparison_h1,
                    "title": f"Multi-Solver H1 Error Comparison - {config_name}",
                    "xlabel": "Epochs",
                    "ylabel": "H1 Error Norm",
                    "log_scale": True,
                    "save_path": comp_h1_path
                }
            })

    # 1.6. Plot Multi-Method 1D Slice Comparison
    if predictions:
        comp_slices_path = os.path.join(output_dir, "comparison_slices_t.png")
        tasks.append({
            "method": "plot_slices_comparison",
            "kwargs": {
                "predictions": predictions,
                "example": example,
                "epsilon": epsilon,
                "t_slices": [0.25, 0.5, 0.75],
                "save_path": comp_slices_path
            }
        })

    # 1.7. Plot Multi-Method 3D Solution Comparison
    if predictions:
        comp_3d_path = os.path.join(output_dir, "comparison_solution_3d.png")
        tasks.append({
            "method": "plot_3d_comparison",
            "kwargs": {
                "predictions": predictions,
                "example": example,
                "epsilon": epsilon,
                "n_points_x": n_points_x,
                "n_points_t": n_points_t,
                "title": f"Multi-Solver 3D Solution Comparison - {config_name}",
                "save_path": comp_3d_path
            }
        })

    # 2. Plot Loss vs error for each method
    for algo, loss_history in losses.items():
        if len(loss_history) > 0:
            h1_vals = h1_errors.get(algo, np.array([]))
            epochs_loss = np.arange(1, len(loss_history) + 1)
            epochs_h1 = np.arange(0, len(h1_vals)) * h1_calc_every

            lve_plot_path = os.path.join(output_dir, f"{algo.lower()}_loss_vs_error.png")
            tasks.append({
                "method": "plot_loss_vs_error",
                "kwargs": {
                    "epochs_loss": epochs_loss,
                    "loss_vals": loss_history,
                    "epochs_h1": epochs_h1,
                    "h1_vals": h1_vals,
                    "example": example,
                    "epsilon": epsilon,
                    "rpinn": rpinn,
                    "title": f"Loss vs Error - {algo} - {config_name}",
                    "save_path": lve_plot_path
                }
            })

    # 3. Plot pcolors and 3D surfaces: Prediction, Exact, and Absolute Error
    for algo, pred_data in predictions.items():
        x = pred_data["x"]
        t = pred_data["t"]
        z_pred = pred_data["z"]

        if len(x) > 0 and len(t) > 0 and len(z_pred) > 0:
            # 3a. Plot Prediction Pcolor & 3D Surface
            pred_plot_path = os.path.join(output_dir, f"{algo.lower()}_solution_pcolor.png")
            tasks.append({
                "method": "plot_pcolor",
                "kwargs": {
                    "z": z_pred, "x": x, "t": t,
                    "n_points_x": n_points_x, "n_points_t": n_points_t,
                    "title": f"{algo} solution - {config_name}",
                    "save_path": pred_plot_path
                }
            })
            pred_plot_path_3d = os.path.join(output_dir, f"{algo.lower()}_solution_3d.png")
            tasks.append({
                "method": "plot_3d_surface",
                "kwargs": {
                    "z": z_pred, "x": x, "t": t,
                    "n_points_x": n_points_x, "n_points_t": n_points_t,
                    "title": f"{algo} solution 3D - {config_name}",
                    "zlabel": "u(x,t)",
                    "save_path": pred_plot_path_3d
                }
            })

            # 3b. Calculate Exact Solution & Plot Pcolor & 3D Surface
            z_exact = numpy_exact_solution(x, t, config_data)
            exact_plot_path = os.path.join(output_dir, f"{algo.lower()}_exact_pcolor.png")
            tasks.append({
                "method": "plot_pcolor",
                "kwargs": {
                    "z": z_exact, "x": x, "t": t,
                    "n_points_x": n_points_x, "n_points_t": n_points_t,
                    "title": f"Exact solution - {config_name}",
                    "save_path": exact_plot_path
                }
            })
            exact_plot_path_3d = os.path.join(output_dir, f"{algo.lower()}_exact_3d.png")
            tasks.append({
                "method": "plot_3d_surface",
                "kwargs": {
                    "z": z_exact, "x": x, "t": t,
                    "n_points_x": n_points_x, "n_points_t": n_points_t,
                    "title": f"Exact solution 3D - {config_name}",
                    "zlabel": "u(x,t)",
                    "save_path": exact_plot_path_3d
                }
            })

            # 3c. Calculate and Plot Absolute Error Pcolor & 3D Surface
            z_error = np.abs(z_exact - z_pred)
            error_plot_path = os.path.join(output_dir, f"{algo.lower()}_error_pcolor.png")
            tasks.append({
                "method": "plot_pcolor",
                "kwargs": {
                    "z": z_error, "x": x, "t": t,
                    "n_points_x": n_points_x, "n_points_t": n_points_t,
                    "title": f"Absolute Error - {algo} - {config_name}",
                    "save_path": error_plot_path
                }
            })
            error_plot_path_3d = os.path.join(output_dir, f"{algo.lower()}_error_3d.png")
            tasks.append({
                "method": "plot_3d_surface",
                "kwargs": {
                    "z": z_error, "x": x, "t": t,
                    "n_points_x": n_points_x, "n_points_t": n_points_t,
                    "title": f"Absolute Error 3D - {algo} - {config_name}",
                    "zlabel": "error",
                    "save_path": error_plot_path_3d
                }
            })

    # 4. Plot Rational activation functions and KAN edge activations
    for algo in ["pinn", "kan", "iga"]:
        file_path = os.path.join(output_dir, f"{algo}.npz")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                rational_keys = [k for k in data.files if k.startswith("rational_") and k.endswith("_num")]
                if rational_keys:
                    rational_subfolder = os.path.join(output_dir, "rational_plots")
                    os.makedirs(rational_subfolder, exist_ok=True)
                    for num_key in rational_keys:
                        base_name = num_key[len("rational_"):-len("_num")]
                        den_key = f"rational_{base_name}_den"
                        if den_key in data.files:
                            num_coeffs = data[num_key]
                            den_coeffs = data[den_key]
                            plot_path = os.path.join(rational_subfolder, f"rational_activation_{base_name}.png")
                            tasks.append({
                                "method": "plot_rational_function",
                                "kwargs": {
                                    "num_coeffs": num_coeffs,
                                    "den_coeffs": den_coeffs,
                                    "title": f"Learned Rational ({base_name}) - {algo.upper()} - {config_name}",
                                    "save_path": plot_path
                                }
                            })
                
                # Check for KAN edge activations
                if "kan_x_eval" in data.files:
                    kan_subfolder = os.path.join(output_dir, "kan_plots")
                    os.makedirs(kan_subfolder, exist_ok=True)
                    
                    x_eval = data["kan_x_eval"]
                    layer_phi_keys = [k for k in data.files if k.startswith("kan_layer_") and k.endswith("_phi")]
                    for phi_key in sorted(layer_phi_keys):
                        layer_idx = phi_key.split("_")[2]
                        phi = data[phi_key]  # shape: (out_features, in_features, N_eval)
                        
                        # Save layer grid plot
                        grid_plot_path = os.path.join(kan_subfolder, f"kan_layer_{layer_idx}_grid.png")
                        tasks.append({
                            "method": "plot_kan_layer_grid",
                            "kwargs": {
                                "x": x_eval,
                                "phi": phi,
                                "title": f"KAN Layer {layer_idx} Activations - {config_name}",
                                "save_path": grid_plot_path
                            }
                        })
                        
                        # Save individual edge plots (only for top 3 most active/varying edges in the layer)
                        out_features, in_features, _ = phi.shape
                        edges_activity = []
                        for i in range(out_features):
                            for j in range(in_features):
                                activity = np.std(phi[i, j])
                                edges_activity.append((activity, i, j))
                        
                        # Sort by activity descending and take top 3
                        edges_activity.sort(key=lambda item: item[0], reverse=True)
                        top_edges = edges_activity[:3]
                        
                        for activity, i, j in top_edges:
                            edge_plot_path = os.path.join(kan_subfolder, f"kan_layer_{layer_idx}_edge_{i}_{j}.png")
                            tasks.append({
                                "method": "plot_kan_edge",
                                "kwargs": {
                                    "x": x_eval,
                                    "y": phi[i, j],
                                    "title": f"KAN Layer {layer_idx} Edge (In {j} -> Out {i}) - {config_name}",
                                    "save_path": edge_plot_path
                                }
                            })
            except Exception as e:
                print(f"Error reading activation functions for {algo}: {e}")

    # Add Boundary Layer Slice Plot Task
    if predictions:
        tasks.append({
            "method": "plot_boundary_layer_slice",
            "kwargs": {
                "predictions": predictions,
                "example": config_data.get("EXAMPLE", 3),
                "epsilon": config_data.get("EPSILON", 0.01),
                "x_cut": 0.5,
                "title": f"Boundary Layer Profile - {config_name}",
                "save_path": os.path.join(output_dir, "comparison_boundary_layer_slice.png")
            }
        })

    # Parallel Execution
    num_tasks = len(tasks)
    print(f"\nSubmitting {num_tasks} plotting tasks to ProcessPoolExecutor...")
    start_time = time.time()

    # Use default max_workers (usually CPU count)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_plot_task, tasks))

    end_time = time.time()
    successful = sum(1 for r in results if r[0])
    failed = num_tasks - successful

    print(f"\n=== Parallel Visualization Completed in {end_time - start_time:.2f} seconds ===")
    print(f"Successful tasks: {successful} / {num_tasks}")
    if failed > 0:
        print(f"Failed tasks: {failed}")
        for idx, (success, err_msg) in enumerate(results):
            if not success:
                task = tasks[idx]
                kwargs = task.get("kwargs", {}) if isinstance(task, dict) else {}
                save_path = kwargs.get("save_path") if isinstance(kwargs, dict) else None
                method_name = task.get("method") if isinstance(task, dict) else "unknown"
                print(f"\n--- Task {idx} Failed: {method_name} -> {save_path} ---")
                print(err_msg)

if __name__ == "__main__":
    main()
