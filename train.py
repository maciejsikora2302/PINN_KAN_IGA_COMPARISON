import argparse
import os
import time
import sys
import platform
import yaml
import torch
from src.solvers.pinn import PINNExperiment
from src.solvers.kan import KANExperiment
from src.solvers.iga import IGAExperiment

def main():
    parser = argparse.ArgumentParser(description="Run PINN, KAN, and IGA solvers based on a YAML configuration.")
    parser.add_argument("--config", type=str, help="Configuration filename or path.")
    parser.add_argument("config_pos", type=str, nargs="?", help="Configuration filename or path (positional).")
    parser.add_argument("--wall-time", nargs="?", const=5.0, type=float, help="Stop training if wall-clock time in minutes is exceeded.")
    parser.add_argument("--solvers", type=str, nargs="+", help="Specify which solvers to run (pinn, kan, iga). Overrides YAML config.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip running solvers whose outputs (.npz files) already exist.")
    parser.add_argument("--optimized", action="store_true", help="Enable performance optimizations (simultaneous autograd, fast Kronecker/sparse Gram solvers, tensor-contracted IGA).")
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

    # Load base config to read default solvers
    from config import SharedConfig
    base_config = SharedConfig()
    base_config.load_config(config_path)

    # Determine optimization mode
    is_optimized = args.optimized or getattr(base_config, "OPTIMIZED", False)
    if is_optimized:
        base_config.OPTIMIZED = True
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass

    # Determine active solvers
    if args.solvers:
        active_solvers = [s.lower() for s in args.solvers]
    else:
        active_solvers = [s.lower() for s in getattr(base_config, "SOLVERS", ["pinn", "kan", "iga"])]

    # Determine subfolder name based on config file basename without extension
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join("output", config_name)
    os.makedirs(output_dir, exist_ok=True)

    # Handle skip-existing logic
    if args.skip_existing:
        skipped = []
        filtered_solvers = []
        for s in active_solvers:
            if os.path.exists(os.path.join(output_dir, f"{s}.npz")):
                skipped.append(s)
            else:
                filtered_solvers.append(s)
        if skipped:
            print(f"Skipping existing solver outputs: {', '.join(skipped)}")
        active_solvers = filtered_solvers

    if not active_solvers:
        print("No active solvers left to run. Exiting.")
        return

    print(f"=== Starting Solver Run for Configuration: {config_name} ===")
    print(f"Config path: {config_path}")
    print(f"Output folder: {output_dir}")
    print(f"Active solvers: {', '.join(active_solvers)}\n")

    # Record start time & date
    start_date = time.strftime("%Y-%m-%d %H:%M:%S")
    start_time_total = time.time()

    pinn_solver = None
    kan_solver = None
    iga_solver = None

    elapsed_pinn = 0.0
    elapsed_kan = 0.0
    elapsed_iga = 0.0

    # Run PINN
    if "pinn" in active_solvers:
        print("--- Running PINN Solver ---")
        start_time_pinn = time.time()
        wall_time_sec = args.wall_time * 60.0 if args.wall_time is not None else None
        pinn_solver = PINNExperiment(config_path, wall_time_limit=wall_time_sec, optimized=is_optimized)
        pinn_solver.train()
        pinn_solver.save_outcomes(os.path.join(output_dir, "pinn.npz"))
        elapsed_pinn = time.time() - start_time_pinn

    # Run KAN Solver
    if "kan" in active_solvers:
        print("\n--- Running KAN Solver ---")
        start_time_kan = time.time()
        wall_time_sec = args.wall_time * 60.0 if args.wall_time is not None else None
        kan_solver = KANExperiment(config_path, wall_time_limit=wall_time_sec, optimized=is_optimized)
        kan_solver.train()
        kan_solver.save_outcomes(os.path.join(output_dir, "kan.npz"))
        elapsed_kan = time.time() - start_time_kan

    # Run IGA Solver
    if "iga" in active_solvers:
        print("\n--- Running IGA Solver ---")
        start_time_iga = time.time()
        iga_solver = IGAExperiment(config_path, optimized=is_optimized)
        iga_solver.train()
        iga_solver.save_outcomes(os.path.join(output_dir, "iga.npz"))
        elapsed_iga = time.time() - start_time_iga

    elapsed_total = time.time() - start_time_total
    print(f"\n=== Solver Runs Completed. Results saved in {output_dir} ===")

    # Get environment metadata
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"

    environment_info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": str(torch.__version__),
        "device": device_name,
        "optimized": is_optimized
    }

    # Count trainable parameters
    def count_parameters(model):
        if model is None:
            return 0
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load existing metadata if it exists to preserve results of other solvers
    metadata_path = os.path.join(output_dir, "metadata.yaml")
    existing_metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                existing_metadata = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load existing metadata.yaml: {e}")

    existing_results = existing_metadata.get("results", {})

    # Compile solver results and complexities
    solvers_metadata = {}

    if pinn_solver is not None:
        solvers_metadata["PINN"] = {
            "trainable_parameters": count_parameters(pinn_solver.model),
            "layers": pinn_solver.config.LAYERS,
            "neurons_per_layer": pinn_solver.config.NEURONS_PER_LAYER,
            "activation": str(pinn_solver.config.ACTIVATION),
            "sampler_type": str(pinn_solver.config.SAMPLER_TYPE),
            "epochs_trained": getattr(pinn_solver, "epochs_trained", pinn_solver.config.EPOCHS),
            "epochs_total": getattr(pinn_solver, "epochs_total", pinn_solver.config.EPOCHS),
            "final_loss": float(pinn_solver.final_loss) if pinn_solver.final_loss is not None else None,
            "final_h1_error": float(pinn_solver.final_h1_error) if pinn_solver.final_h1_error is not None else None,
            "final_l2_error": float(pinn_solver.final_l2_error) if getattr(pinn_solver, "final_l2_error", None) is not None else None,
            "final_linf_error": float(pinn_solver.final_linf_error) if getattr(pinn_solver, "final_linf_error", None) is not None else None,
            "elapsed_seconds": elapsed_pinn
        }
    elif "PINN" in existing_results:
        solvers_metadata["PINN"] = existing_results["PINN"]

    if kan_solver is not None:
        solvers_metadata["KAN"] = {
            "trainable_parameters": count_parameters(kan_solver.model),
            "layers": kan_solver.config.KAN_LAYERS,
            "neurons_per_layer": kan_solver.config.KAN_NEURONS_PER_LAYER,
            "spline_type": str(kan_solver.config.KAN_SPLINE_TYPE),
            "sampler_type": str(kan_solver.config.SAMPLER_TYPE),
            "epochs_trained": getattr(kan_solver, "epochs_trained", getattr(kan_solver.config, "KAN_EPOCHS", None) or kan_solver.config.EPOCHS),
            "epochs_total": getattr(kan_solver, "epochs_total", getattr(kan_solver.config, "KAN_EPOCHS", None) or kan_solver.config.EPOCHS),
            "final_loss": float(kan_solver.final_loss) if kan_solver.final_loss is not None else None,
            "final_h1_error": float(kan_solver.final_h1_error) if kan_solver.final_h1_error is not None else None,
            "final_l2_error": float(kan_solver.final_l2_error) if getattr(kan_solver, "final_l2_error", None) is not None else None,
            "final_linf_error": float(kan_solver.final_linf_error) if getattr(kan_solver, "final_linf_error", None) is not None else None,
            "elapsed_seconds": elapsed_kan
        }
    elif "KAN" in existing_results:
        solvers_metadata["KAN"] = existing_results["KAN"]

    if iga_solver is not None:
        solvers_metadata["IGA"] = {
            "method": str(iga_solver.config.IGA_METHOD),
            "mesh_type": str(iga_solver.config.IGA_MESH_TYPE),
            "elements": iga_solver.config.IGA_ELEMENTS,
            "degree": iga_solver.config.IGA_DEGREE,
            "degrees_of_freedom": (iga_solver.config.IGA_ELEMENTS + iga_solver.config.IGA_DEGREE) ** 2,
            "final_loss": float(iga_solver.final_loss) if iga_solver.final_loss is not None else None,
            "final_h1_error": float(iga_solver.final_h1_error) if iga_solver.final_h1_error is not None else None,
            "final_l2_error": float(iga_solver.final_l2_error) if getattr(iga_solver, "final_l2_error", None) is not None else None,
            "final_linf_error": float(iga_solver.final_linf_error) if getattr(iga_solver, "final_linf_error", None) is not None else None,
            "elapsed_seconds": elapsed_iga
        }
    elif "IGA" in existing_results:
        solvers_metadata["IGA"] = existing_results["IGA"]

    # Gather config
    config_dict = base_config.to_dict()

    # Re-read on-disk metadata to merge any solvers concurrently completed
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                latest_disk_data = yaml.safe_load(f) or {}
                latest_results = latest_disk_data.get("results", {})
                for k, v in latest_results.items():
                    if k not in solvers_metadata or solvers_metadata[k] is None:
                        solvers_metadata[k] = v
        except Exception as e:
            print(f"Warning: Failed to merge existing metadata.yaml: {e}")

    metadata = {
        "config": config_dict,
        "execution": {
            "start_date": start_date,
            "elapsed_seconds_total": elapsed_total,
        },
        "environment": environment_info,
        "results": solvers_metadata
    }

    # Write atomically
    temp_metadata_path = metadata_path + ".tmp"
    with open(temp_metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    for _ in range(5):
        try:
            if os.path.exists(metadata_path):
                os.replace(temp_metadata_path, metadata_path)
            else:
                os.rename(temp_metadata_path, metadata_path)
            break
        except Exception:
            time.sleep(0.05)
    print(f"Metadata saved successfully to {metadata_path}")

if __name__ == "__main__":
    main()
