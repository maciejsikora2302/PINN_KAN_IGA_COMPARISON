import argparse
import os
import time
import sys
import platform
import yaml
import torch
from config import RunConfig
from model import SolverOutcome
from src.solvers.pinn import PINNExperiment
from src.solvers.kan import KANExperiment
from src.solvers.iga import IGAExperiment

def main():
    parser = argparse.ArgumentParser(description="Execute single-method training for PINN, KAN, or IGA.")
    parser.add_argument("--config", type=str, help="Configuration filename or path.")
    parser.add_argument("config_pos", type=str, nargs="?", help="Configuration filename or path (positional).")
    parser.add_argument("--wall-time", nargs="?", const=5.0, type=float, help="Stop training if wall-clock time in minutes is exceeded.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip running if output outcomes.npz already exists.")
    parser.add_argument("--optimized", action="store_true", help="Enable performance optimizations.")
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

    # Load configuration
    config = RunConfig()
    config.load_config(config_path)
    config.validate_config()

    is_optimized = args.optimized or getattr(config, "OPTIMIZED", False)
    if is_optimized:
        config.OPTIMIZED = True
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass

    # Resolve output directory
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    outcomes_path = os.path.join(output_dir, "outcomes.npz")
    metadata_path = os.path.join(output_dir, "metadata.yaml")

    # Skip existing if requested
    if args.skip_existing and os.path.exists(outcomes_path) and os.path.exists(metadata_path):
        print(f"Skipping already completed run at {output_dir}")
        return

    solver_type = config.SOLVER.lower()
    print(f"================================================================================")
    print(f"=== Starting Training Run: {config.problem_id} / {config.METHOD_NAME} ===")
    print(f"  Solver:     {solver_type.upper()}")
    print(f"  Problem:    {config.PROBLEM_NAME} (Epsilon: {config.EPSILON})")
    print(f"  Config:     {config_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"================================================================================\n")

    start_date = time.strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.perf_counter()
    wall_time_sec = args.wall_time * 60.0 if args.wall_time is not None else None

    # Instantiate experiment
    if solver_type == "pinn":
        experiment = PINNExperiment(config_path, wall_time_limit=wall_time_sec, optimized=is_optimized)
        model_save_path = os.path.join(output_dir, "model.pt")
    elif solver_type == "kan":
        experiment = KANExperiment(config_path, wall_time_limit=wall_time_sec, optimized=is_optimized)
        model_save_path = os.path.join(output_dir, "model.pt")
    elif solver_type == "iga":
        experiment = IGAExperiment(config_path, optimized=is_optimized)
        model_save_path = os.path.join(output_dir, "iga_coeffs.npy")
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    # Train and obtain strongly-typed outcome
    outcome: SolverOutcome = experiment.train()

    # Save artifacts
    outcome.save(outcomes_path)
    experiment.save_model(model_save_path)

    total_elapsed = time.perf_counter() - start_time

    # Collect environment info
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

    # Structured metadata from typed SolverMetrics
    solver_metadata = outcome.metrics.to_dict()
    solver_metadata["solver"] = solver_type
    solver_metadata["method_name"] = config.METHOD_NAME
    solver_metadata["sampler_or_mesh"] = config.SAMPLER_TYPE if solver_type != "iga" else config.IGA_MESH_TYPE

    metadata = {
        "config": config.to_dict(),
        "execution": {
            "start_date": start_date,
            "elapsed_seconds_total": total_elapsed,
        },
        "environment": environment_info,
        "results": {
            solver_type.upper(): solver_metadata
        }
    }

    # Save metadata atomically
    temp_metadata_path = metadata_path + ".tmp"
    with open(temp_metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    if os.path.exists(metadata_path):
        os.replace(temp_metadata_path, metadata_path)
    else:
        os.rename(temp_metadata_path, metadata_path)

    print(f"\nTraining completed successfully in {total_elapsed:.2f}s.")
    print(f"  Metrics:")
    print(f"    - H1 Error:   {outcome.final_h1_error:.6e}")
    print(f"    - L2 Error:   {outcome.final_l2_error:.6e}")
    print(f"    - Linf Error: {outcome.final_linf_error:.6e}")
    print(f"    - DoFs/Params:{outcome.metrics.trainable_parameters_or_dofs}")
    print(f"  Artifacts saved:")
    print(f"    - Outcomes: {outcomes_path}")
    print(f"    - Metadata: {metadata_path}")
    print(f"    - Model:    {model_save_path}")
    print(f"================================================================================\n")

if __name__ == "__main__":
    main()
