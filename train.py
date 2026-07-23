import argparse
import os
import time
import sys
import platform
import yaml
import torch
from pinn import PINNExperiment
from kan import KANExperiment
from iga import IGAExperiment

def main():
    parser = argparse.ArgumentParser(description="Run PINN, KAN, and IGA solvers based on a YAML configuration.")
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

    # Determine subfolder name based on config file basename without extension
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join("output", config_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Solver Run for Configuration: {config_name} ===")
    print(f"Config path: {config_path}")
    print(f"Output folder: {output_dir}\n")

    # Record start time & date
    start_date = time.strftime("%Y-%m-%d %H:%M:%S")
    start_time_total = time.time()

    # Run PINN
    print("--- Running PINN Solver ---")
    start_time_pinn = time.time()
    pinn_solver = PINNExperiment(config_path)
    pinn_solver.train()
    pinn_solver.save_outcomes(os.path.join(output_dir, "pinn.npz"))
    elapsed_pinn = time.time() - start_time_pinn

    # Run KAN Mock
    print("\n--- Running KAN Solver (Mock) ---")
    start_time_kan = time.time()
    kan_solver = KANExperiment(config_path)
    kan_solver.train()
    kan_solver.save_outcomes(os.path.join(output_dir, "kan.npz"))
    elapsed_kan = time.time() - start_time_kan

    # Run IGA Mock
    print("\n--- Running IGA Solver (Mock) ---")
    start_time_iga = time.time()
    iga_solver = IGAExperiment(config_path)
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
        "device": device_name
    }

    # Count trainable parameters
    def count_parameters(model):
        if model is None:
            return 0
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Compile solver results and complexities
    solvers_metadata = {
        "PINN": {
            "trainable_parameters": count_parameters(pinn_solver.model),
            "layers": pinn_solver.config.LAYERS,
            "neurons_per_layer": pinn_solver.config.NEURONS_PER_LAYER,
            "activation": str(pinn_solver.config.ACTIVATION),
            "final_loss": float(pinn_solver.final_loss) if pinn_solver.final_loss is not None else None,
            "final_h1_error": float(pinn_solver.final_h1_error) if pinn_solver.final_h1_error is not None else None,
            "elapsed_seconds": elapsed_pinn
        },
        "KAN": {
            "trainable_parameters": count_parameters(kan_solver.model),
            "layers": kan_solver.config.KAN_LAYERS,
            "neurons_per_layer": kan_solver.config.KAN_NEURONS_PER_LAYER,
            "final_loss": float(kan_solver.final_loss) if kan_solver.final_loss is not None else None,
            "final_h1_error": float(kan_solver.final_h1_error) if kan_solver.final_h1_error is not None else None,
            "elapsed_seconds": elapsed_kan
        },
        "IGA": {
            "elements": iga_solver.config.IGA_ELEMENTS,
            "degree": iga_solver.config.IGA_DEGREE,
            "degrees_of_freedom": (iga_solver.config.IGA_ELEMENTS + iga_solver.config.IGA_DEGREE) ** 2,
            "final_loss": float(iga_solver.final_loss) if iga_solver.final_loss is not None else None,
            "final_h1_error": float(iga_solver.final_h1_error) if iga_solver.final_h1_error is not None else None,
            "elapsed_seconds": elapsed_iga
        }
    }

    # Gather config (use pinn_solver config as reference)
    config_dict = pinn_solver.config.to_dict()

    metadata = {
        "config": config_dict,
        "execution": {
            "start_date": start_date,
            "elapsed_seconds_total": elapsed_total,
        },
        "environment": environment_info,
        "results": solvers_metadata
    }

    metadata_path = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Metadata saved successfully to {metadata_path}")

if __name__ == "__main__":
    main()
