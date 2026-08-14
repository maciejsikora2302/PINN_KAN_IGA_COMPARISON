import os
import sys
import subprocess
import time
import re
import argparse
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def natural_sort_key(filename):
    if filename == "test_config.yaml":
        return (0, 0, filename)
    match = re.search(r'exp(\d+)', filename)
    if match:
        return (1, int(match.group(1)), filename)
    if "test" in filename.lower():
        return (3, 0, filename)
    return (2, 0, filename)

def natural_sort(files):
    return sorted(files, key=natural_sort_key)

def get_config_solvers(config_path, cli_solvers=None):
    """Determine active solvers for a config file."""
    if cli_solvers:
        return [s.lower() for s in cli_solvers]
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if "SOLVERS" in data:
            s_val = data["SOLVERS"]
            if isinstance(s_val, str):
                return [s_val.lower()]
            return [str(s).lower() for s in s_val]
    except Exception:
        pass
    return ["pinn", "kan", "iga"]

def run_training_step(python_exe, config_path, solvers, wall_time=None, skip_existing=False, optimized=False):
    """Run train.py for a specific subset of solvers."""
    train_cmd = [python_exe, "train.py", "--config", config_path]
    if wall_time is not None:
        train_cmd.extend(["--wall-time", str(wall_time)])
    if solvers:
        train_cmd.extend(["--solvers"] + solvers)
    if skip_existing:
        train_cmd.append("--skip-existing")
    if optimized:
        train_cmd.append("--optimized")
        
    start_t = time.time()
    res = subprocess.run(train_cmd)
    elapsed = time.time() - start_t
    return res.returncode, elapsed

def run_plot_step(python_exe, config_path):
    """Run plot_parallel.py for a configuration."""
    start_t = time.time()
    plot_cmd = [python_exe, "plot_parallel.py", "--config", config_path]
    res = subprocess.run(plot_cmd)
    elapsed = time.time() - start_t
    return res.returncode, elapsed

def process_single_config_sequential(config, config_path, python_exe, wall_time, solvers, skip_existing, optimized, idx, total_count):
    """Sequential runner for a single configuration (all solvers then plot)."""
    print("=" * 80)
    print(f"[{idx}/{total_count}] Processing Configuration (Sequential): {config} (Optimized: {optimized})")
    print("=" * 80)
    
    train_rc, train_time = run_training_step(python_exe, config_path, solvers, wall_time, skip_existing, optimized)
    if train_rc != 0:
        print(f"Error: Training failed for configuration {config} (Exit code: {train_rc})")
        return config, {"status": "Failed (Training)", "time": train_time}
        
    plot_rc, plot_time = run_plot_step(python_exe, config_path)
    total_config_time = train_time + plot_time
    if plot_rc != 0:
        print(f"Error: Plotting failed for configuration {config} (Exit code: {plot_rc})")
        return config, {"status": "Failed (Plotting)", "time": total_config_time}
    else:
        print(f"Successfully processed configuration {config} in {total_config_time:.2f}s")
        return config, {"status": "Success", "time": total_config_time}

def run_concurrent_pipeline(configs, config_dir, python_exe, wall_time, cli_solvers, skip_existing, optimized, max_workers):
    """
    Executes benchmark suite concurrently with hardware-aware division:
    - GPU Task Stream: PINN and KAN training run sequentially (1 task at a time on GPU)
      to avoid context switching and maximize GPU throughput.
    - CPU Worker Pool: IGA tasks run concurrently across CPU workers.
    - GPU stream and CPU pool run in parallel simultaneously.
    - Plotting: Triggered per config as soon as all active solvers for that config complete.
    """
    cpu_workers = max_workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"--- Hardware-Aware Concurrent Pipeline ---")
    print(f"  GPU Stream: 1 dedicated worker for PINN/KAN tasks (sequential GPU execution)")
    print(f"  CPU Stream: {cpu_workers} worker threads for IGA tasks (parallel CPU execution)")
    print(f"  Plotting:   Auto-dispatched per configuration upon solver completion")
    print("=" * 57 + "\n")

    results = {}
    lock = threading.Lock()

    # Partition tasks per configuration
    config_tasks = {}
    gpu_tasks = []
    cpu_tasks = []

    for idx, config in enumerate(configs, 1):
        config_path = os.path.join(config_dir, config)
        solvers = get_config_solvers(config_path, cli_solvers)
        
        gpu_solvers = [s for s in solvers if s in ("pinn", "kan")]
        cpu_solvers = [s for s in solvers if s not in ("pinn", "kan")]  # e.g. "iga"
        
        config_tasks[config] = {
            "path": config_path,
            "idx": idx,
            "gpu_needed": bool(gpu_solvers),
            "gpu_solvers": gpu_solvers,
            "gpu_done": not bool(gpu_solvers),
            "gpu_success": True,
            "gpu_time": 0.0,
            "cpu_needed": bool(cpu_solvers),
            "cpu_solvers": cpu_solvers,
            "cpu_done": not bool(cpu_solvers),
            "cpu_success": True,
            "cpu_time": 0.0,
            "wall_time_start": time.time(),
        }

        if gpu_solvers:
            gpu_tasks.append((idx, config, config_path, gpu_solvers))
        if cpu_solvers:
            cpu_tasks.append((idx, config, config_path, cpu_solvers))

    def check_and_trigger_plot(config):
        """Called under lock to verify if all solvers for config have completed."""
        t_info = config_tasks[config]
        if not t_info["gpu_done"] or not t_info["cpu_done"]:
            return None
        return t_info

    def execute_plotting(config, t_info):
        """Execute plotting step for a fully solved configuration."""
        config_path = t_info["path"]
        training_success = t_info["gpu_success"] and t_info["cpu_success"]
        
        if not training_success:
            total_elapsed = max(t_info["gpu_time"], t_info["cpu_time"])
            with lock:
                print(f"[FAIL] Training failed for configuration {config}")
                results[config] = {"status": "Failed (Training)", "time": total_elapsed}
            return

        with lock:
            print(f"[PLOT] Starting plots for configuration {config}...")
            
        plot_rc, plot_time = run_plot_step(python_exe, config_path)
        total_elapsed = max(t_info["gpu_time"], t_info["cpu_time"]) + plot_time

        with lock:
            if plot_rc == 0:
                print(f"[DONE] Configuration {config} finished successfully in {total_elapsed:.2f}s")
                results[config] = {"status": "Success", "time": total_elapsed}
            else:
                print(f"[FAIL] Plotting failed for configuration {config} (Exit code: {plot_rc})")
                results[config] = {"status": "Failed (Plotting)", "time": total_elapsed}

    def gpu_worker_loop():
        """Single GPU worker executing PINN and KAN tasks sequentially."""
        total_gpu = len(gpu_tasks)
        for g_idx, (idx, config, config_path, gpu_solvers) in enumerate(gpu_tasks, 1):
            with lock:
                print(f"[GPU] [{g_idx}/{total_gpu}] Starting PINN/KAN for {config} ({', '.join(gpu_solvers)})")
            
            rc, elapsed = run_training_step(python_exe, config_path, gpu_solvers, wall_time, skip_existing, optimized)
            
            ready_info = None
            with lock:
                t_info = config_tasks[config]
                t_info["gpu_done"] = True
                t_info["gpu_success"] = (rc == 0)
                t_info["gpu_time"] = elapsed
                if rc == 0:
                    print(f"[GPU] [{g_idx}/{total_gpu}] Completed PINN/KAN for {config} in {elapsed:.2f}s")
                else:
                    print(f"[GPU-ERR] [{g_idx}/{total_gpu}] PINN/KAN failed for {config} (Exit code: {rc})")
                ready_info = check_and_trigger_plot(config)
                
            if ready_info is not None:
                execute_plotting(config, ready_info)

    def cpu_worker_task(c_idx, total_cpu, idx, config, config_path, cpu_solvers):
        """Individual CPU worker task for IGA execution."""
        with lock:
            print(f"[CPU] [{c_idx}/{total_cpu}] Starting IGA for {config}")
        
        rc, elapsed = run_training_step(python_exe, config_path, cpu_solvers, wall_time, skip_existing, optimized)
        
        ready_info = None
        with lock:
            t_info = config_tasks[config]
            t_info["cpu_done"] = True
            t_info["cpu_success"] = (rc == 0)
            t_info["cpu_time"] = elapsed
            if rc == 0:
                print(f"[CPU] [{c_idx}/{total_cpu}] Completed IGA for {config} in {elapsed:.2f}s")
            else:
                print(f"[CPU-ERR] [{c_idx}/{total_cpu}] IGA failed for {config} (Exit code: {rc})")
            ready_info = check_and_trigger_plot(config)
            
        if ready_info is not None:
            execute_plotting(config, ready_info)

    # Check for any configs that needed neither GPU nor CPU
    for config, t_info in config_tasks.items():
        if not t_info["gpu_needed"] and not t_info["cpu_needed"]:
            execute_plotting(config, t_info)

    # Launch GPU thread and CPU thread pool concurrently
    gpu_thread = threading.Thread(target=gpu_worker_loop)
    gpu_thread.start()

    total_cpu = len(cpu_tasks)
    with ThreadPoolExecutor(max_workers=cpu_workers) as cpu_executor:
        cpu_futures = [
            cpu_executor.submit(cpu_worker_task, c_idx, total_cpu, idx, config, config_path, cpu_solvers)
            for c_idx, (idx, config, config_path, cpu_solvers) in enumerate(cpu_tasks, 1)
        ]
        for f in as_completed(cpu_futures):
            f.result()

    # Wait for GPU thread to complete
    gpu_thread.join()

    return results

def main():
    config_dir = "training_config"
    if not os.path.exists(config_dir):
        print(f"Error: Configuration directory '{config_dir}' not found.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run all solver configurations and generate plots.")
    parser.add_argument("--start", "-s", type=str, help="Experiment name, prefix, or number to start from.")
    parser.add_argument("--wall-time", "-w", nargs="?", const=5.0, type=float, help="Stop training early after a specified wall time in minutes (default 5.0).")
    parser.add_argument("--solvers", type=str, nargs="+", help="Specify which solvers to run (pinn, kan, iga). Overrides config defaults.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip running solvers whose outputs (.npz files) already exist.")
    parser.add_argument("--optimized", action="store_true", help="Enable performance optimizations across all solver executions.")
    parser.add_argument("--concurrent-execution", "-c", action="store_true", help="Run independent configuration runs concurrently (sequential GPU for PINN/KAN, parallel CPU for IGA).")
    parser.add_argument("--max-workers", "-j", type=int, default=None, help="Maximum number of parallel worker processes for CPU concurrent execution.")
    args = parser.parse_args()

    # Find all yaml/yml files, skipping common.yaml and original.yaml
    configs = []
    for f in os.listdir(config_dir):
        if (f.endswith(".yaml") or f.endswith(".yml")) and f not in ("common.yaml", "original.yaml"):
            configs.append(f)
    
    configs = natural_sort(configs)

    if not configs:
        print(f"No configuration files found in '{config_dir}' (excluding common.yaml and original.yaml).")
        return

    # Handle start-from logic
    skipped_configs = []
    run_configs = list(configs)

    if args.start:
        start_idx = None
        # 1. Exact match
        for idx, c in enumerate(configs):
            if c == args.start:
                start_idx = idx
                break
        
        # 2. Prefix match
        if start_idx is None:
            for idx, c in enumerate(configs):
                if c.startswith(args.start):
                    start_idx = idx
                    break
        
        # 3. Number match (e.g. "5" -> "exp5_")
        if start_idx is None and args.start.isdigit():
            target_prefix = f"exp{args.start}_"
            for idx, c in enumerate(configs):
                if c.startswith(target_prefix):
                    start_idx = idx
                    break

        if start_idx is None:
            print(f"Error: Starting configuration matching '{args.start}' was not found.")
            print("Available configurations in deterministic order:")
            for c in configs:
                print(f"  - {c}")
            sys.exit(1)
            
        skipped_configs = configs[:start_idx]
        run_configs = configs[start_idx:]

    print("=== Deterministic Configuration Run Order Evaluation ===")
    if skipped_configs:
        print(f"Skipped configurations ({len(skipped_configs)}):")
        for c in skipped_configs:
            print(f"  [SKIP] {c}")
    print(f"Configurations to run ({len(run_configs)}):")
    for c in run_configs:
        print(f"  [RUN]  {c}")
    print(f"Optimization mode: {'OPTIMIZED' if args.optimized else 'BASELINE'}")
    print(f"Execution mode:    {'CONCURRENT (Split GPU/CPU Stream)' if args.concurrent_execution else 'SEQUENTIAL'}")
    print("=" * 57 + "\n")

    configs = run_configs
    
    results = {}
    python_exe = sys.executable  # Use current Python interpreter
    total_start = time.time()

    if args.concurrent_execution:
        results = run_concurrent_pipeline(
            configs=configs,
            config_dir=config_dir,
            python_exe=python_exe,
            wall_time=args.wall_time,
            cli_solvers=args.solvers,
            skip_existing=args.skip_existing,
            optimized=args.optimized,
            max_workers=args.max_workers
        )
    else:
        for idx, config in enumerate(configs, 1):
            config_path = os.path.join(config_dir, config)
            _, res_dict = process_single_config_sequential(
                config=config,
                config_path=config_path,
                python_exe=python_exe,
                wall_time=args.wall_time,
                solvers=args.solvers,
                skip_existing=args.skip_existing,
                optimized=args.optimized,
                idx=idx,
                total_count=len(configs)
            )
            results[config] = res_dict

    # Generate summary tables and comparison plots
    print("\n" + "=" * 80)
    print("GENERATING GLOBAL SUMMARY TABLES AND COMPARISON PLOTS")
    print("=" * 80)
    
    summary_tables_cmd = [python_exe, "generate_summary_tables.py"]
    print(f"Running: {' '.join(summary_tables_cmd)}")
    summary_tables_res = subprocess.run(summary_tables_cmd)
    
    comparison_suite_cmd = [python_exe, "plot_comparison_suite.py"]
    print(f"Running: {' '.join(comparison_suite_cmd)}")
    comparison_suite_res = subprocess.run(comparison_suite_cmd)
    
    # Validation
    print("\n--- Validating Generated Summary & Plot Files ---")
    validation_ok = True
    
    expected_files = [
        os.path.join("output", "summary.md"),
        os.path.join("output", "summary_table.tex"),
    ]
    for ef in expected_files:
        if os.path.exists(ef) and os.path.getsize(ef) > 0:
            print(f"  [OK]  {ef} generated successfully.")
        else:
            print(f"  [ERR] {ef} is missing or empty.")
            validation_ok = False
            
    comparison_dir = os.path.join("output", "comparison_suite")
    if os.path.exists(comparison_dir) and os.path.isdir(comparison_dir):
        png_files = [f for f in os.listdir(comparison_dir) if f.endswith(".png")]
        if png_files:
            print(f"  [OK]  {comparison_dir} contains {len(png_files)} plot(s): {', '.join(png_files[:3])}...")
        else:
            print(f"  [WARN] {comparison_dir} exists but contains no plots.")
    else:
        print(f"  [ERR] {comparison_dir} was not created.")
        validation_ok = False
        
    if validation_ok:
        print("Validation Successful: All summary files and plots are generated correctly.")
    else:
        print("Validation Warning: Some expected summary tables or comparison plots were missing.")

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("RUN ALL CONFIGS SUMMARY")
    print("=" * 80)
    print(f"Total time elapsed: {total_time:.2f} seconds\n")
    
    success_count = 0
    for config, info in results.items():
        status = info.get("status", "Unknown")
        elapsed = info.get("time", 0.0)
        print(f"  - {config:<25}: {status:<20} (took {elapsed:.2f}s)")
        if status == "Success":
            success_count += 1
            
    print("-" * 80)
    print(f"Completed: {success_count}/{len(configs)} configurations succeeded.")
    print("=" * 80)

if __name__ == "__main__":
    main()
