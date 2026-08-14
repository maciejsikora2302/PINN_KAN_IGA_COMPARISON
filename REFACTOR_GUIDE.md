# REFACTOR GUIDE: PINN, KAN & IGA Benchmark Suite Refactoring Blueprint

This document is the **definitive, step-by-step instruction guide** for the AI agent to implement the refactoring of the PINN-KAN-IGA comparison framework.

---

## Table of Contents
1. [Core Architectural Specifications](#1-core-architectural-specifications)
2. [Phase 1: Codebase Restructuring & Root Cleanup](#phase-1-codebase-restructuring--root-cleanup)
3. [Phase 2: Unified Configuration Engine & Declarative Generator](#phase-2-unified-configuration-engine--declarative-generator)
4. [Phase 3: Single-Method Compute-Only Training Pipeline](#phase-3-single-method-compute-only-training-pipeline)
5. [Phase 4: Decoupled Parallel Per-Run Visualizations](#phase-4-decoupled-parallel-per-run-visualizations)
6. [Phase 5: Problem-Level Cross-Method Comparison Suite](#phase-5-problem-level-cross-method-comparison-suite)
7. [Phase 6: Global Multi-Problem Benchmark Suite](#phase-6-global-multi-problem-benchmark-suite)
8. [Phase 7: Master Orchestrator (`run_all.py`) & Verification](#phase-7-master-orchestrator-run_allpy--verification)

---

## 1. Core Architectural Specifications

### Benchmark Problems Matrix (from `abstract.pdf`)
1. **Poisson Sine**: $-\Delta u = f(x, y)$ with $u(x, y) = \sin(2\pi x)\sin(2\pi y)$ on $\Omega = (0, 1)^2$.
2. **Poisson Exp**: $-\Delta u = f(x, y)$ with $u(x, y) = \sin(2\pi x)\sin(2\pi y)\exp(\pi(x - 2y))$ on $\Omega = (0, 1)^2$.
3. **Eriksson-Johnson Advection-Diffusion**: $u_t - \epsilon(u_{xx} + u_{yy}) = 0$ with $\epsilon \in \{1.0, 0.1, 0.01, 0.001\}$ on $\Omega = (0, 1)^2$.

### Solver Methods Matrix
- **PINN**: Standard residual loss on Uniform & Boundary Layer samplers.
- **R-PINN**: Riesz-representative Gram-inverted robust loss on Uniform & Boundary Layer samplers.
- **NURBS-KAN**: Kolmogorov-Arnold Network with NURBS activations on Uniform & Boundary Layer samplers.
- **R-NURBS-KAN**: Robust Gram loss with NURBS-KAN on Uniform & Boundary Layer samplers.
- **Standard IGA-FEM**: Higher-order B-spline Galerkin on Uniform mesh.
- **SUPG IGA-FEM**: Streamline-Upwind Petrov-Galerkin on Uniform & Boundary Layer Adaptive meshes.
- **IGA-RM / iGRM**: Isogeometric Residual Minimization on Uniform & Boundary Layer Adaptive meshes.

### Output Directory Structure
```
output/
├── <problem_id>/                             # e.g., poisson_sine, eriksson_johnson_eps0.01
│   ├── <method_name>/                        # e.g., pinn_uniform, r_nurbs_kan_boundary, iga_supg_adaptive
│   │   ├── outcomes.npz                      # Arrays, histories, errors, time
│   │   ├── metadata.yaml                     # Config & metrics
│   │   ├── model.pt / iga_coeffs.npy         # Weights
│   │   ├── loss_curve.png
│   │   ├── h1_error_curve.png
│   │   ├── prediction_contour.png
│   │   ├── error_contour.png
│   │   └── solution_slices.png
│   └── comparisons/                          # Problem-level comparison suite
│       ├── convergence_overlay_epochs.png
│       ├── convergence_overlay_time.png
│       ├── convergence_overlay_progress.png
│       ├── side_by_side_solution_grid.png
│       ├── side_by_side_error_grid.png
│       ├── boundary_layer_slice_comparison.png
│       └── summary_table.md / .csv
└── global_comparisons/                       # Global multi-problem synthesis
    ├── global_metrics_table.md / .csv
    ├── pareto_efficiency_h1_vs_time.png
    ├── pareto_efficiency_h1_vs_dofs.png
    ├── robust_loss_ablation.png
    ├── uniform_vs_adaptive_sampling_impact.png
    └── abstract_summary_overview.png
```

---

## Phase 1: Codebase Restructuring & Root Cleanup

### Objective
Remove top-level wrapper scripts (`pinn.py`, `kan.py`, `iga.py`) from the root directory and cleanly consolidate them into `src/solvers/`.

### Instructions for Agent:
1. **Consolidate PINN**:
   - Check `pinn.py`. The exact solution helpers in `pinn.py` delegate directly to `src.problems.get_problem`. Move any required utility functions into `src/solvers/pinn/pinn_solver.py` or export them cleanly from `src/solvers/pinn/__init__.py`.
   - Update `src/solvers/pinn/__init__.py` to export: `PINNExperiment`, `PINN`, `f`, `df`, `dfdx`, `dfdt`.
   - Delete root `pinn.py`.
2. **Consolidate KAN**:
   - `kan.py` at root simply re-exports from `src.solvers.kan`. Ensure `src/solvers/kan/__init__.py` exports `KANExperiment`, `KANModel`, `KAN`, `KANLinear`, `NURBSLinear`.
   - Delete root `kan.py`.
3. **Consolidate IGA**:
   - `iga.py` at root re-exports from `src.solvers.iga`. Ensure `src/solvers/iga/__init__.py` exports `IGAExperiment`, `BaseIGASolver`, `StandardIGASolver`, `SUPGIGASolver`, `ResidualMinimizationIGASolver`, `bspline_basis`, `bspline_basis_deriv`.
   - Delete root `iga.py`.
4. **Update Imports Across Workspace**:
   - Change `from pinn import ...` $\rightarrow$ `from src.solvers.pinn import ...`
   - Change `from kan import ...` $\rightarrow$ `from src.solvers.kan import ...`
   - Change `from iga import ...` $\rightarrow$ `from src.solvers.iga import ...`

---

## Phase 2: Unified Configuration Engine & Declarative Generator

### Objective
Create a clean, declarative configuration engine where:
$$\text{Run Config} = \text{Problem Spec} + \text{Method Spec}$$
Eliminate interactive prompts, manual `--force` flags, and ambiguous multi-solver configs.

### Instructions for Agent:
1. **Refactor `config.py`**:
   - Define dataclasses or dictionary schemas:
     - `ProblemConfig`:
       - Fields: `PROBLEM_NAME` (`"poisson_sine" | "poisson_exp" | "eriksson_johnson"`), `EPSILON` (float, default 0.01), `LENGTH` (float, 1.0), `TOTAL_TIME` (float, 1.0), `EXAMPLE` (optional legacy integer).
       - Computed property `problem_id`:
         - If `"poisson_sine"`: `"poisson_sine"`
         - If `"poisson_exp"`: `"poisson_exp"`
         - If `"eriksson_johnson"`: `f"eriksson_johnson_eps{EPSILON}"` (e.g. `eriksson_johnson_eps0.01`).
     - `MethodConfig`:
       - Fields: `SOLVER` (`"pinn" | "kan" | "iga"`), `METHOD_NAME` (string identifier e.g. `"pinn_uniform"`, `"r_nurbs_kan_boundary"`, `"iga_supg_adaptive"`).
       - Neural network hyperparams: `RPINN` (0 or 1), `SAMPLER_TYPE` (`"uniform" | "boundary_layer"`), `SAMPLER_GAMMA` (float, 3.0), `ACTIVATION` (`"tanh" | "sin"`), `LAYERS`, `NEURONS_PER_LAYER`, `KAN_LAYERS`, `KAN_NEURONS_PER_LAYER`, `KAN_SPLINE_TYPE`, `KAN_GRID_SIZE`, `KAN_SPLINE_ORDER`, `EPOCHS`, `LEARNING_RATE`.
       - IGA hyperparams: `IGA_METHOD` (`"standard" | "supg" | "igrm"`), `IGA_MESH_TYPE` (`"uniform" | "adaptive"`), `IGA_DEGREE`, `IGA_ELEMENTS`, `IGA_ADAPTIVE_GAMMA`, `IGA_TEST_DEGREE_ENRICHMENT`.
     - `RunConfig`:
       - Combines `ProblemConfig` and `MethodConfig`.
       - Computed property `output_dir`: `os.path.join("output", self.problem_id, self.METHOD_NAME)`.
       - Methods: `load_yaml(path)`, `save_yaml(path)`, `validate()`, `to_dict()`.
2. **Create `sweeps/benchmark_matrix.yaml`**:
   - Define the complete set of problems and methods from `abstract.pdf`:
     - Global parameters (domain size, common points/elements).
     - Problem list (Poisson Sine, Poisson Exp, Eriksson-Johnson for $\epsilon \in \{1.0, 0.1, 0.01, 0.001\}$).
     - Method list per problem:
       - `pinn_uniform` (PINN, uniform sampler)
       - `pinn_boundary` (PINN, boundary layer sampler)
       - `r_pinn_uniform` (R-PINN, uniform sampler)
       - `r_pinn_boundary` (R-PINN, boundary layer sampler)
       - `nurbs_kan_uniform` (NURBS-KAN, uniform sampler)
       - `nurbs_kan_boundary` (NURBS-KAN, boundary layer sampler)
       - `r_nurbs_kan_uniform` (R-NURBS-KAN, uniform sampler)
       - `r_nurbs_kan_boundary` (R-NURBS-KAN, boundary layer sampler)
       - `iga_standard_uniform` (Standard IGA-FEM, uniform mesh)
       - `iga_supg_uniform` (SUPG IGA-FEM, uniform mesh)
       - `iga_supg_adaptive` (SUPG IGA-FEM, boundary-layer adaptive mesh)
       - `iga_igrm_adaptive` (iGRM IGA-FEM, boundary-layer adaptive mesh)
3. **Create `sweeps/test_matrix.yaml`**:
   - Minimal test matrix:
     - Problem: Poisson Sine (`EXAMPLE: 1`).
     - Methods: `pinn_uniform` (5 epochs), `nurbs_kan_uniform` (5 epochs), `iga_standard_uniform` (4 elements).
4. **Refactor `manage_configs.py`**:
   - CLI usage: `python manage_configs.py [sweep_file]` (default `sweeps/benchmark_matrix.yaml`).
   - Cleanly parses the matrix and writes single-run YAMLs into:
     - `training_config/<problem_id>/<method_name>.yaml`
     - When given `sweeps/test_matrix.yaml`, writes into `training_config/test/<method_name>.yaml`.
   - Completely non-interactive, overwrites generated YAMLs deterministically.

---

## Phase 3: Single-Method Compute-Only Training Pipeline

### Objective
Ensure `train.py` executes exactly 1 method run per invocation, has **zero** plotting dependencies, and saves standardized output metrics.

### Instructions for Agent:
1. **Refactor `train.py`**:
   - Parse `--config <path>` (e.g., `training_config/poisson_sine/pinn_uniform.yaml`).
   - Optional CLI flags: `--wall-time <minutes>`, `--optimized`, `--skip-existing`.
   - Load `RunConfig`.
   - Target output folder: `config.output_dir` (i.e. `output/<problem_id>/<method_name>/`).
   - If `--skip-existing` is set and `outcomes.npz` already exists in `config.output_dir`, exit immediately.
   - Dispatch exclusively to:
     - `pinn` $\rightarrow$ `PINNExperiment(config, wall_time_limit, optimized)`
     - `kan` $\rightarrow$ `KANExperiment(config, wall_time_limit, optimized)`
     - `iga` $\rightarrow$ `IGAExperiment(config, optimized)`
   - Execute `.train()`.
   - Save outputs:
     - `outcomes.npz`: Arrays (`x`, `t`, `z_pred`), convergence histories (`loss_history`, `h1_error_history`, `h1_time_history`, `h1_epoch_history`, `h1_progress_history`), final metrics (`final_l2_error`, `final_h1_error`, `final_linf_error`, `elapsed_seconds`).
     - `metadata.yaml`: Environment info, model/solver specs, number of parameters / DoFs, exact elapsed time, final metrics.
     - Model checkpoint / coefficients (`model.pt` or `iga_coeffs.npy`).
   - Print clean summary and exit (NO matplotlib plotting calls).
2. **Standardize Solver Metrics & Return Objects**:
   - In `model.py`, introduce `SolverMetrics` and `SolverOutcome` dataclasses to encapsulate solution grids, numerical predictions, full convergence histories, exact error norms, execution timings, and extra diagnostics (e.g., KAN edge activations, IGA knot vectors).
   - In `src/solvers/iga/base.py`, define `IGASolution` and `IGAMetrics` dataclasses for the internal IGA solvers.
   - All experiment classes (`PINNExperiment`, `KANExperiment`, `IGAExperiment`) return a strongly-typed `SolverOutcome` instance from `.train()`.
   - `train.py` consumes `outcome: SolverOutcome` uniformly, saving `.npz` artifacts and atomic `metadata.yaml` with zero custom branching or guessing.

---

## Phase 4: Decoupled Parallel Per-Run Visualizations

### Objective
Create a standalone visualizer that generates all 5 publication-quality diagnostic plots for a single method run concurrently using `ProcessPoolExecutor`.

### Instructions for Agent:
1. **Create `src/visualization/run_visualizer.py`**:
   - Define dedicated plotting routines (using `matplotlib` with `Agg` backend):
     - `plot_loss_curve(loss_history, save_path)`: Semilogy plot of training loss vs epochs.
     - `plot_h1_convergence(h1_history, epoch_history, time_history, save_path)`: Semilogy curve of $H^1$ semi-norm error vs epochs / wall-clock seconds.
     - `plot_2d_solution(x, t, z_pred, title, save_path)`: 2D contour/heatmap of predicted solution $u_h(x, y)$.
     - `plot_2d_error(x, t, z_pred, z_exact, title, save_path)`: 2D contour of pointwise absolute error $|u - u_h|$ in $\log_{10}$ scale.
     - `plot_1d_slices(x, t, z_pred, z_exact, problem, save_path)`: 1D cross-section curves comparing $u_h(x, y)$ vs $u(x, y)$ along $x=0.5$ and near the boundary $y=0.95, y=1.0$.
2. **Create `plot_run.py`**:
   - CLI usage: `python plot_run.py --output-dir <path>` or `python plot_run.py --config <path>`.
   - Loads `outcomes.npz` and `metadata.yaml` from the target directory.
   - Computes analytical exact solution using `src.problems.get_problem`.
   - Launches all 5 plotting tasks in parallel using `concurrent.futures.ProcessPoolExecutor`.
   - Saves:
     - `loss_curve.png`
     - `h1_error_curve.png`
     - `prediction_contour.png`
     - `error_contour.png`
     - `solution_slices.png`
     directly inside `output/<problem_id>/<method_name>/`.

---

## Phase 5: Problem-Level Cross-Method Comparison Suite

### Objective
Aggregate all methods tested on a single problem and produce a comprehensive comparison suite in `output/<problem_id>/comparisons/`.

### Instructions for Agent:
1. **Create `src/visualization/problem_comparison.py`**:
   - `class ProblemComparator`:
     - Scans `output/<problem_id>/` for all method subdirectories (ignoring `comparisons/`).
     - Loads `metadata.yaml` and `outcomes.npz` for each method.
     - Generates in `output/<problem_id>/comparisons/`:
       1. **`convergence_overlay_epochs.png`**: Overlaid $H^1$ error vs. epochs for PINN, R-PINN, NURBS-KAN, R-NURBS-KAN.
       2. **`convergence_overlay_time.png`**: Overlaid $H^1$ error vs. wall-clock time (seconds) across all neural methods + horizontal dashed reference lines for IGA-FEM methods.
       3. **`convergence_overlay_progress.png`**: Overlaid $H^1$ error vs. training progress percentage (0–100%).
       4. **`side_by_side_solution_grid.png`**: Multi-panel figure comparing Exact Solution against each numerical method.
       5. **`side_by_side_error_grid.png`**: Side-by-side $\log_{10}$ pointwise error heatmaps with synchronized color scale.
       6. **`boundary_layer_slice_comparison.png`**: Multi-method 1D cross-sectional cut across the singular boundary layer ($y=0.95, y=1.0$) highlighting spurious oscillations / diffusion smearing vs. exact resolution.
       7. **`summary_table.md` & `summary_table.csv`**: Formatted Markdown and CSV tables comparing $L_2, H^1, L_\infty$ errors, parameter count / DoFs, and runtimes for this problem.
2. **Create `plot_problem_comparisons.py`**:
   - CLI usage: `python plot_problem_comparisons.py --problem-dir output/<problem_id>`.
   - Instantiates `ProblemComparator` and executes the comparison generation.

---

## Phase 6: Global Multi-Problem Benchmark Suite

### Objective
Synthesize benchmark results across all problems into global comparisons in `output/global_comparisons/` to support the paper claims in `abstract.pdf`.

### Instructions for Agent:
1. **Create `src/visualization/global_comparison.py`**:
   - `class GlobalComparator`:
     - Scans `output/` across all problem folders.
     - Generates in `output/global_comparisons/`:
       1. **`global_metrics_table.md` & `.csv`**: Complete summary table of all problems and methods.
       2. **`pareto_efficiency_h1_vs_time.png`**: Pareto efficiency frontiers ($H^1$ error vs. wall-clock time) for all methods across all problems.
       3. **`pareto_efficiency_h1_vs_dofs.png`**: Model compactness analysis ($H^1$ error vs. number of trainable parameters / DoFs).
       4. **`robust_loss_ablation.png`**: Bar/box comparison demonstrating convergence speed and accuracy advantage of robust Gram loss vs standard residual loss.
       5. **`uniform_vs_adaptive_sampling_impact.png`**: Bar comparison showing error reduction from boundary-layer adaptive collocation / meshes on singular problems.
       6. **`abstract_summary_overview.png`**: Publication-ready multi-panel figure synthesizing all experimental findings.
2. **Update `plot_comparison_suite.py` and `generate_summary_tables.py`**:
   - Make them lightweight CLI entry points calling `src/visualization/global_comparison.py`.

---

## Phase 7: Master Orchestrator (`run_all.py`) & Verification

### Objective
Create a unified orchestrator that coordinates training, parallel per-run plotting, problem comparisons, and global comparisons with a fast `--test` mode.

### Instructions for Agent:
1. **Refactor `run_all.py`**:
   - CLI flags:
     - `--test`: Runs exclusively minimal test configs (`training_config/test/`).
     - `--config-dir <dir>`: Runs all configs in specified directory.
     - `--solvers <list>`: Filter solvers (e.g. `pinn`, `kan`, `iga`).
     - `--skip-existing`: Skip already trained methods.
     - `--optimized`: Enable performance optimizations.
     - `--wall-time <minutes>`: Stop run if wall-clock time is exceeded.
   - **Hardware-Aware Scheduling**:
     - GPU Worker Queue: PINN and KAN runs execute sequentially on GPU to avoid CUDA memory thrashing.
     - CPU Thread Pool: IGA runs execute concurrently in parallel across CPU cores.
   - **Automatic Multi-Stage Workflow**:
     - Step 1: Execute single-method `train.py`.
     - Step 2: Trigger `plot_run.py` immediately in the background upon method completion.
     - Step 3: Trigger `plot_problem_comparisons.py` when all methods for a given problem complete.
     - Step 4: Trigger `plot_comparison_suite.py` and `generate_summary_tables.py` when all problems finish.
2. **Execute Full Verification Checklist**:
   ```powershell
   # Step 1: Generate test configs
   python manage_configs.py sweeps/test_matrix.yaml

   # Step 2: Run minimal end-to-end smoke test
   python run_all.py --test

   # Step 3: Verify outputs
   # Check that output/test_poisson_sine/ contains method subfolders, outcomes.npz, metadata.yaml, 5 png plots
   # Check that output/test_poisson_sine/comparisons/ contains convergence overlays and summary_table.md
   # Check that output/global_comparisons/ contains global Pareto plots and global_metrics_table.md
   ```
