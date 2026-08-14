# Comprehensive Benchmark & Framework: PINN vs. KAN vs. IGA

A unified, high-performance research platform for solving singularly perturbed partial differential equations (PDEs) with boundary layers ($0 < \epsilon \ll 1$) using Physics-Informed Neural Networks (PINN), Kolmogorov-Arnold Networks (KAN & NURBS-KAN), and Isogeometric Analysis (IGA-FEM).

---

## 1. Theoretical Foundations

This section outlines the mathematical formulations and numerical principles governing all tested solver families.

### 1.1 Physics-Informed Neural Networks (PINN)
*Placeholder: Comprehensive formulation of standard PINNs.*
- **Scope**: Multi-Layer Perceptron (MLP) architecture parameterized by weights $\theta$.
- **Collocation & Loss**: Automatic differentiation for strong-form differential operators $\mathcal{L}[u_\theta] = f$.
- **Boundary Pinning**: Exact satisfaction of homogeneous Dirichlet boundary conditions via distance pinning:
  $$u_\theta(x, y) = D(x, y) \cdot \mathcal{N}_\theta(x, y) + g(x, y)$$
  where $D(x, y) = x(1-x)y(1-y)$ vanishes on $\partial\Omega$.

### 1.2 Robust Variational PINNs (R-PINN)
*Placeholder: Mathematical formulation of the Petrov-Galerkin variational loss.*
- **Scope**: Overcoming spectral bias and loss landscape ill-conditioning in convection-diffusion singularly perturbed regimes.
- **Dual Norm & Gram Inversion**: Minimizing residuals in the dual Sobolev norm $\|R\|_{V^*}$ via the inverted Gram operator $G^{-1}$:
  $$\mathcal{L}_{\text{robust}}(\theta) = \langle R(u_\theta), G^{-1} R(u_\theta) \rangle$$

### 1.3 Kolmogorov-Arnold Networks (KAN & NURBS-KAN)
*Placeholder: Mathematical formulation of learnable 1D edge activations and Rational Splines.*
- **Scope**: Replacing fixed node activations with learnable 1D univariate spline functions $\phi_{i, j}(x)$ on network edges based on the Kolmogorov-Arnold representation theorem:
  $$u_i = \sum_{j=1}^{N_{\text{in}}} \phi_{i, j}(x_j)$$
- **NURBS Parameterization**: Combining rational B-spline basis functions with learnable projective weights $w_k$ to achieve sharp local boundary-layer gradient resolution.

### 1.4 Robust Kolmogorov-Arnold Networks (R-KAN & R-NURBS-KAN)
*Placeholder: Combining KAN representation capacity with robust Petrov-Galerkin loss functionals.*
- **Scope**: Pairing rational spline edge activations with the inverse Gram matrix $G^{-1}$ operator.

### 1.5 Isogeometric Analysis (IGA-FEM)
*Placeholder: Formulation of Standard Galerkin, SUPG, and iGRM solvers.*
- **Scope**:
  1. **Standard Galerkin IGA**: Tensor-product B-spline basis functions $N_{i, p}(x) N_{j, p}(y)$ with exact geometric representation.
  2. **Streamline-Upwind Petrov-Galerkin (SUPG)**: Residual-based stabilization adding directional diffusion $\tau (\mathbf{b} \cdot \nabla v)$ along flow streamlines.
  3. **Isogeometric Residual Minimization (iGRM)**: Discontinuous Petrov-Galerkin inspired dual-norm residual minimization with enriched test spaces $V_h \subset H^1_0(\Omega)$ ($p_v = p_u + \Delta p$).

> [!NOTE]
> **Why $H^1$ Error is a Single Point / Floor for IGA Solvers**
> - **PINN & KAN** are iterative neural networks optimized via gradient descent across $N_{\text{epochs}}$, recording an evolving convergence history.
> - **IGA** solves a direct sparse linear system ($K\mathbf{u} = F$) in a single computational step ($\sim 0.03\text{s}$), producing its exact discrete projection instantly. In time-convergence charts, it is represented as an immediate accuracy floor.

### 1.6 Singularly Perturbed Benchmark Problems ($0 < \epsilon \ll 1$)
*Placeholder: Mathematical definitions and boundary layer characteristics.*
- **Problem 1: Poisson High-Frequency Sine** ($-\epsilon \Delta u = f$) with oscillatory solution $u(x, y) = \sin(\pi x)\sin(\pi y)$.
- **Problem 2: Poisson Exponential Gradient** ($-\epsilon \Delta u = f$) with steep localized gradients.
- **Problem 3: Eriksson-Johnson Convection-Diffusion** ($-\epsilon \Delta u + u_y = f$) with an exponential boundary layer of width $\mathcal{O}(\epsilon)$ near $y = 1$.

---

## 2. Visualization & Metrics Guide

### 2.1 Standardized Color Palette & Line Styles
To ensure clear visual distinction across method families and sampling distributions:

| Method Family | Formulation | Mesh / Collocation | Color Hex | Line Style | Marker |
|---|---|---|---|---|---|
| **PINN Family** | Standard | Uniform | `#1f77b4` (Classic Blue) | Solid (`-`) | Circle (`o`) |
| **PINN Family** | Standard | Boundary Layer | `#4ba3e3` (Light Blue) | Dashed (`--`) | Square (`s`) |
| **PINN Family** | Robust (R-PINN) | Uniform | `#004c6d` (Navy Blue) | Dash-dot (`-.`) | Diamond (`D`) |
| **PINN Family** | Robust (R-PINN) | Boundary Layer | `#257d9d` (Medium Navy) | Dotted (`:`) | Triangle Down (`v`) |
| **KAN Family** | Standard NURBS-KAN | Uniform | `#2ca02c` (Medium Green) | Solid (`-`) | Triangle Up (`^`) |
| **KAN Family** | Standard NURBS-KAN | Boundary Layer | `#66c2a5` (Mint Green) | Dashed (`--`) | Plus (`P`) |
| **KAN Family** | Robust R-NURBS-KAN | Uniform | `#005a24` (Forest Green) | Dash-dot (`-.`) | Cross (`X`) |
| **KAN Family** | Robust R-NURBS-KAN | Boundary Layer | `#238b45` (Emerald Green) | Dotted (`:`) | Star (`*`) |
| **IGA Family** | Standard Galerkin | Uniform | `#9467bd` (Purple) | Solid (`-`) | Circle (`o`) |
| **IGA Family** | SUPG Stabilized | Uniform | `#8c564b` (Brown) | Dashed (`--`) | Square (`s`) |
| **IGA Family** | SUPG Stabilized | Adaptive Mesh | `#d62728` (Crimson Red) | Dash-dot (`-.`) | Triangle Up (`^`) |
| **IGA Family** | iGRM Minimization | Adaptive Mesh | `#e377c2` (Magenta Pink) | Dotted (`:`) | Diamond (`D`) |

---

### 2.2 Pareto Efficiency Frontiers
**Pareto frontier plots** (`pareto_efficiency_h1_vs_time.png` and `pareto_efficiency_h1_vs_dofs.png`) visualize multi-objective trade-offs:
- **Objective 1 (Vertical Axis)**: Minimize $H^1$ Semi-Norm Error (Log Scale).
- **Objective 2 (Horizontal Axis)**: Minimize Wall-Clock Compute Time (s) or Model Size (DoFs/Parameters).
- **Pareto Optimal Point**: A method is Pareto optimal if no other method achieves higher accuracy with lower compute/memory resources. Points positioned towards the **bottom-left** represent the dominant, optimal approaches.

---

### 2.3 Computational Efficiency Score (CES)

To determine which method yields the highest return on compute and parameter investment, we define the **Computational Efficiency Score (CES)**:

$$
\text{CES} = -\log_{10}\left( H^1_{\text{error}} \cdot \sqrt{\text{Elapsed Time (s)}} \cdot \sqrt[4]{\text{DoFs or Params}} \right)
$$

#### Interpretation & Guidance:
- **Higher is Better**: A higher score indicates superior numerical accuracy achieved per unit of execution runtime and parameter footprint.
- **Accuracy Penalty ($H^1_{\text{error}}$)**: Primary objective; lower error increases the score.
- **Time Penalty ($\sqrt{\text{Time}}$)**: Sublinear penalty for wall-clock training/solving duration.
- **Complexity Penalty ($\sqrt[4]{\text{DoFs}}$)**: Fourth-root penalty for memory footprint and parameter count.
- **Use in Benchmarking**: Reported in all summary tables and ranked in `computational_efficiency_ranking.png` to guide solver selection.

---

### 2.4 Diagnostic Suite (6 Plots per Method Run)
Every single method run in `output/<problem>/<method>/` produces 6 diagnostic artifacts:
1. `loss_curve.png`: Semilogarithmic convergence of interior PDE residual loss $\log_{10}(\mathcal{L})$.
2. `h1_error_curve.png`: Dual-panel $H^1$ semi-norm error history vs. Epochs and vs. Wall-Clock Time (s).
3. `prediction_contour.png`: High-resolution 2D filled contour field of numerical solution $u_h(x, y)$.
4. `error_contour.png`: 2D pointwise absolute error heatmap $\log_{10}|u(x, y) - u_h(x, y)|$.
5. `solution_slices.png`: 1D cross-sectional profiles along $x = 0.5$ and singular boundary $y \approx 0.95$.
6. `surface_3d.png`: 3D perspective surface elevation comparing $u_h(x, y)$ vs. exact $u(x, y)$ and 3D error elevation.

---

## 3. Directory & Framework Architecture

```
.
├── sweeps/
│   ├── benchmark_matrix.yaml   # Full 72-run benchmark definition (6 problems x 12 methods)
│   └── test_matrix.yaml        # Fast smoke-test matrix (3 problems x 7 methods)
├── training_config/            # Declaratively compiled YAML configs (generated)
├── src/
│   ├── problems/               # Base problem definitions & exact solutions
│   ├── samplers/               # Uniform, boundary-layer, and adaptive coordinate samplers
│   ├── solvers/                # PINN, KAN (NURBS), and IGA (Galerkin, SUPG, iGRM) implementations
│   └── visualization/          # Standalone, decoupled visualizer modules
├── manage_configs.py           # Declarative configuration generator
├── train.py                    # Single-method compute-only training runner
├── plot_run.py                 # Parallel per-run diagnostic visualizer
├── plot_problem_comparisons.py # Problem-level cross-method comparison aggregator
├── plot_comparison_suite.py    # Global multi-problem Pareto & synthesis suite generator
├── run_all.py                  # Master benchmark orchestrator
└── output/                     # Benchmark artifacts directory
    ├── <problem_id>/
    │   ├── <method_name>/      # Raw outcomes.npz, metadata.yaml, model checkpoint, 6 diagnostic plots
    │   └── comparisons/        # Problem-level summary tables, 1D slices, separated uniform/boundary grids
    └── global_comparisons/     # Pareto curves, CES ranking, ablation studies, global synthesis table
```

---

## 4. How to Use the Benchmark Suite

### 4.1 Generate Configurations
```bash
# Compile clean configuration files from declarative matrix sweeps
python manage_configs.py sweeps/benchmark_matrix.yaml
python manage_configs.py sweeps/test_matrix.yaml
```

### 4.2 Execute Training & Plotting
```bash
# 1. Train a single configuration (compute-only)
python train.py --config training_config/poisson_sine/pinn_uniform.yaml

# 2. Generate 6 parallel diagnostic plots for a single run
python plot_run.py --output-dir output/poisson_sine/pinn_uniform

# 3. Generate problem-level cross-method comparison suite
python plot_problem_comparisons.py --problem-dir output/poisson_sine

# 4. Generate global multi-problem Pareto & synthesis suite
python plot_comparison_suite.py --output-dir output
```

### 4.3 Master Orchestrator (`run_all.py`)
```bash
# Run isolated smoke test (outputs strictly to output/test_runs/ without touching production data)
python run_all.py --test

# Run full benchmark suite across all problems and methods
python run_all.py

# Run full benchmark with high-performance PyTorch 2.0 compile & fast linear solvers
python run_all.py --optimized
```
