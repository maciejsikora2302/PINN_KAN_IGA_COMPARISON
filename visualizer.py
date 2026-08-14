import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Any, Optional

class Visualizer:
    """
    A separate class for visualizing simulation and experiment results.
    """
    def __init__(self):
        pass

    def plot(self, 
             x: Union[List[float], Any], 
             y: Union[List[float], Any], 
             title: str = "Simulation Plot", 
             xlabel: str = "X", 
             ylabel: str = "Y", 
             save_path: Optional[str] = None) -> None:
        """
        What:
            Plots a simple 1D line graph given X and Y coordinates.
        Why:
            Provides a basic, generic visualization tool for plotting one-dimensional data series 
            such as individual metrics, profiles, or custom signals over time or space.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label="Simulation Data", linewidth=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_comparison(self,
                        curves: dict[str, dict[str, Any]],
                        title: str = "Comparison Plot",
                        xlabel: str = "X",
                        ylabel: str = "Y",
                        log_scale: bool = False,
                        save_path: Optional[str] = None) -> None:
        """
        What:
            Plots multiple curves overlaid on the same 1D graph.
        Why:
            Enables direct performance and trajectory comparisons across different solvers 
            (e.g., PINN vs KAN vs IGA) under identical simulation parameters. It is useful 
            for evaluating speed of convergence or final asymptotic values side-by-side.
        """
        plt.figure(figsize=(9, 6))
        for label, data in curves.items():
            y = data.get('y')
            if y is not None and len(y) > 0:
                x = data.get('x')
                if x is None or len(x) == 0:
                    x = range(len(y))
                plt.plot(x, y, label=label, linewidth=2)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_prediction_contour(self,
                                x: Any,
                                t: Any,
                                z: Any,
                                title: str = "Prediction Contour",
                                xlabel: str = "X",
                                ylabel: str = "T",
                                save_path: Optional[str] = None) -> None:
        """
        What:
            Generates a 2D color contour plot from scattered data using spatial triangulation.
        Why:
            Allows visualizing a 2D surface projection (e.g., predicting the state variable 
            over space X and time T) directly from unstructured coordinate lists without 
            requiring rigid grid reshaping.
        """
        if len(x) == 0 or len(t) == 0 or len(z) == 0:
            print(f"Skipping contour plot '{title}': Data is empty.")
            return

        plt.figure(figsize=(8, 6))
        # Since x, t might be flattened grids, use tripcolor for triangulation
        sc = plt.tripcolor(x, t, z, cmap="viridis")
        plt.colorbar(sc, label="Prediction Value")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Contour plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_running_average(self,
                             y: Any,
                             window: int = 100,
                             title: str = "Loss function (running average)",
                             xlabel: str = "Epoch",
                             ylabel: str = "Loss",
                             save_path: Optional[str] = None) -> None:
        """
        What:
            Plots the raw epoch loss along with its moving average (running average) 
            computed over a sliding window.
        Why:
            High-frequency oscillations in loss histories can obscure overall training trends. 
            Overlaying a running average smooths out temporary noise and gradient fluctuations, 
            making it easier to verify if the optimization process is stably converging or plateauing.
        """
        if len(y) == 0:
            return
        
        cumsum = np.cumsum(np.insert(y, 0, 0))
        avg_y = (cumsum[window:] - cumsum[:-window]) / float(window)
        
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(y, alpha=0.3, label="Raw Loss")
        plt.plot(range(window - 1, len(y)), avg_y, label=f"Running Average (window={window})", linewidth=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Running average plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_loss_vs_error(self,
                           epochs_loss: Any,
                           loss_vals: Any,
                           epochs_h1: Any,
                           h1_vals: Any,
                           example: int,
                           epsilon: float,
                           rpinn: int,
                           title: str = "Loss vs error",
                           save_path: Optional[str] = None) -> None:
        """
        What:
            Plots the PDE loss (represented as `sqrt(loss)`) alongside the H1 validation error 
            norm across training epochs on a logarithmic or semi-logarithmic scale.
        Why:
            This visualization directly compares the solver's internal training objective (the PDE residual loss) 
            with the true external accuracy metric (the H1 error compared against the analytical solution). 
            It is critical to verify that minimizing the PDE loss actually reduces the true physical error 
            rather than leading to overfitting or optimization decoupling.
        """
        if len(loss_vals) == 0:
            return

        plt.figure(figsize=(8, 6), dpi=100)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("value")

        # sqrt(loss) is plotted as loglog
        plt.loglog(epochs_loss, np.sqrt(loss_vals), label="sqrt(loss)")

        if rpinn == 0:
            plt.semilogy(epochs_loss, np.square(loss_vals), label="loss")

        if len(h1_vals) > 0:
            if example == 1:
                plt.loglog(epochs_h1, h1_vals, label="H1 error")
            elif example == 2:
                plt.semilogy(epochs_h1, h1_vals, label="H1 error")
            elif example == 3:
                plt.semilogy(epochs_h1, epsilon * np.array(h1_vals), label=f"H1 error * {epsilon}")

        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss vs error plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_pcolor(self,
                    z: Any,
                    x: Any,
                    t: Any,
                    n_points_x: int,
                    n_points_t: int,
                    title: str = "PINN solution",
                    xlabel: str = "x",
                    ylabel: str = "t",
                    save_path: Optional[str] = None) -> None:
        """
        What:
            Generates a structured 2D pseudocolor plot (pcolor) from reshaped spatial-temporal matrices.
        Why:
            Visualizes continuous 2D fields (such as the PDE approximate solution, exact analytical solution, 
            or spatial error maps) over the entire domain. This shows the spatial and temporal distribution 
            of predictions and errors, highlighting where the neural network models perform well and where 
            boundary or internal region errors accumulate.
        """
        if len(x) == 0 or len(t) == 0 or len(z) == 0:
            return

        plt.figure(figsize=(8, 6), dpi=100)
        X = x.reshape(n_points_t, n_points_x)
        T = t.reshape(n_points_t, n_points_x)
        Z = z.reshape(n_points_t, n_points_x)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        c = plt.pcolor(T, X, Z, shading='auto')
        plt.colorbar(c)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pcolor plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_3d_surface(self,
                        z: Any,
                        x: Any,
                        t: Any,
                        n_points_x: int,
                        n_points_t: int,
                        title: str = "3D Surface",
                        xlabel: str = "t",
                        ylabel: str = "x",
                        zlabel: str = "value",
                        save_path: Optional[str] = None) -> None:
        """
        What:
            Generates a structured 3D surface plot from reshaped spatial-temporal matrices.
        Why:
            Enables visual inspection of the solution elevation surface, helping to identify
            sharp gradients, oscillations, or peak errors in 3D perspective.
        """
        if len(x) == 0 or len(t) == 0 or len(z) == 0:
            return

        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        X = x.reshape(n_points_t, n_points_x)
        T = t.reshape(n_points_t, n_points_x)
        Z = z.reshape(n_points_t, n_points_x)

        surf = ax.plot_surface(T, X, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D surface plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()


    def plot_rational_function(self,
                               num_coeffs: np.ndarray,
                               den_coeffs: np.ndarray,
                               title: str = "Learnable Rational Activation",
                               save_path: Optional[str] = None) -> None:
        """
        What:
            Plots the learned rational function R(x) = P(x) / (1 + |Q(x)|) over a range of inputs.
        Why:
            Allows inspecting the shape of the dynamically learned activation functions,
            helping to understand how the network adapts its non-linearities for the PDE problem.
        """
        x = np.linspace(-3.0, 3.0, 400)
        
        # Compute numerator
        num = 0.0
        for i, coeff in enumerate(num_coeffs):
            num = num + coeff * (x ** i)
            
        # Compute denominator
        den_poly = np.zeros_like(x)
        for i, coeff in enumerate(den_coeffs, start=1):
            den_poly = den_poly + coeff * (x ** i)
        den = 1.0 + np.abs(den_poly)
        
        y = num / den
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label="Learned Rational", linewidth=2.5, color='royalblue')
        
        # Also plot a comparison activation function like Tanh for reference
        plt.plot(x, np.tanh(x), '--', label="Tanh (Reference)", color='gray', alpha=0.7)
        
        plt.title(title)
        plt.xlabel("Input x")
        plt.ylabel("Activation f(x)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rational function plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_kan_edge(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      title: str = "KAN Edge Activation",
                      save_path: Optional[str] = None) -> None:
        """
        Plots a single 1D KAN edge activation function.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label="Learned Activation", linewidth=2.5, color='forestgreen')
        plt.plot(x, np.zeros_like(x), '--', color='gray', alpha=0.5)
        plt.title(title)
        plt.xlabel("Input x")
        plt.ylabel("Activation phi(x)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"KAN edge plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_kan_layer_grid(self,
                            x: np.ndarray,
                            phi: np.ndarray,  # shape (out_features, in_features, N_eval)
                            title: str = "KAN Layer Grid Activations",
                            save_path: Optional[str] = None) -> None:
        """
        Plots all KAN edge activations for a layer in a grid layout (out_features x in_features).
        """
        out_features, in_features, _ = phi.shape
        
        # Limit grid size to avoid generating unreadable giant grids
        max_out = min(16, out_features)
        max_in = min(8, in_features)
        
        fig, axes = plt.subplots(max_out, max_in, figsize=(2.5 * max_in, 2.0 * max_out), sharex=True, sharey=False)
        fig.suptitle(title, fontsize=14, y=1.02)
        
        # Ensure axes is 2D even if 1x1
        if max_out == 1 and max_in == 1:
            axes = np.array([[axes]])
        elif max_out == 1:
            axes = axes[np.newaxis, :]
        elif max_in == 1:
            axes = axes[:, np.newaxis]
            
        for i in range(max_out):
            for j in range(max_in):
                ax = axes[i, j]
                ax.plot(x, phi[i, j], color='forestgreen', linewidth=1.5)
                ax.plot(x, np.zeros_like(x), '--', color='gray', alpha=0.3, linewidth=0.8)
                ax.grid(True, which="both", ls="--", alpha=0.3)
                if i == 0:
                    ax.set_title(f"In {j}")
                if j == 0:
                    ax.set_ylabel(f"Out {i}")
                    
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"KAN layer grid plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def get_exact_solution(self, x: np.ndarray, t: np.ndarray, example: int, epsilon: float) -> np.ndarray:
        """Computes exact analytical solution for 1D slices comparison."""
        if example == 1:
            return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * t)
        elif example == 2:
            return -np.exp(np.pi * (x - 2.0 * t)) * np.sin(2.0 * np.pi * x) * np.sin(np.pi * t)
        elif example == 3:
            r1 = (1.0 + np.sqrt(1.0 + 4.0 * epsilon * epsilon * np.pi * np.pi)) / (2.0 * epsilon)
            r2 = (1.0 - np.sqrt(1.0 + 4.0 * epsilon * epsilon * np.pi * np.pi)) / (2.0 * epsilon)
            res_t = (np.exp(r1 * (t - 1.0)) - np.exp(r2 * (t - 1.0))) / (np.exp(-r1) - np.exp(-r2))
            res_x_dx = np.sin(np.pi * x)
            return res_t * res_x_dx
        else:
            raise ValueError(f"Unknown Example index: {example}")

    def plot_slices_comparison(self,
                               predictions: dict[str, dict[str, Any]],
                               example: int,
                               epsilon: float,
                               t_slices: List[float] = [0.25, 0.5, 0.75],
                               save_path: Optional[str] = None) -> None:
        """
        What:
            Plots 1D slices of predictions from multiple solvers (PINN, KAN, IGA)
            against the exact analytical solution at specific time slices.
        Why:
            Allows direct comparison of how the solvers match the physical boundary
            conditions and internal behaviors across the spatial domain at snapshots in time.
        """
        num_slices = len(t_slices)
        if num_slices == 0 or len(predictions) == 0:
            return

        fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 4.5), sharey=True)
        if num_slices == 1:
            axes = [axes]

        for idx, t_val in enumerate(t_slices):
            ax = axes[idx]
            
            # Find the actual closest t_val from any of the predictions to get the grid's t coordinates
            first_algo = list(predictions.keys())[0]
            t_coords = predictions[first_algo]["t"]
            unique_t = np.unique(t_coords)
            t_val_actual = unique_t[np.argmin(np.abs(unique_t - t_val))]
            
            # Plot exact solution first for reference
            first_x = predictions[first_algo]["x"]
            indices = np.where(np.isclose(t_coords, t_val_actual))[0]
            x_slice = first_x[indices]
            sort_idx = np.argsort(x_slice)
            x_sorted = x_slice[sort_idx]
            
            t_sorted = np.full_like(x_sorted, t_val_actual)
            z_exact = self.get_exact_solution(x_sorted, t_sorted, example, epsilon)
            
            ax.plot(x_sorted, z_exact, 'k--', label="Exact", linewidth=2.5, zorder=2)
            
            # Plot each algorithm's prediction
            for algo, pred_data in predictions.items():
                algo_x = pred_data["x"]
                algo_t = pred_data["t"]
                algo_z = pred_data["z"]
                
                algo_indices = np.where(np.isclose(algo_t, t_val_actual))[0]
                algo_x_slice = algo_x[algo_indices]
                algo_z_slice = algo_z[algo_indices]
                
                algo_sort_idx = np.argsort(algo_x_slice)
                algo_x_sorted = algo_x_slice[algo_sort_idx]
                algo_z_sorted = algo_z_slice[algo_sort_idx]
                
                ax.plot(algo_x_sorted, algo_z_sorted, label=algo, linewidth=2, zorder=3)
                
            ax.set_title(f"Slice at t = {t_val_actual:.2f}")
            ax.set_xlabel("x")
            if idx == 0:
                ax.set_ylabel("u(x, t)")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            if idx == num_slices - 1:
                ax.legend(loc="best")
                
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"1D slices comparison plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_3d_comparison(self,
                           predictions: dict[str, dict[str, Any]],
                           example: int,
                           epsilon: float,
                           n_points_x: int,
                           n_points_t: int,
                           title: str = "3D Solution Comparison",
                           save_path: Optional[str] = None) -> None:
        """
        What:
            Generates a multi-panel 3D surface plot comparing the predictions of 
            different solvers in a 2x2 grid alongside the Exact analytical solution.
        Why:
            Enables a direct visual comparison of the entire 3D surface shape 
            and qualitative features across different methods in a single figure.
        """
        algos = list(predictions.keys())
        
        # Configure a 2x2 grid (Exact, PINN, KAN, IGA)
        fig = plt.figure(figsize=(10, 9), dpi=100)
        fig.suptitle(title, fontsize=14, y=0.98)
        
        # 1. Plot Exact reference first at top-left (2, 2, 1)
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        first_algo = algos[0]
        x = predictions[first_algo]["x"]
        t = predictions[first_algo]["t"]
        
        X = x.reshape(n_points_t, n_points_x)
        T = t.reshape(n_points_t, n_points_x)
        Z_exact = self.get_exact_solution(x, t, example, epsilon).reshape(n_points_t, n_points_x)
        
        surf = ax.plot_surface(T, X, Z_exact, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_title("Exact Solution")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_zlabel("u")
        
        # 2. Plot each algorithm's prediction
        for idx, algo in enumerate(algos, start=2):
            ax = fig.add_subplot(2, 2, idx, projection='3d')
            algo_x = predictions[algo]["x"]
            algo_t = predictions[algo]["t"]
            algo_z = predictions[algo]["z"]
            
            Algo_X = algo_x.reshape(n_points_t, n_points_x)
            Algo_T = algo_t.reshape(n_points_t, n_points_x)
            Algo_Z = algo_z.reshape(n_points_t, n_points_x)
            
            surf = ax.plot_surface(Algo_T, Algo_X, Algo_Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_title(f"{algo} Prediction")
            ax.set_xlabel("t")
            ax.set_ylabel("x")
            ax.set_zlabel("u")
            
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D comparison plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_boundary_layer_slice(
        self,
        predictions: dict[str, dict[str, Any]],
        example: int = 3,
        epsilon: float = 0.01,
        x_cut: float = 0.5,
        title: str = "Boundary Layer Profile (Cut at x=0.5)",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plots 1D cut-slice profiles along y (t) at x = x_cut.
        Subplot 1: Full domain y in [0, 1].
        Subplot 2: Zoom into boundary layer y in [max(0.0, 1.0 - 5*epsilon), 1.0].
        Overlays curves for Exact solution and active solvers (PINN, KAN, IGA).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        t_fine = np.linspace(0.0, 1.0, 500)
        x_fine = np.full_like(t_fine, x_cut)

        from src.problems import get_problem
        problem = get_problem(example, epsilon)
        z_exact = problem.exact_solution(x_fine, t_fine)

        ax1.plot(t_fine, z_exact, 'k--', label="Exact", linewidth=2.5)
        ax2.plot(t_fine, z_exact, 'k--', label="Exact", linewidth=2.5)

        colors = {"PINN": "tab:blue", "KAN": "tab:orange", "IGA": "tab:green"}

        for name, data in predictions.items():
            if not data or "x" not in data or "t" not in data or "z" not in data:
                continue
            x_arr = np.array(data["x"]).flatten()
            t_arr = np.array(data["t"]).flatten()
            z_arr = np.array(data["z"]).flatten()

            unique_x = np.unique(x_arr)
            closest_x = unique_x[np.argmin(np.abs(unique_x - x_cut))]
            mask = np.isclose(x_arr, closest_x, atol=1e-2)

            if not np.any(mask):
                continue

            t_slice = t_arr[mask]
            z_slice = z_arr[mask]
            sort_idx = np.argsort(t_slice)
            t_slice = t_slice[sort_idx]
            z_slice = z_slice[sort_idx]

            color = colors.get(name, None)
            ax1.plot(t_slice, z_slice, label=name, color=color, linewidth=2, alpha=0.85)
            ax2.plot(t_slice, z_slice, label=name, color=color, linewidth=2, alpha=0.85)

        ax1.set_title(f"{title} - Full Domain y in [0, 1]")
        ax1.set_xlabel("y (advection coordinate)")
        ax1.set_ylabel("u(x=0.5, y)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        y_min_zoom = max(0.0, 1.0 - 5.0 * epsilon)
        ax2.set_xlim(y_min_zoom, 1.0)
        ax2.set_title(f"Boundary Layer Zoom y in [{y_min_zoom:.3f}, 1.0]")
        ax2.set_xlabel("y (advection coordinate)")
        ax2.set_ylabel("u(x=0.5, y)")
        ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
        ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boundary layer slice plot saved successfully to {save_path}")
        else:
            plt.show()
        plt.close()






