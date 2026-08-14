import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication style defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def plot_loss_curve(loss_history: np.ndarray, save_path: str, title: str = "Training Loss Convergence") -> str:
    """
    Plots training loss history on semilogy scale vs epochs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    loss = np.asarray(loss_history).flatten()
    epochs = np.arange(1, len(loss) + 1)

    # Filter invalid/zero/negative values for log plot
    valid_mask = np.isfinite(loss) & (loss > 0)
    if np.any(valid_mask):
        ax.semilogy(epochs[valid_mask], loss[valid_mask], color="#1f77b4", label="Total Loss", lw=2.0)
    else:
        ax.plot(epochs, loss, color="#1f77b4", label="Total Loss", lw=2.0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Log Scale)")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_h1_convergence(
    h1_history: np.ndarray,
    epoch_history: np.ndarray | None = None,
    time_history: np.ndarray | None = None,
    save_path: str = "h1_error_curve.png",
    title: str = "$H^1$ Semi-Norm Error Convergence"
) -> str:
    """
    Plots H^1 semi-norm error history vs epochs and vs wall-clock time in dual subplots.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    h1 = np.asarray(h1_history).flatten()
    valid_mask = np.isfinite(h1) & (h1 > 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: vs Epochs
    ax1 = axes[0]
    if epoch_history is not None and len(epoch_history) == len(h1):
        epochs = np.asarray(epoch_history).flatten()
    else:
        epochs = np.arange(1, len(h1) + 1)

    if np.any(valid_mask):
        ax1.semilogy(epochs[valid_mask], h1[valid_mask], color="#d62728", marker="o", markersize=3, lw=1.8, label="$H^1$ Error")
    else:
        ax1.plot(epochs, h1, color="#d62728", lw=1.8, label="$H^1$ Error")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("$H^1$ Semi-Norm Error")
    ax1.set_title("$H^1$ Error vs Epochs", fontweight="bold")
    ax1.grid(True)
    ax1.legend(frameon=True)

    # Subplot 2: vs Wall-clock Time
    ax2 = axes[1]
    if time_history is not None and len(time_history) == len(h1):
        time_sec = np.asarray(time_history).flatten()
        if np.any(valid_mask):
            ax2.semilogy(time_sec[valid_mask], h1[valid_mask], color="#2ca02c", marker="s", markersize=3, lw=1.8, label="$H^1$ Error")
        else:
            ax2.plot(time_sec, h1, color="#2ca02c", lw=1.8, label="$H^1$ Error")
        ax2.set_xlabel("Wall-Clock Time (seconds)")
    else:
        if np.any(valid_mask):
            ax2.semilogy(epochs[valid_mask], h1[valid_mask], color="#2ca02c", lw=1.8, label="$H^1$ Error")
        ax2.set_xlabel("Evaluation Step")

    ax2.set_ylabel("$H^1$ Semi-Norm Error")
    ax2.set_title("$H^1$ Error vs Elapsed Time", fontweight="bold")
    ax2.grid(True)
    ax2.legend(frameon=True)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _format_grid(x: np.ndarray, t: np.ndarray, z: np.ndarray):
    """Reshape flat 1D coordinates and values into 2D structured mesh for contour plotting."""
    x = np.asarray(x).flatten()
    t = np.asarray(t).flatten()
    z = np.asarray(z).flatten()

    unique_x = np.unique(np.round(x, decimals=7))
    unique_t = np.unique(np.round(t, decimals=7))

    nx, nt = len(unique_x), len(unique_t)
    if nx * nt == len(z):
        # Structured grid
        X = x.reshape(nx, nt)
        T = t.reshape(nx, nt)
        Z = z.reshape(nx, nt)
        return X, T, Z
    else:
        # Fallback to sqrt square or interpolation
        dim = int(np.sqrt(len(z)))
        if dim * dim == len(z):
            return x.reshape(dim, dim), t.reshape(dim, dim), z.reshape(dim, dim)
        return None, None, None


def plot_2d_solution(
    x: np.ndarray,
    t: np.ndarray,
    z_pred: np.ndarray,
    title: str = "Numerical Solution $u_h(x, y)$",
    save_path: str = "prediction_contour.png"
) -> str:
    """
    Plots 2D contour/heatmap of predicted solution.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    X, T, Z = _format_grid(x, t, z_pred)
    if X is not None:
        cntr = ax.contourf(X, T, Z, levels=60, cmap="viridis")
        fig.colorbar(cntr, ax=ax, label="Solution $u_h$")
        ax.contour(X, T, Z, levels=12, colors="k", alpha=0.25, linewidths=0.5)
    else:
        scatter = ax.scatter(x, t, c=z_pred, cmap="viridis", s=25, edgecolors="none")
        fig.colorbar(scatter, ax=ax, label="Solution $u_h$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$ (or $t$)")
    ax.set_title(title, pad=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_2d_error(
    x: np.ndarray,
    t: np.ndarray,
    z_pred: np.ndarray,
    z_exact: np.ndarray,
    title: str = "Pointwise Absolute Error $\\log_{10}|u - u_h|$",
    save_path: str = "error_contour.png"
) -> str:
    """
    Plots 2D contour of pointwise absolute error |u - u_h| in log10 scale.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    err = np.abs(np.asarray(z_exact).flatten() - np.asarray(z_pred).flatten())
    # Floor small values for log stability
    err_log10 = np.log10(np.clip(err, 1e-16, None))

    X, T, Z_err = _format_grid(x, t, err_log10)
    if X is not None:
        cntr = ax.contourf(X, T, Z_err, levels=60, cmap="inferno")
        cbar = fig.colorbar(cntr, ax=ax)
        cbar.set_label("$\\log_{10}|u - u_h|$")
        ax.contour(X, T, Z_err, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    else:
        scatter = ax.scatter(x, t, c=err_log10, cmap="inferno", s=25, edgecolors="none")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("$\\log_{10}|u - u_h|$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$ (or $t$)")
    ax.set_title(title, pad=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_1d_slices(
    x: np.ndarray,
    t: np.ndarray,
    z_pred: np.ndarray,
    z_exact: np.ndarray,
    problem_name: str = "Problem",
    save_path: str = "solution_slices.png"
) -> str:
    """
    1D cross-section curves comparing u_h vs u along x=0.5 and near boundary (y=0.95, y=1.0 or t slices).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    x_arr = np.asarray(x).flatten()
    t_arr = np.asarray(t).flatten()
    z_p = np.asarray(z_pred).flatten()
    z_e = np.asarray(z_exact).flatten()

    unique_x = np.unique(np.round(x_arr, decimals=5))
    unique_t = np.unique(np.round(t_arr, decimals=5))

    # Slice 1: Cut along x approx 0.5 across all y/t
    ax1 = axes[0]
    mid_x = unique_x[np.argmin(np.abs(unique_x - 0.5))]
    mask_x = np.isclose(x_arr, mid_x, atol=1e-3)
    
    if np.any(mask_x):
        t_sub = t_arr[mask_x]
        sort_idx = np.argsort(t_sub)
        ax1.plot(t_sub[sort_idx], z_e[mask_x][sort_idx], "k-", lw=2.0, label="Exact $u(x=0.5, y)$")
        ax1.plot(t_sub[sort_idx], z_p[mask_x][sort_idx], "r--", lw=2.0, label="Pred $u_h(x=0.5, y)$")
    ax1.set_xlabel("$y$ (or $t$)")
    ax1.set_ylabel("Solution $u$")
    ax1.set_title(f"1D Slice along $x \\approx {mid_x:.2f}$", fontweight="bold")
    ax1.grid(True)
    ax1.legend(frameon=True)

    # Slice 2: Cut near boundary (top 95% or max y/t)
    ax2 = axes[1]
    target_t = unique_t[-1] if len(unique_t) > 0 else 1.0
    if len(unique_t) > 2:
        # Check near boundary layer y=0.95 or y=1.0
        target_t = unique_t[np.argmin(np.abs(unique_t - 0.95))]
    
    mask_t = np.isclose(t_arr, target_t, atol=1e-3)
    if np.any(mask_t):
        x_sub = x_arr[mask_t]
        sort_idx = np.argsort(x_sub)
        ax2.plot(x_sub[sort_idx], z_e[mask_t][sort_idx], "k-", lw=2.0, label=f"Exact $u(x, y={target_t:.2f})$")
        ax2.plot(x_sub[sort_idx], z_p[mask_t][sort_idx], "b--", lw=2.0, label=f"Pred $u_h(x, y={target_t:.2f})$")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("Solution $u$")
    ax2.set_title(f"1D Slice near Boundary ($y \\approx {target_t:.2f}$)", fontweight="bold")
    ax2.grid(True)
    ax2.legend(frameon=True)

    fig.suptitle(f"1D Cross-Section Profiles - {problem_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_3d_surface(
    x: np.ndarray,
    t: np.ndarray,
    z_pred: np.ndarray,
    z_exact: np.ndarray,
    title: str = "3D Solution & Pointwise Error Elevation",
    save_path: str = "surface_3d.png"
) -> str:
    """
    Renders publication-quality 3D elevation surface plots comparing Predicted vs. Exact and Error.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.interpolate import griddata

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig = plt.figure(figsize=(15, 6), dpi=300)

    x_f = np.asarray(x).flatten()
    t_f = np.asarray(t).flatten()
    zp_f = np.asarray(z_pred).flatten()
    ze_f = np.asarray(z_exact).flatten()
    err_f = np.abs(ze_f - zp_f)

    # Standardize onto regular 100x100 grid for surface plotting
    xi = np.linspace(0.0, 1.0, 100)
    ti = np.linspace(0.0, 1.0, 100)
    XI, TI = np.meshgrid(xi, ti)

    points = np.column_stack((x_f, t_f))
    ZI_pred = griddata(points, zp_f, (XI, TI), method="cubic")
    ZI_exact = griddata(points, ze_f, (XI, TI), method="cubic")
    ZI_err = griddata(points, err_f, (XI, TI), method="cubic")

    # Fallback to nearest if NaN at edges
    if np.any(np.isnan(ZI_pred)):
        ZI_pred = griddata(points, zp_f, (XI, TI), method="nearest")
        ZI_exact = griddata(points, ze_f, (XI, TI), method="nearest")
        ZI_err = griddata(points, err_f, (XI, TI), method="nearest")

    # Subplot 1: Exact and Predicted Surfaces
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf_pred = ax1.plot_surface(
        XI, TI, ZI_pred,
        cmap="viridis",
        alpha=0.85,
        edgecolor="none",
        antialiased=True,
        label="Predicted $u_h$"
    )
    # Overlay exact solution wireframe
    ax1.plot_wireframe(XI, TI, ZI_exact, color="black", rstride=8, cstride=8, alpha=0.4, linewidth=0.8)

    ax1.set_xlabel("$x$", labelpad=8)
    ax1.set_ylabel("$y$ (or $t$)", labelpad=8)
    ax1.set_zlabel("$u(x, y)$", labelpad=8)
    ax1.set_title("Numerical Solution $u_h$ (Surface) vs. Exact $u$ (Wireframe)", pad=14, fontweight="bold")
    ax1.view_init(elev=28, azim=-125)
    cbar1 = fig.colorbar(surf_pred, ax=ax1, shrink=0.55, aspect=12, pad=0.08)
    cbar1.set_label("$u_h(x, y)$")

    # Subplot 2: Pointwise Error Elevation
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf_err = ax2.plot_surface(
        XI, TI, ZI_err,
        cmap="inferno",
        alpha=0.9,
        edgecolor="none",
        antialiased=True
    )
    ax2.set_xlabel("$x$", labelpad=8)
    ax2.set_ylabel("$y$ (or $t$)", labelpad=8)
    ax2.set_zlabel("Error $|u - u_h|$", labelpad=8)
    ax2.set_title("Pointwise Absolute Error $|u(x, y) - u_h(x, y)|$", pad=14, fontweight="bold")
    ax2.view_init(elev=28, azim=-125)
    cbar2 = fig.colorbar(surf_err, ax=ax2, shrink=0.55, aspect=12, pad=0.08)
    cbar2.set_label("Absolute Error")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

