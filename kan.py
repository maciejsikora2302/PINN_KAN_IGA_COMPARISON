import os
import math
import time
import yaml
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ExperimentInterface
from config import KANConfig
from pinn import (
    exact_solution,
    exact_solution_dx,
    exact_solution_dt,
    exact_solution_dx2,
    exact_solution_dt2,
    shift_EJ,
    shift_EJ_dx,
    shift_EJ_dt,
    shift_EJ_dx2,
    shift_EJ_dt2,
    device,
)

# Differentiation Utilities specifically for KAN
def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value


def dfdt(model: nn.Module, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    return df(model(x, t), t, order=order)


def dfdx(model: nn.Module, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    return df(model(x, t), x, order=order)


# Efficient KAN Implementation from Blealtan/efficient-kan
class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: Any = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1.0, 1.0],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def evaluate_edges(self, x_range: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the 1D activation function on each edge (input j -> output i) for a range of input values.
        Returns a tensor of shape (out_features, in_features, N_eval).
        """
        # x_range: (N_eval,)
        x_in = x_range.unsqueeze(1).repeat(1, self.in_features) # (N_eval, in_features)
        bases = self.b_splines(x_in) # (N_eval, in_features, grid_size + spline_order)
        
        # base_activation part: (out_features, in_features, N_eval)
        base_act = self.base_activation(x_range) # (N_eval,)
        base_val = self.base_weight.unsqueeze(-1) * base_act.unsqueeze(0).unsqueeze(0)
        
        # B-splines part: (out_features, in_features, N_eval)
        spline_val = torch.einsum('ijm,kjm->ijk', self.scaled_spline_weight, bases)
        
        return base_val + spline_val


    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: Any = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1.0, 1.0],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x


# Physics-Informed Boundary Condition Enforced (Pinned) KAN model
class KANModel(nn.Module):
    def __init__(self, layers_hidden: List[int], grid_size: int = 5, spline_order: int = 3, pinning: bool = True):
        super().__init__()
        self.pinning = pinning
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_stack = torch.cat([x, t], dim=1)
        logits = self.kan(x_stack)
        if self.pinning:
            logits = logits * (x - 0.0) * (x - 1.0) * (t - 0.0) * (t - 1.0)
        return logits


class KANExperiment(ExperimentInterface):
    """
    KAN Experiment implementation for a 2D Poisson solver.
    Uses Kolmogorov-Arnold Networks (KAN) as the neural PDE representation.
    """
    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)
        self.model = None
        self.loss_history = []
        self.h1_error_history = []
        self.final_loss = None
        self.final_interior_loss = None
        
        self.x_grid: Any = None
        self.t_grid: Any = None
        self.final_h1_error = None

    def load_config(self, config_path: str) -> None:
        """Loads configuration from YAML file into a KANConfig instance."""
        self.config = KANConfig()
        self.config.load_config(config_path)
        self.config.validate_config()

    def train(self) -> None:
        """Runs the 2D Poisson solver training with a KAN."""
        if not self.config:
            raise ValueError("Configuration has not been loaded.")

        # 1. Setup Train Collocation Points Grid
        x_domain = [0.0, self.config.LENGTH]
        t_domain = [0.0, self.config.TOTAL_TIME]

        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=self.config.N_POINTS_X, requires_grad=True)
        t_raw = torch.linspace(t_domain[0], t_domain[1], steps=self.config.N_POINTS_T, requires_grad=True)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        self.x_grid = grids[0].flatten().reshape(-1, 1).to(device)
        self.t_grid = grids[1].flatten().reshape(-1, 1).to(device)

        # 2. Build Discrete Negative Laplacian Matrix G for RPINN
        G_LU = None
        if self.config.RPINN == 1:
            G = torch.eye(self.config.N_POINTS_X * self.config.N_POINTS_T)
            
            def linearized(ix, iy):
                return ix * self.config.N_POINTS_T + iy

            def nearby(ix, iy):
                return [(ix + 1, iy), (ix - 1, iy), (ix, iy - 1), (ix, iy + 1)]

            for ix in range(self.config.N_POINTS_X):
                for iy in range(self.config.N_POINTS_T):
                    i = linearized(ix, iy)
                    G[i, i] = 1

            for ix in range(1, self.config.N_POINTS_X - 1):
                for iy in range(1, self.config.N_POINTS_T - 1):
                    i = linearized(ix, iy)
                    G[i, i] = 4
                    for jx, jy in nearby(ix, iy):
                        j = linearized(jx, jy)
                        G[i, j] = -1

            hx = 1.0 / self.config.N_POINTS_X
            hy = 1.0 / self.config.N_POINTS_T
            G = G / (hx * hy)
            G = G.to(device)
            G_LU = torch.linalg.lu_factor(G)

        # 3. Model construction
        # Set layer sizes: input is (x,t) so dim=2. Hidden layers defined by neurons/layer and count. Output is 1.
        layers_hidden = [2] + [self.config.KAN_NEURONS_PER_LAYER] * self.config.KAN_LAYERS + [1]
        
        self.model = KANModel(
            layers_hidden=layers_hidden,
            grid_size=5,
            spline_order=3,
            pinning=True
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        # Loss function closure
        def compute_interior_loss(model, x, t):
            if self.config.EXAMPLE == 1:
                f1 = -4.0 * torch.pi * torch.pi * torch.sin(2.0 * torch.pi * x) * torch.sin(2.0 * torch.pi * t)
                f2 = -4.0 * torch.pi * torch.pi * torch.sin(2.0 * torch.pi * x) * torch.sin(2.0 * torch.pi * t)
                rhs = -f1 - f2
                loss = dfdt(model, x, t, order=2) + dfdx(model, x, t, order=2) + rhs

            elif self.config.EXAMPLE == 2:
                f1 = exact_solution_dx2(x, t, self.config)
                f2 = exact_solution_dt2(x, t, self.config)
                rhs = -f1 - f2
                loss = dfdt(model, x, t, order=2) + dfdx(model, x, t, order=2) + rhs

            elif self.config.EXAMPLE == 3:
                loss = dfdt(model, x, t, order=1) - self.config.EPSILON * dfdt(model, x, t, order=2) - self.config.EPSILON * dfdx(model, x, t, order=2)
                loss += shift_EJ_dt(x, t) - self.config.EPSILON * shift_EJ_dt2(x, t) - self.config.EPSILON * shift_EJ_dx2(x, t)
            else:
                raise ValueError(f"Unknown example index: {self.config.EXAMPLE}")

            if self.config.RPINN == 1:
                Ginv_loss = torch.linalg.lu_solve(*G_LU, loss.reshape(-1, 1))
                loss_val = torch.dot(loss.reshape(-1), Ginv_loss.reshape(-1))
            else:
                loss_val = loss.pow(2).sum()

            return loss_val

        # Compute metric function
        def compute_h1_norm(model, x, t, int_loss_val):
            N = self.config.N_POINTS_X * self.config.N_POINTS_T
            dzdx = exact_solution_dx(x, t, self.config) - dfdx(model, x, t, order=1)
            dzdt = exact_solution_dt(x, t, self.config) - dfdt(model, x, t, order=1)

            if self.config.EXAMPLE == 3:
                dzdx = dzdx - shift_EJ_dx(x, t)
                dzdt = dzdt - shift_EJ_dt(x, t)

            h1_norm = math.sqrt((dzdx.detach().pow(2).sum() + dzdt.detach().pow(2).sum()) / N)
            return h1_norm

        self.loss_history = []
        self.h1_error_history = []

        print(f"=== Starting KAN training ===")
        print_every = max(1, self.config.EPOCHS // 200)

        start_time = time.time()
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            optimizer.zero_grad()
            
            loss_val = compute_interior_loss(self.model, self.x_grid, self.t_grid)
            loss_val.backward()
            optimizer.step()
            
            self.loss_history.append(float(loss_val.item()))
            
            if epoch % self.config.H1_CALC_EVERY == 0:
                h1 = compute_h1_norm(self.model, self.x_grid, self.t_grid, loss_val)
                self.h1_error_history.append(h1)

            if (epoch + 1) % print_every == 0 or epoch == 0 or (epoch + 1) == self.config.EPOCHS:
                elapsed = time.time() - start_time
                progress = (epoch + 1) / self.config.EPOCHS
                
                # Format elapsed time
                el_min, el_sec = divmod(int(elapsed), 60)
                el_hr, el_min = divmod(el_min, 60)
                el_str = f"{el_hr:02d}:{el_min:02d}:{el_sec:02d}" if el_hr > 0 else f"{el_min:02d}:{el_sec:02d}"
                
                # Calculate and format ETA
                if progress > 0:
                    eta = elapsed / progress * (1.0 - progress)
                    eta_min, eta_sec = divmod(int(eta), 60)
                    eta_hr, eta_min = divmod(eta_min, 60)
                    eta_str = f"{eta_hr:02d}:{eta_min:02d}:{eta_sec:02d}" if eta_hr > 0 else f"{eta_min:02d}:{eta_sec:02d}"
                else:
                    eta_str = "--:--"
                
                bar_len = 20
                filled_len = int(bar_len * progress)
                bar = '=' * filled_len + '>' + '.' * (bar_len - filled_len - 1)
                bar = bar[:bar_len]
                h1_str = f" | H1 Error: {self.h1_error_history[-1]:.6e}" if self.h1_error_history else ""
                print(f"\rKAN Training: [{bar}] {epoch + 1}/{self.config.EPOCHS} ({progress*100:.1f}%) | {el_str} < {eta_str} | Loss: {loss_val.item():.6e}{h1_str}", end="", flush=True)

        print()
        self.model.eval()
        self.final_loss = float(compute_interior_loss(self.model, self.x_grid, self.t_grid).item())
        self.final_interior_loss = self.final_loss
        self.final_h1_error = self.h1_error_history[-1] if self.h1_error_history else None
        
        print("KAN training complete.")

    def save_model(self, path: str) -> None:
        """Saves PyTorch weights model state dictionary to path."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"KAN Model saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        """Saves final loss values, logs, and simulated prediction coordinates to a NumPy .npz file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        with torch.no_grad():
            assert self.x_grid is not None
            assert self.t_grid is not None
            z_pred = self.model(self.x_grid, self.t_grid).flatten().cpu().numpy()
            if self.config.EXAMPLE == 3:
                z_shift = shift_EJ(self.x_grid, self.t_grid).flatten().cpu().numpy()
                z_pred = z_pred + z_shift

        # Extract KAN edge activations if model has the KAN network
        kan_curves = {}
        if hasattr(self.model, "kan"):
            try:
                # Evaluate activations over [-1.5, 1.5]
                x_eval = torch.linspace(-1.5, 1.5, 200, device=device)
                kan_curves["kan_x_eval"] = x_eval.cpu().numpy()
                
                for idx, layer in enumerate(self.model.kan.layers):
                    if isinstance(layer, KANLinear):
                        # Shape: (out_features, in_features, N_eval)
                        phi = layer.evaluate_edges(x_eval)
                        kan_curves[f"kan_layer_{idx}_phi"] = phi.cpu().numpy()
            except Exception as e:
                print(f"Warning: Failed to extract KAN edge activations: {e}")
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            final_loss=np.array(self.final_loss),
            final_interior_loss=np.array(self.final_interior_loss),
            loss_history=np.array(self.loss_history),
            h1_error_history=np.array(self.h1_error_history),
            x=self.x_grid.flatten().detach().cpu().numpy(),
            t=self.t_grid.flatten().detach().cpu().numpy(),
            z_pred=z_pred,
            **kan_curves
        )
        print(f"KAN Outcomes saved successfully to {path}")
