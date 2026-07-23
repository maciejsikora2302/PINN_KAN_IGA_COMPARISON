import os
import math
import time
import yaml
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from model import ExperimentInterface
from config import PINNConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learnable activation function from lain_poisson.py
#
# Comparison of formulations:
#
# Original formulation (from lainpoisson_iga2026.py):
#   den = 1.0
#   for i, coeff in enumerate(self.den_coeffs, start=1):
#       den = den + coeff * (x ** i)
#   den = torch.clamp(den, min=self.eps)
#
# Current formulation:
#   den_poly = 0.0
#   for i, coeff in enumerate(self.den_coeffs, start=1):
#       den_poly = den_poly + coeff * (x ** i)
#   den = 1.0 + torch.abs(den_poly)
#
# Why the current version is better:
# 1. No Discontinuous Gradients: torch.clamp introduces a hard threshold (min=eps).
#    When the denominator falls below eps, the gradient of the clamped output with
#    respect to the coefficients becomes zero, which halts gradient descent updates.
# 2. Smoothness and Positivity Guarantee: 1.0 + torch.abs(den_poly) mathematically
#    guarantees that den >= 1.0 everywhere. It avoids division-by-zero or negative
#    values smoothly without flat regions, ensuring continuous gradients.
class LearnableRational(nn.Module):
    def __init__(self, num_order=2, den_order=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.num_coeffs = nn.Parameter(torch.randn(num_order + 1))
        self.den_coeffs = nn.Parameter(torch.randn(den_order))

    def forward(self, x):
        num = 0.0
        for i, coeff in enumerate(self.num_coeffs):
            num = num + coeff * (x ** i)

        den_poly = torch.zeros_like(x)
        for i, coeff in enumerate(self.den_coeffs, start=1):
            den_poly = den_poly + coeff * (x ** i)
        den = 1.0 + torch.abs(den_poly)
        return num / den



# PINN Model from lain_poisson.py
class PINN(nn.Module):
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False):
        super().__init__()
        self.pinning = pinning
        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):
        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        if self.pinning:
            logits *= (x - 0.0) * (x - 1.0) * (t - 0.0) * (t - 1.0)

        return logits


# Differentiation Utilities
def f(pinn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return pinn(x, t)


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


def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    f_value = f(pinn, x, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    f_value = f(pinn, x, t)
    return df(f_value, x, order=order)


# Exact solutions and derivative formulations
def exact_solution(x, y, config) -> torch.Tensor:
    if config.EXAMPLE == 1:
        return torch.sin(2 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)
    if config.EXAMPLE == 2:
        return -torch.exp(torch.pi * (x - 2 * y)) * torch.sin(2 * torch.pi * x) * torch.sin(torch.pi * y)
    if config.EXAMPLE == 3:
        r1 = (1.0 + math.sqrt(1.0 + 4.0 * config.EPSILON * config.EPSILON * math.pi * math.pi)) / (2.0 * config.EPSILON)
        r2 = (1.0 - math.sqrt(1.0 + 4.0 * config.EPSILON * config.EPSILON * math.pi * math.pi)) / (2.0 * config.EPSILON)
        res_t = (torch.exp(r1 * (y - 1.0)) - torch.exp(r2 * (y - 1.0))) / (math.exp(-r1) - math.exp(-r2))
        res_x_dx = torch.sin(math.pi * x)
        return res_t.mul(res_x_dx)
    raise ValueError(f"Unknown Example index: {config.EXAMPLE}")


def exact_solution_dx(x, y, config) -> torch.Tensor:
    if config.EXAMPLE == 1:
        return 2 * torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)
    if config.EXAMPLE == 2:
        exp1 = -torch.pi * torch.exp(torch.pi * (x - 2 * y)) * torch.sin(torch.pi * y)
        sin1 = torch.sin(2 * torch.pi * x) + 2.0 * torch.cos(2 * torch.pi * x)
        return exp1 * sin1
    if config.EXAMPLE == 3:
        r1 = (1.0 + math.sqrt(1.0 + 4.0 * config.EPSILON * config.EPSILON * math.pi * math.pi)) / (2.0 * config.EPSILON)
        r2 = (1.0 - math.sqrt(1.0 + 4.0 * config.EPSILON * config.EPSILON * math.pi * math.pi)) / (2.0 * config.EPSILON)
        res_t = (torch.exp(r1 * (y - 1.0)) - torch.exp(r2 * (y - 1.0))) / (math.exp(-r1) - math.exp(-r2))
        res_x_dx = math.pi * torch.cos(math.pi * x)
        return res_t.mul(res_x_dx)
    raise ValueError(f"Unknown Example index: {config.EXAMPLE}")


def exact_solution_dt(x, y, config) -> torch.Tensor:
    if config.EXAMPLE == 1:
        return 2 * torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(2.0 * torch.pi * y)
    if config.EXAMPLE == 2:
        exp1 = -torch.pi * torch.exp(torch.pi * (x - 2 * y)) * torch.sin(2.0 * torch.pi * x)
        sin1 = torch.cos(torch.pi * y) - 2.0 * torch.sin(torch.pi * y)
        return exp1 * sin1
    if config.EXAMPLE == 3:
        r1 = (1.0 + math.sqrt(1.0 + 4.0 * config.EPSILON * config.EPSILON * math.pi * math.pi)) / (2.0 * config.EPSILON)
        r2 = (1.0 - math.sqrt(1.0 + 4.0 * config.EPSILON * config.EPSILON * math.pi * math.pi)) / (2.0 * config.EPSILON)
        res_t_dt = (r1 * torch.exp(r1 * (y - 1.0)) - r2 * torch.exp(r2 * (y - 1.0))) / (math.exp(-r1) - math.exp(-r2))
        res_x = torch.sin(math.pi * x)
        return res_t_dt.mul(res_x)
    raise ValueError(f"Unknown Example index: {config.EXAMPLE}")


def exact_solution_dx2(x, y, config) -> torch.Tensor:
    if config.EXAMPLE == 1:
        return -4 * torch.pi * torch.pi * torch.sin(2 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)
    if config.EXAMPLE == 2:
        exp1 = -torch.pi * torch.pi * torch.exp(torch.pi * (x - 2 * y)) * torch.sin(torch.pi * y)
        sin1 = 4.0 * torch.cos(2 * torch.pi * x) - 3.0 * torch.sin(2 * torch.pi * x)
        return exp1 * sin1
    raise ValueError(f"Unknown Example index: {config.EXAMPLE}")


def exact_solution_dt2(x, y, config) -> torch.Tensor:
    if config.EXAMPLE == 1:
        return 4 * torch.pi * torch.pi * torch.sin(2 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)
    if config.EXAMPLE == 2:
        exp1 = torch.pi * torch.pi * torch.exp(torch.pi * (x - 2 * y)) * torch.sin(2.0 * torch.pi * x)
        sin1 = 4.0 * torch.cos(torch.pi * y) - 3.0 * torch.sin(torch.pi * y)
        return exp1 * sin1
    raise ValueError(f"Unknown Example index: {config.EXAMPLE}")


# Shift function formulas for Eriksson-Johnson problem (Example 3)
def shift_EJ(x, t) -> torch.Tensor:
    return torch.sin(math.pi * x) * (1.0 - t)


def shift_EJ_dx(x, t) -> torch.Tensor:
    return math.pi * torch.cos(math.pi * x) * (1.0 - t)


def shift_EJ_dt(x, t) -> torch.Tensor:
    return -torch.sin(math.pi * x)


def shift_EJ_dx2(x, t) -> torch.Tensor:
    return -math.pi * math.pi * torch.sin(math.pi * x) * (1.0 - t)


def shift_EJ_dt2(x, t) -> torch.Tensor:
    return torch.zeros_like(x)


class PINNExperiment(ExperimentInterface):
    """
    PINN Experiment implementation for a 2D Poisson solver.
    Allows running the solver, validating, and saving models/outcomes.
    """

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)
        self.model = None
        self.loss_history = []
        self.h1_error_history = []
        self.final_loss = None
        self.final_interior_loss = None
        
        # Grid variables
        self.x_grid: Any = None
        self.t_grid: Any = None
        self.final_h1_error = None
        
    def load_config(self, config_path: str) -> None:
        """Loads configuration from YAML file into a PINNConfig instance."""
        self.config = PINNConfig()
        self.config.load_config(config_path)
        self.config.validate_config()

    def train(self) -> None:
        """Runs the 2D Poisson training/simulation logic."""
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

        # 2. Build Discrete Negative Laplacian Matrix G
        G_LU = None
        if self.config.RPINN == 1:
            G = torch.eye(self.config.N_POINTS_X * self.config.N_POINTS_T)
            
            # The original code in lainpoisson_iga2026.py used 'ix * N_POINTS_X + iy'.
            # That formulation is incorrect (buggy) when the spatial and temporal grids are non-square
            # (i.e. N_POINTS_X != N_POINTS_T) because the grids are mesh-gridded with dimensions (N_POINTS_X, N_POINTS_T)
            # and then flattened. Consequently, the stride of the ix dimension in the flattened 1D array is N_POINTS_T.
            # We correct this to use N_POINTS_T as the stride multiplier.
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
        if self.config.ACTIVATION == "rational":
            act_func = LearnableRational(num_order=2, den_order=2).to(device)
        elif self.config.ACTIVATION == "tanh":
            act_func = nn.Tanh()
        elif self.config.ACTIVATION == "sin":
            # Sin activation support if requested
            class SinActivation(nn.Module):
                def forward(self, x):
                    return torch.sin(x)
            act_func = SinActivation()
        else:
            raise ValueError(f"Activation function {self.config.ACTIVATION} not supported.")

        self.model = PINN(
            num_hidden=self.config.LAYERS, 
            dim_hidden=self.config.NEURONS_PER_LAYER, 
            act=act_func, 
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

        print(f"=== Starting training for activation: {self.config.ACTIVATION} ===")
        
        # Determine print interval dynamically based on total epochs to keep output clean and fast
        print_every = max(1, self.config.EPOCHS // 200)
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            optimizer.zero_grad()
            
            loss_val = compute_interior_loss(self.model, self.x_grid, self.t_grid)
            loss_val.backward()
            optimizer.step()
            
            # Record metrics
            self.loss_history.append(float(loss_val.item()))
            
            # Periodically compute H1 norms (similar to lain_poisson.py log step)
            # Checked using config.H1_CALC_EVERY to save on autograd overhead
            if epoch % self.config.H1_CALC_EVERY == 0:
                h1 = compute_h1_norm(self.model, self.x_grid, self.t_grid, loss_val)
                self.h1_error_history.append(h1)

            # Logging & Progress Bar with timing info
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
                print(f"\rPINN Training: [{bar}] {epoch + 1}/{self.config.EPOCHS} ({progress*100:.1f}%) | {el_str} < {eta_str} | Loss: {loss_val.item():.6e}{h1_str}", end="", flush=True)

        print() # New line after completing training
        self.model.eval()
        self.final_loss = float(compute_interior_loss(self.model, self.x_grid, self.t_grid).item())
        self.final_interior_loss = self.final_loss
        self.final_h1_error = self.h1_error_history[-1] if self.h1_error_history else None
        
        print("PINN training complete.")

    def save_model(self, path: str) -> None:
        """Saves PyTorch weights model state dictionary to path."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        """Saves final loss values, logs, and simulated prediction coordinates to a NumPy .npz file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        with torch.no_grad():
            assert self.x_grid is not None
            assert self.t_grid is not None
            z_pred = f(self.model, self.x_grid, self.t_grid).flatten().cpu().numpy()
            # If Example 3, we add the shift function
            if self.config.EXAMPLE == 3:
                z_shift = shift_EJ(self.x_grid, self.t_grid).flatten().cpu().numpy()
                z_pred = z_pred + z_shift
        
        # Extract LearnableRational coefficients if they exist in the model
        rational_data = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LearnableRational):
                # Ensure unique and filesystem-friendly name keys
                safe_name = name.replace(".", "_") if name else "act"
                rational_data[f"rational_{safe_name}_num"] = module.num_coeffs.detach().cpu().numpy()
                rational_data[f"rational_{safe_name}_den"] = module.den_coeffs.detach().cpu().numpy()

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
            **rational_data
        )
        print(f"Outcomes saved successfully to {path}")
