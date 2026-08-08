import os
import math
import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from model import ExperimentInterface
from config import PINNConfig
from src.problems import get_problem, BasePDEProblem
from src.samplers import UniformGridSampler, BoundaryLayerSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def f(pinn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return pinn(x, t)

def df(output: torch.Tensor, input_tensor: torch.Tensor, order: int = 1) -> torch.Tensor:
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input_tensor,
            grad_outputs=torch.ones_like(input_tensor),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value

def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    return df(f(pinn, x, t), t, order=order)

def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    return df(f(pinn, x, t), x, order=order)


class PINNExperiment(ExperimentInterface):
    """
    Refactored PINN Experiment implementation.
    Fully decoupled from specific problem formulas via the PDEProblem strategy pattern.
    """

    def __init__(self, config_path: str | None = None):
        self.model = None
        self.problem: BasePDEProblem = None
        self.loss_history = []
        self.h1_error_history = []
        self.final_loss = None
        self.final_interior_loss = None
        self.x_grid: Any = None
        self.t_grid: Any = None
        self.final_h1_error = None
        super().__init__(config_path)

    def load_config(self, config_path: str) -> None:
        self.config = PINNConfig()
        self.config.load_config(config_path)
        self.config.validate_config()
        self.problem = get_problem(self.config.EXAMPLE, self.config.EPSILON)

    def train(self) -> None:
        if not self.config or not self.problem:
            raise ValueError("Configuration or PDE problem has not been loaded.")

        # 1. Setup Sampler & Collocation Points
        sampler = UniformGridSampler()
        self.x_grid, self.t_grid = sampler.sample_points(
            self.config.N_POINTS_X,
            self.config.N_POINTS_T,
            self.config.LENGTH,
            self.config.TOTAL_TIME,
            device=device
        )

        # 2. Build Discrete Negative Laplacian Matrix G if RPINN requested
        G_LU = None
        if self.config.RPINN == 1:
            G_LU = sampler.build_gram_matrix(self.config.N_POINTS_X, self.config.N_POINTS_T, device=device)

        # 3. Model construction
        if self.config.ACTIVATION == "sin":
            class SinActivation(nn.Module):
                def forward(self, x):
                    return torch.sin(x)
            act_func = SinActivation()
        else:
            act_func = nn.Tanh()

        self.model = PINN(
            num_hidden=self.config.LAYERS,
            dim_hidden=self.config.NEURONS_PER_LAYER,
            act=act_func,
            pinning=True
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        # Loss function closure using PDEProblem strategy pattern
        def compute_interior_loss(model, x, t):
            loss = self.problem.compute_strong_residual(model, x, t)

            if self.config.RPINN == 1:
                Ginv_loss = torch.linalg.lu_solve(*G_LU, loss.reshape(-1, 1))
                loss_val = torch.dot(loss.reshape(-1), Ginv_loss.reshape(-1))
            else:
                loss_val = loss.pow(2).sum()

            return loss_val

        # Compute H1 norm error using PDEProblem derivatives
        def compute_h1_norm(model, x, t):
            N = self.config.N_POINTS_X * self.config.N_POINTS_T
            dzdx = self.problem.exact_dx(x, t) - dfdx(model, x, t, order=1) - self.problem.shift_dx(x, t)
            dzdt = self.problem.exact_dy(x, t) - dfdt(model, x, t, order=1) - self.problem.shift_dy(x, t)

            h1_norm = math.sqrt((dzdx.detach().pow(2).sum() + dzdt.detach().pow(2).sum()) / N)
            return h1_norm

        self.loss_history = []
        self.h1_error_history = []

        print(f"=== Starting training for activation: {self.config.ACTIVATION} ===")
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
                h1 = compute_h1_norm(self.model, self.x_grid, self.t_grid)
                self.h1_error_history.append(h1)

            if (epoch + 1) % print_every == 0 or epoch == 0 or (epoch + 1) == self.config.EPOCHS:
                elapsed = time.time() - start_time
                progress = (epoch + 1) / self.config.EPOCHS

                el_min, el_sec = divmod(int(elapsed), 60)
                el_hr, el_min = divmod(el_min, 60)
                el_str = f"{el_hr:02d}:{el_min:02d}:{el_sec:02d}" if el_hr > 0 else f"{el_min:02d}:{el_sec:02d}"

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

        print()
        self.model.eval()
        self.final_loss = float(compute_interior_loss(self.model, self.x_grid, self.t_grid).item())
        self.final_interior_loss = self.final_loss
        self.final_h1_error = self.h1_error_history[-1] if self.h1_error_history else None

        print("PINN training complete.")

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()
        with torch.no_grad():
            assert self.x_grid is not None
            assert self.t_grid is not None
            z_pred = f(self.model, self.x_grid, self.t_grid).flatten().cpu().numpy()
            # Add shift function from PDEProblem if present
            z_shift = self.problem.shift_function(self.x_grid, self.t_grid).flatten().cpu().numpy()
            z_pred = z_pred + z_shift

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            final_loss=np.array(self.final_loss),
            final_interior_loss=np.array(self.final_interior_loss),
            loss_history=np.array(self.loss_history),
            h1_error_history=np.array(self.h1_error_history),
            x=self.x_grid.flatten().detach().cpu().numpy(),
            t=self.t_grid.flatten().detach().cpu().numpy(),
            z_pred=z_pred
        )
        print(f"Outcomes saved successfully to {path}")
