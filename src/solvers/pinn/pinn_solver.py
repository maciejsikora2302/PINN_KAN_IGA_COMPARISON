import os
import sys
import math
import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from model import ExperimentInterface
from config import PINNConfig
from src.problems import get_problem, BasePDEProblem
from src.samplers import get_sampler

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

    def __init__(self, config_path: str | None = None, wall_time_limit: float | None = None, optimized: bool | None = None):
        self.model = None
        self.problem: BasePDEProblem = None
        self.loss_history = []
        self.h1_error_history = []
        self.h1_time_history = []
        self.h1_epoch_history = []
        self.h1_progress_history = []
        self.final_loss = None
        self.final_interior_loss = None
        self.x_grid: Any = None
        self.t_grid: Any = None
        self.final_h1_error = None
        self.wall_time_limit = wall_time_limit
        self.optimized = optimized
        self.epochs_trained = 0
        self.epochs_total = 0
        self.elapsed_seconds = 0.0
        super().__init__(config_path)

    def load_config(self, config_path: str) -> None:
        self.config = PINNConfig()
        self.config.load_config(config_path)
        if self.optimized is not None:
            self.config.OPTIMIZED = self.optimized
        prob_name = getattr(self.config, "PROBLEM_NAME", None) or self.config.EXAMPLE
        self.problem = get_problem(prob_name, self.config.EPSILON)

    def train(self) -> None:
        if not self.config or not self.problem:
            raise ValueError("Configuration or PDE problem has not been loaded.")

        # 1. Setup Sampler & Collocation Points
        sampler = get_sampler(
            getattr(self.config, "SAMPLER_TYPE", "uniform"),
            getattr(self.config, "SAMPLER_GAMMA", 3.0)
        )
        self.x_grid, self.t_grid = sampler.sample_points(
            self.config.N_POINTS_X,
            self.config.N_POINTS_T,
            self.config.LENGTH,
            self.config.TOTAL_TIME,
            device=device
        )

        # 2. Build Discrete Negative Laplacian Matrix G / Solver if RPINN requested
        gram_solver = None
        is_optimized = getattr(self.config, "OPTIMIZED", False)
        if self.config.RPINN == 1:
            gram_solver = sampler.build_gram_solver(
                self.config.N_POINTS_X,
                self.config.N_POINTS_T,
                device=device,
                optimized=is_optimized
            )

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

        if hasattr(torch, "compile") and torch.cuda.is_available() and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model)
                print("PINN model compiled successfully using torch.compile.")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Running standard model.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        # Loss function closure using PDEProblem strategy pattern
        def compute_interior_loss(model, x, t):
            loss = self.problem.compute_strong_residual(model, x, t, optimized=is_optimized)

            if self.config.RPINN == 1:
                Ginv_loss = gram_solver(loss)
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
        self.h1_time_history = []
        self.h1_epoch_history = []
        self.h1_progress_history = []

        if self.config.RPINN == 1 and getattr(self.config, "RPINN_EPOCHS", None) is not None:
            pinn_epochs = self.config.RPINN_EPOCHS
        else:
            pinn_epochs = self.config.EPOCHS

        print(f"=== Starting PINN training (RPINN: {self.config.RPINN}, Epochs: {pinn_epochs}, Activation: {self.config.ACTIVATION}) ===")
        print_every = max(1, pinn_epochs // 200)
        self.epochs_total = pinn_epochs
        self.epochs_trained = 0
        start_time = time.perf_counter()

        for epoch in range(pinn_epochs):
            self.epochs_trained = epoch + 1
            if self.wall_time_limit is not None and (time.perf_counter() - start_time) > self.wall_time_limit:
                print(f"\n[Wall Clock Limit Breached] Stopping PINN training early at epoch {self.epochs_trained}")
                break

            self.model.train()
            optimizer.zero_grad()

            use_amp = torch.cuda.is_available()
            device_type = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=torch.bfloat16):
                loss_val = compute_interior_loss(self.model, self.x_grid, self.t_grid)
            loss_val.backward()
            optimizer.step()

            self.loss_history.append(float(loss_val.item()))

            if epoch % self.config.H1_CALC_EVERY == 0 or (epoch + 1) == pinn_epochs:
                elapsed_now = time.perf_counter() - start_time
                h1 = compute_h1_norm(self.model, self.x_grid, self.t_grid)
                self.h1_error_history.append(h1)
                self.h1_time_history.append(elapsed_now)
                self.h1_epoch_history.append(epoch + 1)
                self.h1_progress_history.append(((epoch + 1) / pinn_epochs) * 100.0)

            if (epoch + 1) % print_every == 0 or epoch == 0 or (epoch + 1) == pinn_epochs:
                elapsed = time.perf_counter() - start_time
                progress = (epoch + 1) / pinn_epochs

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
                print(f"\rPINN Training: [{bar}] {epoch + 1}/{pinn_epochs} ({progress*100:.1f}%) | {el_str} < {eta_str} | Loss: {loss_val.item():.6e}{h1_str}", end="", flush=True)

        self.elapsed_seconds = time.perf_counter() - start_time
        print()
        self.model.eval()
        with torch.no_grad():
            z_pred_tensor = f(self.model, self.x_grid, self.t_grid).flatten()
            z_shift_tensor = self.problem.shift_function(self.x_grid, self.t_grid).flatten()
            z_num_tensor = z_pred_tensor + z_shift_tensor
            z_exact_tensor = self.problem.exact_solution(self.x_grid, self.t_grid).flatten()
            err_tensor = z_exact_tensor - z_num_tensor
            self.final_l2_error = float(torch.sqrt(torch.mean(err_tensor ** 2)).item())
            self.final_linf_error = float(torch.max(torch.abs(err_tensor)).item())

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
            h1_time_history=np.array(self.h1_time_history),
            h1_epoch_history=np.array(self.h1_epoch_history),
            h1_progress_history=np.array(self.h1_progress_history),
            elapsed_seconds=np.array(self.elapsed_seconds),
            final_h1_error=np.array(self.final_h1_error if self.final_h1_error is not None else 0.0),
            final_l2_error=np.array(self.final_l2_error if getattr(self, "final_l2_error", None) is not None else 0.0),
            final_linf_error=np.array(self.final_linf_error if getattr(self, "final_linf_error", None) is not None else 0.0),
            epochs_trained=np.array(self.epochs_trained),
            epochs_total=np.array(self.epochs_total),
            x=self.x_grid.flatten().detach().cpu().numpy(),
            t=self.t_grid.flatten().detach().cpu().numpy(),
            z_pred=z_pred
        )
        print(f"Outcomes saved successfully to {path}")
