import os
import sys
import math
import time
from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ExperimentInterface, SolverMetrics, SolverOutcome
from config import KANConfig
from src.problems import get_problem, BasePDEProblem
from src.samplers import get_sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            grid_val: Any = self.grid
            grid_tensor: torch.Tensor = grid_val
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    grid_tensor.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid_val: Any = self.grid
        grid: torch.Tensor = grid_val
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
        return output.reshape(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def evaluate_edges(self, x_range: torch.Tensor) -> torch.Tensor:
        x_in = x_range.unsqueeze(1).repeat(1, self.in_features)
        bases = self.b_splines(x_in)
        base_act = self.base_activation(x_range)
        base_val = self.base_weight.unsqueeze(-1) * base_act.unsqueeze(0).unsqueeze(0)
        spline_val = torch.einsum('ijm,kjm->ijk', self.scaled_spline_weight, bases)
        return base_val + spline_val


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
        super().__init__()
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

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


from .kan_model import KANModel

class KANExperiment(ExperimentInterface):
    """
    Refactored KAN Experiment implementation supporting both NURBS and B-spline activation functions.
    Fully decoupled from specific problem formulas via the PDEProblem strategy pattern.
    """

    def __init__(self, config_path: str | None = None, wall_time_limit: float | None = None, optimized: bool | None = None):
        self.model = None
        self.problem: Optional[BasePDEProblem] = None
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
        self.config = KANConfig()
        self.config.load_config(config_path)
        if self.optimized is not None:
            self.config.OPTIMIZED = self.optimized
        prob_name = getattr(self.config, "PROBLEM_NAME", None)
        if prob_name is None:
            prob_name = getattr(self.config, "EXAMPLE", 1)
        if prob_name is None:
            prob_name = 1
        self.problem = get_problem(prob_name, self.config.EPSILON)

    def train(self) -> SolverOutcome:
        if not self.config or not self.problem:
            raise ValueError("Configuration or PDE problem has not been loaded.")

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

        gram_solver = None
        is_optimized = getattr(self.config, "OPTIMIZED", False)
        if self.config.RPINN == 1:
            gram_solver = sampler.build_gram_solver(
                self.config.N_POINTS_X,
                self.config.N_POINTS_T,
                device=device,
                optimized=is_optimized
            )

        layers_hidden = [2] + [self.config.KAN_NEURONS_PER_LAYER] * self.config.KAN_LAYERS + [1]
        spline_type = getattr(self.config, "KAN_SPLINE_TYPE", "nurbs")
        grid_size = getattr(self.config, "KAN_GRID_SIZE", 5)
        spline_order = getattr(self.config, "KAN_SPLINE_ORDER", 3)

        self.model = KANModel(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
            pinning=True,
            spline_type=spline_type
        ).to(device)

        if hasattr(torch, "compile") and torch.cuda.is_available() and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model)
                print("KAN model compiled successfully using torch.compile.")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Running standard model.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.KAN_LEARNING_RATE)

        def compute_interior_loss(model, x, t):
            loss = self.problem.compute_strong_residual(model, x, t, optimized=is_optimized)
            if self.config.RPINN == 1 and gram_solver is not None:
                Ginv_loss = gram_solver(loss)
                loss_val = torch.dot(loss.reshape(-1), Ginv_loss.reshape(-1))
            else:
                loss_val = loss.pow(2).sum()
            return loss_val

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

        if self.config.RPINN == 1 and getattr(self.config, "KAN_RPINN_EPOCHS", None) is not None:
            kan_epochs = self.config.KAN_RPINN_EPOCHS
        elif getattr(self.config, "KAN_EPOCHS", None) is not None:
            kan_epochs = self.config.KAN_EPOCHS
        elif self.config.RPINN == 1 and getattr(self.config, "RPINN_EPOCHS", None) is not None:
            kan_epochs = self.config.RPINN_EPOCHS
        else:
            kan_epochs = self.config.EPOCHS

        print(f"=== Starting KAN training (Spline Type: {spline_type}, RPINN: {self.config.RPINN}, Epochs: {kan_epochs}) ===")
        print_every = max(1, kan_epochs // 200)
        self.epochs_total = kan_epochs
        self.epochs_trained = 0
        start_time = time.perf_counter()

        for epoch in range(kan_epochs):
            self.epochs_trained = epoch + 1
            if self.wall_time_limit is not None and (time.perf_counter() - start_time) > self.wall_time_limit:
                print(f"\n[Wall Clock Limit Breached] Stopping KAN training early at epoch {self.epochs_trained}")
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

            if epoch % self.config.H1_CALC_EVERY == 0 or (epoch + 1) == kan_epochs:
                elapsed_now = time.perf_counter() - start_time
                h1 = compute_h1_norm(self.model, self.x_grid, self.t_grid)
                self.h1_error_history.append(h1)
                self.h1_time_history.append(elapsed_now)
                self.h1_epoch_history.append(epoch + 1)
                self.h1_progress_history.append(((epoch + 1) / kan_epochs) * 100.0)

            if (epoch + 1) % print_every == 0 or epoch == 0 or (epoch + 1) == kan_epochs:
                elapsed = time.perf_counter() - start_time
                progress = (epoch + 1) / kan_epochs

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
                print(f"\rKAN Training: [{bar}] {epoch + 1}/{kan_epochs} ({progress*100:.1f}%) | {el_str} < {eta_str} | Loss: {loss_val.item():.6e}{h1_str}", end="", flush=True)

        self.elapsed_seconds = time.perf_counter() - start_time
        print()
        self.model.eval()
        with torch.no_grad():
            z_pred_tensor = self.model(self.x_grid, self.t_grid).flatten()
            z_shift_tensor = self.problem.shift_function(self.x_grid, self.t_grid).flatten()
            z_num_tensor = z_pred_tensor + z_shift_tensor
            z_exact_tensor = self.problem.exact_solution(self.x_grid, self.t_grid).flatten()
            err_tensor = z_exact_tensor - z_num_tensor
            self.final_l2_error = float(torch.sqrt(torch.mean(err_tensor ** 2)).item())
            self.final_linf_error = float(torch.max(torch.abs(err_tensor)).item())

        self.final_loss = float(compute_interior_loss(self.model, self.x_grid, self.t_grid).item())
        self.final_interior_loss = self.final_loss
        self.final_h1_error = self.h1_error_history[-1] if self.h1_error_history else None

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        metrics = SolverMetrics(
            final_loss=self.final_loss,
            final_interior_loss=self.final_interior_loss,
            final_h1_error=self.final_h1_error if self.final_h1_error is not None else 0.0,
            final_l2_error=self.final_l2_error if self.final_l2_error is not None else 0.0,
            final_linf_error=self.final_linf_error if self.final_linf_error is not None else 0.0,
            trainable_parameters_or_dofs=trainable_params,
            elapsed_seconds=self.elapsed_seconds,
            epochs_trained=self.epochs_trained,
            epochs_total=self.epochs_total
        )

        kan_curves = {}
        if hasattr(self.model, "kan"):
            try:
                x_eval = torch.linspace(-1.5, 1.5, 200, device=device)
                kan_curves["kan_x_eval"] = x_eval.cpu().numpy()
                for idx, layer_module in enumerate(self.model.kan.layers):
                    layer: Any = layer_module
                    if hasattr(layer, "evaluate_edges"):
                        out_edges = layer.evaluate_edges(x_eval)
                        if isinstance(out_edges, tuple):
                            phi, w = out_edges
                            kan_curves[f"kan_layer_{idx}_phi"] = phi.cpu().numpy()
                            kan_curves[f"kan_layer_{idx}_nurbs_weights"] = w.cpu().numpy()
                        else:
                            kan_curves[f"kan_layer_{idx}_phi"] = out_edges.cpu().numpy()
            except Exception as e:
                print(f"Warning: Failed to extract KAN edge activations: {e}")

        self.outcome = SolverOutcome(
            x_grid=self.x_grid.flatten().detach().cpu().numpy(),
            t_grid=self.t_grid.flatten().detach().cpu().numpy(),
            z_pred=z_num_tensor.detach().cpu().numpy(),
            loss_history=self.loss_history,
            h1_error_history=self.h1_error_history,
            h1_time_history=self.h1_time_history,
            h1_epoch_history=self.h1_epoch_history,
            h1_progress_history=self.h1_progress_history,
            metrics=metrics,
            extra_data=kan_curves
        )

        print("KAN training complete.")
        return self.outcome

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"KAN Model saved successfully to {path}")

    def save_outcomes(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()
        with torch.no_grad():
            assert self.x_grid is not None
            assert self.t_grid is not None
            z_pred = self.model(self.x_grid, self.t_grid).flatten().cpu().numpy()
            z_shift = self.problem.shift_function(self.x_grid, self.t_grid).flatten().cpu().numpy()
            z_pred = z_pred + z_shift

        kan_curves = {}
        if hasattr(self.model, "kan"):
            try:
                x_eval = torch.linspace(-1.5, 1.5, 200, device=device)
                kan_curves["kan_x_eval"] = x_eval.cpu().numpy()
                for idx, layer in enumerate(self.model.kan.layers):
                    if hasattr(layer, "evaluate_edges"):
                        out_edges = layer.evaluate_edges(x_eval)
                        if isinstance(out_edges, tuple):
                            phi, w = out_edges
                            kan_curves[f"kan_layer_{idx}_phi"] = phi.cpu().numpy()
                            kan_curves[f"kan_layer_{idx}_nurbs_weights"] = w.cpu().numpy()
                        else:
                            kan_curves[f"kan_layer_{idx}_phi"] = out_edges.cpu().numpy()
            except Exception as e:
                print(f"Warning: Failed to extract KAN edge activations: {e}")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_dict: Any = {
            "final_loss": np.array(self.final_loss),
            "final_interior_loss": np.array(self.final_interior_loss),
            "loss_history": np.array(self.loss_history),
            "h1_error_history": np.array(self.h1_error_history),
            "h1_time_history": np.array(self.h1_time_history),
            "h1_epoch_history": np.array(self.h1_epoch_history),
            "h1_progress_history": np.array(self.h1_progress_history),
            "elapsed_seconds": np.array(self.elapsed_seconds),
            "final_h1_error": np.array(self.final_h1_error if self.final_h1_error is not None else 0.0),
            "final_l2_error": np.array(self.final_l2_error if getattr(self, "final_l2_error", None) is not None else 0.0),
            "final_linf_error": np.array(self.final_linf_error if getattr(self, "final_linf_error", None) is not None else 0.0),
            "epochs_trained": np.array(self.epochs_trained),
            "epochs_total": np.array(self.epochs_total),
            "x": self.x_grid.flatten().detach().cpu().numpy(),
            "t": self.t_grid.flatten().detach().cpu().numpy(),
            "z_pred": z_pred,
        }
        save_dict.update(kan_curves)
        np.savez(
            path,
            **save_dict
        )
        print(f"KAN Outcomes saved successfully to {path}")
