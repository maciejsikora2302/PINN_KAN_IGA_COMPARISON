import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class SolverMetrics:
    """
    Standardized numerical performance and complexity metrics for all solver experiments.
    """
    final_loss: float
    final_interior_loss: float
    final_h1_error: float
    final_l2_error: float
    final_linf_error: float
    trainable_parameters_or_dofs: int
    elapsed_seconds: float
    epochs_trained: int
    epochs_total: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolverOutcome:
    """
    Standardized, strongly-typed cargo container for all solver experiment outputs.
    """
    x_grid: np.ndarray
    t_grid: np.ndarray
    z_pred: np.ndarray
    loss_history: List[float]
    h1_error_history: List[float]
    h1_time_history: List[float]
    h1_epoch_history: List[int]
    h1_progress_history: List[float]
    metrics: SolverMetrics
    extra_data: Dict[str, Any] = field(default_factory=dict)

    # Convenience properties delegating to metrics
    @property
    def final_h1_error(self) -> float:
        return self.metrics.final_h1_error

    @property
    def final_l2_error(self) -> float:
        return self.metrics.final_l2_error

    @property
    def final_linf_error(self) -> float:
        return self.metrics.final_linf_error

    @property
    def elapsed_seconds(self) -> float:
        return self.metrics.elapsed_seconds

    def save(self, path: str) -> None:
        """Saves outcomes to .npz format with consistent keys across all solvers."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_dict = {
            "x": self.x_grid.flatten(),
            "t": self.t_grid.flatten(),
            "z_pred": self.z_pred.flatten(),
            "loss_history": np.array(self.loss_history),
            "h1_error_history": np.array(self.h1_error_history),
            "h1_time_history": np.array(self.h1_time_history),
            "h1_epoch_history": np.array(self.h1_epoch_history),
            "h1_progress_history": np.array(self.h1_progress_history),
            "final_loss": np.array(self.metrics.final_loss),
            "final_interior_loss": np.array(self.metrics.final_interior_loss),
            "final_h1_error": np.array(self.metrics.final_h1_error),
            "final_l2_error": np.array(self.metrics.final_l2_error),
            "final_linf_error": np.array(self.metrics.final_linf_error),
            "elapsed_seconds": np.array(self.metrics.elapsed_seconds),
            "epochs_trained": np.array(self.metrics.epochs_trained),
            "epochs_total": np.array(self.metrics.epochs_total),
        }
        for k, v in self.extra_data.items():
            if isinstance(v, np.ndarray):
                save_dict[k] = v
            else:
                save_dict[k] = np.array(v)
        np.savez(path, **save_dict)


class ExperimentInterface(ABC):
    """
    Abstract base interface for running physics simulation experiments using
    different numerical/machine learning methods.
    """

    def __init__(self, config_path: str | None = None):
        self.config: Any = None
        self.outcome: Optional[SolverOutcome] = None
        if config_path:
            self.load_config(config_path)

    @abstractmethod
    def load_config(self, config_path: str) -> None:
        """Loads configuration settings from a YAML file."""
        pass

    @abstractmethod
    def train(self) -> SolverOutcome:
        """
        Performs the training loop or runs the physics simulation.
        Returns a strongly-typed SolverOutcome dataclass instance.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Saves the model weights/parameters/state to a file."""
        pass

    def save_outcomes(self, path: str) -> None:
        """Saves the outcome/results/metrics of the simulation to a file."""
        if self.outcome is None:
            raise ValueError("Experiment has not been run yet. Call train() first.")
        self.outcome.save(path)
