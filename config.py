import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class ProblemConfig:
    PROBLEM_NAME: str = "poisson_sine"          # poisson_sine | poisson_exp | eriksson_johnson
    EPSILON: float = 0.01
    LENGTH: float = 1.0
    TOTAL_TIME: float = 1.0
    EXAMPLE: int | None = None               # Legacy fallback

    @property
    def problem_id(self) -> str:
        name = self.PROBLEM_NAME.lower().strip()
        if "eriksson" in name or "johnson" in name or self.EXAMPLE == 3:
            eps_str = f"{self.EPSILON:.4f}".rstrip('0').rstrip('.') if self.EPSILON < 1.0 else f"{self.EPSILON:.1f}"
            return f"eriksson_johnson_eps{eps_str}"
        elif "poisson_exp" in name or self.EXAMPLE == 2:
            return "poisson_exp"
        else:
            return "poisson_sine"

@dataclass
class MethodConfig:
    SOLVER: str = "pinn"                         # pinn | kan | iga
    METHOD_NAME: str = "pinn_uniform"
    RPINN: int = 0                               # 0 = standard loss, 1 = robust Gram loss
    SAMPLER_TYPE: str = "uniform"                # uniform | boundary_layer
    SAMPLER_GAMMA: float = 3.0
    ACTIVATION: str = "tanh"                     # tanh | sin
    
    # PINN Architecture
    LAYERS: int = 2
    NEURONS_PER_LAYER: int = 100
    
    # KAN Architecture
    KAN_LAYERS: int = 2
    KAN_NEURONS_PER_LAYER: int = 10
    KAN_SPLINE_TYPE: str = "nurbs"               # nurbs | bspline
    KAN_GRID_SIZE: int = 5
    KAN_SPLINE_ORDER: int = 3
    
    # IGA Configuration
    IGA_METHOD: str = "standard"                 # standard | supg | igrm
    IGA_MESH_TYPE: str = "uniform"               # uniform | adaptive
    IGA_DEGREE: int = 3
    IGA_ELEMENTS: int = 32
    IGA_ADAPTIVE_GAMMA: float = 3.0
    IGA_TEST_DEGREE_ENRICHMENT: int = 1
    
    # Training & Evaluation Settings
    EPOCHS: int = 10000
    RPINN_EPOCHS: int | None = None
    KAN_EPOCHS: int | None = None
    KAN_RPINN_EPOCHS: int | None = None
    LEARNING_RATE: float = 0.001
    KAN_LEARNING_RATE: float = 0.001
    H1_CALC_EVERY: int = 100
    N_POINTS_X: int = 100
    N_POINTS_T: int = 100
    OPTIMIZED: bool = False
    DEBUG: bool = False

class RunConfig(ProblemConfig, MethodConfig):
    """
    Combined Configuration object representing 1 PDE Problem + 1 Specific Solver Method.
    Provides deterministic output directory resolution and validation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.OUTPUT_DIR = None

    @property
    def output_dir(self) -> str:
        if self.OUTPUT_DIR:
            return self.OUTPUT_DIR
        return os.path.join("output", self.problem_id, self.METHOD_NAME)

    def load_config(self, path: str) -> None:
        """Loads configuration from YAML with fallback to common.yaml."""
        # 1. Load common.yaml if available
        common_path = os.path.join(os.path.dirname(os.path.abspath(path)), "common.yaml")
        if not os.path.exists(common_path):
            common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_config", "common.yaml")

        if os.path.exists(common_path):
            with open(common_path, 'r') as f:
                common_data = yaml.safe_load(f) or {}
            for key, value in common_data.items():
                setattr(self, key, value)

        # 2. Load the specific configuration file
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            setattr(self, key, value)

        # Handle PROBLEM_NAME vs legacy EXAMPLE resolution
        if data.get('PROBLEM_NAME'):
            self.PROBLEM_NAME = data['PROBLEM_NAME']
            name = self.PROBLEM_NAME.lower().strip()
            if "poisson_sin" in name:
                self.EXAMPLE = 1
            elif "poisson_exp" in name:
                self.EXAMPLE = 2
            elif "eriksson" in name or "johnson" in name:
                self.EXAMPLE = 3
        elif 'EXAMPLE' in data and data['EXAMPLE'] is not None:
            self.EXAMPLE = data['EXAMPLE']
            if self.EXAMPLE == 1:
                self.PROBLEM_NAME = "poisson_sine"
            elif self.EXAMPLE == 2:
                self.PROBLEM_NAME = "poisson_exp"
            elif self.EXAMPLE == 3:
                self.PROBLEM_NAME = "eriksson_johnson"

        # Auto-infer METHOD_NAME from filename if missing
        if not hasattr(self, 'METHOD_NAME') or not self.METHOD_NAME:
            self.METHOD_NAME = os.path.splitext(os.path.basename(path))[0]

        # Auto-infer SOLVER if missing
        if not hasattr(self, 'SOLVER') or not self.SOLVER:
            name_lower = self.METHOD_NAME.lower()
            if "pinn" in name_lower:
                self.SOLVER = "pinn"
            elif "kan" in name_lower:
                self.SOLVER = "kan"
            elif "iga" in name_lower:
                self.SOLVER = "iga"
            else:
                self.SOLVER = "pinn"

        # Epoch resolution fallbacks
        if self.SOLVER == "kan":
            if self.RPINN == 1 and self.KAN_RPINN_EPOCHS is not None:
                self.EPOCHS = self.KAN_RPINN_EPOCHS
            elif self.KAN_EPOCHS is not None:
                self.EPOCHS = self.KAN_EPOCHS
        elif self.SOLVER == "pinn":
            if self.RPINN == 1 and self.RPINN_EPOCHS is not None:
                self.EPOCHS = self.RPINN_EPOCHS

    def to_dict(self) -> dict[str, Any]:
        """Serializes public configuration parameters to dictionary."""
        d = {}
        for k in dir(self):
            if not k.startswith('_') and not callable(getattr(self, k)):
                if k in ("output_dir", "problem_id"):
                    d[k] = getattr(self, k)
                else:
                    d[k] = getattr(self, k)
        return d

    def validate_config(self) -> None:
        """Validates that necessary fields are set."""
        if not self.SOLVER or self.SOLVER not in ("pinn", "kan", "iga"):
            raise ValueError(f"Invalid SOLVER '{self.SOLVER}'. Must be 'pinn', 'kan', or 'iga'.")
        if not self.METHOD_NAME:
            raise ValueError("METHOD_NAME must be specified.")
        valid_problems = ("poisson_sine", "poisson_exp", "eriksson_johnson")
        if self.PROBLEM_NAME.lower().strip() not in valid_problems and self.EXAMPLE not in (1, 2, 3):
            raise ValueError(f"Invalid PROBLEM_NAME '{self.PROBLEM_NAME}'. Must be one of {valid_problems}.")

    def __str__(self) -> str:
        params = [f"  {k} = {v}" for k, v in sorted(self.to_dict().items())]
        return "RunConfig:\n" + "=" * 50 + "\n" + "\n".join(params) + "\n" + "=" * 50

def load_config(path: str) -> RunConfig:
    """Convenience helper to load and return a validated RunConfig instance."""
    cfg = RunConfig()
    cfg.load_config(path)
    return cfg

# Backward compatibility alias
Config = RunConfig
SharedConfig = RunConfig
PINNConfig = RunConfig
KANConfig = RunConfig
IGAConfig = RunConfig

