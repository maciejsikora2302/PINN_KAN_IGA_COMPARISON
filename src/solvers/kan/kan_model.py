from typing import Any

import torch
from torch import nn

from .kan_linear import KANLinear
from .nurbs import NURBSLinear


class KAN(nn.Module):
    """
    Multi-Layer Kolmogorov-Arnold Network supporting either NURBS or polynomial B-splines.
    """

    def __init__(
        self,
        layers_hidden: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: Any = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] = [-1.0, 1.0],
        spline_type: str = "nurbs",
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.spline_type = spline_type.lower()

        layer_class = NURBSLinear if self.spline_type == "nurbs" else KANLinear

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                layer_class(
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


class KANModel(nn.Module):
    """
    Physics-Informed Boundary Condition Enforced (Pinned) KAN model container.
    Enforces u(0)=0, u(1)=0 on spatial and temporal boundaries.
    """

    def __init__(
        self,
        layers_hidden: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        pinning: bool = True,
        spline_type: str = "nurbs",
    ):
        super().__init__()
        self.pinning = pinning
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
            spline_type=spline_type,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_stack = torch.cat([x, t], dim=1)
        logits = self.kan(x_stack)
        if self.pinning:
            logits = logits * (x - 0.0) * (x - 1.0) * (t - 0.0) * (t - 1.0)
        return logits
