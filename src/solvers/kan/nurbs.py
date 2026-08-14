import math
from typing import Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class NURBSLinear(nn.Module):
    """
    Non-Uniform Rational B-Spline (NURBS) KAN Layer.
    Replaces standard polynomial B-splines with trainable rational B-spline activation functions
    incorporating learnable positive projective weights w_i > 0.
    """

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
        super(NURBSLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_bases = grid_size + spline_order

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
            torch.Tensor(out_features, in_features, self.num_bases)
        )
        # Raw unconstrained projective weights initialized near 0.5413 so softplus(0.5413) ~ 1.0
        self.raw_weights = nn.Parameter(
            torch.full((out_features, in_features, self.num_bases), 0.5413)
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
                    - 0.5
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

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates 1D Cox-de Boor B-spline basis functions N_{i,p}(x)."""
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
            self.num_bases,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.num_bases,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    @property
    def positive_weights(self) -> torch.Tensor:
        """Computes positive projective weights w_i > 0 via Softplus."""
        return F.softplus(self.raw_weights) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # 1. Evaluate B-spline bases N(x) -> (batch, in_features, num_bases)
        bases = self.b_splines(x)

        # 2. Get positive projective weights w -> (out_features, in_features, num_bases)
        w = self.positive_weights
        scaled_w = self.scaled_spline_weight

        # 3. Fused tensor contraction: eliminates intermediate 4D tensor allocation
        # Numerator: sum_m (C_{o,i,m} * w_{o,i,m} * N_{b,i,m})
        num = torch.einsum('bim,oim->boi', bases, scaled_w * w)
        # Denominator: sum_m (w_{o,i,m} * N_{b,i,m}) + eps
        den = torch.einsum('bim,oim->boi', bases, w) + 1e-8

        # 4. Compute edge activation sum over inputs
        spline_output = torch.sum(num / den, dim=-1)  # (batch, out_features)

        # 5. Add residual linear base activation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        output = base_output + spline_output

        return output.reshape(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def evaluate_edges(self, x_range: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates 1D NURBS edge activations phi_{i,j}(x) and exports projective weights w.
        Returns:
            phi: (out_features, in_features, N_eval)
            w: (out_features, in_features, num_bases)
        """
        x_in = x_range.unsqueeze(1).repeat(1, self.in_features)
        bases = self.b_splines(x_in)  # (N_eval, in_features, num_bases)
        w = self.positive_weights  # (out_features, in_features, num_bases)

        bases_exp = bases.unsqueeze(1)
        weighted_bases = bases_exp * w.unsqueeze(0)
        den = torch.sum(weighted_bases, dim=-1, keepdim=True) + 1e-8
        rational_bases = weighted_bases / den

        spline_val = torch.einsum('ijm,kijm->ijk', self.scaled_spline_weight, rational_bases)

        base_act = self.base_activation(x_range)
        base_val = self.base_weight.unsqueeze(-1) * base_act.unsqueeze(0).unsqueeze(0)

        phi = base_val + spline_val
        return phi, w
