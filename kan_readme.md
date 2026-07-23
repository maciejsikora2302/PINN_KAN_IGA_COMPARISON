# Kolmogorov-Arnold Networks (KAN) Implementation & Edge Activations

This document provides a mathematical and technical overview of the Kolmogorov-Arnold Network (KAN) architecture implemented in this repository, focusing on how learnable edge activations are parameterized, evaluated, and visualized.

---

## 1. Mathematical Formulation

Unlike Multi-Layer Perceptrons (MLPs), which apply fixed non-linear activation functions at nodes and learn weights on edges, Kolmogorov-Arnold Networks (KANs) place **learnable 1D activation functions on the edges** (connections) between nodes, and sum their outputs at the nodes.

For a KAN layer with \(N_{\text{in}}\) inputs and \(N_{\text{out}}\) outputs, the transition from input vector \(\mathbf{x} \in \mathbb{R}^{N_{\text{in}}}\) to output \(\mathbf{y} \in \mathbb{R}^{N_{\text{out}}}\) is given by:

\[
y_i = \sum_{j=1}^{N_{\text{in}}} \phi_{i, j}(x_j)
\]

where \(\phi_{i, j}(x)\) is the learnable activation function mapping input component \(j\) to output component \(i\).

### The Edge Activation Function \(\phi_{i, j}(x)\)

Each edge activation function \(\phi_{i, j}(x)\) is composed of two components:
1. **Base Activation**: A fixed global activation function scaled by a learnable weight.
2. **Spline Activation**: A linear combination of B-spline basis functions scaled by learnable spline coefficients.

Mathematically, it is defined as:

\[
\phi_{i, j}(x) = w_{\text{base}, i, j} \cdot b(x) + w_{\text{spline}, i, j} \cdot s(x)
\]

where:
- \(b(x) = \text{SiLU}(x) = \frac{x}{1 + e^{-x}}\) is the base activation function.
- \(w_{\text{base}, i, j}\) is the learnable base weight.
- \(s(x) = \sum_{m} c_{i, j, m} \cdot B_m(x)\) is the spline curve.
- \(B_m(x)\) are the B-spline basis functions of order \(k\) (typically \(k=3\), cubic splines) defined over a local grid.
- \(c_{i, j, m}\) (stored as `spline_weight` / `scaled_spline_weight`) are the learnable spline coefficients.

---

## 2. Implementation in `kan.py`

The core elements are implemented as PyTorch modules:

### `KANLinear`
This layer represents a single KAN layer. It maintains the learnable parameters:
- `base_weight`: Tensor of shape `(out_features, in_features)` representing \(w_{\text{base}, i, j}\).
- `spline_weight`: Tensor of shape `(out_features, in_features, grid_size + spline_order)` representing \(c_{i, j, m}\).
- `grid`: The spline knots grid of shape `(in_features, grid_size + 2 * spline_order + 1)`.

The forward pass is optimized using matrix multiplications:
```python
base_output = F.linear(self.base_activation(x), self.base_weight)
spline_output = F.linear(
    self.b_splines(x).view(x.size(0), -1),
    self.scaled_spline_weight.view(self.out_features, -1),
)
output = base_output + spline_output
```

### `KAN`
A multi-layer KAN constructed by stacking multiple `KANLinear` layers:
```python
self.layers = nn.ModuleList()
for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
    self.layers.append(KANLinear(in_features, out_features, ...))
```

---

## 3. How Edges are Evaluated and Saved

To visualize the learned shape of the edge activations \(\phi_{i, j}(x)\) without executing full forward passes or implementing B-splines in pure NumPy, a PyTorch evaluation helper is used inside `KANLinear` during outcome saving:

```python
@torch.no_grad()
def evaluate_edges(self, x_range: torch.Tensor) -> torch.Tensor:
    # Evaluate B-spline bases for all input features simultaneously
    x_in = x_range.unsqueeze(1).repeat(1, self.in_features)  # (N_eval, in_features)
    bases = self.b_splines(x_in)  # (N_eval, in_features, grid_size + spline_order)
    
    # Compute base activation contribution: (out_features, in_features, N_eval)
    base_act = self.base_activation(x_range)
    base_val = self.base_weight.unsqueeze(-1) * base_act.unsqueeze(0).unsqueeze(0)
    
    # Compute spline contribution: (out_features, in_features, N_eval)
    spline_val = torch.einsum('ijm,kjm->ijk', self.scaled_spline_weight, bases)
    
    # Total edge activation
    return base_val + spline_val
```

During training execution:
- At the end of KAN training, `save_outcomes` evaluates all layers' edge activations over the domain \([-1.5, 1.5]\) with 200 points.
- The evaluation input coordinates (`kan_x_eval`) and output evaluations (`kan_layer_{idx}_phi` of shape `(out_features, in_features, N_eval)`) are written directly to `kan.npz`.

---

## 4. Visualization of Edge Activations

The `plot_parallel.py` and `visualizer.py` scripts handle rendering these activation functions:

1. **Individual Edge Plots**:
   To avoid generating thousands of image files for wider KAN layers, only the **top 3 most active edge activations** (those with the highest standard deviation / variation across the evaluated range) are plotted individually for each layer, saved to `output/{config_name}/kan_plots/kan_layer_{l}_edge_{i}_{j}.png`.
2. **Layer-wide Grid Plots**:
   To understand the layer as a whole, a matrix grid of size \(N_{\text{out}} \times N_{\text{in}}\) is plotted (limited to a maximum of \(16 \times 8\) to keep the visualization readable). Each subplot cell represents the respective edge activation \(\phi_{i, j}(x)\), saved as `output/{config_name}/kan_plots/kan_layer_{l}_grid.png`.

These plots help identify:
- Which inputs have the strongest influence on each node (via higher amplitude activation curves).
- The level of non-linearity learned by the splines (e.g., whether they behave linearly, quadratically, or oscillate).

---

## 5. Model Sizing and Architecture

Since KANs define learnable B-splines on every connection rather than simple scalar weights, they have a much higher parameter density per neuron/layer compared to PINNs. To prevent memory overhead and long training times, KAN's structure is scaled down:
- **`KAN_LAYERS`**: Represents the number of hidden layers (defaults to `LAYERS`).
- **`KAN_NEURONS_PER_LAYER`**: Represents the number of hidden neurons per layer. It defaults to `max(5, NEURONS_PER_LAYER // 5)` if not explicitly provided, scaling KAN width to roughly 1/5th of the PINN width.

These are configured explicitly in `training_config/common.yaml` or overridden on a per-experiment basis.
