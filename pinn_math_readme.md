# Mathematical Formulation of PINN implementation (`pinn.py`)

This document outlines the mathematical formulations and numerical methods implemented in `pinn.py`. The purpose of this guide is to provide a clear mathematical baseline to facilitate the translation of these concepts into the KAN (Kolmogorov-Arnold Network) framework.

## 1. Network Architecture & Boundary Condition Enforcement (Pinning)

The basic model $u_\theta(x, t)$ consists of an input layer, several hidden layers, and a single output layer. 
To strictly enforce homogeneous Dirichlet boundary conditions at the boundaries of the domain $(x,t) \in [0,1]^2$, the network employs a **"pinning"** technique. The output of the neural network $N_\theta(x, t)$ is multiplied by a distance function $D(x,t)$ that strictly vanishes on the boundary.

$$
u_\theta(x, t) = N_\theta(x, t) \cdot x(x-1)t(t-1)
$$

Where $N_\theta(x, t)$ will be replaced by a KAN model in the new framework, while the pinning factor $x(x-1)t(t-1)$ should remain exactly the same to guarantee zero boundary conditions.

## 2. Learnable Rational Activation Function

Instead of a standard fixed activation function (like Tanh), the code supports a **Learnable Rational Activation Function**:

$$
\sigma(x) = \frac{P(x)}{Q(x)} = \frac{a_0 + a_1 x + a_2 x^2}{1.0 + |b_1 x + b_2 x^2|}
$$

- $a_0, a_1, a_2$ are learnable numerator coefficients (`num_coeffs`).
- $b_1, b_2$ are learnable denominator coefficients (`den_coeffs`).
- The denominator employs $1.0 + |\cdot|$ to ensure $Q(x) \ge 1.0$ everywhere, completely avoiding division by zero or discontinuous gradients without relying on hard clamping.

*(Note: When transitioning to KAN, KAN natively learns activation functions on edges via B-splines, so this rational activation might become redundant or can be used as the base function in KAN edges).*

## 3. Loss Formulations (PDE Residuals)

The interior loss evaluates the strong form of the PDE residuals. The residual $L_i$ at collocation point $(x_i, t_i)$ is formulated depending on the chosen example.

### Examples 1 & 2 (Poisson Equation)
The governing equation is the 2D Poisson problem:
$$
\Delta u(x,t) = f(x,t) \implies \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial t^2} - f(x,t) = 0
$$
Since the code constructs the right-hand side `rhs` using the exact solution derivatives $f(x,t) = \Delta u_{exact}(x,t)$, the loss residual $L(x,t)$ is defined as:
$$
L(x,t) = \frac{\partial^2 u_\theta}{\partial t^2} + \frac{\partial^2 u_\theta}{\partial x^2} - \Delta u_{exact}(x,t)
$$

### Example 3 (Eriksson-Johnson Problem)
This represents a convection-diffusion equation (often written as $u_y - \epsilon \Delta u = 0$):
$$
\frac{\partial u}{\partial t} - \epsilon \left( \frac{\partial^2 u}{\partial t^2} + \frac{\partial^2 u}{\partial x^2} \right) = 0
$$
The solution incorporates a shift function $S(x,t) = \sin(\pi x)(1 - t)$. Therefore, the full approximation is $u(x,t) = u_\theta(x,t) + S(x,t)$. 
The residual is defined as:
$$
L(x,t) = \left( \frac{\partial u_\theta}{\partial t} - \epsilon \Delta u_\theta \right) + \left( \frac{\partial S}{\partial t} - \epsilon \Delta S \right)
$$

## 4. RPINN Formulation (Discrete $H^{-1}$ Norm Loss)

By default, standard PINNs minimize the $L^2$ norm of the PDE residual: $\mathcal{L} = \sum L(x_i, t_i)^2$. 

However, `pinn.py` implements an alternative loss metric controlled by the `RPINN=1` flag. This corresponds to minimizing the discrete $H^{-1}$ norm of the residual, which helps combat spectral bias and improves convergence for problems with high-frequency solutions.

**Construction of the Discrete Negative Laplacian ($G$):**
A matrix $G$ of size $(N_x \cdot N_t) \times (N_x \cdot N_t)$ is constructed using a 5-point finite difference stencil:
- Diagonal elements: $G_{i,i} = 4$
- Adjacent elements: $G_{i,j} = -1$ (for $j$ neighboring $i$ in the grid)
- Boundary points are set to identity.
- The matrix is scaled by $1 / (h_x h_y)$, where $h_x = 1/N_x$ and $h_y = 1/N_t$.

**Loss Calculation:**
Given the flattened residual vector $\vec{L}$:
1. We solve the linear system $G \vec{w} = \vec{L}$ (where $\vec{w} = G^{-1}\vec{L}$) using an LU decomposition (`G_LU`).
2. The final scalar loss is calculated as the dot product:
$$
\mathcal{L} = \vec{L}^T G^{-1} \vec{L} = \vec{L}^T \vec{w}
$$
This mathematical approach effectively calculates the weighted $H^{-1}$ Sobolev norm of the error residual. **This is a critical algorithm to port identically over to the KAN framework.**

## 5. Evaluation Metric ($H^1$ Semi-Norm Error)

To evaluate the solver's accuracy, the code calculates the $H^1$ semi-norm of the error:
$$
\text{Error}_{H^1} = \sqrt{ \frac{1}{N} \sum_{i=1}^N \left( \left( \frac{\partial u_{exact}}{\partial x} - \frac{\partial u_\theta}{\partial x} \right)^2 + \left( \frac{\partial u_{exact}}{\partial t} - \frac{\partial u_\theta}{\partial t} \right)^2 \right) }
$$
For Example 3, the gradients of the shift function $S(x,t)$ are correctly subtracted to align the continuous PINN output with the exact solution definitions.
