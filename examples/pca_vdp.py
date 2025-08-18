from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_datasets.van_der_pol import VanDerPolDataset, van_der_pol
from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step
from function_encoder.coefficients import least_squares, lasso
import tqdm

if torch.cuda.is_available():
    device = "cuda:5"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

def basis_function_factory() -> NeuralODE:
    """
    Factory function to create a new basis function.
    This allows us to easily create new basis functions with the same architecture.
    """
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
        integrator=rk4_step,
    )

def input_grid(y0,dt):
    # Get range from current batch
    y0_min = y0.min(dim=0).values.min(dim=0).values
    y0_max = y0.max(dim=0).values.min(dim=0).values
    dt_min = dt.min().item()
    dt_max = dt.max().item()

    # Grid resolution
    n_points = 30

    # 2D state space grid
    y0_1 = torch.linspace(y0_min[0].item(), y0_max[0].item(), n_points)
    y0_2 = torch.linspace(y0_min[1].item(), y0_max[1].item(), n_points)
    dt_vals = torch.linspace(dt_min, dt_max, n_points)

    # Meshgrid for 3D space
    Y0_1, Y0_2, DT = torch.meshgrid(y0_1, y0_2, dt_vals, indexing="ij")  # [n, n, n]

    # Stack into shape [n*n*n, 3]
    grid = torch.stack([Y0_1, Y0_2, DT], dim=-1).reshape(-1, 3)  # [n^3, 3]
    grid = grid.unsqueeze(0).to(y0.device)  # [1, n^3, 3]

    y0_grid = grid[..., :2]  # [1, N, 2]
    dt_grid = grid[..., 2]   # [1, N]

    return (y0_grid, dt_grid)

def loss_function(model, batch, ortho_lambda=0.02):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    
    # Count non-zero coefficients per sample
    nonzero_counts = (coefficients.abs() > 1e-6).sum(dim=-1)  # Shape: [batch_size]
    avg_nonzero = nonzero_counts.float().mean().item()

    pred = model((y0, dt), coefficients=coefficients)
    pred_loss = torch.nn.functional.mse_loss(pred, y1)

    ortho_loss = 0.0
    num_heads = model.basis_functions.num_heads()

    if num_heads > 1:
        # phi shape is [batch, points, features, basis], e.g., [1, 200, 1, num_heads]
        x_grid, dt_grid = input_grid(y0,dt)
        phi = model.basis_functions((x_grid, dt_grid))  # [n_points, num_heads]
        phi = phi / (torch.norm(phi, p=2, dim=(1, 2), keepdim=True) + 1e-8)

        old_phi = phi[..., -2:-1].squeeze(1)    # all previous basis funcs other than the last
        new_phi = phi[..., -1:].squeeze(1)    # the latest basis function

        # Compute cross-gram matrix: [1, num_old]
        cross_gram = torch.einsum('bmdk,bmdl->bkl', new_phi, old_phi)  # inner product
        ortho_loss = torch.sum((cross_gram - torch.diag_embed(torch.diagonal(cross_gram, dim1=-2, dim2=-1))) ** 2)

    return pred_loss + ortho_lambda * ortho_loss


# Setting
MAX_BASIS_SIZE = 20
LOSS_THRESHOLD = 1e-3

# Load dataset
dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
n_basis = 10
basis_functions = BasisFunctions(
    *[basis_function_factory() for _ in range(n_basis)]
)

print(f"Training mode: Progressive")
model = FunctionEncoder(basis_functions).to(device)

# Train the first basis function
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{MAX_BASIS_SIZE}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix_str(f"loss: {loss:.2e}")


# Initialize head count
num_heads = 1

# Train remaining basis functions manually
while num_heads <= MAX_BASIS_SIZE:
    num_heads += 1

    # Freeze all existing params
    for p in model.parameters(): 
        p.requires_grad = False
    new_fn = basis_function_factory().to(device)
    # Create new basis function and add to model
    for p in new_fn.parameters(): 
        p.requires_grad = True
    model.basis_functions.basis_functions.append(new_fn)

    # Only train the trainable parameters
    optimizer = torch.optim.Adam([p for p in new_fn.parameters() if p.requires_grad], lr=5e-4)
    
    with tqdm.tqdm(range(num_epochs), desc=f"basis {num_heads}/{MAX_BASIS_SIZE}") as tqdm_bar:
        for epoch in tqdm_bar:
            batch = next(dataloader_iter)
            loss = train_step(model, optimizer, batch, loss_function)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

    if loss <= LOSS_THRESHOLD:
        print(f"Reached target loss with {num_heads} basis functions.")
        break

    if num_heads == MAX_BASIS_SIZE:
        print("Reached maximum basis size without reaching loss threshold.")
        break



# Plot a grid of evaluations

import matplotlib.pyplot as plt


model.eval()
with torch.no_grad():
    # Generate a single batch of functions for plotting
    dataloader = DataLoader(dataset, batch_size=9)
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)

    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    mu = mu.to(device)
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    # Precompute the coefficients for the batch
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        for j in range(3):

            # Plot a single trajectory
            _mu = mu[i * 3 + j]
            _y0 = torch.empty(1, 2, device=device).uniform_(
                *dataloader.dataset.y0_range
            )
            # We use the coefficients that we computed before
            _c = coefficients[i * 3 + j].unsqueeze(0)
            s = 0.1  # Time step for simulation
            n = int(10 / s)
            _dt = torch.tensor([s], device=device)

            # Integrate the true trajectory
            x = _y0.clone()
            y = [x]
            for k in range(n):
                x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
                y.append(x)
            y = torch.cat(y, dim=0)
            y = y.detach().cpu().numpy()

            # Integrate the predicted trajectory
            x = _y0.clone()
            x = x.unsqueeze(1)
            _dt = _dt.unsqueeze(0)
            pred = [x]
            for k in range(n):
                x = model((x, _dt), coefficients=_c) + x
                pred.append(x)
            pred = torch.cat(pred, dim=1)
            pred = pred.detach().cpu().numpy()

            ax[i, j].set_xlim(-5, 5)
            ax[i, j].set_ylim(-5, 5)
            (_t,) = ax[i, j].plot(y[:, 0], y[:, 1], label="True")
            (_p,) = ax[i, j].plot(pred[0, :, 0], pred[0, :, 1], label="Predicted")

    fig.legend(
        handles=[_t, _p],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )

    plt.show()

    # save the model

# torch.save(model.state_dict(), "van_der_pol_model.pth")
