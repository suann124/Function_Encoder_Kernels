"""
Progressive Basis Functions Training for Van der Pol Oscillator

This example demonstrates progressive training of basis functions, where new basis
functions are added one by one. The key fixes made to match the batch approach:

1. **Basis Function Architecture**: Changed from simple MLP [3, 32, 1] to NeuralODE
   with MLP [3, 64, 64, 2] to match van_der_pol.py

2. **Input Format**: Changed from concatenated (y0, dt) to tuple format ((y0, dt))
   to match the expected input format

3. **Coefficients Method**: Added lasso coefficients method to match van_der_pol.py

4. **Training Strategy**: Maintain progressive training but allow new basis functions
   to adapt to existing ones during their training phase

5. **Dataset**: Increased n_points from 100 to 1000 and added dt_range to match

The progressive approach maintains its core structure: adding one basis function at
a time, but now with proper architecture and training that allows for optimal
performance matching the batch approach.
"""

import torch

from torch.utils.data import DataLoader
from my_datasets.van_der_pol import VanDerPolDataset

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import train_step
from function_encoder.coefficients import lasso


import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(42)

# Load dataset
dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
def basis_function_factory():
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
        integrator=rk4_step,
    )


num_basis = 10
# Only use one basis function initially for progressive training
basis_functions = BasisFunctions(basis_function_factory())

model = FunctionEncoder(basis_functions, coefficients_method=lasso).to(device)

# Train model

losses = []  # For plotting.
scores = []  # For plotting.
dataloader_coeffs = DataLoader(dataset, batch_size=100)
dataloader_coeffs_iter = iter(dataloader_coeffs)


def compute_explained_variance(model):
    _, _, _, _, y0_example, dt_example, y1_example = next(dataloader_coeffs_iter)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    # Use the same input format as the batch approach
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)
    
    # Handle the case where we have fewer basis functions than expected
    n_basis_current = coefficients.shape[1]
    
    # Compute covariance matrix of coefficients
    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = (
        torch.matmul(coefficients_centered.T, coefficients_centered)
        / (coefficients.shape[0] - 1)  # Use n-1 for unbiased estimate
    )

    # Get eigenvalues of covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order
    
    # Pad eigenvalues to match expected number of basis functions
    if len(eigenvalues) < num_basis:
        padding = torch.zeros(num_basis - len(eigenvalues), device=eigenvalues.device)
        eigenvalues = torch.cat([eigenvalues, padding])
    
    # Compute explained variance ratio
    explained_variance_ratio = eigenvalues / (torch.sum(eigenvalues) + 1e-8)  # Add small epsilon to avoid division by zero

    # Get eigenvalues of gram matrix (averaged over batch)
    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(G.mean(dim=0))
    gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order
    
    # Pad gram eigenvalues to match expected number of basis functions
    if len(gram_eigenvalues) < num_basis:
        padding = torch.zeros(num_basis - len(gram_eigenvalues), device=gram_eigenvalues.device)
        gram_eigenvalues = torch.cat([gram_eigenvalues, padding])
    
    # Normalize gram eigenvalues for comparison
    gram_eigenvalues = gram_eigenvalues / (torch.sum(gram_eigenvalues) + 1e-8)

    return explained_variance_ratio, eigenvalues, gram_eigenvalues


def loss_function(model, batch):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    # Use the same input format as the batch approach
    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    y_pred = model((y0, dt), coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y1)

    return pred_loss

# Train the first basis function
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{num_basis}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        losses.append(loss)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


model.eval()
with torch.no_grad():
    explained_variance_ratio, *_ = compute_explained_variance(model)
    scores.append(explained_variance_ratio)

# Train the remaining basis functions progressively
for k in range(num_basis - 1):

    # Freeze all existing parameters except the new basis function
    for param in model.parameters():
        param.requires_grad = False

    # Create a new basis function and add it to the model
    new_basis_function = basis_function_factory()
    for param in new_basis_function.parameters():
        param.requires_grad = True

    new_basis_function = new_basis_function.to(device)
    model.basis_functions.basis_functions.append(new_basis_function)

    # Select only the trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    with tqdm.tqdm(range(num_epochs), desc=f"basis {k + 2}/{num_basis}") as tqdm_bar:
        for epoch in tqdm_bar:
            batch = next(dataloader_iter)
            loss = train_step(model, optimizer, batch, loss_function)
            losses.append(loss)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

    model.eval()
    with torch.no_grad():
        explained_variance_ratio, *_ = compute_explained_variance(model)
        scores.append(explained_variance_ratio)

# Plot results

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    dataloader_eval = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader_eval))

    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    idx = torch.argsort(y0, dim=1, descending=False)
    y0 = torch.gather(y0, dim=1, index=idx)
    y1 = torch.gather(y1, dim=1, index=idx)

    # Use the same input format as the batch approach
    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    y_pred = model((y0, dt), coefficients)

    y0 = y0.squeeze(0).cpu().numpy()
    y_pred = y_pred.squeeze(0).cpu().numpy()
    y1 = y1.squeeze(0).cpu().numpy()

    y0_example = y0_example.squeeze(0).cpu().numpy()
    y1_example = y1_example.squeeze(0).cpu().numpy()

    # Plot the results
    fig, ax = plt.subplots()
    # For 2D system, plot x1 vs x2 (phase space)
    ax.plot(y0[:, 0], y0[:, 1], label="True", linewidth=2)
    ax.plot(y_pred[:, 0], y_pred[:, 1], label="Predicted", linewidth=2, linestyle='--')
    ax.scatter(y0_example[:, 0], y0_example[:, 1], label="Data", color="red", alpha=0.7)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Phase Space Trajectory: x1 vs x2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    # Visualize individual basis functions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Create a proper 2D grid for phase space visualization
    # We'll create a grid of initial conditions and show how each basis function
    # transforms them over a short time step
    x1_range = torch.linspace(-2, 2, 20).to(device)
    x2_range = torch.linspace(-2, 2, 20).to(device)
    dt_plot = torch.tensor(0.1).to(device)
    
    for i, basis_fn in enumerate(model.basis_functions.basis_functions):
        if i >= num_basis or i >= len(axes):
            break
            
        # Create a grid of initial conditions
        x1_grid, x2_grid = torch.meshgrid(x1_range, x2_range, indexing='ij')
        x1_flat = x1_grid.flatten().unsqueeze(0).unsqueeze(-1)  # [1, 400, 1]
        x2_flat = x2_grid.flatten().unsqueeze(0).unsqueeze(-1)  # [1, 400, 1]
        
        # Combine x1 and x2 into initial state y0: [1, 400, 2]
        y0_plot = torch.cat([x1_flat, x2_flat], dim=-1)
        
        # Get basis function output - NeuralODE expects (y0, dt) as separate inputs
        basis_output = basis_fn((y0_plot, dt_plot))  # [1, 400, 2]
        
        # Reshape back to grid for visualization
        output_x1 = basis_output[0, :, 0].reshape(20, 20).detach().cpu().numpy()
        output_x2 = basis_output[0, :, 1].reshape(20, 20).detach().cpu().numpy()
        
        # Plot as a quiver plot showing the vector field
        axes[i].quiver(x1_grid.cpu().numpy(), x2_grid.cpu().numpy(), 
                      output_x1, output_x2, alpha=0.6)
        axes[i].set_xlabel("x1")
        axes[i].set_ylabel("x2")
        axes[i].set_title(f"Basis Function {i+1}")
        axes[i].set_xlim(-2, 2)
        axes[i].set_ylim(-2, 2)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Plot loss and explained variance
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot loss
    ax1.plot(losses)
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    ax1.set_yscale("log")

    # Plot explained variance ratio
    for i in range(len(scores)):
        scores[i] = scores[i].cpu().numpy()
        ax2.plot(
            range(1, len(scores[i]) + 1),
            scores[i],
            marker="o",
            label=f"k = {i + 1}",
        )
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True)

    # Plot the eigenvalues of the coefficients
    _, eigenvalues, gram_eigenvalues = compute_explained_variance(model)
    eigenvalues = eigenvalues.cpu().numpy()
    gram_eigenvalues = gram_eigenvalues.cpu().numpy()

    ax3.plot(
        range(1, len(eigenvalues) + 1),
        eigenvalues,
        marker="o",
        label="Covariance Matrix",
    )
    ax3.plot(
        range(1, len(gram_eigenvalues) + 1),
        gram_eigenvalues,
        marker="o",
        label="Gram Matrix",
    )
    ax3.set_xlabel("Eigenvalue Index")
    ax3.set_ylabel("Eigenvalue")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
