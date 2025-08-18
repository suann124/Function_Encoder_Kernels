"""
Progressive Basis Functions Training for Van der Pol Oscillator
Simple version matching polynomial_pca.py structure
"""

import torch
from torch.utils.data import DataLoader
from my_datasets.van_der_pol import VanDerPolDataset, van_der_pol

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import train_step

import tqdm

if torch.cuda.is_available():
    device = "cuda:4"
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

num_basis = 5
# Only use one basis function initially for progressive training
basis_functions = BasisFunctions(basis_function_factory())
model = FunctionEncoder(basis_functions).to(device)

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
    
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)

    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = (
        torch.matmul(coefficients_centered.T, coefficients_centered)
        / coefficients.shape[0]
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order

    explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(G.mean(dim=0))
    gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order

    return explained_variance_ratio, eigenvalues, gram_eigenvalues

def loss_function(model, batch):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)
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

    # Generate and plot an actual trajectory
    
    # Get coefficients for a single function
    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    
    # Generate a trajectory by integration
    _mu = torch.tensor(1.0, device=device)  # Van der Pol parameter
    _y0 = torch.tensor([[-2.0, 0.0]], device=device)  # Initial condition
    _c = coefficients[0:1]  # Use first set of coefficients
    
    s = 0.1  # Time step for simulation
    n = int(10 / s)  # Number of steps for 10 time units
    _dt = torch.tensor([s], device=device)
    
    # Integrate the true trajectory
    x_true = _y0.clone()
    y_true = [x_true]
    for k in range(n):
        x_true = rk4_step(van_der_pol, x_true, _dt, mu=_mu) + x_true
        y_true.append(x_true)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    
    # Integrate the predicted trajectory
    x_pred = _y0.clone().unsqueeze(1)  # Add batch dimension
    _dt_batch = _dt.unsqueeze(0)
    y_pred = [x_pred]
    for k in range(n):
        x_pred = model((x_pred, _dt_batch), coefficients=_c) + x_pred
        y_pred.append(x_pred)
    y_pred = torch.cat(y_pred, dim=1).squeeze(0).detach().cpu().numpy()
    
    # Plot the trajectories
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(y_true[:, 0], y_true[:, 1], label="True", linewidth=2, color='blue')
    ax.plot(y_pred[:, 0], y_pred[:, 1], label="Predicted", linewidth=2, linestyle='--', color='orange')
    ax.scatter(_y0[0, 0].cpu(), _y0[0, 1].cpu(), s=100, c='green', marker='o', label="Start", zorder=5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Phase Space Trajectory: x1 vs x2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.show()

    # Visualize individual basis functions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    x1_range = torch.linspace(-2, 2, 20).to(device)
    x2_range = torch.linspace(-2, 2, 20).to(device)
    dt_plot = torch.full((1, 400), 0.1, device=device)  # Make it (1, 400) to match y0_plot
    
    for i, basis_fn in enumerate(model.basis_functions.basis_functions):
        if i >= num_basis or i >= len(axes):
            break
            
        x1_grid, x2_grid = torch.meshgrid(x1_range, x2_range, indexing='ij')
        x1_flat = x1_grid.flatten().unsqueeze(0).unsqueeze(-1)
        x2_flat = x2_grid.flatten().unsqueeze(0).unsqueeze(-1)
        y0_plot = torch.cat([x1_flat, x2_flat], dim=-1)
        
        basis_output = basis_fn((y0_plot, dt_plot))
        output_x1 = basis_output[0, :, 0].reshape(20, 20).detach().cpu().numpy()
        output_x2 = basis_output[0, :, 1].reshape(20, 20).detach().cpu().numpy()
        
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