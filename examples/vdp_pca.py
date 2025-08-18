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
from my_datasets.van_der_pol import VanDerPolDataset, van_der_pol

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
    
    # Compute explained variance ratio - only use the actual eigenvalues, not the padding
    actual_eigenvalues = eigenvalues[:n_basis_current]
    explained_variance_ratio = actual_eigenvalues / (torch.sum(actual_eigenvalues) + 1e-8)
    
    # Pad explained variance ratio to match expected number of basis functions
    if len(explained_variance_ratio) < num_basis:
        padding = torch.zeros(num_basis - len(explained_variance_ratio), device=explained_variance_ratio.device)
        explained_variance_ratio = torch.cat([explained_variance_ratio, padding])

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
basis_losses = []  # Track losses for each basis function separately

with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{num_basis}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        basis_losses.append(loss)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

# Store the final loss for this basis function
losses.append(basis_losses[-1])

model.eval()
with torch.no_grad():
    explained_variance_ratio, *_ = compute_explained_variance(model)
    # print(f"After training basis 1: explained_variance_ratio shape: {explained_variance_ratio.shape}, values: {explained_variance_ratio}")
    scores.append(explained_variance_ratio)

# Train the remaining basis functions progressively
for k in range(num_basis - 1):

    # Freeze all existing parameters except the new basis function
    for param in model.parameters():    
        param.requires_grad = False

    # Create a new basis function and add it to the model
    new_basis_function = basis_function_factory().to(device)
    for param in new_basis_function.parameters():
        param.requires_grad = True

    model.basis_functions.basis_functions.append(new_basis_function)

    # Select only the trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    basis_losses = []  # Reset for each new basis function
    with tqdm.tqdm(range(num_epochs), desc=f"basis {k + 2}/{num_basis}") as tqdm_bar:
        for epoch in tqdm_bar:
            batch = next(dataloader_iter)
            loss = train_step(model, optimizer, batch, loss_function)
            basis_losses.append(loss)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})
    
    # Store the final loss for this basis function
    losses.append(basis_losses[-1])

    model.eval()
    with torch.no_grad():
        explained_variance_ratio, *_ = compute_explained_variance(model)
        # print(f"After training basis {k + 2}: explained_variance_ratio shape: {explained_variance_ratio.shape}, values: {explained_variance_ratio}")
        scores.append(explained_variance_ratio)

# Plot results

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    dataloader_eval = DataLoader(dataset, batch_size=9)
    batch = next(iter(dataloader_eval))

    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    mu = mu.to(device)
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    # Use the same input format as the batch approach
    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
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

    # Plot training progress and analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training loss progression across basis functions
    axes[0].plot(range(1, len(losses) + 1), losses, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Basis Function Number')
    axes[0].set_ylabel('Final MSE Loss')
    axes[0].set_yscale('log')
    axes[0].set_title('Training Loss Progression\n(Each point = final loss after training a basis function)')
    axes[0].grid(True, alpha=0.3)
    
    # Add step lines to emphasize the progression
    for i in range(len(losses) - 1):
        axes[0].plot([i+1, i+2], [losses[i], losses[i+1]], 'r--', alpha=0.7, linewidth=1)
    
    # Plot 2: Explained variance ratio for different k values
    # print(f"Number of scores: {len(scores)}")
    # print(f"Scores shapes: {[s.shape for s in scores]}")
    if len(scores) > 0:
        scores_tensor = torch.stack(scores)  # Convert list of tensors to tensor
        # print(f"Scores tensor shape: {scores_tensor.shape}")
        
        # Plot each progressive training stage
        for k in range(min(len(scores), num_basis)):
            # Only plot the actual basis functions that exist at this stage
            n_basis_at_stage = k + 1
            axes[1].plot(range(1, n_basis_at_stage + 1), 
                         scores_tensor[k][:n_basis_at_stage].cpu(), 
                         marker='o', label=f'k = {k+1}', linewidth=2, markersize=8)
        
        # Set x-axis to show only the relevant range
        axes[1].set_xlim(0.5, min(len(scores) + 0.5, num_basis + 0.5))
    else:
        axes[1].text(0.5, 0.5, 'No scores available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('No Data Available')
    axes[1].set_xlabel('Eigenvalue Index')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_yscale('log')
    axes[1].set_title(f'Explained Variance Ratio vs. Eigenvalue Index\n(Progressive Training: {len(scores)} basis functions trained)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Eigenvalues comparison
    with torch.no_grad():
        explained_variance_ratio, eigenvalues, gram_eigenvalues = compute_explained_variance(model)
        # print(f"Final eigenvalues shape: {eigenvalues.shape}, values: {eigenvalues}")
        # print(f"Final gram_eigenvalues shape: {gram_eigenvalues.shape}, values: {gram_eigenvalues}")
        
        # Only plot the actual eigenvalues that exist (not the padding)
        n_basis_current = len(model.basis_functions.basis_functions)
        actual_eigenvalues = eigenvalues[:n_basis_current]
        actual_gram_eigenvalues = gram_eigenvalues[:n_basis_current]
        
        axes[2].plot(range(1, n_basis_current + 1), actual_eigenvalues.cpu(), 
                     marker='o', label='Covariance Matrix', linewidth=2, markersize=8)
        axes[2].plot(range(1, n_basis_current + 1), actual_gram_eigenvalues.cpu(), 
                     marker='s', label='Gram Matrix', linewidth=2, markersize=8)
        
        # Set x-axis to show only the relevant range
        axes[2].set_xlim(0.5, n_basis_current + 0.5)
        axes[2].set_xlabel('Eigenvalue Index')
        axes[2].set_ylabel('Eigenvalue')
        axes[2].set_yscale('log')
        axes[2].set_title('Eigenvalue vs. Eigenvalue Index')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Plot individual basis functions visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Visualize each basis function
    for i in range(min(num_basis, len(axes))):
        if i < len(model.basis_functions.basis_functions):
            basis_func = model.basis_functions.basis_functions[i]
            
            # Create test input
            test_y0 = torch.tensor([[1.0, 0.0]], device=device)  # Example initial condition
            test_dt = torch.tensor([[0.1]], device=device)
            
            # Get basis function output
            with torch.no_grad():
                output = basis_func((test_y0, test_dt))
            
            # Plot the output (assuming 2D output)
            if output.shape[-1] == 2:
                axes[i].scatter(output[0, :, 0].cpu(), output[0, :, 1].cpu(), alpha=0.6)
                axes[i].set_title(f'Basis Function {i+1}')
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('y')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'Basis {i+1}\n{output.shape}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Basis Function {i+1}')
    
    # Hide unused subplots
    for i in range(num_basis, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


    # Plot the results
    # Visualize individual basis functions
    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
