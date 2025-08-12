import argparse
from enum import Enum, auto

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import islice
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Assuming the project structure from your files
from my_datasets.van_der_pol import VanDerPolDataset, van_der_pol

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step
from function_encoder.coefficients import least_squares, lasso

# --- Configuration ---
class TrainingMode(Enum):
    """Enumeration for the different training modes."""
    CLASSICAL = auto()
    LASSO = auto()
    PROGRESSIVE = auto()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Function Encoder on the Van der Pol dataset.")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=[mode.name.lower() for mode in TrainingMode],
        help="Select the training mode: classical, lasso, or progressive."
    )
    parser.add_argument('--n_basis', type=int, default=10, help="Number of basis functions.")
    parser.add_argument('--num_epochs', type=int, default=2000, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size.")
    args = parser.parse_args()
    args.mode = TrainingMode[args.mode.upper()]
    return args

def basis_function_factory():
    """Creates a single Neural ODE basis function."""
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
        integrator=rk4_step,
    )

def get_device():
    """Gets the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --- Loss Functions ---

def standard_loss(model, batch, device):
    """Standard MSE prediction loss."""
    _, y0, dt, y1, y0_example, dt_example, y1_example = [b.to(device) for b in batch]
    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt), coefficients=coefficients)
    return torch.nn.functional.mse_loss(pred, y1)

def progressive_loss(model, batch, device, ortho_lambda=0.01):
    """MSE loss with an added orthogonality penalty for progressive training."""
    mse_loss = standard_loss(model, batch, device)

    ortho_loss = 0.0
    num_heads = model.basis_functions.num_heads()

    if num_heads > 1:
        # Create a grid of initial conditions to evaluate orthogonality
        y0_grid = torch.linspace(-2, 2, 100, device=device).unsqueeze(-1).repeat(1, 2)
        y0_grid = y0_grid.unsqueeze(0) # Add batch dimension
        dt_grid = torch.full((1, 100, 1), 0.1, device=device)

        # phi shape: [batch, points, features, basis] -> [1, 100, 2, num_heads]
        phi = model.basis_functions((y0_grid, dt_grid))
        # Normalize each basis function's output over the grid
        phi = phi / (torch.norm(phi, p=2, dim=(1, 2), keepdim=True) + 1e-8)

        # Separate the newly added basis function from the frozen ones
        old_phi = phi[..., :-1]
        new_phi = phi[..., -1:]

        # The inner product is a sum over the grid points and features
        # [1, 100, 2, 1] * [1, 100, 2, num_heads-1] -> [1, 100, 2, num_heads-1]
        # Sum over points and features to get the dot product -> [1, num_heads-1]
        cross_gram = torch.sum(new_phi * old_phi, dim=(1, 2))
        ortho_loss = torch.sum(cross_gram**2)

    return mse_loss + ortho_lambda * ortho_loss

# --- Plotting ---

def plot_results(model, dataset, device):
    """Generates and displays a grid of trajectory plots."""
    print("\nGenerating plots...")
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=9)
        batch = next(iter(dataloader))
        mu, y0, _, _, y0_example, dt_example, y1_example = [b.to(device) for b in batch]

        coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)

        fig, ax = plt.subplots(3, 3, figsize=(12, 12))
        ax = ax.flatten()

        for i in range(9):
            _mu = mu[i]
            _y0 = torch.empty(1, 2, device=device).uniform_(*dataset.y0_range)
            _c = coefficients[i].unsqueeze(0)
            
            s = 0.1  # Time step for simulation
            n_steps = int(10 / s)
            _dt = torch.tensor([s], device=device)

            # True trajectory
            true_traj = [_y0.clone()]
            for _ in range(n_steps):
                true_traj.append(rk4_step(van_der_pol, true_traj[-1], _dt, mu=_mu) + true_traj[-1])
            true_traj = torch.cat(true_traj).detach().cpu().numpy()

            # Predicted trajectory
            pred_traj = [_y0.unsqueeze(1).clone()]
            for _ in range(n_steps):
                pred_traj.append(model((pred_traj[-1], _dt.unsqueeze(0)), coefficients=_c) + pred_traj[-1])
            pred_traj = torch.cat(pred_traj, dim=1).squeeze(0).detach().cpu().numpy()

            ax[i].set_xlim(-5, 5)
            ax[i].set_ylim(-5, 5)
            ax[i].plot(true_traj[:, 0], true_traj[:, 1], label="True", color='blue')
            ax[i].plot(pred_traj[:, 0], pred_traj[:, 1], label="Predicted", color='red', linestyle='--')
            ax[i].set_title(f"μ = {_mu.item():.2f}")
            ax[i].grid(True)

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.97))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# --- Main Execution ---

def main():
    args = parse_args()
    device = get_device()
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Running experiment in '{args.mode.name}' mode on '{device}' device.")

    # --- Data ---
    dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
    dataloader = DataLoader(dataset, batch_size=50)
    dataloader_iter = iter(dataloader)    

    # --- Model and Training Strategy ---
    model = None
    optimizer = None

    if args.mode in [TrainingMode.CLASSICAL, TrainingMode.LASSO]:
        print(f"Initializing model with {args.n_basis} basis functions.")
        basis_functions = BasisFunctions(*[basis_function_factory() for _ in range(args.n_basis)])
        
        coeff_method = least_squares
        if args.mode == TrainingMode.LASSO:
            print("Using Lasso for coefficient computation.")
            coeff_method = lasso

        model = FunctionEncoder(basis_functions, coefficients_method=coeff_method).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Standard training loop
        with tqdm.trange(args.num_epochs, desc="Training") as tqdm_bar:
            for epoch in tqdm_bar:
                batch = next(dataloader_iter)
                loss = train_step(model, optimizer, batch, lambda m, b: standard_loss(m, b, device))
                tqdm_bar.set_postfix_str(f"loss: {loss:.2e}")

    elif args.mode == TrainingMode.PROGRESSIVE:
        print("Initializing model for progressive training, starting with 1 basis function.")
        
        # Start with one basis function
        basis_functions = BasisFunctions(basis_function_factory()).to(device)
        model = FunctionEncoder(basis_functions).to(device)

        for i in range(1, args.n_basis + 1):
            # Freeze parameters of all previous basis functions
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze parameters of the last (newest) basis function
            for param in model.basis_functions.basis_functions[-1].parameters():
                param.requires_grad = True

            # Optimizer for only the trainable (new) parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
            
            desc = f"Training Basis {i}/{args.n_basis}"
            with tqdm.trange(args.num_epochs, desc=desc) as pbar:
                for epoch in pbar:
                    for batch in dataloader:
                        loss = train_step(model, optimizer, batch, lambda m, b: progressive_loss(m, b, device))
                    pbar.set_postfix_str(f"loss: {loss:.2e}")
            
            # Add a new basis function for the next round, unless it's the last one
            if i < args.n_basis:
                print(f"Freezing basis {i} and adding basis {i+1}.")
                new_fn = basis_function_factory().to(device)
                model.basis_functions.basis_functions.append(new_fn)

    # --- Evaluation ---
    if model:
        model_path = f"van_der_pol_{args.mode.name.lower()}.pth"
        print(f"\nTraining complete. Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        plot_results(model, dataset, device)

if __name__ == '__main__':
    main()