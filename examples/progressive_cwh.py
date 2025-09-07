"""
Progressive Basis Functions Training for Van der Pol Oscillator
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from my_datasets.cwh import CWHDataset, cwh, cwh_torch, CWHParams

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
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    print(f"Using GPU {gpu_id}: {gpu_name}")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Metal (MPS) backend")
else:
    device = "cpu"
    print("Using CPU")

torch.manual_seed(42)

# Load dataset
dataset = CWHDataset(n_points=1000, n_example_points=100)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
def basis_function_factory():
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[7, 64, 64, 6])),  # 7 = 6 state + 1 time
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
    orbital_rate, _, _, _, y0_example, dt_example, y1_example = next(dataloader_coeffs_iter)
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
    orbital_rate, y0, dt, y1, y0_example, dt_example, y1_example = batch
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
    # add state back in mse loss

@torch.no_grad()
def prediction_subset(
    model,
    y0,                  # (1,1000,6)
    dt,                  # (1,1000)
    coeffs=None,
    n_steps: int = 25,
    select_idx=None,
    model_outputs_derivative: bool = True,
):
    device = y0.device
    # flatten y0: (1,1000,6) -> (1000,6)
    B = y0.shape[1]
    x_pred = y0.reshape(B, 6)
    dt_step = dt.reshape(B, 1)    # reshape dt: (1,1000) -> (1000,1)
    # pick indices to store
    if select_idx is None:
        M = min(3, B)
        select_idx = torch.arange(M, device=device)
    else:
        select_idx = torch.as_tensor(select_idx, device=device, dtype=torch.long)

    traj_sel = [x_pred.index_select(0, select_idx)]
    model.eval()

    for _ in range(n_steps):
        x_in  = x_pred.unsqueeze(1)                 # (1000,1,6)
        dt_in = dt_step                             # (1000,1)

        out = model((x_in, dt_in), coefficients=coeffs)  # (1000,1,6) or (1000,6)
        dx  = out[:,0,:] if out.ndim == 3 else out       # (1000,6)

        step = dx * dt_step if model_outputs_derivative else dx
        x_pred = x_pred + step                           # (1000,6)

        traj_sel.append(x_pred.index_select(0, select_idx))

    # -> (M, n_steps+1, 6)
    y_plot = torch.stack(traj_sel, dim=1).detach().cpu().numpy()
    which  = select_idx.detach().cpu().tolist()
    return y_plot, which

# ========================== Training ==============================

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

# =====================Plot results ==========================
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    dataloader_eval = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader_eval))

    orbital_rate, y0, dt, y1, y0_example, dt_example, y1_example = batch
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
    params = CWHParams(orbital_rate=orbital_rate[0].item())  # Use orbital rate from batch

    _c = coefficients[0:1]  # Use first set of coefficients
    
    s = 20.0          # Time step for simulation
    n = 25           # Number of steps

    # --- Integrate the true trajectory ---
    x_true = y0[0,0].clone().cpu()      # shape (6,)
    y_true = [x_true.numpy()]
    for k in range(n):
        x_np = x_true.detach().cpu().numpy()  
        u = np.zeros(3) 

        # Manual RK4 with numpy cwh
        k1 = cwh(x_np, u, params)
        k2 = cwh(x_np + s*k1/2, u, params)
        k3 = cwh(x_np + s*k2/2, u, params)
        k4 = cwh(x_np + s*k3, u, params)
        
        dx = s * (k1 + 2*k2 + 2*k3 + k4) / 6
        # dx = rk4_step(cwh, x_true, s, u=u, params=params)
        x_true = x_true + torch.from_numpy(dx)
        y_true.append(x_true.numpy())
    # Stack along the time dimension -> shape (n+1, 6)
    y_true = np.stack(y_true, axis=0)

    # --- Integrate the predicted trajectory ---
    x_pred = y0[0,0].clone()      # shape (6,)
    y_pred = [x_pred]
    _dt = torch.tensor([s], device=device)  # shape (1,)

    for k in range(n):
        # model expects batch dimensions: (batch, state_dim) and (batch, 1)
        x_in = x_pred.unsqueeze(0).unsqueeze(0)        # shape (1,1, 6)
        dt_in = _dt.unsqueeze(0)         #[1,1]
        dx_pred = model((x_in, dt_in), coefficients=_c)[0]
        x_pred = x_pred + dx_pred
        y_pred.append(x_pred.view(-1))
    # Stack along time dimension -> shape (n+1, 6)
    y_pred = torch.stack(y_pred, dim=0).detach().cpu().numpy()

    # y_pred, which = prediction_subset(
    #     model,
    #     y0=y0, 
    #     dt=dt, 
    #     coeffs=_c,
    #     n_steps=n,
    #     select_idx=[0, 3, 7],   # choose any few to visualize; or leave None to take first few
    #     model_outputs_derivative=False,

    # )
    
    # Plot the trajectories
    fig, ax = plt.subplots(figsize=(3, 5.5))
    ax.plot(y_true[:, 0], y_true[:, 1], label="True", linewidth=2, color='blue')
    curve = y_pred[0] #(26,6)
    ax.plot(curve[:, 0], curve[:, 1], label="Predicted", linewidth=2, linestyle='--', color='orange')
    start_state = y0[0, 0]  # Extract the (6,) state vector
    ax.scatter(start_state[0].cpu(), start_state[1].cpu(), s=100, c='green', marker='o', label="Predicted", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CWH System Phase Space Trajectory: x vs y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # FONTS: 8pt, 300dpi, .png, timesnewroman, no titles, shared axes and labels for multiplots, annotations instead of titles for multiplots, if shared legends put outside all plots, fig.legend(loc="outside right upper")
    plt.show()

    # Visualize individual basis functions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Create 6D input grid (we'll visualize in x-y plane, setting other dimensions to 0)
    x1_range = torch.linspace(-2, 2, 20).to(device)
    x2_range = torch.linspace(-2, 2, 20).to(device)
    dt_plot = torch.full((1, 400), 0.1, device=device)  # Make it (1, 400) to match y0_plot
    
    for i, basis_fn in enumerate(model.basis_functions.basis_functions):
        if i >= num_basis or i >= len(axes):
            break
            
        x1_grid, x2_grid = torch.meshgrid(x1_range, x2_range, indexing='xy')
        x1_flat = x1_grid.flatten().unsqueeze(0).unsqueeze(-1)
        x2_flat = x2_grid.flatten().unsqueeze(0).unsqueeze(-1)
        # Create 6D input with zeros for z, x_dot, y_dot, z_dot
        zeros = torch.zeros_like(x1_flat)
        y0_plot = torch.cat([x1_flat, x2_flat, zeros, zeros, zeros, zeros], dim=-1)
        
        basis_output = basis_fn((y0_plot, dt_plot))
        output_x1 = basis_output[0, :, 0].reshape(20, 20).detach().cpu().numpy()
        output_x2 = basis_output[0, :, 1].reshape(20, 20).detach().cpu().numpy()
        
        axes[i].streamplot(x1_grid.cpu().numpy(), x2_grid.cpu().numpy(), 
                   output_x1, output_x2, density=1.5, color='blue')
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
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

with torch.no_grad():
    x0 = y0.reshape(-1,6)[0:1].unsqueeze(1)  # (1,1,6)
    eps = torch.randn_like(x0) * 1e-3
    g0  = model.basis_functions((x0, dt.reshape(-1,1)[0:1]))  # expect (1,1,K,6) or (1,1,6,K)
    g1  = model.basis_functions(((x0+eps), dt.reshape(-1,1)[0:1]))
    print("‖g1-g0‖ =", (g1-g0).norm().item())