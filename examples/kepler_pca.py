import torch
from torch.utils.data import DataLoader

from my_datasets.kepler import KeplerDataset, kepler

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step
from function_encoder.utils.experiment_saver import ExperimentSaver, create_visualization_data_dynamics

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

# Load dataset

dataset = KeplerDataset(
    integrator=rk4_step,
    n_points=1000,
    n_example_points=100,
    dt_range=(0.1, 0.1),
    device=torch.device(device),
)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model


def basis_function_factory():
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[5, 128, 128, 4])),
        integrator=rk4_step,
    )


num_basis = 10

# Start with one basis function for progressive training
basis_functions = BasisFunctions(basis_function_factory())

model = FunctionEncoder(basis_functions).to(device)

# Train model

losses = []  # For plotting
scores = []  # For plotting
dataloader_coeffs = DataLoader(dataset, batch_size=100)
dataloader_coeffs_iter = iter(dataloader_coeffs)


def compute_explained_variance(model):
    _, _, _, _, example_y0, example_dt, example_y1 = next(dataloader_coeffs_iter)
    # Data is already on the correct device from the dataset

    coefficients, G = model.compute_coefficients((example_y0, example_dt), example_y1)

    # Compute covariance matrix of coefficients (like in polynomial_pca)
    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = (
        torch.matmul(coefficients_centered.T, coefficients_centered)
        / coefficients.shape[0] 
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order

    # Compute explained variance from Gram matrix eigenvalues
    K = G.mean(dim=0)
    gram_eigenvalues, _ = torch.linalg.eigh(K)
    gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order

    explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

    return explained_variance_ratio, eigenvalues, gram_eigenvalues


def loss_function(model, batch):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    # Data is already on the correct device from the dataset

    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt), coefficients=coefficients)

    pred_loss = torch.nn.functional.mse_loss(pred, y1)

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

    # Create a new basis function and add it to the model
    new_basis_function = basis_function_factory()
    new_basis_function = new_basis_function.to(device)
    model.basis_functions.basis_functions.append(new_basis_function)

    # Freeze all existing basis function parameters except the new one
    for i, basis_func in enumerate(model.basis_functions.basis_functions):
        if i < len(model.basis_functions.basis_functions) - 1:  # Freeze all except the last (newest)
            for param in basis_func.parameters():
                param.requires_grad = False
        else:  # Keep the newest basis function trainable
            for param in basis_func.parameters():
                param.requires_grad = True

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
    # Generate a batch for visualization
    dataloader_eval = DataLoader(dataset, batch_size=9)
    batch = next(iter(dataloader_eval))

    M_central, y0, dt, y1, y0_example, dt_example, y1_example = batch
    # Data is already on the correct device from the dataset

    # Compute coefficients
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)

    # Plot 1: MSE Loss and Eigenvalue Analysis (like polynomial_pca.py)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot loss
    ax1.plot(losses)
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("Training Step")
    ax1.grid(True)
    ax1.set_yscale("log")
    ax1.set_title("Training Loss")

    # Plot explained variance ratio progression
    for i in range(len(scores)):
        scores_np = scores[i].cpu().numpy()
        ax2.plot(
            range(1, len(scores_np) + 1),
            scores_np,
            marker="o",
            label=f"k = {i + 1}",
        )
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True)
    ax2.set_title("Explained Variance Progression")

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
        marker="s",
        label="Gram Matrix",
    )
    ax3.set_xlabel("Eigenvalue Index")
    ax3.set_ylabel("Eigenvalue")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True)
    ax3.set_title("Eigenvalue Comparison")

    plt.tight_layout()
    plt.show()

    # Plot 2: Orbital Dynamics
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Plot orbital trajectories in all subplots
    trajectory_axes = axes.flatten()

    for plot_idx, ax_traj in enumerate(trajectory_axes):
        if plot_idx >= min(8, len(M_central)):
            ax_traj.set_visible(False)
            continue

        i = plot_idx
        _M_central = M_central[i]

        # Generate initial conditions using orbital parameters
        from my_datasets.kepler import generate_kepler_states_batch

        _y0 = generate_kepler_states_batch(
            _M_central.item(),
            dataset.a_range,
            dataset.e_range,
            1,
            device=torch.device(device),
        )

        _c = coefficients[i].unsqueeze(0)
        s = 0.1  # Time step for simulation
        n = int(10.0 / s)  # Simulate for 2 time units
        _dt = torch.tensor([s], device=device)

        # Integrate the true trajectory
        x = _y0.clone()
        y_true = [x]
        for k in range(n):
            x = rk4_step(kepler, x, _dt, M_central=_M_central) + x
            y_true.append(x)
        y_true = torch.cat(y_true, dim=0)
        y_true = y_true.detach().cpu().numpy()

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

        # Plot trajectories
        ax_traj.plot(y_true[:, 0], y_true[:, 1], "b-", alpha=0.8, linewidth=1.5, label="True")
        ax_traj.plot(pred[0, :, 0], pred[0, :, 1], "r--", alpha=0.9, linewidth=2, label="Predicted")

        # Mark initial positions and central body
        ax_traj.plot(y_true[0, 0], y_true[0, 1], "go", markersize=4)
        ax_traj.plot(0, 0, "ko", markersize=6)

        ax_traj.set_xlim(-4, 4)
        ax_traj.set_ylim(-4, 4)
        ax_traj.set_aspect("equal")
        ax_traj.set_xlabel("X Position")
        ax_traj.set_ylabel("Y Position")
        ax_traj.set_title(f"M={_M_central.item():.2f}")
        ax_traj.grid(True, alpha=0.3)

    # Add overall legend for the dynamics plot
    fig.legend(
        handles=[plt.Line2D([0], [0], color='b', linewidth=1.5, label='True'),
                plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Predicted'),
                plt.Line2D([0], [0], color='g', marker='o', linestyle='None', markersize=4, label='Start'),
                plt.Line2D([0], [0], color='k', marker='o', linestyle='None', markersize=6, label='Central Body')],
        labels=['True', 'Predicted', 'Start', 'Central Body'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=4,
        frameon=False,
    )

    plt.suptitle("Kepler Problem: Orbital Dynamics", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), "kepler_pca_model.pth")

# Save experiment data
saver = ExperimentSaver()

# Prepare trajectory data for saving
trajectories_true = []
trajectories_pred = []
initial_conditions = []
system_params = []

# Regenerate data for visualization following the plotting code above
for plot_idx in range(min(8, len(M_central))):
    i = plot_idx
    _M_central = M_central[i]

    # Generate initial conditions
    from my_datasets.kepler import generate_kepler_states_batch
    _y0 = generate_kepler_states_batch(
        _M_central.item(),
        dataset.a_range,
        dataset.e_range,
        1,
        device=torch.device(device),
    )

    _c = coefficients[i].unsqueeze(0)
    s = 0.1
    n = int(10 / s)
    _dt = torch.tensor([s], device=device)

    # True trajectory
    x = _y0.clone()
    y_true = [x]
    for k in range(n):
        x = rk4_step(kepler, x, _dt, M_central=_M_central) + x
        y_true.append(x)
    y_true_traj = torch.cat(y_true, dim=0).detach().cpu().numpy()

    # Predicted trajectory
    x = _y0.clone().unsqueeze(1)
    _dt = _dt.unsqueeze(0)
    pred = [x]
    for k in range(n):
        x = model((x, _dt), coefficients=_c) + x
        pred.append(x)
    pred_traj = torch.cat(pred, dim=1)[0].detach().cpu().numpy()

    trajectories_true.append(y_true_traj)
    trajectories_pred.append(pred_traj)
    initial_conditions.append(_y0[0].cpu().numpy())
    system_params.append(_M_central.item())

viz_data = create_visualization_data_dynamics(
    trajectories_true=trajectories_true,
    trajectories_pred=trajectories_pred,
    initial_conditions=initial_conditions,
    system_params=system_params
)

# Prepare and save experiment data
experiment_data = saver.prepare_progressive_data(
    problem_type="kepler",
    num_basis=num_basis,
    losses=losses,
    scores=scores,
    eigenvalues=eigenvalues,
    gram_eigenvalues=gram_eigenvalues,
    visualization_data=viz_data,
    dataset_params={
        "name": "kepler_dt01",
        "n_points": 1000,
        "n_example_points": 100,
        "dt_range": (0.1, 0.1)
    },
    training_params={
        "num_epochs": num_epochs,
        "learning_rate": 1e-3,
        "batch_size": 50
    }
)

saver.save_experiment("kepler", "progressive", experiment_data, dataset_name="dt01")

print(
    f"Training completed with {len(model.basis_functions.basis_functions)} basis functions"
)
print(
    f"Final explained variance ratios: {scores[-1][:5].cpu().numpy()}"
)  # Show first 5
