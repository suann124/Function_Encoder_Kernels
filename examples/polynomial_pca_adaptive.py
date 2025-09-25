import torch

from torch.utils.data import DataLoader
from my_datasets.polynomial import PolynomialDataset

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import train_step
from function_encoder.utils.experiment_saver import ExperimentSaver, create_visualization_data_polynomial


import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(42)

# Load dataset
dataset = PolynomialDataset(n_points=100, n_example_points=100, degree=3)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
def basis_function_factory():
    return MLP(layer_sizes=[1, 32, 1])


num_basis = 15  # Maximum number of basis functions to try
# Only use one basis function initially for progressive training
basis_functions = BasisFunctions(basis_function_factory())

model = FunctionEncoder(basis_functions).to(device)

# Train model

losses = []  # For plotting.
scores = []  # For plotting.
dataloader_coeffs = DataLoader(dataset, batch_size=100)
dataloader_coeffs_iter = iter(dataloader_coeffs)


def compute_explained_variance(model):
    _, _, example_X, example_y = next(dataloader_coeffs_iter)
    example_X = example_X.to(device)
    example_y = example_y.to(device)
    coefficients, G = model.compute_coefficients(example_X, example_y)

    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = (
        torch.matmul(coefficients_centered.T, coefficients_centered)
        / coefficients.shape[0]
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order

    explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

    # eigenvectors = eigenvectors.flip(1)  # Flip to descending order
    # fpc_scores = torch.matmul(coefficients_centered, eigenvectors)

    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(G.mean(dim=0))
    gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order

    return explained_variance_ratio, eigenvalues, gram_eigenvalues


def loss_function(model, batch):
    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y)

    return pred_loss


def find_optimal_basis_count(scores, threshold=0.01):
    """Find optimal number of basis functions based on explained variance ratio plateau."""
    if len(scores) < 2:
        return len(scores)

    for i in range(1, len(scores)):
        # Check if the first eigenvalue (most important) has plateaued
        current_first_ev = scores[i][0].item()
        previous_first_ev = scores[i-1][0].item()

        # If improvement is less than threshold, we've reached plateau
        improvement = (current_first_ev - previous_first_ev) / previous_first_ev
        if improvement < threshold:
            return i  # Return previous index (before plateau)

    return len(scores)  # Use all if no plateau detected


# Train the first basis function
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{num_basis}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        losses.append(loss)  # Only the final loss
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


model.eval()
with torch.no_grad():
    explained_variance_ratio, *_ = compute_explained_variance(model)
    scores.append(explained_variance_ratio)

# Train the remaining basis functions progressively with early stopping
optimal_basis_count = None
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

    # Check if we should stop training (early stopping)
    if optimal_basis_count is None:
        current_optimal = find_optimal_basis_count(scores)
        if current_optimal < len(scores):  # Plateau detected
            optimal_basis_count = current_optimal
            print(f"Early stopping: Optimal basis count detected as {optimal_basis_count}")
            break

# If no early stopping occurred, use all trained basis functions
if optimal_basis_count is None:
    optimal_basis_count = len(scores)

print(f"Using {optimal_basis_count} basis functions for final prediction")

# Create optimal model for prediction
def create_optimal_model(full_model, num_basis_to_use):
    """Create a model with only the first num_basis_to_use basis functions."""
    optimal_basis_functions = BasisFunctions()
    for i in range(num_basis_to_use):
        optimal_basis_functions.basis_functions.append(full_model.basis_functions.basis_functions[i])

    optimal_model = FunctionEncoder(optimal_basis_functions).to(device)
    return optimal_model

optimal_model = create_optimal_model(model, optimal_basis_count)

# Plot results

import matplotlib.pyplot as plt

# Publication formatting
plt.rcParams.update({
    'font.size': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'lines.markersize': 3,
    'legend.fontsize': 8,
    'legend.handlelength': 1.0,
    'legend.handletextpad': 0.3,
    'legend.columnspacing': 0.5
})

model.eval()
optimal_model.eval()
with torch.no_grad():
    dataloader_eval = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader_eval))

    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    idx = torch.argsort(X, dim=1, descending=False)
    X = torch.gather(X, dim=1, index=idx)
    y = torch.gather(y, dim=1, index=idx)

    # Full model prediction
    coefficients_full, _ = model.compute_coefficients(example_X, example_y)
    y_pred_full = model(X, coefficients_full)

    # Optimal model prediction
    coefficients_opt, _ = optimal_model.compute_coefficients(example_X, example_y)
    y_pred_opt = optimal_model(X, coefficients_opt)

    X = X.squeeze(0).cpu().numpy()
    y_pred_full = y_pred_full.squeeze(0).cpu().numpy()
    y_pred_opt = y_pred_opt.squeeze(0).cpu().numpy()
    y = y.squeeze(0).cpu().numpy()

    example_X = example_X.squeeze(0).cpu().numpy()
    example_y = example_y.squeeze(0).cpu().numpy()

    # Plot the results comparing optimal vs full model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))

    # Full model
    ax1.plot(X, y, label="True", linewidth=2)
    ax1.plot(X, y_pred_full, label="Predicted (Full)", linestyle='--')
    ax1.scatter(example_X, example_y, label="Data", color="red", s=20)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Full Model ({len(model.basis_functions.basis_functions)} basis)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Optimal model
    ax2.plot(X, y, label="True", linewidth=2)
    ax2.plot(X, y_pred_opt, label="Predicted (Optimal)", linestyle='--')
    ax2.scatter(example_X, example_y, label="Data", color="red", s=20)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Optimal Model ({optimal_basis_count} basis)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_outputs/poly_adaptive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualize optimal basis functions
    fig, axes = plt.subplots(2, int((optimal_basis_count + 1) // 2), figsize=(8, 3))
    if optimal_basis_count == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    X_plot = torch.linspace(-1, 1, 100).unsqueeze(1).unsqueeze(0).to(device)
    for i in range(optimal_basis_count):
        if i >= len(axes):
            break
        basis_fn = optimal_model.basis_functions.basis_functions[i]
        basis_output = basis_fn(X_plot)
        axes[i].plot(X_plot[0].cpu().numpy(), basis_output[0].detach().cpu().numpy())
        axes[i].set_title(f"φ{i+1}")
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('φ(x)')
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(optimal_basis_count, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('plot_outputs/poly_adaptive_basis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot loss and explained variance with optimal point marked
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    # Plot loss
    ax1.plot(losses)
    ax1.axvline(x=optimal_basis_count * num_epochs, color='red', linestyle='--',
                label=f'Optimal ({optimal_basis_count} basis)')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    ax1.set_yscale("log")
    ax1.legend()

    # Plot explained variance ratio
    for i in range(len(scores)):
        scores[i] = scores[i].cpu().numpy()
        ax2.plot(
            range(1, len(scores[i]) + 1),
            scores[i],
            marker="o",
            markersize=3,
            label=f"k = {i + 1}",
        )
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_yscale("log")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)

    # Plot the eigenvalues of the coefficients
    _, eigenvalues, gram_eigenvalues = compute_explained_variance(optimal_model)
    eigenvalues = eigenvalues.cpu().numpy()
    gram_eigenvalues = gram_eigenvalues.cpu().numpy()

    ax3.plot(
        range(1, len(eigenvalues) + 1),
        eigenvalues,
        marker="o",
        markersize=3,
        label="Optimal Model",
    )
    ax3.set_xlabel("Eigenvalue Index")
    ax3.set_ylabel("Eigenvalue")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('plot_outputs/poly_adaptive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Print summary
print(f"\nSummary:")
print(f"- Total basis functions trained: {len(scores)}")
print(f"- Optimal basis functions: {optimal_basis_count}")
print(f"- Efficiency gain: {(len(scores) - optimal_basis_count) / len(scores) * 100:.1f}% reduction")

# Compute MSE for both models
with torch.no_grad():
    mse_full = torch.nn.functional.mse_loss(torch.tensor(y_pred_full), torch.tensor(y)).item()
    mse_opt = torch.nn.functional.mse_loss(torch.tensor(y_pred_opt), torch.tensor(y)).item()
    print(f"- Full model MSE: {mse_full:.6f}")
    print(f"- Optimal model MSE: {mse_opt:.6f}")
    print(f"- Performance retention: {(1 - (mse_opt - mse_full) / mse_full) * 100:.1f}%")