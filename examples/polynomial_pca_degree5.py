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

# Load dataset - DEGREE 5
dataset = PolynomialDataset(n_points=100, n_example_points=100, degree=5)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
def basis_function_factory():
    return MLP(layer_sizes=[1, 32, 1])


num_basis = 10
# Only use one basis function initially for progressive training
basis_functions = BasisFunctions(basis_function_factory())

model = FunctionEncoder(basis_functions).to(device)

# Train model

losses = []  # For plotting.
scores = []  # For plotting.
dataloader_coeffs = DataLoader(dataset, batch_size=100)
dataloader_coeffs_iter = iter(dataloader_coeffs)


def compute_explained_variance(model):
    X, y, X_example, y_example = next(dataloader_coeffs_iter)
    X, y, X_example, y_example = X.to(device), y.to(device), X_example.to(device), y_example.to(device)

    coefficients, G = model.compute_coefficients(X_example, y_example)

    # Compute covariance matrix of coefficients
    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = torch.matmul(coefficients_centered.T, coefficients_centered) / coefficients.shape[0]

    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order

    # Compute explained variance from Gram matrix eigenvalues
    K = G.mean(dim=0)
    gram_eigenvalues, _ = torch.linalg.eigh(K)
    gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order

    explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

    return explained_variance_ratio, eigenvalues, gram_eigenvalues


def loss_function(model, batch):
    X, y, X_example, y_example = batch
    X, y, X_example, y_example = X.to(device), y.to(device), X_example.to(device), y_example.to(device)

    coefficients, _ = model.compute_coefficients(X_example, y_example)
    pred = model(X, coefficients=coefficients)

    pred_loss = torch.nn.functional.mse_loss(pred, y)

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

# Generate data for visualization and saving
model.eval()
with torch.no_grad():
    # Generate a batch for visualization
    dataloader_eval = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader_eval))

    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    # Sort X for plotting
    idx = torch.argsort(X, dim=1, descending=False)
    X_sorted = torch.gather(X, dim=1, index=idx)
    y_sorted = torch.gather(y, dim=1, index=idx)

    # Compute coefficients
    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X_sorted, coefficients)

    # Get final eigenvalues for saving
    explained_variance_ratio, eigenvalues, gram_eigenvalues = compute_explained_variance(model)

    # Convert to numpy for visualization
    X_sorted_np = X_sorted.squeeze(0).cpu().numpy()
    y_sorted_np = y_sorted.squeeze(0).cpu().numpy()
    y_pred_np = y_pred.squeeze(0).cpu().numpy()
    X_example_np = example_X.squeeze(0).cpu().numpy()
    y_example_np = example_y.squeeze(0).cpu().numpy()

    # Create visualization data
    viz_data = create_visualization_data_polynomial(
        X_sorted=X_sorted_np,
        y_sorted=y_sorted_np,
        y_pred=y_pred_np,
        example_X=X_example_np,
        example_y=y_example_np
    )

    # Save experiment data
    saver = ExperimentSaver()

    experiment_data = saver.prepare_progressive_data(
        problem_type="polynomial",
        num_basis=num_basis,
        losses=losses,
        scores=scores,
        eigenvalues=eigenvalues,
        gram_eigenvalues=gram_eigenvalues,
        visualization_data=viz_data,
        dataset_params={
            "name": "polynomial_d5",
            "degree": 5,
            "n_points": 100,
            "n_example_points": 100
        },
        training_params={
            "num_epochs": num_epochs,
            "learning_rate": 1e-3,
            "batch_size": 50
        }
    )

    saver.save_experiment("polynomial", "progressive", experiment_data, dataset_name="polynomial_d5")

    print(f"Training completed with {len(model.basis_functions.basis_functions)} basis functions")
    print(f"Final explained variance ratios: {scores[-1][:5].cpu().numpy()}")  # Show first 5