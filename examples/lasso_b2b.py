import torch

from torch.utils.data import DataLoader

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.polynomial import PolynomialDataset

from function_encoder.model.mlp import MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import train_step

import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

import tqdm
from tqdm import trange


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(42)

# Load dataset

dataset = PolynomialDataset(n_points=100, n_example_points=10)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model

basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)

model = FunctionEncoder(basis_functions).to(device)

# Train model


def loss_function(model, batch):
    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y)
    norm_loss = basis_normalization_loss(G)

    return pred_loss + norm_loss


num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


# Compute the coefficients for a big batch of data
dataloader = DataLoader(dataset, batch_size=500)
dataloader_iter = iter(dataloader)
batch = next(dataloader_iter)
X, y, example_X, example_y = batch
X = X.to(device)
y = y.to(device)
example_X = example_X.to(device)
example_y = example_y.to(device)
coefficients, G = model.compute_coefficients(example_X, example_y)

coefficients = coefficients.detach().cpu()
# Compute the PCA of the coefficients
pca = PCA(n_components=8)
pca.fit(coefficients.cpu().numpy().reshape(-1, coefficients.size(-1)))
# Compute the angles between the PCA components and the reduced set of basis functions
angles = subspace_angles(pca.components_.T, np.identity(8))

# Print the angles in degrees
print("Angles between PCA components and coefficients (in radians):")
for angle in angles:
    print(f"{angle:.4f} radians, {angle * 180 / 3.141592653589793:.4f} degrees")

# Plot an evaluation of the model

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))

    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    idx = torch.argsort(X, dim=1, descending=False)
    X = torch.gather(X, dim=1, index=idx)
    y = torch.gather(y, dim=1, index=idx)

    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    X = X.squeeze(0).cpu().numpy()
    y_pred = y_pred.squeeze(0).cpu().numpy()
    y = y.squeeze(0).cpu().numpy()

    example_X = example_X.squeeze(0).cpu().numpy()
    example_y = example_y.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(X, y, label="True")
    ax.plot(X, y_pred, label="Predicted")
    ax.scatter(example_X, example_y, label="Data", color="red")
    ax.legend()
    plt.show()
