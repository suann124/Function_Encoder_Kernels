import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import random

# Add subspace_angles import
from scipy.linalg import subspace_angles

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# Add function_encoder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import train_step

import tqdm

# Parameters
LOSS_THRESHOLD = 0.003
MAX_BASIS_SIZE = 20
TRAIN_EPOCHS = 200
BATCH_SIZE = 128

# Device config
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load dataset
ds = load_dataset("ajthor/derivative_polynomial")
ds = ds.with_format("torch", device=device)
train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=False)
dataloader_iter = iter(train_loader)

# Extract full training data for PCA analysis
X_train = ds["train"]["X"]
f_train = ds["train"]["f"]
X_train = X_train.unsqueeze(-1)
f_train = f_train.unsqueeze(-1)

# Convert to CPU numpy for PCA
X_np = X_train.cpu().numpy()
f_np = f_train.cpu().numpy()

# Train PCA on function values
flat_f = f_np.reshape(f_np.shape[0], -1)
pca = PCA(n_components=MAX_BASIS_SIZE)
pca.fit(flat_f)
pca_components = pca.components_  # shape (n_components, dim)
pca_explained = pca.explained_variance_ratio_

losses = []
cosine_similarities = []

# Create model

def basis_function_factory():
    return MLP(layer_sizes=[1, 32, 1])

# Only use one basis function initially for progressive training
basis_functions = BasisFunctions(basis_function_factory())
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
    # norm_loss = basis_normalization_loss(G)

    return pred_loss  # + norm_loss


# Train the first basis function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(TRAIN_EPOCHS), desc=f"basis 1/{MAX_BASIS_SIZE}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
losses.append(loss)

# Train the remaining basis functions progressively
for k in range(MAX_BASIS_SIZE - 1):

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

    with tqdm.tqdm(range(TRAIN_EPOCHS), desc=f"basis {k + 2}/{MAX_BASIS_SIZE}") as tqdm_bar:
        for epoch in tqdm_bar:
            batch = next(dataloader_iter)
            loss = train_step(model, optimizer, batch, loss_function)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


    # Compare current coefficients with PCA components (cosine similarity)
    learned_basis = normalize(coeffs_np, axis=0)  # (N, num_heads)
    flat_pca = normalize(pca.transform(flat_f), axis=0)
    sim = [np.abs(np.dot(learned_basis[:, i], flat_pca[:, i])) for i in range(num_heads)]
    cosine_similarities.append(sim)

    print(f"Train MSE: {loss:.6f}")

    if loss <= LOSS_THRESHOLD:
        print(f"Reached target loss with {num_heads} basis functions.")
        break

    if num_heads == MAX_BASIS_SIZE:
        print("Reached maximum basis size without reaching loss threshold.")
        break

    input_basis.freeze_heads(list(range(num_heads)))
    input_basis.add_head()
    num_heads += 1

# Plot residual loss
def plot_losses():
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Number of Basis Functions")
    plt.ylabel("Training MSE")
    plt.title("Residual Loss vs Number of Learned Basis Functions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# PCA cumulative variance
def plot_pca_variance():
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca_explained), marker='o', label="PCA")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("PCA Variance Explained")
    plt.grid(True)
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.show()

# ==========================================================================================
# Plot similarity to PCA
def plot_similarity():
    avg_sim = [np.mean(s) for s in cosine_similarities]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(avg_sim)+1), avg_sim, marker='o')
    plt.xlabel("Number of Basis Functions")
    plt.ylabel("Avg Cosine Similarity to PCA Components")
    plt.title("Alignment with Principal Components")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============== Compare PCA eigenfunctions and learned basis =====================
from sklearn.metrics.pairwise import cosine_similarity

# Evaluate learned basis functions at dense grid
x_grid = torch.linspace(-1, 1, 200).unsqueeze(1).to(device)
with torch.no_grad():
    learned_basis_vals = input_encoder.basis_functions(x_grid).cpu().numpy()  # (200, num_heads)

# Interpolate PCA eigenfunctions to x_grid
pca_eigenfunctions = pca.components_.reshape((MAX_BASIS_SIZE, *f_np.shape[1:]))  # (n_components, T, 1)
X_grid_np = x_grid.cpu().numpy().flatten()

interp_pca_vals = np.array([
    np.interp(X_grid_np, X_np[0].flatten(), ef.flatten())
    for ef in pca_eigenfunctions[:num_heads]
])  # (num_heads, 200)

# Normalize
# Ensure the data is 2D before normalization
learned_norm = normalize(learned_basis_vals.T.reshape(num_heads, -1))  # (num_heads, 200)
pca_norm = normalize(interp_pca_vals.reshape(num_heads, -1))           # (num_heads, 200)

# Cosine similarity
sim_basis = cosine_similarity(learned_norm, pca_norm)  # (num_heads, num_heads)

# ========================Plot==============================
plot_losses()
plot_pca_variance()
plot_similarity()

# Plot similarity heatmap
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(sim_basis, annot=True, cmap='viridis', xticklabels=False, yticklabels=False)
plt.xlabel("PCA Eigenfunctions")
plt.ylabel("Learned Basis Functions")
plt.title("Cosine Similarity: Learned Basis vs PCA Eigenfunctions")
plt.tight_layout()
plt.show()


# === Subspace angle comparison ===
print("\n=== Subspace Angle Comparison ===")
# Flatten and normalize both coefficient matrices
learned_subspace = normalize(learned_basis.T)  # (num_heads, N)
pca_subspace = normalize(flat_pca[:, :num_heads].T)  # (num_heads, N)

angles_rad = subspace_angles(learned_subspace.T, pca_subspace.T)
angles_deg = np.degrees(angles_rad)

for i, angle in enumerate(angles_deg):
    print(f"Principal angle {i+1}: {angle:.4f} degrees")



