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
from function_encoder.model.mlp import MLP, MultiHeadedMLP_Freeze
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit

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

# DataLoader
train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=False)

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

# Initialize encoders
input_basis = MultiHeadedMLP_Freeze(layer_sizes=[1, 32, 1], num_heads=1)
input_encoder = FunctionEncoder(input_basis).to(device)

losses = []
cosine_similarities = []
num_heads = 1

# Loss function
def input_loss(model, batch):
    X, y = batch["X"].to(device), batch["f"].to(device)
    X, y = X.unsqueeze(-1), y.unsqueeze(-1)
    coeffs = model.compute_coefficients(X, y)
    y_pred = model(X, coeffs)
    pred_loss = torch.nn.functional.mse_loss(y_pred, y)
    norm_loss = basis_normalization_loss(model.basis_functions(X))
    return pred_loss + norm_loss

# Iteratively train heads
while num_heads <= MAX_BASIS_SIZE:
    print(f"\n=== Iteration with {num_heads} basis functions ===")

    input_encoder = fit(model=input_encoder, ds=train_loader, loss_function=input_loss, epochs=TRAIN_EPOCHS)

    # Compute training loss
    input_encoder.eval()
    with torch.no_grad():
        coeffs = input_encoder.compute_coefficients(X_train.to(device), f_train.to(device))
        coeffs_np = coeffs.cpu().numpy()
        loss = torch.nn.functional.mse_loss(input_encoder(X_train, coeffs), f_train).item()

    losses.append(loss)

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



