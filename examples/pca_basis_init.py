import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys
import os

# Add function_encoder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function_encoder.model.mlp import MultiHeadedMLP_Freeze
from function_encoder.function_encoder import FunctionEncoder

# Set seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Helper: Batch evaluation of basis functions ===
def evaluate_basis_batch(encoder, Y_list):
    Y_tensor = torch.stack([y.unsqueeze(-1).to(device) for y in Y_list])  # (N, T, 1)
    N, T, _ = Y_tensor.shape
    Y_flat = Y_tensor.view(-1, 1)  # (N*T, 1)
    with torch.no_grad():
        B_flat = encoder.basis_functions(Y_flat).squeeze()  # (N*T, 4)
    return B_flat.view(N, T, -1)  # (N, T, 4)

# === Load dataset ===
ds = load_dataset("ajthor/derivative_polynomial")
ds = ds.with_format("torch", device=device)

# Get the actual data from the dataset
f_np = ds["train"]["f"].unsqueeze(-1).cpu().numpy()  # Input functions
X_np = ds["train"]["X"].unsqueeze(-1).cpu().numpy()  # Input coordinates
Y_np = ds["train"]["Y"].unsqueeze(-1).cpu().numpy()  # Output coordinates
Tf_np = ds["train"]["Tf"].unsqueeze(-1).cpu().numpy()  # TRUE derivatives (not computed)

Y_train = ds["train"]["Y"]

print(f"Data shapes:")
print(f"f_np: {f_np.shape}")
print(f"X_np: {X_np.shape}")  
print(f"Y_np: {Y_np.shape}")
print(f"Tf_np: {Tf_np.shape}")

# === PCA on function values ===
flat_f = f_np.reshape(f_np.shape[0], -1)
pca = PCA(n_components=4)
pca.fit(flat_f)
pca_components = pca.components_  # (n_components, T)

# === Initialize FunctionEncoder ===
input_basis = MultiHeadedMLP_Freeze(layer_sizes=[1, 32, 1], num_heads=4)
input_encoder = FunctionEncoder(input_basis).to(device)

# === Train heads to match PCA basis ===
def match_heads_to_pca(input_encoder, pca_components, X_np):
    x_tensor = torch.tensor(X_np[0], dtype=torch.float32).unsqueeze(-1).to(device)  # (T, 1)
    targets = torch.tensor(pca_components[:4], dtype=torch.float32).to(device)  # (4, T)
    
    optimizer = torch.optim.Adam(input_encoder.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    input_encoder.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        out = input_encoder.basis_functions(x_tensor).squeeze().permute(1, 0)  # (4, T)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    input_encoder.eval()

match_heads_to_pca(input_encoder, pca_components, X_np)

# === Evaluate and plot basis functions ===
x_tensor = torch.tensor(X_np[0], dtype=torch.float32).unsqueeze(-1).to(device)
with torch.no_grad():
    learned_basis = input_encoder.basis_functions(x_tensor).squeeze().cpu().numpy()  # (T, 4)

print("\nVisualizingasis functions vs PCA components...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i in range(4):
    ax = axes[i]
    ax.plot(X_np[0].flatten(), learned_basis[:, i], label=f"Learned Head {i}", linewidth=2)
    ax.plot(X_np[0].flatten(), pca_components[i], '--', label=f"PCA {i}", alpha=0.7)
    ax.legend()
    ax.set_title(f"Head {i} vs PCA Component")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Evaluate prediction utility of learned basis on training data ===
print("Evaluating prediction performance of learned basis...")

from sklearn.metrics import mean_squared_error

# Compute input coefficients on a single representative input grid
X_tensor_single = torch.tensor(X_np[0], dtype=torch.float32).unsqueeze(-1).to(device)  # (T, 1)
phi = input_encoder.basis_functions(X_tensor_single).squeeze().detach()  # (T, 4)
phi = normalize(phi.cpu().numpy(), axis=0)
phi = torch.tensor(phi, dtype=torch.float32, device=device)

reconstruction_errors = []
for i in range(f_np.shape[0]):
    f_target = torch.tensor(f_np[i].squeeze(), dtype=torch.float32, device=device)
    coeffs = torch.matmul(phi.T, f_target)
    f_recon = torch.matmul(phi, coeffs)
    error = mean_squared_error(f_target.cpu().numpy(), f_recon.cpu().numpy())
    reconstruction_errors.append(error)

avg_error = np.mean(reconstruction_errors)
print(f"Average reconstruction MSE across training set: {avg_error:.6f}")

# === Use operator and output encoder to predict Tf from input f ===
print("Training operator and output encoder...")

from function_encoder.model.mlp import MLP

output_basis = MultiHeadedMLP_Freeze(layer_sizes=[1, 32, 1], num_heads=4)
output_encoder = FunctionEncoder(output_basis).to(device)
operator = MLP(layer_sizes=[4, 32, 4], activation=torch.nn.ReLU()).to(device)

optimizer = torch.optim.Adam(list(output_encoder.parameters()) + list(operator.parameters()), lr=1e-3)
loss_fn = torch.nn.MSELoss()

X_tensor = torch.tensor(X_np[0], dtype=torch.float32).unsqueeze(-1).to(device)  # (T, 1)
Y_tensor = Y_train[0].unsqueeze(-1).to(device)  # (T, 1)
f_tensor = torch.tensor(f_np.squeeze(), dtype=torch.float32).to(device)  # (N, T)

# *** FIX: Use the actual derivative data from the dataset, not computed gradients ***
Tf_tensor = torch.tensor(Tf_np.squeeze(), dtype=torch.float32).to(device)  # (N, T)

print(f"Input f shape: {f_tensor.shape}")
print(f"Target Tf shape: {Tf_tensor.shape}")
print(f"Sample of true derivatives: {Tf_tensor[0][:5]}")

# Train operator and output encoder
print("Fitting operator model on actual derivative data...")
for epoch in range(500):
    output_encoder.train()
    optimizer.zero_grad()

    # Compute input coefficients
    from sklearn.preprocessing import normalize as sk_normalize
    phi = input_encoder.basis_functions(X_tensor).squeeze().detach()  # (T, 4)
    phi_np = sk_normalize(phi.cpu().numpy(), axis=0)
    phi = torch.tensor(phi_np, dtype=torch.float32, device=device)

    coeffs_in = torch.stack([phi.T @ f_tensor[i] for i in range(f_tensor.shape[0])])  # (N, 4)

    # Pass through operator
    coeffs_out = operator(coeffs_in)

    # Decode output in batch (vectorized)
    basis_out_all = evaluate_basis_batch(output_encoder, Y_train)  # (N, T, 4)
    y_pred = torch.einsum('nd,ntd->nt', coeffs_out, basis_out_all)  # (N, T)

    loss = loss_fn(y_pred, Tf_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Operator Loss: {loss.item():.6f}")

# === Visualization ===
print("\nGenerating visualization...")

i = 0  # Use first sample for visualization
f_input = f_tensor[i]
Tf_true = Tf_tensor[i]

with torch.no_grad():
    input_coeff = (phi.T @ f_input).unsqueeze(0)  # (1, 4)
    output_coeff = operator(input_coeff)          # (1, 4)
    basis_out = output_encoder.basis_functions(Y_tensor).squeeze()  # (T, 4)
    Tf_pred = torch.matmul(output_coeff, basis_out.T).squeeze().cpu().numpy()  # (T,)

X_plot = X_tensor.squeeze().cpu().numpy()
f_plot = f_input.cpu().numpy()
Tf_true_plot = Tf_true.cpu().numpy()

fig = plt.figure(figsize=(12, 8))

# Plot 1: Input function
ax1 = fig.add_subplot(211)
ax1.plot(X_plot, f_plot, label="Input Function f", linewidth=2, color='blue')
ax1.scatter(X_plot[::5], f_plot[::5], label="Input Data Points", color="red", s=30, alpha=0.7)
ax1.legend(fontsize=12)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('f(x)', fontsize=14)
ax1.set_title('Input Function', fontsize=16)
ax1.grid(True, alpha=0.3)

# Plot 2: Derivative prediction
ax2 = fig.add_subplot(212)
Y_plot = Y_tensor.squeeze().cpu().numpy()
ax2.plot(Y_plot, Tf_true_plot, label="True Derivative", linewidth=2, color='orange')
ax2.plot(Y_plot, Tf_pred, label="Predicted Derivative", linewidth=2, linestyle='--', color='green')
ax2.legend(fontsize=12)
ax2.set_xlabel('y', fontsize=14)
ax2.set_ylabel("f'(y)", fontsize=14)
ax2.set_title(f'Derivative Prediction (Loss: {loss.item():.6f})', fontsize=16)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Additional diagnostic: Check a few more samples ===
print("\nTesting on additional samples...")
for test_idx in [1, 2, 3]:
    f_test = f_tensor[test_idx]
    Tf_test_true = Tf_tensor[test_idx]
    
    with torch.no_grad():
        input_coeff_test = (phi.T @ f_test).unsqueeze(0)
        output_coeff_test = operator(input_coeff_test)
        Tf_test_pred = torch.matmul(output_coeff_test, basis_out.T).squeeze()
        
        mse = torch.nn.functional.mse_loss(Tf_test_pred, Tf_test_true)
        print(f"Sample {test_idx} - MSE: {mse.item():.6f}")