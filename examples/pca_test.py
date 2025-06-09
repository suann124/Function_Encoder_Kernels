import torch
from torch.utils.data import DataLoader
from datasets.polynomial import PolynomialDataset
import sys, os, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import subspace_angles
import seaborn as sns
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

dataset = PolynomialDataset(n_points=100, n_example_points=10)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

X_list, f_list = [], []
for i, d in enumerate(dataset):
    X_list.append(d[0].unsqueeze(0))
    f_list.append(d[1].unsqueeze(0))
    if i == 99: break  # sample 100 examples max

X_all = torch.cat(X_list, dim=0).to(device)
f_all = torch.cat(f_list, dim=0).to(device)
flat_f = f_all.cpu().numpy().reshape(f_all.shape[0], -1)

# — Precompute the function-evaluation grid once
x_grid   = torch.linspace(-1, 1, 200).unsqueeze(1).to(device)
X_grid_np = x_grid.cpu().numpy().flatten()

MAX_BASIS_SIZE = 10
LOSS_THRESHOLD = 0.003

def basis_function_factory():
    return MLP(layer_sizes=[1, 32, 1])

basis_functions = BasisFunctions(basis_function_factory())
model = FunctionEncoder(basis_functions).to(device)

def loss_function(model, batch, ortho_lambda=0.01):
    X, y, example_X, example_y = [b.to(device) for b in batch]
    coeffs, _ = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coeffs)
    mse = torch.nn.functional.mse_loss(y_pred, y)

    ortho_loss = 0.0
    num_heads = model.basis_functions.num_heads()

    if num_heads > 1:
        x_grid = torch.linspace(-1, 1, 200).unsqueeze(1).to(device)
        phi = model.basis_functions(x_grid)  # [n_points, num_heads]
        phi = phi / (phi.norm(dim=0, keepdim=True) + 1e-8)

        old_phi = phi[:, :, -2:-1].squeeze(1)    # all but the latest
        new_phi = phi[:, :, -1:].squeeze(1)    # the latest basis function

        # Compute cross-gram matrix: [1, num_old]
        cross_gram = new_phi.T @ old_phi
        ortho_loss = torch.norm(cross_gram)

    return mse + ortho_lambda * ortho_loss

def pca_interpolate(X_all, f_all, x_grid, num_heads):
    """Works with your existing imports - no new dependencies"""
    
    grid_np = x_grid.cpu().numpy().flatten()
    
    # Interpolate all functions to same grid
    func_matrix = []
    for i in range(len(f_all)):
        x_coords = X_all[i].cpu().numpy().flatten()
        func_vals = f_all[i].cpu().numpy().flatten()
        func_interp = np.interp(grid_np, x_coords, func_vals)
        func_matrix.append(func_interp)
    
    func_matrix = np.array(func_matrix)
    
    # PCA on interpolated functions  
    pca = PCA(n_components=num_heads)
    pca.fit(func_matrix)
    
    return pca.components_, pca.explained_variance_ratio_


losses = []
sim_matrices = []
angles_history = []

# Train the first basis function
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{MAX_BASIS_SIZE}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})
    losses.append(loss)

# Initialize head count
num_heads = 1

# Compute function-space PCA & metrics of first basis
pca_components, pca_explained = pca_interpolate(X_all, f_all, x_grid, num_heads)

with torch.no_grad():
    learned_vals = model.basis_functions(x_grid).cpu().numpy().squeeze()  # [200, 1]

learned_norm = normalize(learned_vals.reshape(1,-1))     # [1, 200]
pca_norm = normalize(pca_components.reshape(1,-1))         # [1, 200] 
sim_matrix = cosine_similarity(learned_norm, pca_norm)
angles_func = np.degrees(subspace_angles(learned_norm.T, pca_norm.T))


print(f"\n-- After {num_heads} basis function(s) --")
print("Cosine similarity matrix:")
print(sim_matrix)
for i, ang in enumerate(angles_func, start=1):
    print(f"  Function-space angle {i}: {ang:.4f}°")
print("—" * 40)


# Train remaining basis functions manually
while num_heads <= MAX_BASIS_SIZE:
    num_heads += 1

    # Freeze all existing params
    for p in model.parameters(): 
        p.requires_grad = False
    new_fn = basis_function_factory().to(device)

    # Create new basis function and add to model
    for p in new_fn.parameters(): 
        p.requires_grad = True
    model.basis_functions.basis_functions.append(new_fn)

    # Only train the trainable parameters
    optimizer = torch.optim.Adam([p for p in new_fn.parameters() if p.requires_grad], lr=1e-3)
    
    with tqdm.tqdm(range(num_epochs), desc=f"basis {num_heads}/{MAX_BASIS_SIZE}") as tqdm_bar:
        for epoch in tqdm_bar:
            batch = next(dataloader_iter)
            loss = train_step(model, optimizer, batch, loss_function)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})
        losses.append(loss)

    # Compute PCA and metrics for the new basis
    pca_components, pca_explained = pca_interpolate(X_all, f_all, x_grid, num_heads)

    with torch.no_grad():
        learned_vals = model.basis_functions(x_grid).cpu().numpy().squeeze()

    learned_norm = normalize(learned_vals.T)     # [num_heads, 200]
    pca_norm = normalize(pca_components)         # [num_heads, 200] 

    sim_matrix = cosine_similarity(learned_norm, pca_norm)
    angles_func = np.degrees(subspace_angles(learned_norm.T, pca_norm.T))

    print(f"\n-- After {num_heads} basis function(s) --")
    print("Cosine similarity matrix:")
    print(sim_matrix)

    print(f"PCA explained variance: {pca_explained}")
    for i, ang in enumerate(angles_func, start=1):
        print(f"  Function-space angle {i}: {ang:.4f}°")
    print("—" * 40)

    if loss <= LOSS_THRESHOLD:
        print(f"Reached target loss with {num_heads} basis functions.")
        break

    if num_heads == MAX_BASIS_SIZE:
        print("Reached maximum basis size without reaching loss threshold.")
        break


# ====================================================================================
# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].plot(range(1, num_heads + 1), losses, marker='o')
axs[0].set_title("Residual Loss vs Basis Size")
axs[0].set_xlabel("Basis Size")
axs[0].set_ylabel("MSE")
axs[0].grid()

axs[1].plot(np.cumsum(pca_explained), marker='o')
axs[1].set_title("PCA Cumulative Variance")
axs[1].set_xlabel("Components")
axs[1].set_ylabel("Cumulative Variance")
axs[1].grid()

# =============================Cosine Similarities Heatmap=================================
basis_labels = [f"B{i+1}" for i in range(num_heads)]
pca_labels = [f"PC{i+1}\n({var:.1%})" for i, var in enumerate(pca_explained[:num_heads])]

sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=pca_labels, yticklabels=basis_labels, ax=axs[2])

axs[2].set_xlabel("PCA Components")
axs[2].set_ylabel("Learned Basis Functions")
axs[2].set_title("Cosine Similarity: Basis vs PCA")

plt.tight_layout()
plt.show()

# # ===================================Principal Angles=================================
# # Coefficient subspace comparison
# coeff_ls = []
# with torch.no_grad():
#      for i in range(X_all.shape[0]):
#         Xi = X_all[i].unsqueeze(0)  # shape: [1, 100, 1]
#         yi = f_all[i].unsqueeze(0)  # shape: [1, 100, 1]
#         coeffs, _ = model.compute_coefficients(Xi, yi)
#         coeff_ls.append(coeffs.cpu())
# coeffs_np = torch.cat(coeff_ls, dim=0).numpy()
# learned_subspace = normalize(coeffs_np.T)
# pca_subspace = normalize(pca.transform(flat_f)[:, :num_heads].T)
# angles_deg = np.degrees(subspace_angles(learned_subspace.T, pca_subspace.T))

# print("\n=== Principal Angles ===")
# for i, angle in enumerate(angles_deg):
#     print(f"Principal angle {i+1}: {angle:.4f}°")

# ===============================plotting basis functions==============================
def plot_learned_basis():
    x_grid = torch.linspace(-1, 1, 200).unsqueeze(1).to(device)
    
    with torch.no_grad():
        basis_vals = model.basis_functions(x_grid).cpu().numpy().squeeze()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_heads, 8)):
        axes[i].plot(x_grid.cpu().numpy().flatten(), basis_vals[:, i])
        axes[i].set_title(f'Learned Basis Function {i+1}')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

plot_learned_basis()