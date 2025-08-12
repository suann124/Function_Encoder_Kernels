# %%
import torch
from torch.utils.data import DataLoader
from my_datasets.polynomial import PolynomialDataset
import sys, os, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import subspace_angles
import seaborn as sns
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step
from function_encoder.coefficients import lasso
from function_encoder.losses import basis_normalization_loss


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

# — Precompute the function-evaluation grid once
x_grid   = torch.linspace(-1, 1, 200).unsqueeze(1).to(device)
X_grid_np = x_grid.cpu().numpy().flatten()

MAX_BASIS_SIZE = 10
LOSS_THRESHOLD = 0.003

def basis_function_factory():
    return MLP(layer_sizes=[1, 32, 1])

basis_functions = BasisFunctions(basis_function_factory())
# model = FunctionEncoder(basis_functions, coefficients_method=lasso).to(device)
model = FunctionEncoder(basis_functions).to(device)


def loss_function(model, batch, ortho_lambda=0.02):
    X, y, example_X, example_y = [b.to(device) for b in batch]
    coeffs, _ = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coeffs)
    mse = torch.nn.functional.mse_loss(y_pred, y)

    ortho_loss = 0.0
    num_heads = model.basis_functions.num_heads()

    if num_heads > 1:
        x_grid = torch.linspace(-1, 1, 200).unsqueeze(1).unsqueeze(0).to(device)

        # phi shape is [batch, points, features, basis], e.g., [1, 200, 1, num_heads]
        phi = model.basis_functions(x_grid)  # [n_points, num_heads]
        phi = phi / (torch.norm(phi, p=2, dim=(1, 2), keepdim=True) + 1e-8)

        old_phi = phi[..., -2:-1].squeeze(1)    # all previous basis funcs other than the last
        new_phi = phi[..., -1:].squeeze(1)    # the latest basis function

        # Compute cross-gram matrix: [1, num_old]
        # cross_gram = new_phi.T @ old_phi
        # cross_gram = torch.einsum('bmdk,bmdl->bmkl', new_phi, old_phi)  # inner product
        cross_gram = torch.einsum('bmdk,bmdl->bkl', new_phi, old_phi)  # inner product
        ortho_loss = torch.sum((cross_gram - torch.diag_embed(torch.diagonal(cross_gram, dim1=-2, dim2=-1))) ** 2)

        
    return mse + ortho_lambda * ortho_loss

def pca_interpolate(X_all, f_all, x_grid, num_heads):    
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
explained_variances_history = []
angles_history = []
final_sim_matrix = None # Will store the last similarity matrix for plotting

# Train the first basis function
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{MAX_BASIS_SIZE}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})
    losses.append(loss)

# For a single function, it explains 100% of its own variance
explained_variances_history.append(np.array([1.0]))
angles_history.append(np.array([0.0])) # A 1D space is perfectly aligned with itself
final_sim_matrix = np.array([[1.0]])

print(f"\n-- After 1 basis function(s) --")
print(f"Explained variance of learned subspace: [1.0]")
print("—" * 40)

# Initialize head count
num_heads = 1

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
    optimizer = torch.optim.Adam([p for p in new_fn.parameters() if p.requires_grad], lr=5e-4)
    
    with tqdm.tqdm(range(num_epochs), desc=f"basis {num_heads}/{MAX_BASIS_SIZE}") as tqdm_bar:
        for epoch in tqdm_bar:
            batch = next(dataloader_iter)
            loss = train_step(model, optimizer, batch, loss_function)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})
        losses.append(loss)

    # --- Test 1: PCA on the Learned Basis Functions ---
    with torch.no_grad():
        # Get ALL learned functions. Squeeze to remove batch and feature dims.
        learned_vals = model.basis_functions(x_grid.unsqueeze(0)).cpu().numpy().squeeze()

    # Perform PCA on the current set of learned functions.
    # We transpose because sklearn PCA expects samples as rows.
    pca_on_learned = PCA()
    pca_on_learned.fit(learned_vals.T)
    explained_variance_ratio = pca_on_learned.explained_variance_ratio_
    explained_variances_history.append(explained_variance_ratio)

    # 2. Get the new principal components from this PCA
    learned_pca_components = pca_on_learned.components_ # Shape: [num_heads, 200]

    # 3. Calculate Cosine Similarity
    # Compare learned functions (rows) to their own PCs (rows)
    final_sim_matrix = cosine_similarity(learned_vals.T, learned_pca_components)

    # 4. Calculate Subspace Angles
    # Compare the subspace of learned functions (columns) to the subspace of their PCs (columns)
    angles_func = np.degrees(subspace_angles(learned_vals, learned_pca_components.T))
    angles_history.append(angles_func)

    print(f"\n-- After {num_heads} basis function(s) --")
    print(f"Explained variance of learned subspace: {np.round(explained_variance_ratio, 4)}")
    print(f"Cumulative variance: {np.round(np.cumsum(explained_variance_ratio), 4)}")
    for i, ang in enumerate(angles_func, start=1):
        print(f"  Subspace angle {i}: {ang:.4f}°")
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

for i, variances in enumerate(explained_variances_history):
    axs[1].plot(np.cumsum(variances), marker='o', linestyle='-', label=f'{i+2} Basis Functions' if i > 0 else f'{i+1} Basis Function')
axs[1].set_title("Test 1: PCA on Learned Functions")
axs[1].set_xlabel("Components")
axs[1].set_ylabel("Cumulative Variance")
axs[1].grid()

# =============================Cosine Similarities Heatmap=================================
basis_labels = [f"B{i+1}" for i in range(num_heads)]
pca_labels = [f"PC{i+1}" for i in range(num_heads)]
sns.heatmap(final_sim_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=pca_labels, yticklabels=basis_labels, ax=axs[2])

axs[2].set_xlabel("PCA Components")
axs[2].set_ylabel("Learned Basis Functions")
axs[2].set_title("Cosine Similarity: Basis vs PCA")

plt.tight_layout()
plt.show(block=False)

# # ===================================Principal Angles=================================
# Pad each angle list to the max number of basis functions
max_len = max(len(a) for a in angles_history)
angles_padded = [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in angles_history]
angles_array = np.vstack(angles_padded)  # shape [num_heads, max_len]

# Plot each principal angle over basis size
plt.figure()
for i in range(angles_array.shape[1]):
    plt.plot(range(1, angles_array.shape[0] + 1), angles_array[:, i],'o', label=f"Angle {i+1}")

plt.xlabel("Basis Size")
plt.ylabel("Principal Angle (°)")
plt.title("Principal Angles vs Basis Size")
plt.legend()
plt.grid()
plt.show(block=False)

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
    plt.show(block=False)

plot_learned_basis()

# %% 
# ----------------- Test 2 --------------------------
print("===================================================================")
print("Test 2: Comparing PCA coefficients of Progressive vs Classical Models")
print("===================================================================")

def loss_function_classical(model, batch):
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


basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)
classical_model = FunctionEncoder(basis_functions).to(device)

optimizer = torch.optim.Adam(classical_model.parameters(), lr=1e-3)

num_epochs = 1000 # May need more epochs for all functions to converge
with tqdm.tqdm(range(num_epochs), desc="Training Classical Model") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(classical_model, optimizer, batch, loss_function_classical)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


with torch.no_grad():
    # For the progressive model
    progressive_coeffs, _ = model.compute_coefficients(X_all, f_all)
    progressive_coeffs = progressive_coeffs.cpu().numpy()

    # For the classical model
    f_all_squeezed = f_all.squeeze(-1) if f_all.dim() > 3 else f_all
    classical_coeffs, _ = classical_model.compute_coefficients(X_all, f_all_squeezed)

    classical_coeffs = classical_coeffs.cpu().numpy()

# Perform PCA on the coefficients
pca_progressive = PCA()
pca_progressive.fit(progressive_coeffs)

pca_classical = PCA()
pca_classical.fit(classical_coeffs)

# Plot the explained variance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(pca_progressive.explained_variance_ratio_, marker='o')
plt.title('Progressive Training: PCA on Coefficients')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.yscale('log')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(pca_classical.explained_variance_ratio_, marker='o')
plt.title('Classical Training: PCA on Coefficients')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()

print("END")
    # %%
