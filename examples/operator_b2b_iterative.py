import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Add function_encoder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function_encoder.model.mlp import MLP, MultiHeadedMLP_Freeze
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit

# Parameters for iterative basis function addition
LOSS_THRESHOLD = 0.003  # Target loss threshold
MAX_BASIS_SIZE = 40    # Maximum number of basis functions (heads)
TRAIN_EPOCHS = 200     # Epochs per training run

# Device config
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load dataset
ds = load_dataset("ajthor/derivative")
ds = ds.with_format("torch", device=device)

# Create data loaders
train_loader = DataLoader(ds["train"], batch_size=50)
test_loader = DataLoader(ds["test"], batch_size=50)

# Function to evaluate operator prediction
def evaluate_operator(input_encoder, output_encoder, operator, test_loader, num_heads):
    input_encoder.eval()
    output_encoder.eval()
    operator.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            X, f = batch["X"].to(device), batch["f"].to(device)
            Y, Tf = batch["Y"].to(device), batch["Tf"].to(device)
            
            X = X.unsqueeze(-1)
            f = f.unsqueeze(-1)
            Y = Y.unsqueeze(-1)
            Tf = Tf.unsqueeze(-1)
            
            # Get input coefficients
            input_coefficients = input_encoder.compute_coefficients(X, f)
            
            # Pad input coefficients to full size
            pad_size = MAX_BASIS_SIZE - input_coefficients.shape[-1]
            input_coefficients_padded = torch.nn.functional.pad(input_coefficients, (0, pad_size), mode='constant', value=0)
            output_coefficients = operator(input_coefficients_padded)
            
            # Predict Tf using output coefficients
            Tf_pred = output_encoder(Y, output_coefficients[..., :num_heads])
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(Tf_pred, Tf)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def train_encoder(encoder, loss_function, train_loader):
    return fit(model=encoder, ds=train_loader, loss_function=loss_function, epochs=TRAIN_EPOCHS)

def iterative_training_progressive():
    input_basis = MultiHeadedMLP_Freeze(layer_sizes=[1, 32, 1], num_heads=1)
    output_basis = MultiHeadedMLP_Freeze(layer_sizes=[1, 32, 1], num_heads=1)

    input_encoder = FunctionEncoder(input_basis)
    output_encoder = FunctionEncoder(output_basis)

    def input_loss(model, batch):
        X, y = batch["X"].to(device), batch["f"].to(device)
        X, y = X.unsqueeze(-1), y.unsqueeze(-1)
        coeffs = model.compute_coefficients(X, y)
        y_pred = model(X, coeffs)
        pred_loss = torch.nn.functional.mse_loss(y_pred, y)
        norm_loss = basis_normalization_loss(model.basis_functions(X))
        return pred_loss + norm_loss

    def output_loss(model, batch):
        X, y = batch["Y"].to(device), batch["Tf"].to(device)
        X, y = X.unsqueeze(-1), y.unsqueeze(-1)
        coeffs = model.compute_coefficients(X, y)
        y_pred = model(X, coeffs)
        pred_loss = torch.nn.functional.mse_loss(y_pred, y)
        norm_loss = basis_normalization_loss(model.basis_functions(X))
        return pred_loss + norm_loss

    operator = MLP(layer_sizes=[MAX_BASIS_SIZE, 32, MAX_BASIS_SIZE], activation=torch.nn.ReLU())

    def operator_loss(model, batch):
        X, f = batch["X"].to(device), batch["f"].to(device)
        Y, Tf = batch["Y"].to(device), batch["Tf"].to(device)
        X, f, Y, Tf = X.unsqueeze(-1), f.unsqueeze(-1), Y.unsqueeze(-1), Tf.unsqueeze(-1)

        input_coeffs = input_encoder.compute_coefficients(X, f)

        # Pad input coefficients to full size
        pad_size = MAX_BASIS_SIZE - input_coeffs.shape[-1]
        input_coeffs_padded = torch.nn.functional.pad(input_coeffs, (0, pad_size), mode='constant', value=0)

        # Pass through operator and slice output to current head size
        output_coeffs = model(input_coeffs_padded)[..., :num_heads]
        Tf_pred = output_encoder(Y, output_coeffs)

        return torch.nn.functional.mse_loss(Tf_pred, Tf)

    num_heads = 1
    losses = []

    while num_heads <= MAX_BASIS_SIZE:
        print(f"\n=== Iteration with {num_heads} basis functions ===")

        input_encoder = train_encoder(input_encoder, input_loss, train_loader)
        output_encoder = train_encoder(output_encoder, output_loss, train_loader)
        operator = fit(model=operator, ds=train_loader, loss_function=operator_loss, epochs=TRAIN_EPOCHS)

        train_loss = evaluate_operator(input_encoder,
                                    output_encoder,
                                    operator,
                                    train_loader,    # ← train, not test
                                    num_heads)

        test_loss  = evaluate_operator(input_encoder,
                                    output_encoder,
                                    operator,
                                    test_loader,
                                    num_heads)

        print(f"train MSE: {train_loss:.5f} | test MSE: {test_loss:.5f}")

        # # What should I use as threshold?
        # loss = operator_loss(operator, ds["train"])
        losses.append((num_heads, train_loss))
        # print(f"Operator loss: {loss:.6f}")

        if train_loss <= LOSS_THRESHOLD:
            print(f"Reached target loss with {num_heads} basis functions.")
            break

        if num_heads == MAX_BASIS_SIZE:
            print("Reached maximum basis size without reaching loss threshold.")
            break

        # Freeze current heads and add new ones
        input_basis.freeze_heads(list(range(num_heads)))
        output_basis.freeze_heads(list(range(num_heads)))
        input_basis.add_head()
        output_basis.add_head()

        num_heads += 1

    # Plot loss history
    plt.figure(figsize=(10, 6))
    sizes, vals = zip(*losses)
    plt.plot(sizes, vals, 'o-', label='Operator Loss', color='red')
    plt.xlabel("Number of Basis Functions")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of Basis Functions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return input_encoder, output_encoder, operator, num_heads, train_loss

# Run the iterative training
input_function_encoder, output_function_encoder, operator, final_basis_size, final_operator_loss = iterative_training_progressive()

# Plot a test case
print("\nGenerating visualization...")
input_function_encoder.eval()
output_function_encoder.eval()
operator.eval()
# %%
# Get a test sample
point = ds["test"].take(1)[0]

X = point["X"]
f = point["f"]
Y = point["Y"]
Tf = point["Tf"]

idx = torch.argsort(X, dim=0).squeeze()
X = X[idx]
f = f[idx]

idx = torch.argsort(Y, dim=0).squeeze()
Y = Y[idx]
Tf = Tf[idx]

X_tensor = X.unsqueeze(-1).unsqueeze(0)
f_tensor = f.unsqueeze(-1).unsqueeze(0)
Y_tensor = Y.unsqueeze(-1).unsqueeze(0)
Tf_tensor = Tf.unsqueeze(-1).unsqueeze(0)

with torch.no_grad():
    input_coefficients = input_function_encoder.compute_coefficients(X_tensor, f_tensor)
    pad_size = MAX_BASIS_SIZE - input_coefficients.shape[-1]
    input_coefficients_padded = torch.nn.functional.pad(input_coefficients, (0, pad_size), mode='constant', value=0)
    output_coefficients = operator(input_coefficients_padded)[..., :final_basis_size]
    Tf_pred = output_function_encoder(Y_tensor, output_coefficients)

# Detach from device and squeeze
X = X.cpu().detach().numpy()
f = f.cpu().detach().numpy()
Y = Y.cpu().detach().numpy()
Tf = Tf.cpu().detach().numpy()
Tf_pred = Tf_pred.squeeze().cpu().detach().numpy()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

ax.plot(X, f, label="Original", linewidth=2)
ax.scatter(X, f, label="Data", color="red", s=30, alpha=0.7)

ax.plot(Y, Tf, label="True", linewidth=2)
ax.plot(Y, Tf_pred, label="Predicted", linewidth=2, linestyle='--')

ax.legend(fontsize=12)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(f'Results with {final_basis_size} basis functions\nOperator Loss: {final_operator_loss:.6f}', fontsize=16)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal results: {final_basis_size} basis functions")
print(f"Operator loss: {final_operator_loss:.6f}")
# %%
