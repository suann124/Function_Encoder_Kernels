import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset, concatenate_datasets
from vae_sarah import create_model, train, loss_function, fidelity_loss
from vae_sarah import evaluate
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# --- 1. Load data ---
# columns 0, 1, 2 are [amplitude, frequency, phase]
# columns 3, 4 are [Y_imag, Y_real]
DATASET_NAME = "suann124/shaken-lattice-control" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# --- 2. Separate into control (beta) and state (alpha)
print(f"Loading dataset '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME)
print("Dataset loaded successfully.")

def prepare_data(dataset_split):
    """Prepares the dataset for the surrogate model."""
    # Input X: Control parameters
    X = torch.tensor(dataset_split['X_controls'], dtype=torch.float32)

    # Output Y: Final state coefficients
    Y_real = torch.tensor(dataset_split['Y_real'], dtype=torch.float32)
    Y_imag = torch.tensor(dataset_split['Y_imag'], dtype=torch.float32)
    Y = torch.cat([Y_real, Y_imag], dim=1)
    
    return X, Y

full_split = concatenate_datasets([dataset[split] for split in dataset])
full_dataset = prepare_data(full_split)

controls, Y = full_dataset
print(f"Y is type: {type(Y)}")
print(f"Y shape: {Y.shape}")

# --- 3. Normalize the data
alpha_scaler = StandardScaler()
beta_scaler = StandardScaler()

# Fit on the entire dataset for simplicity, but fitting on a training set is better practice
Y_scaled = alpha_scaler.fit_transform(Y)
C_scaled = beta_scaler.fit_transform(controls)

# --- 4. Convert to PyTorch Tensors
alpha_tensor = torch.tensor(Y_scaled, dtype=torch.float32)
beta_tensor = torch.tensor(C_scaled, dtype=torch.float32)

# --- 5. Create Dataset and DataLoaders
dataset = TensorDataset(alpha_tensor, beta_tensor)

# Split into training and testing sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# ========= Model Config ==========
# Define model parameters
ALPHA_SIZE = 22 # Dimension of [Y_imag, Y_real]
BETA_SIZE = 3  # Dimension of [amplitude, frequency, phase]
LATENT_SIZE = 16 # Example latent dimension, you can tune this
HIDDEN_SIZES = [64, 64] # Example hidden layer sizes

# Create the model instance
model = create_model(
    alpha_size=ALPHA_SIZE,
    beta_size=BETA_SIZE,
    hidden_sizes=HIDDEN_SIZES,
    latent_size=LATENT_SIZE,
)

print(model)

# =========== Training the VAE ==========
# --- Setup for training
N_EPOCHS = 500
LEARNING_RATE = 1e-4
MODEL_NAME = "interferometry_cvae"
CHECKPOINT_DIR = "./checkpoints"

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Tensorboard writer for logging loss
# To view logs, run `tensorboard --logdir=runs` in your terminal
summary_writer = SummaryWriter(f"runs/{MODEL_NAME}")

# --- Call the training function
train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
    optimizer=optimizer,
    n_epochs=N_EPOCHS,
    summary_writer=summary_writer,
    model_name=MODEL_NAME,
    params={}, # Not used in the provided train function
    checkpoint_dir=CHECKPOINT_DIR,
    checkpoint_interval=100,
    fidelity_weight=0.1
    )

# ========= Evaluation of the VAE ==========
# Evaluate on the entire test dataset
model.eval()
total_loss = 0.0
num_batches = 0
total_fid_loss = 0.0

with torch.no_grad():
    for alpha_batch, beta_batch in test_dataloader:
        alpha_batch = alpha_batch.to(DEVICE)
        beta_batch = beta_batch.to(DEVICE)
        z, mu, logvar = model(alpha_batch, beta_batch)
        recon_alpha = model.inverse(beta_batch, z)
        loss = loss_function(model, (alpha_batch, beta_batch), fidelity_weight=0.1)          
        total_loss += loss.item()
        num_batches += 1
        fid_loss = fidelity_loss(alpha_batch, recon_alpha).item()
        total_fid_loss += fid_loss
        print(f"Test Loss: {loss:.6f} | Fidelity Loss: {fid_loss:.6f}")

average_test_loss = total_loss / num_batches
print(f"Average Test Loss: {average_test_loss:.6f} | Fidelity Loss: {total_fid_loss/num_batches:.6f}")