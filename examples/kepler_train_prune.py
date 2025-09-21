import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import tqdm
from copy import deepcopy

from my_datasets.kepler import KeplerDataset, kepler, generate_kepler_states_batch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step
from function_encoder.inner_products import standard_inner_product
from function_encoder.utils.experiment_saver import ExperimentSaver, create_visualization_data_dynamics

if torch.cuda.is_available():
    device = "cuda:2"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)


class KeplerTrainPruneAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def create_kepler_dataset(self):
        """Create Kepler dataset similar to kepler_pca.py"""
        return KeplerDataset(
            integrator=rk4_step,
            n_points=1000,
            n_example_points=100,
            dt_range=(0.1, 0.1),
            device=torch.device(self.device),
        )

    def basis_function_factory(self):
        """Create NeuralODE basis function like in kepler_pca.py"""
        return NeuralODE(
            ode_func=ODEFunc(model=MLP(layer_sizes=[5,64, 64, 4])),
            integrator=rk4_step,
        )

    def load_pretrained_model(self, model_path: str, num_basis: int):
        """Load a pre-trained model to save computation time."""
        print(f"Loading pre-trained model from {model_path}...")

        # Create model structure similar to kepler_pca.py
        basis_functions = BasisFunctions(*[self.basis_function_factory() for _ in range(num_basis)])
        model = FunctionEncoder(basis_functions).to(self.device)

        # Load saved state
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Successfully loaded pre-trained model!")
        except Exception as e:
            print(f"Warning: Could not load pre-trained model: {e}")
            print("Will train from scratch...")
            return self.train_full_model(num_basis, self.create_kepler_dataset())

        return model

    def train_full_model(self,
                        num_basis: int,
                        dataset: KeplerDataset,
                        num_epochs: int = 1000,
                        batch_size: int = 50) -> FunctionEncoder:
        """Train a model with all basis functions simultaneously (batch training)."""

        print(f"Training full model with {num_basis} basis functions using batch training...")

        # Create model with all basis functions
        all_basis_functions = BasisFunctions(*[self.basis_function_factory() for _ in range(num_basis)])
        model = FunctionEncoder(all_basis_functions).to(self.device)

        # Setup training
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_iter = iter(dataloader)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        # Training loop - train all basis functions together
        with tqdm.tqdm(range(num_epochs), desc="Training full model (batch)") as pbar:
            for epoch in pbar:
                batch = next(dataloader_iter)
                loss = train_step(model, optimizer, batch, self.loss_function)
                losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.2e}"})

        return model

    def analyze_basis_importance(self,
                               model: FunctionEncoder,
                               dataset: KeplerDataset,
                               num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze basis importance using PCA on coefficients."""

        print("Analyzing basis importance with PCA...")

        model.eval()
        dataloader = DataLoader(dataset, batch_size=num_samples)
        batch = next(iter(dataloader))

        with torch.no_grad():
            _, y0, dt, y1, y0_example, dt_example, y1_example = batch
            # Data is already on the correct device from the dataset

            # Compute coefficients for all samples
            coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)
            coefficients_np = coefficients.cpu().numpy()

            # Center the coefficients
            coefficients_centered = coefficients_np - np.mean(coefficients_np, axis=0)

            # Compute covariance matrix
            cov_matrix = np.cov(coefficients_centered.T)

            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Compute explained variance ratio
            explained_variance_ratio = eigenvalues / eigenvalues.sum()

            return eigenvalues, eigenvectors, explained_variance_ratio

    def cos_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify_redundant_basis(self,
                               eigenvalues: np.ndarray,
                               eigenvectors: np.ndarray,
                               explained_variance_ratio: np.ndarray,
                               model: FunctionEncoder,
                               variance_threshold: float = 0.99) -> List[int]:
        """Identify which basis functions to keep based on PCA analysis."""

        # Finding number of basis needed: Cumulative variance threshold
        cumsum_var = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1

        print(f"Need {n_components} components to explain {variance_threshold*100}% variance")

        # Find which original basis contribute most to top PCs
        n_basis = eigenvectors.shape[0]

        # Use eigenvectors as coefficients and compute functional similarity with basis functions
        top_k_eig_idx = np.argsort(eigenvalues)[::-1][:n_components]
        important_pcs = eigenvectors[:, top_k_eig_idx]

        print("Using eigenvalue-weighted PCA loadings to identify important basis functions...")
        n_basis = eigenvectors.shape[0]
        weighted_eig = np.zeros(n_basis)
        for i in range(n_components):
            weighted_eig += np.abs(eigenvectors[:, i]) * eigenvalues[i]   
            best_aligned_basis = np.argsort(weighted_eig)[::-1][:n_components]   
            
        # # Create test input for functional evaluation - match the Kepler dataset format
        # # Use a batch of test trajectories
        # test_y0 = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=self.device)  # [1, 4] initial condition
        # test_dt = torch.tensor([[0.1]], device=self.device)  # [1, 1] time step
        # test_input = (test_y0.unsqueeze(1), test_dt.unsqueeze(1))  # [1, 1, 4], [1, 1, 1] to match expected format

        # basis_alignment = np.zeros((n_basis, n_components))

        # # Evaluate individual basis functions
        # with torch.no_grad():
        #     basis_evals = []
        #     for i, basis_func in enumerate(model.basis_functions.basis_functions):
        #         basis_eval = basis_func(test_input)
        #         basis_evals.append(basis_eval)

        #     # Use coefficient-based alignment instead of functional alignment
        #     # This is simpler and more direct
        #     for j in range(n_components):
        #         for i in range(n_basis):
        #             # Just use the coefficient weights directly as alignment scores
        #             basis_alignment[i, j] = np.abs(important_pcs[i, j])

        # print("Basis alignment:", basis_alignment)

        # # Find best aligned basis for each PC
        # alignment_abs = np.abs(basis_alignment)
        # best_aligned_basis = np.argmax(alignment_abs, axis=0)

        # print(f"Best aligned basis: {best_aligned_basis}")
        return sorted(best_aligned_basis.tolist())

    def prune_model(self,
                   model: FunctionEncoder,
                   keep_indices: List[int]) -> FunctionEncoder:
        """Create a pruned model keeping only specified basis functions."""

        print(f"Pruning model to keep {len(keep_indices)} basis functions...")

        # Create new model with fewer basis functions
        pruned_basis_functions = BasisFunctions(*[self.basis_function_factory() for _ in range(len(keep_indices))])
        pruned_model = FunctionEncoder(pruned_basis_functions).to(self.device)

        # Copy weights from original model for kept basis
        with torch.no_grad():
            for new_idx, old_idx in enumerate(keep_indices):
                old_basis = model.basis_functions.basis_functions[old_idx]
                new_basis = pruned_model.basis_functions.basis_functions[new_idx]

                # Copy all parameters
                old_state = old_basis.state_dict()
                new_basis.load_state_dict(old_state)

        return pruned_model

    def fine_tune_pruned_model(self,
                             model: FunctionEncoder,
                             dataset: KeplerDataset,
                             num_epochs: int = 1000,
                             batch_size: int = 50) -> Tuple[FunctionEncoder, List[float]]:
        """Fine-tune the pruned model."""

        print("Fine-tuning pruned model...")
        model_to_tune = deepcopy(model)

        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_iter = iter(dataloader)
        optimizer = torch.optim.Adam(model_to_tune.parameters(), lr=5e-4)
        losses = []

        with tqdm.tqdm(range(num_epochs), desc="Fine-tuning") as pbar:
            for epoch in pbar:
                batch = next(dataloader_iter)
                loss = train_step(model_to_tune, optimizer, batch, self.loss_function)
                losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.2e}"})

        return model_to_tune, losses

    def compare_models(self,
                      original_model: FunctionEncoder,
                      pruned_model: FunctionEncoder,
                      pruned_model_refined: FunctionEncoder,
                      dataset: KeplerDataset,
                      num_test_samples: int = 100):
        """Compare performance of original vs pruned model."""

        print("\nComparing model performance...")

        test_loader = DataLoader(dataset, batch_size=num_test_samples)
        batch = next(iter(test_loader))

        _, y0, dt, y1, y0_example, dt_example, y1_example = batch
        # Data is already on the correct device from the dataset

        original_model.eval()
        pruned_model.eval()
        pruned_model_refined.eval()

        with torch.no_grad():
            # Original model predictions
            coeffs_orig, _ = original_model.compute_coefficients((y0_example, dt_example), y1_example)
            y_pred_orig = original_model((y0, dt), coeffs_orig)
            mse_orig = torch.nn.functional.mse_loss(y_pred_orig, y1).item()

            # Pruned model predictions
            coeffs_pruned, _ = pruned_model.compute_coefficients((y0_example, dt_example), y1_example)
            y_pred_pruned = pruned_model((y0, dt), coeffs_pruned)
            mse_pruned = torch.nn.functional.mse_loss(y_pred_pruned, y1).item()

            # Pruned Refined model predictions
            coeffs_pruned_refined, _ = pruned_model_refined.compute_coefficients((y0_example, dt_example), y1_example)
            y_pred_pruned_refined = pruned_model_refined((y0, dt), coeffs_pruned_refined)
            mse_pruned_refined = torch.nn.functional.mse_loss(y_pred_pruned_refined, y1).item()

        print(f"Original model MSE: {mse_orig:.2e}")
        print(f"Pruned model MSE: {mse_pruned:.2e}")
        print(f"Pruned Refined model MSE: {mse_pruned_refined:.2e}")
        print(f"Performance ratio (refined): {mse_pruned_refined/mse_orig:.3f}")
        print(f"Compression ratio (refined): {len(pruned_model_refined.basis_functions.basis_functions)}/{len(original_model.basis_functions.basis_functions)}")

        return {
            'mse_original': mse_orig,
            'mse_pruned': mse_pruned,
            'mse_pruned_refined': mse_pruned_refined,
            'y_pred_original': y_pred_orig,
            'y_pred_pruned': y_pred_pruned,
            'y_pred_pruned_refined': y_pred_pruned_refined,
            'coeffs_original': coeffs_orig,
            'coeffs_pruned': coeffs_pruned,
            'coeffs_pruned_refined': coeffs_pruned_refined
        }

    def loss_function(self, model, batch):
        """Loss function for training - adapted from kepler_pca.py"""
        _, y0, dt, y1, y0_example, dt_example, y1_example = batch
        # Data is already on the correct device from the dataset

        coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
        pred = model((y0, dt), coefficients=coefficients)

        pred_loss = torch.nn.functional.mse_loss(pred, y1)
        return pred_loss

    def compute_explained_variance(self, model, dataset):
        """Compute explained variance like in kepler_pca.py"""
        dataloader_coeffs = DataLoader(dataset, batch_size=100)
        batch = next(iter(dataloader_coeffs))
        _, _, _, _, example_y0, example_dt, example_y1 = batch
        # Data is already on the correct device from the dataset

        coefficients, G = model.compute_coefficients((example_y0, example_dt), example_y1)

        # Compute covariance matrix of coefficients (like in kepler_pca)
        coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
        coefficients_cov = (
            torch.matmul(coefficients_centered.T, coefficients_centered)
            / coefficients.shape[0]
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
        eigenvalues = eigenvalues.flip(0)  # Flip to descending order

        # Compute explained variance from Gram matrix eigenvalues
        K = G.mean(dim=0)
        gram_eigenvalues, _ = torch.linalg.eigh(K)
        gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order

        explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

        return explained_variance_ratio, eigenvalues, gram_eigenvalues

    def visualize_results(self,
                         original_model: FunctionEncoder,
                         pruned_model: FunctionEncoder,
                         pruned_model_refined: FunctionEncoder,
                         eigenvalues: np.ndarray,
                         explained_variance_ratio: np.ndarray,
                         keep_indices: List[int],
                         comparison_results: dict,
                         dataset: KeplerDataset):
        """Visualize the pruning results - exactly matching polynomial_prune.py layout."""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Eigenvalue spectrum
        ax = axes[0, 0]
        ax.semilogy(eigenvalues, 'b.-', label='Eigenvalues')
        ax.axvline(x=len(keep_indices)-1, color='r', linestyle='--', label=f'Cutoff (n={len(keep_indices)})')
        ax.set_xlabel('Component')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('PCA Eigenvalue Spectrum')
        ax.legend()
        ax.grid(True)

        # 2. Cumulative explained variance
        ax = axes[0, 1]
        cumsum_var = np.cumsum(explained_variance_ratio)
        ax.plot(cumsum_var, 'g.-')
        ax.axhline(y=0.99, color='r', linestyle='--', label='99% threshold')
        ax.axvline(x=len(keep_indices)-1, color='r', linestyle='--')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Variance Explained')
        ax.legend()
        ax.grid(True)

        # 3. Basis function importance
        ax = axes[0, 2]
        n_basis = len(original_model.basis_functions.basis_functions)
        basis_indices = np.arange(n_basis)
        colors = ['red' if i in keep_indices else 'blue' for i in basis_indices]
        ax.bar(basis_indices, np.ones(n_basis), color=colors)
        ax.set_xlabel('Basis Function Index')
        ax.set_ylabel('Selected')
        ax.set_title('Selected Basis Functions (Red = Kept)')

        # 4. Orbital trajectory comparison (like kepler_pca.py)
        ax = axes[1, 0]
        test_sample = next(iter(DataLoader(dataset, batch_size=1)))
        M_central, y0, dt, y1, example_y0, example_dt, example_y1 = test_sample

        original_model.eval()
        pruned_model.eval()
        pruned_model_refined.eval()

        # Recompute predictions for the local test sample
        with torch.no_grad():
            coeffs_orig, _ = original_model.compute_coefficients((example_y0, example_dt), example_y1)
            coeffs_pruned, _ = pruned_model.compute_coefficients((example_y0, example_dt), example_y1)
            coeffs_pruned_refined, _ = pruned_model_refined.compute_coefficients((example_y0, example_dt), example_y1)

        # Generate orbital trajectory like in kepler_pca.py
        from my_datasets.kepler import generate_kepler_states_batch, kepler

        _M_central = M_central[0]
        _y0 = generate_kepler_states_batch(
            _M_central.item(),
            dataset.a_range,
            dataset.e_range,
            1,
            device=torch.device(device),
        )

        _c_orig = coeffs_orig[0].unsqueeze(0)
        _c_pruned = coeffs_pruned[0].unsqueeze(0)
        _c_pruned_refined = coeffs_pruned_refined[0].unsqueeze(0)
        s = 0.1  # Time step for simulation
        n = int(10.0 / s)  # Simulate for 2 time units
        _dt = torch.tensor([s], device=device)

        # Integrate the true trajectory
        x = _y0.clone()
        y_true = [x]
        for k in range(n):
            x = rk4_step(kepler, x, _dt, M_central=_M_central) + x
            y_true.append(x)
        y_true = torch.cat(y_true, dim=0)
        y_true = y_true.detach().cpu().numpy()

        # Integrate original model prediction
        x = _y0.clone()
        x = x.unsqueeze(1)
        _dt = _dt.unsqueeze(0)
        pred_orig = [x]
        for k in range(n):
            x = original_model((x, _dt), coefficients=_c_orig) + x
            pred_orig.append(x)
        pred_orig = torch.cat(pred_orig, dim=1)
        pred_orig = pred_orig.detach().cpu().numpy()

        # Integrate pruned model prediction
        x = _y0.clone()
        x = x.unsqueeze(1)
        pred_pruned = [x]
        for k in range(n):
            x = pruned_model((x, _dt), coefficients=_c_pruned) + x
            pred_pruned.append(x)
        pred_pruned = torch.cat(pred_pruned, dim=1)
        pred_pruned = pred_pruned.detach().cpu().numpy()

        # Integrate pruned refined model prediction
        x = _y0.clone()
        x = x.unsqueeze(1)
        pred_pruned_refined = [x]
        for k in range(n):
            x = pruned_model_refined((x, _dt), coefficients=_c_pruned_refined) + x
            pred_pruned_refined.append(x)
        pred_pruned_refined = torch.cat(pred_pruned_refined, dim=1)
        pred_pruned_refined = pred_pruned_refined.detach().cpu().numpy()

        # Plot orbital trajectories in phase space (x vs y)
        ax.plot(y_true[:, 0], y_true[:, 1], "k-", alpha=0.8, linewidth=1.5, label="True")
        ax.plot(pred_orig[0, :, 0], pred_orig[0, :, 1], "b--", alpha=0.9, linewidth=2, label="Original")
        ax.plot(pred_pruned[0, :, 0], pred_pruned[0, :, 1], "g--", alpha=0.9, linewidth=2, label="Pruned")
        ax.plot(pred_pruned_refined[0, :, 0], pred_pruned_refined[0, :, 1], "r:", alpha=0.9, linewidth=2, label="Pruned & Refined")

        # Mark initial positions and central body
        ax.plot(y_true[0, 0], y_true[0, 1], "go", markersize=4, label="Start")
        ax.plot(0, 0, "ko", markersize=6, label="Central Body")

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"Orbital Trajectory (M={_M_central.item():.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 5. Coefficient comparison
        ax = axes[1, 1]
        coeffs_orig = comparison_results['coeffs_original'][0].cpu().numpy()
        coeffs_pruned = comparison_results['coeffs_pruned'][0].cpu().numpy()
        coeffs_pruned_refined = comparison_results['coeffs_pruned_refined'][0].cpu().numpy()

        x_pos = np.arange(len(coeffs_orig))
        ax.bar(x_pos - 0.2, coeffs_orig, 0.4, label='Original', alpha=0.7)

        x_pos_pruned = np.arange(len(coeffs_pruned))
        ax.bar(x_pos_pruned + 0.2, coeffs_pruned, 0.4, label='Pruned', alpha=0.7)

        x_pos_pruned = np.arange(len(coeffs_pruned_refined))
        ax.bar(x_pos_pruned + 0.4, coeffs_pruned_refined, 0.4, label='Pruned & Refined', alpha=0.7)

        ax.set_xlabel('Basis Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Coefficient Comparison')
        ax.legend()

        # 6. Performance summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""Performance Summary:

Original Model:
- Basis functions: {len(original_model.basis_functions.basis_functions)}
- MSE: {comparison_results['mse_original']:.2e}

Pruned Model:
- Basis functions: {len(pruned_model.basis_functions.basis_functions)}
- MSE: {comparison_results['mse_pruned']:.2e}

Pruned & Refined Model:
- Basis functions: {len(pruned_model_refined.basis_functions.basis_functions)}
- MSE: {comparison_results['mse_pruned_refined']:.2e}

Compression: {len(pruned_model_refined.basis_functions.basis_functions)}/{len(original_model.basis_functions.basis_functions)} = {len(pruned_model.basis_functions.basis_functions)/len(original_model.basis_functions.basis_functions):.1%}
Performance ratio: {comparison_results['mse_pruned_refined']/comparison_results['mse_original']:.3f}"""

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


# ============================== Main ===============================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize
    analyzer = KeplerTrainPruneAnalyzer(device=device)

    # Create dataset
    dataset = analyzer.create_kepler_dataset()

    # Step 1: Train from scratch using batch training
    num_basis = 10  # Match the number from kepler_pca.py
    full_model = analyzer.train_full_model(num_basis, dataset, num_epochs=1000)
    # Save for future use
    torch.save(full_model.state_dict(), "kepler_train_prune_model.pth")

    # Step 2: Analyze basis importance
    eigenvalues, eigenvectors, explained_var = analyzer.analyze_basis_importance(full_model, dataset)

    # Step 3: Identify which basis to keep
    keep_indices = analyzer.identify_redundant_basis(eigenvalues, eigenvectors, explained_var, full_model, variance_threshold=0.95)
    print(f"\nKeeping basis functions at indices: {keep_indices}")

    # Step 4: Create pruned model
    pruned_model = analyzer.prune_model(full_model, keep_indices)

    # Step 5: Fine-tune pruned model
    pruned_model_refined, finetune_losses = analyzer.fine_tune_pruned_model(pruned_model, dataset, num_epochs=1000)

    # Step 6: Compare performance
    comparison_results = analyzer.compare_models(full_model, pruned_model, pruned_model_refined, dataset)

    # Step 7: Visualize results
    analyzer.visualize_results(full_model, pruned_model, pruned_model_refined, eigenvalues, explained_var,
                              keep_indices, comparison_results, dataset)

    # Step 8: Analyze explained variance like in kepler_pca.py
    print("\nAnalyzing explained variance (original method)...")
    explained_variance_ratio_orig, eigenvalues_orig, gram_eigenvalues_orig = analyzer.compute_explained_variance(full_model, dataset)
    explained_variance_ratio_pruned, eigenvalues_pruned, gram_eigenvalues_pruned = analyzer.compute_explained_variance(pruned_model_refined, dataset)

    print(f"Original model explained variance ratios: {explained_variance_ratio_orig[:5].detach().cpu().numpy()}")
    print(f"Pruned model explained variance ratios: {explained_variance_ratio_pruned[:len(keep_indices)].detach().cpu().numpy()}")

    # Save results
    torch.save(pruned_model_refined.state_dict(), "kepler_pruned_refined_model.pth")

    # Save experiment data
    saver = ExperimentSaver()

    # Regenerate trajectory data for saving (following the visualization code)
    test_sample = next(iter(DataLoader(dataset, batch_size=1)))
    M_central, y0, dt, y1, example_y0, example_dt, example_y1 = test_sample

    full_model.eval()
    pruned_model.eval()
    pruned_model_refined.eval()

    with torch.no_grad():
        coeffs_orig, _ = full_model.compute_coefficients((example_y0, example_dt), example_y1)
        coeffs_pruned, _ = pruned_model.compute_coefficients((example_y0, example_dt), example_y1)
        coeffs_pruned_refined, _ = pruned_model_refined.compute_coefficients((example_y0, example_dt), example_y1)

    _M_central = M_central[0]
    _y0 = generate_kepler_states_batch(
        _M_central.item(),
        dataset.a_range,
        dataset.e_range,
        1,
        device=torch.device(device),
    )

    _c_orig = coeffs_orig[0].unsqueeze(0)
    _c_pruned = coeffs_pruned[0].unsqueeze(0)
    _c_pruned_refined = coeffs_pruned_refined[0].unsqueeze(0)
    s = 0.1
    n = int(10.0 / s)
    _dt = torch.tensor([s], device=device)

    # True trajectory
    x = _y0.clone()
    y_true = [x]
    for k in range(n):
        x = rk4_step(kepler, x, _dt, M_central=_M_central) + x
        y_true.append(x)
    y_true_traj = torch.cat(y_true, dim=0).detach().cpu().numpy()

    # Original model trajectory
    x = _y0.clone().unsqueeze(1)
    _dt = _dt.unsqueeze(0)
    pred_orig = [x]
    for k in range(n):
        x = full_model((x, _dt), coefficients=_c_orig) + x
        pred_orig.append(x)
    pred_orig_traj = torch.cat(pred_orig, dim=1)[0].detach().cpu().numpy()

    # Pruned model trajectory
    x = _y0.clone().unsqueeze(1)
    pred_pruned = [x]
    for k in range(n):
        x = pruned_model((x, _dt), coefficients=_c_pruned) + x
        pred_pruned.append(x)
    pred_pruned_traj = torch.cat(pred_pruned, dim=1)[0].detach().cpu().numpy()

    # Refined model trajectory
    x = _y0.clone().unsqueeze(1)
    pred_refined = [x]
    for k in range(n):
        x = pruned_model_refined((x, _dt), coefficients=_c_pruned_refined) + x
        pred_refined.append(x)
    pred_refined_traj = torch.cat(pred_refined, dim=1)[0].detach().cpu().numpy()

    # Create visualization data
    viz_data = create_visualization_data_dynamics(
        trajectories_true=[y_true_traj],
        trajectories_pred=[pred_orig_traj, pred_pruned_traj, pred_refined_traj],
        initial_conditions=[_y0[0].cpu().numpy()],
        system_params=[_M_central.item()]
    )

    # We don't have separate train_losses for this script, so create empty list
    train_losses = []

    # Prepare experiment data
    experiment_data = saver.prepare_prune_data(
        problem_type="kepler",
        num_basis_original=len(full_model.basis_functions.basis_functions),
        num_basis_pruned=len(pruned_model_refined.basis_functions.basis_functions),
        train_losses=train_losses,
        finetune_losses=finetune_losses,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=explained_var,
        keep_indices=keep_indices,
        comparison_results=comparison_results,
        visualization_data=viz_data,
        dataset_params={
            "name": "kepler_dt01",
            "n_points": 1000,
            "n_example_points": 100,
            "dt_range": (0.1, 0.1)
        },
        training_params={
            "num_epochs_initial": 1000,
            "num_epochs_finetune": 100,
            "learning_rate": 1e-3,
            "batch_size": 50
        }
    )

    saver.save_experiment("kepler","prune", experiment_data, dataset_name="dt01")

    print(f"\nPruning completed!")
    print(f"Original model: {len(full_model.basis_functions.basis_functions)} basis functions")
    print(f"Pruned model: {len(pruned_model_refined.basis_functions.basis_functions)} basis functions")
    print(f"Compression ratio: {len(pruned_model_refined.basis_functions.basis_functions)/len(full_model.basis_functions.basis_functions):.1%}")
    print(f"Performance degradation: {comparison_results['mse_pruned_refined']/comparison_results['mse_original']:.3f}x")