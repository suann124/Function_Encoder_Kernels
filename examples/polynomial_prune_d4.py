import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import tqdm
from copy import deepcopy

from my_datasets.polynomial import PolynomialDataset

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step
from function_encoder.inner_products import standard_inner_product
from function_encoder.utils.experiment_saver import ExperimentSaver, create_visualization_data_polynomial


class TrainPruneAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def train_full_model(self,
                        num_basis: int,
                        dataset: PolynomialDataset,
                        num_epochs: int = 2000,
                        batch_size: int = 50) -> FunctionEncoder:
        """Train a model with all basis functions from scratch."""

        print(f"Training full model with {num_basis} basis functions...")

        # Create model with all basis functions
        def basis_function_factory():
            return MLP(layer_sizes=[1, 32, 1])

        all_basis_functions = BasisFunctions(*[basis_function_factory() for _ in range(num_basis)])
        model = FunctionEncoder(all_basis_functions).to(self.device)

        # Setup training
        dataloader = DataLoader(dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        # Training loop
        with tqdm.tqdm(range(num_epochs), desc="Training full model") as pbar:
            for epoch in pbar:
                batch = next(iter(dataloader))
                loss = train_step(model, optimizer, batch, self.loss_function)
                losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.2e}"})

        return all_basis_functions, model, losses

    def analyze_basis_importance(self,
                               model: FunctionEncoder,
                               dataset: PolynomialDataset,
                               num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze basis importance using PCA on coefficients."""

        print("Analyzing basis importance with PCA...")

        model.eval()
        dataloader = DataLoader(dataset, batch_size=num_samples)
        batch = next(iter(dataloader))

        with torch.no_grad():
            _, _, example_X, example_y = batch
            example_X = example_X.to(self.device)
            example_y = example_y.to(self.device)

            # Compute coefficients for all samples
            coefficients, G = model.compute_coefficients(example_X, example_y)
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

            # Project coefficients onto principal components
            pc_scores = coefficients_centered @ eigenvectors

            return eigenvalues, eigenvectors, explained_variance_ratio

    def cos_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify_redundant_basis(self,
                               eigenvalues: np.ndarray,
                               eigenvectors: np.ndarray,
                               explained_variance_ratio: np.ndarray,
                               basis_funcs: BasisFunctions,
                               variance_threshold: float = 0.99) -> List[int]:
        """Identify which basis functions to keep based on PCA analysis."""

        # Finding number of basis needed: Cumulative variance threshold
        cumsum_var = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1

        print(f"Need {n_components} components to explain {variance_threshold*100}% variance")

        # Find which original basis contribute most to top PCs
        # Look at the loadings (eigenvectors)
        n_basis = eigenvectors.shape[0]
        basis_importance = np.zeros(n_basis)

        # Method 1: Eigenvalues weighted PCs
        weighted_eig = np.zeros(n_basis)
        for i in range(n_components):
            weighted_eig += np.abs(eigenvectors[:, i]) * eigenvalues[i]
            best_aligned_basis = np.argsort(weighted_eig)[::-1][:n_components]

        print(f"Best aligned basis: {best_aligned_basis}")
        return sorted(best_aligned_basis.tolist())

    def prune_model(self,
                   model: FunctionEncoder,
                   keep_indices: List[int]) -> FunctionEncoder:
        """Create a pruned model keeping only specified basis functions."""

        print(f"Pruning model to keep {len(keep_indices)} basis functions...")

        # Create new model with fewer basis functions
        def basis_function_factory():
            return MLP(layer_sizes=[1, 32, 1])

        pruned_basis_functions = BasisFunctions(*[basis_function_factory() for _ in range(len(keep_indices))])
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
                             dataset: PolynomialDataset,
                             num_epochs: int = 500,
                             batch_size: int = 50) -> Tuple[FunctionEncoder, List[float]]:
        """Fine-tune the pruned model."""

        print("Fine-tuning pruned model...")
        model_to_tune = deepcopy(model)

        dataloader = DataLoader(dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(model_to_tune.parameters(), lr=5e-4)
        losses = []

        with tqdm.tqdm(range(num_epochs), desc="Fine-tuning") as pbar:
            for epoch in pbar:
                batch = next(iter(dataloader))
                loss = train_step(model_to_tune, optimizer, batch, self.loss_function)
                losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.2e}"})

        return model_to_tune, losses

    def compare_models(self,
                      original_model: FunctionEncoder,
                      pruned_model: FunctionEncoder,
                      pruned_model_refined: FunctionEncoder,
                      dataset: PolynomialDataset,
                      num_test_samples: int = 100):
        """Compare performance of original vs pruned model."""

        print("\nComparing model performance...")

        test_loader = DataLoader(dataset, batch_size=num_test_samples)
        batch = next(iter(test_loader))

        X, y, example_X, example_y = batch
        X = X.to(self.device)
        y = y.to(self.device)
        example_X = example_X.to(self.device)
        example_y = example_y.to(self.device)

        original_model.eval()
        pruned_model.eval()
        pruned_model_refined.eval()

        with torch.no_grad():
            # Original model predictions
            coeffs_orig, _ = original_model.compute_coefficients(example_X, example_y)
            y_pred_orig = original_model(X, coeffs_orig)
            mse_orig = torch.nn.functional.mse_loss(y_pred_orig, y).item()

            # Pruned model predictions
            coeffs_pruned, _ = pruned_model.compute_coefficients(example_X, example_y)
            y_pred_pruned = pruned_model(X, coeffs_pruned)
            mse_pruned = torch.nn.functional.mse_loss(y_pred_pruned, y).item()

            # Pruned Refined model predictions
            coeffs_pruned_refined, _ = pruned_model_refined.compute_coefficients(example_X, example_y)
            y_pred_pruned_refined = pruned_model_refined(X, coeffs_pruned_refined)
            mse_pruned_refined = torch.nn.functional.mse_loss(y_pred_pruned_refined, y).item()

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
        """Loss function for training."""
        X, y, example_X, example_y = batch
        X = X.to(self.device)
        y = y.to(self.device)
        example_X = example_X.to(self.device)
        example_y = example_y.to(self.device)

        coefficients, G = model.compute_coefficients(example_X, example_y)
        y_pred = model(X, coefficients)

        pred_loss = torch.nn.functional.mse_loss(y_pred, y)
        return pred_loss


# ============================== Main ===============================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize
    analyzer = TrainPruneAnalyzer()

    # Create dataset with degree 4
    n_points = 100
    dataset = PolynomialDataset(n_points=100, n_example_points=100, degree=4)

    # Step 1: Train full model
    num_basis = 20  # Start with many basis functions
    all_basis_functions, full_model, train_losses = analyzer.train_full_model(num_basis, dataset, num_epochs=2000)

    # Step 2: Analyze basis importance
    eigenvalues, eigenvectors, explained_var = analyzer.analyze_basis_importance(full_model, dataset)

    # Step 3: Identify which basis to keep
    keep_indices = analyzer.identify_redundant_basis(eigenvalues, eigenvectors, explained_var, all_basis_functions, variance_threshold=0.99)
    print(f"\nKeeping basis functions at indices: {keep_indices}")

    # Step 4: Create pruned model
    pruned_model = analyzer.prune_model(full_model, keep_indices)

    # Step 5: Fine-tune pruned model
    pruned_model_refined, finetune_losses = analyzer.fine_tune_pruned_model(pruned_model, dataset, num_epochs=500)

    # Step 6: Compare performance
    comparison_results = analyzer.compare_models(full_model, pruned_model, pruned_model_refined, dataset)

    # Step 7: Save experiment data
    saver = ExperimentSaver()

    # Get variables for visualization
    test_sample = next(iter(DataLoader(dataset, batch_size=1)))
    X, y, example_X, example_y = test_sample
    idx = torch.argsort(X[0,:,0])
    X_sorted = X[0,:,0][idx].cpu().numpy()
    y_sorted = y[0,:,0][idx].cpu().numpy()

    # Get predictions for visualization
    with torch.no_grad():
        coeffs_orig, _ = full_model.compute_coefficients(example_X.to(analyzer.device), example_y.to(analyzer.device))
        y_pred_orig = full_model(X.to(analyzer.device), coeffs_orig)[0,:,0][idx].cpu().numpy()

    # Compute basis function outputs for visualization
    X_plot = torch.linspace(-1, 1, 100).unsqueeze(0).unsqueeze(2).to(analyzer.device)

    # Get basis outputs for the original model (all basis functions)
    with torch.no_grad():
        basis_outputs_all = []
        for i in range(len(full_model.basis_functions.basis_functions)):
            basis_output = full_model.basis_functions.basis_functions[i](X_plot)
            basis_outputs_all.append(basis_output[0, :, 0].cpu().numpy())

    # Prepare visualization data
    viz_data = create_visualization_data_polynomial(
        X_sorted=X_sorted,
        y_sorted=y_sorted,
        y_pred=y_pred_orig,
        example_X=example_X[0].cpu().numpy(),
        example_y=example_y[0].cpu().numpy(),
        basis_outputs=basis_outputs_all
    )

    # Prepare and save experiment data
    experiment_data = saver.prepare_prune_data(
        problem_type="polynomial",
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
            "name": "poly_degree4",
            "n_points": n_points,
            "n_example_points": 100,
            "degree": 4
        },
        training_params={
            "num_epochs_initial": 2000,
            "num_epochs_finetune": 500,
            "learning_rate": 1e-3,
            "batch_size": 50
        }
    )

    saver.save_experiment("polynomial", "prune", experiment_data, dataset_name="d4")
    print("Experiment saved successfully!")