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
                               model: FunctionEncoder = None,
                               X: torch.Tensor = None,
                               variance_threshold: float = 0.99) -> List[int]:
        """Identify which basis functions to keep based on PCA analysis."""
        
        # Finding number of basis needed: Cumulative variance threshold
        cumsum_var = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        
        print(f"Need {n_components} components to explain {variance_threshold*100}% variance")
        
        # Find which original basis contribute most to top PCs
        n_basis = eigenvectors.shape[0]
        
        # Convert to torch tensors for cosine similarity computation
        eigenvalues_torch = torch.tensor(eigenvalues, device=self.device, dtype=torch.float32)
        eigenvectors_torch = torch.tensor(eigenvectors, device=self.device, dtype=torch.float32)
        
        # Get top n eigenvectors
        top_eigenvectors = eigenvectors_torch[:, :n_components]  # [num_basis, n_components]
        
        if X is None:
            X = torch.linspace(-1, 1, 100).unsqueeze(0).unsqueeze(2).to(self.device)
        
        # Store model for reconstruction error computation
        self.model = model
        
        # Compute cosine similarity between eigenvector functions and basis functions
        similarity_scores, eigenvector_outputs = self.compute_cosine_similarity_alignment(
            model, basis_funcs, top_eigenvectors, X, n_components
        )
        
        print(f"\nCosine similarity matrix:")
        print(f"Shape: {similarity_scores.shape} (eigenvectors x basis functions)")
        for i in range(n_components):
            print(f"Eigenvector {i+1} cosine similarities: {similarity_scores[i, :].detach().cpu().numpy()}")
        
        # Select basis functions based on multiple methods including reconstruction error
        selected_indices = self.select_basis_functions_from_similarity(
            similarity_scores, eigenvalues_torch[:n_components], n_components,
            model, eigenvector_outputs, X
        )
        
        return sorted(selected_indices)

    def compute_cosine_similarity_alignment(self, 
                                          model: FunctionEncoder,
                                          basis_funcs: BasisFunctions,
                                          top_eigenvectors: torch.Tensor,
                                          X: torch.Tensor,
                                          n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine similarity between eigenvector functions and basis functions."""
        
        num_basis = len(basis_funcs.basis_functions)
        
        # Compute eigenvector outputs
        eigenvector_outputs = []
        for i in range(n_components):
            eigenvector_output = model(
                X, coefficients=top_eigenvectors[:, i:i+1].T
            )
            eigenvector_outputs.append(eigenvector_output)
        
        eigenvector_outputs = torch.stack(eigenvector_outputs, dim=1)
        
        # Get basis function outputs
        basis_outputs = basis_funcs(X)  # [batch_size, n_points, n_features, num_basis]
        
        # Compute cosine similarity between each basis function and each eigenvector function
        similarity_scores = torch.zeros(n_components, num_basis, device=self.device)
        
        for i in range(n_components):
            # eigenvector_outputs[:, i]: [batch_size, n_points, n_features]
            eigenvec_output = eigenvector_outputs[:, i].unsqueeze(-1)  # [batch_size, n_points, n_features, 1]
            
            # Compute inner product with all basis functions using model's inner product
            inner_prod_matrix = model.inner_product(
                basis_outputs, eigenvec_output
            )  # [batch_size, num_basis, 1]
            
            # Compute norms of basis functions
            basis_norms_squared = torch.zeros(X.shape[0], num_basis, device=self.device)
            for j in range(num_basis):
                basis_j = basis_outputs[:, :, :, j:j+1]  # [batch_size, n_points, n_features, 1]
                basis_norm_sq = model.inner_product(basis_j, basis_j)  # [batch_size, 1, 1]
                basis_norms_squared[:, j] = basis_norm_sq.squeeze(-1).squeeze(-1)  # [batch_size]
            
            basis_norms = torch.sqrt(basis_norms_squared)  # [batch_size, num_basis]
            
            # Compute norm of eigenvector function
            eigenvec_norm_sq = model.inner_product(
                eigenvec_output, eigenvec_output
            )  # [batch_size, 1, 1]
            eigenvec_norm = torch.sqrt(
                eigenvec_norm_sq.squeeze(-1).squeeze(-1)
            )  # [batch_size]
            
            # Compute cosine similarity: inner_product / (norm1 * norm2)
            eps = 1e-8
            cosine_sim = inner_prod_matrix.squeeze(-1) / (
                basis_norms * eigenvec_norm.unsqueeze(-1) + eps
            )  # [batch_size, num_basis]
            
            # Average over batch dimension
            similarity_scores[i, :] = cosine_sim.mean(dim=0)  # [num_basis]
        
        return similarity_scores, eigenvector_outputs

    def select_basis_functions_from_similarity(self, 
                                             similarity_scores: torch.Tensor,
                                             eigenvalues: torch.Tensor,
                                             n_components: int,
                                             model: FunctionEncoder,
                                             eigenvector_outputs: torch.Tensor,
                                             X: torch.Tensor) -> List[int]:
        """Select basis functions based on cosine similarity and reconstruction error methods."""
        
        num_basis = similarity_scores.shape[1]
        
        # Method 1: Select based on highest cosine similarity to any eigenvector function
        max_similarity_per_basis = torch.max(torch.abs(similarity_scores), dim=0)[0]  # [num_basis]
        print(f"\nMax absolute cosine similarity per basis function: {max_similarity_per_basis.detach().cpu().numpy()}")
        
        # Method 2: Select based on weighted sum of cosine similarities
        eigenvalue_weights = eigenvalues / eigenvalues.sum()  # Normalize eigenvalues
        weighted_similarity = torch.zeros(num_basis, device=self.device)
        for i in range(n_components):
            weighted_similarity += torch.abs(similarity_scores[i, :]) * eigenvalue_weights[i]
        
        print(f"Weighted cosine similarity scores: {weighted_similarity.detach().cpu().numpy()}")
        
        # Method 3: Reconstruction Error Minimization (Greedy Forward Selection)
        print(f"\n=== METHOD 3: Reconstruction Error Minimization ===")
        selected_indices_method3 = self.greedy_reconstruction_selection(
            model, eigenvector_outputs, X, eigenvalue_weights, n_components, num_basis
        )
        
        # Compare methods
        print(f"\n=== METHOD 1: Select by Maximum Cosine Similarity ===")
        _, selected_indices_method1 = torch.topk(max_similarity_per_basis, k=n_components, largest=True)
        selected_indices_method1 = selected_indices_method1.sort().values
        print(f"Selected basis functions: {selected_indices_method1.tolist()}")
        for idx in selected_indices_method1.tolist():
            max_sim = max_similarity_per_basis[idx].item()
            best_eigenvec = torch.argmax(torch.abs(similarity_scores[:, idx])).item()
            print(f"  Basis {idx+1}: max_similarity={max_sim:.6f} (with eigenvector {best_eigenvec+1})")
        
        print(f"\n=== METHOD 2: Select by Weighted Cosine Similarity ===")
        _, selected_indices_method2 = torch.topk(weighted_similarity, k=n_components, largest=True)
        selected_indices_method2 = selected_indices_method2.sort().values
        print(f"Selected basis functions: {selected_indices_method2.tolist()}")
        for idx in selected_indices_method2.tolist():
            weighted_sim = weighted_similarity[idx].item()
            print(f"  Basis {idx+1}: weighted_similarity={weighted_sim:.6f}")
        
        # Test all methods and choose the best one
        methods = [selected_indices_method1.tolist(), selected_indices_method2.tolist(), selected_indices_method3]
        method_names = ["Max Cosine Similarity", "Weighted Cosine Similarity", "Reconstruction Error"]
        
        best_error = float('inf')
        best_method_idx = 0
        best_indices = selected_indices_method1.tolist()
        
        eigenvector_funcs_list = [eigenvector_outputs[:, i] for i in range(n_components)]
        
        for i, indices in enumerate(methods):
            error = self.compute_reconstruction_error(indices, eigenvector_funcs_list, X, eigenvalue_weights)
            print(f"Method {i+1} ({method_names[i]}) reconstruction error: {error:.6f}")
            if error < best_error:
                best_error = error
                best_method_idx = i
                best_indices = indices
        
        print(f"\nUsing Method {best_method_idx + 1} ({method_names[best_method_idx]}) - lowest reconstruction error")
        print(f"Best reconstruction error: {best_error:.6f}")
        return best_indices

    def greedy_reconstruction_selection(self, model, eigenvector_outputs, X, eigenvalue_weights, n_components, num_basis):
        """Greedy forward selection based on reconstruction error."""
        print("Running greedy forward selection based on reconstruction error...")
        selected_indices = []
        remaining_indices = list(range(num_basis))
        
        eigenvector_funcs_list = [eigenvector_outputs[:, i] for i in range(n_components)]
        
        for step in range(n_components):
            best_error = float("inf")
            best_idx = None
            
            print(f"  Step {step+1}/{n_components}: Evaluating {len(remaining_indices)} candidates...")
            
            for candidate_idx in remaining_indices:
                candidate_subset = selected_indices + [candidate_idx]
                error = self.compute_reconstruction_error(candidate_subset, eigenvector_funcs_list, X, eigenvalue_weights)
                if error < best_error:
                    best_error = error
                    best_idx = candidate_idx
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            print(f"    Selected basis {best_idx+1} (error: {best_error:.6f})")
        
        return sorted(selected_indices)

    def compute_reconstruction_error(self, subset_indices, eigenvector_funcs, X, eigenvalue_weights):
        """Compute reconstruction error for a given subset of basis functions."""
        if len(subset_indices) == 0:
            return float("inf")
        
        # Create temporary model with selected basis functions
        temp_basis_functions = BasisFunctions()
        for idx in subset_indices:
            temp_basis_functions.basis_functions.append(
                self.model.basis_functions.basis_functions[idx]
            )
        temp_model = FunctionEncoder(temp_basis_functions).to(self.device)
        
        total_error = 0.0
        total_norm = 0.0
        
        for i, target_func in enumerate(eigenvector_funcs):
            # Fit coefficients for this target using subset of basis functions
            coefficients, _ = temp_model.compute_coefficients(X, target_func)
            
            # Reconstruct using fitted coefficients
            reconstructed = temp_model(X, coefficients=coefficients)
            
            # Compute L2 error
            error = torch.nn.functional.mse_loss(reconstructed, target_func)
            norm = torch.nn.functional.mse_loss(target_func, torch.zeros_like(target_func))
            
            # Weight by eigenvalue importance
            weight = eigenvalue_weights[i] if i < len(eigenvalue_weights) else 0
            total_error += error * weight
            total_norm += norm * weight
        
        # Return relative error
        return (total_error / (total_norm + 1e-8)).item()
    
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
        print(f"Performance ratio (refned): {mse_pruned_refined/mse_orig:.3f}")
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
    
    def visualize_results(self, 
                         original_model: FunctionEncoder,
                         pruned_model: FunctionEncoder,
                         pruned_model_refined: FunctionEncoder,
                         eigenvalues: np.ndarray,
                         explained_variance_ratio: np.ndarray,
                         keep_indices: List[int],
                         comparison_results: dict,
                         dataset: PolynomialDataset):
        """Visualize the pruning results."""
        
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
        
        # 4. Function approximation comparison
        ax = axes[1, 0]
        test_sample = next(iter(DataLoader(dataset, batch_size=1)))
        X, y, example_X, example_y = test_sample

        idx = torch.argsort(X[0,:,0])
        X_sorted = X[0,:,0][idx].cpu().numpy()
        y_sorted = y[0,:,0][idx].cpu().numpy()

        original_model.eval()
        pruned_model.eval()
        pruned_model_refined.eval()

        # Recompute predictions for the local test sample
        with torch.no_grad():
            coeffs_orig, _ = original_model.compute_coefficients(example_X.to(self.device), example_y.to(self.device))
            y_pred_orig = original_model(X.to(self.device), coeffs_orig)[0,:,0][idx].cpu().numpy()

            coeffs_pruned, _ = pruned_model.compute_coefficients(example_X.to(self.device), example_y.to(self.device))
            y_pred_pruned = pruned_model(X.to(self.device), coeffs_pruned)[0,:,0][idx].cpu().numpy()

            coeffs_pruned_refined, _ = pruned_model_refined.compute_coefficients(example_X.to(self.device), example_y.to(self.device))
            y_pred_pruned_refined = pruned_model_refined(X.to(self.device), coeffs_pruned_refined)[0,:,0][idx].cpu().numpy()

        ax.plot(X_sorted, y_sorted, 'k-', label='True', linewidth=1)
        ax.plot(X_sorted, y_pred_orig, 'b--', label='Original', alpha=0.8)
        ax.plot(X_sorted, y_pred_pruned, 'g--', label='Pruned', alpha=0.8)
        ax.plot(X_sorted, y_pred_pruned_refined, 'r:', label='Pruned & Refined', linewidth=2)
        ax.scatter(example_X[0].cpu(), example_y[0].cpu(), c='red', s=20, zorder=5, alpha=0.5, label='Example Points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Function Approximation Comparison')
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
    analyzer = TrainPruneAnalyzer()
    
    # Create dataset
    n_points=100
    dataset = PolynomialDataset(n_points=100, n_example_points=100, degree=3)
    
    # Step 1: Train full model
    num_basis = 20  # Start with many basis functions
    all_basis_functions, full_model, train_losses = analyzer.train_full_model(num_basis, dataset, num_epochs=1000)
    
    # Step 2: Analyze basis importance
    eigenvalues, eigenvectors, explained_var = analyzer.analyze_basis_importance(full_model, dataset)

    # Step 3: Identify which basis to keep
    # Get sample X data for the new method
    dataloader = DataLoader(dataset, batch_size=1000)
    batch = next(iter(dataloader))
    _, _, sample_X, _ = batch
    sample_X = sample_X.to(analyzer.device)
    
    keep_indices = analyzer.identify_redundant_basis(eigenvalues, eigenvectors, explained_var, all_basis_functions, 
                                                   model=full_model, X=sample_X, variance_threshold=0.99)
    print(f"\nKeeping basis functions at indices: {keep_indices}")
    
    # Step 4: Create pruned model
    pruned_model = analyzer.prune_model(full_model, keep_indices)
    
    # Step 5: Fine-tune pruned model
    pruned_model_refined, finetune_losses = analyzer.fine_tune_pruned_model(pruned_model, dataset, num_epochs=1000)
    
    # Step 6: Compare performance
    comparison_results = analyzer.compare_models(full_model, pruned_model, pruned_model_refined, dataset)
    
    # Print final summary
    print("\n" + "="*60)
    print("POLYNOMIAL PRUNING RESULTS WITH COSINE SIMILARITY")
    print("="*60)
    print(f"Original model: {num_basis} basis functions")
    print(f"Pruned model: {len(keep_indices)} basis functions ({100*len(keep_indices)/num_basis:.1f}% of original)")
    print(f"Selected basis function indices: {keep_indices}")
    print(f"Original model MSE: {comparison_results['mse_original']:.6e}")
    print(f"Pruned model MSE: {comparison_results['mse_pruned']:.6e}")
    print(f"Pruned & Refined model MSE: {comparison_results['mse_pruned_refined']:.6e}")
    print(f"Performance degradation (refined): {100*(comparison_results['mse_pruned_refined'] - comparison_results['mse_original'])/comparison_results['mse_original']:+.2f}%")
    print(f"Model compression ratio: {num_basis/len(keep_indices):.1f}x")
    print("="*60)