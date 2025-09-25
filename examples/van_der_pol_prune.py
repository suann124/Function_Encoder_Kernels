import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import tqdm
from copy import deepcopy

from my_datasets.van_der_pol import VanDerPolDataset, van_der_pol

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step
from function_encoder.coefficients import lasso
from function_encoder.inner_products import standard_inner_product
from function_encoder.utils.experiment_saver import ExperimentSaver, create_visualization_data_dynamics


class VanDerPolPruneAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def train_full_model(self,
                        num_basis: int,
                        dataset: VanDerPolDataset,
                        num_epochs: int = 1000,
                        batch_size: int = 50) -> Tuple[BasisFunctions, FunctionEncoder, List[float]]:
        """Train a model with all basis functions from scratch."""

        print(f"Training full model with {num_basis} basis functions...")

        # Create model with all basis functions (NeuralODE)
        def basis_function_factory():
            return NeuralODE(
                ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
                integrator=rk4_step,
            )

        all_basis_functions = BasisFunctions(*[basis_function_factory() for _ in range(num_basis)])
        model = FunctionEncoder(all_basis_functions).to(self.device)

        # Setup training - reuse dataloader iterator
        dataloader = DataLoader(dataset, batch_size=batch_size)
        data_iter = iter(dataloader)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        # Training loop with proper data iteration
        with tqdm.tqdm(range(num_epochs), desc="Training full model") as pbar:
            for epoch in pbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)  # Reset iterator when exhausted
                    batch = next(data_iter)

                loss = train_step(model, optimizer, batch, self.loss_function)
                losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.2e}"})

                # Clear GPU cache periodically
                if epoch % 100 == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()

        return all_basis_functions, model, losses

    def analyze_basis_importance(self,
                               model: FunctionEncoder,
                               dataset: VanDerPolDataset,
                               num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze basis importance using PCA on coefficients."""

        print("Analyzing basis importance with PCA...")

        model.eval()
        dataloader = DataLoader(dataset, batch_size=num_samples)
        batch = next(iter(dataloader))

        with torch.no_grad():
            mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
            y0 = y0.to(self.device)
            dt = dt.to(self.device)
            y1 = y1.to(self.device)
            y0_example = y0_example.to(self.device)
            dt_example = dt_example.to(self.device)
            y1_example = y1_example.to(self.device)

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
                               basis_funcs: BasisFunctions,
                               model: FunctionEncoder,
                               X: Tuple[torch.Tensor, torch.Tensor] = None,
                               variance_threshold: float = 0.99) -> List[int]:
        """Identify which basis functions to keep based on total contribution to top PCs."""

        # 1. Finding number of components needed (this part is correct)
        cumsum_var = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        print(f"Need {n_components} components to explain {variance_threshold*100}% variance")

        # ============= Method 1: Eigenvalues weighted PCs =============
        print("Using eigenvalue-weighted PCA loadings to identify important basis functions...")
        n_basis = eigenvectors.shape[0]
        weighted_eig = np.zeros(n_basis)
        for i in range(n_components):
            weighted_eig += np.abs(eigenvectors[:, i]) * eigenvalues[i]   
            best_aligned_basis = np.argsort(weighted_eig)[::-1][:n_components]     

        # ============= Method 2: Total contribution to top PCs =============
        # print("Using total contribution to top PCs to identify important basis functions...")
        # 2. Get the top 'n_components' eigenvectors
        # top_eigenvectors = eigenvectors[:, :n_components]

        # # 3. Calculate the total contribution of each original basis function
        # #    by summing its squared loadings across all top components.
        # #    This measures the total energy of each basis in the principal subspace.
        # basis_importance = np.sum(top_eigenvectors**2, axis=1)

        # # 4. Select the indices of the basis functions with the highest importance scores.
        # #    The number of basis functions to keep should be equal to n_components.
        # keep_indices = np.argsort(basis_importance)[::-1][:n_components]

        # print(f"Basis importance scores: {basis_importance}")
        # print(f"Top {n_components} most important basis indices: {np.sort(keep_indices)}")

        # return sorted(keep_indices.tolist())

        # ================= Method 3: PC Components loadings =============
        # print("Using PCA loadings to identify important basis functions...")
        # Find which original basis contribute most to top PCs
        # Look at the loadings (eigenvectors)
        # n_basis = eigenvectors.shape[0]
        # top_k_eig_idx = np.argsort(eigenvalues)[::-1][:n_components] #sort, reverse, slice to k basis
        # important_pcs = eigenvectors[:, top_k_eig_idx]
        # basis_alignment = important_pcs                                            # (n_basis, k)
        # alignment_abs = np.abs(basis_alignment)
        # best_aligned_basis = np.argmax(alignment_abs, axis=0)
        
        # # ================= prints =================
        # # print(basis_alignment)
        # print(f"Best aligned basis: {best_aligned_basis}")
        return sorted(best_aligned_basis.tolist())

    def prune_model(self,
                   model: FunctionEncoder,
                   keep_indices: List[int]) -> FunctionEncoder:
        """Create a pruned model keeping only specified basis functions."""

        print(f"Pruning model to keep {len(keep_indices)} basis functions...")

        # Create new model with fewer basis functions
        def basis_function_factory():
            return NeuralODE(
                ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
                integrator=rk4_step,
            )

        pruned_basis_functions = BasisFunctions(*[basis_function_factory() for _ in range(len(keep_indices))])
        pruned_model = FunctionEncoder(pruned_basis_functions
                                    #    , coefficients_method=lasso
                                    ).to(self.device)

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
                             dataset: VanDerPolDataset,
                             num_epochs: int = 1000,
                             batch_size: int = 50) -> Tuple[FunctionEncoder, List[float]]:
        """Fine-tune the pruned model."""

        print("Fine-tuning pruned model...")
        model_to_tune = deepcopy(model)

        # Reuse dataloader iterator like in train_full_model
        dataloader = DataLoader(dataset, batch_size=batch_size)
        data_iter = iter(dataloader)
        optimizer = torch.optim.Adam(model_to_tune.parameters(), lr=5e-4)
        losses = []

        with tqdm.tqdm(range(num_epochs), desc="Fine-tuning") as pbar:
            for epoch in pbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)  # Reset iterator when exhausted
                    batch = next(data_iter)

                loss = train_step(model_to_tune, optimizer, batch, self.loss_function)
                losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.2e}"})

                # Clear GPU cache periodically
                if epoch % 100 == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()

        return model_to_tune, losses

    def compare_models(self,
                      original_model: FunctionEncoder,
                      pruned_model: FunctionEncoder,
                      pruned_model_refined: FunctionEncoder,
                      dataset: VanDerPolDataset,
                      num_test_samples: int = 100):
        """Compare performance of original vs pruned model."""

        print("\nComparing model performance...")

        test_loader = DataLoader(dataset, batch_size=num_test_samples)
        batch = next(iter(test_loader))

        mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
        y0 = y0.to(self.device)
        dt = dt.to(self.device)
        y1 = y1.to(self.device)
        y0_example = y0_example.to(self.device)
        dt_example = dt_example.to(self.device)
        y1_example = y1_example.to(self.device)

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
        """Loss function for training Van der Pol dataset."""
        mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
        y0 = y0.to(self.device)
        dt = dt.to(self.device)
        y1 = y1.to(self.device)
        y0_example = y0_example.to(self.device)
        dt_example = dt_example.to(self.device)
        y1_example = y1_example.to(self.device)

        coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)
        y_pred = model((y0, dt), coefficients)

        pred_loss = torch.nn.functional.mse_loss(y_pred, y1)
        return pred_loss

    def visualize_results(self,
                         original_model: FunctionEncoder,
                         pruned_model: FunctionEncoder,
                         pruned_model_refined: FunctionEncoder,
                         eigenvalues: np.ndarray,
                         explained_variance_ratio: np.ndarray,
                         keep_indices: List[int],
                         comparison_results: dict,
                         dataset: VanDerPolDataset):
        """Visualize the pruning results for Van der Pol system."""

        # Publication formatting
        plt.rcParams.update({
            'font.size': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'png',
            'lines.markersize': 3,
            'legend.fontsize': 6,
            'legend.handlelength': 1.0,
            'legend.handletextpad': 0.3,
            'legend.columnspacing': 0.5
        })

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Eigenvalue spectrum - show ALL eigenvalues, not just the ones needed
        ax = axes[0, 0]
        plot_eigenvalues = np.maximum(eigenvalues, 1e-50) 
        ax.semilogy(plot_eigenvalues, 'b.-', label='Eigenvalues') 
        ax.axvline(x=len(keep_indices)-1, color='r', linestyle='--', label=f'Cutoff (n={len(keep_indices)})')
        ax.set_xlabel('Component')
        ax.set_ylabel('Eigenvalue')
        ax.set_yscale('log')  # Use log scale to see small eigenvalues
        ax.legend()
        # ax.set_ylim(0, 1e-50)
        ax.grid(True)

        # 2. Cumulative explained variance
        ax = axes[0, 1]
        cumsum_var = np.cumsum(explained_variance_ratio)
        ax.plot(cumsum_var, 'g.-')
        ax.axhline(y=0.99, color='r', linestyle='--', label='99% threshold')
        ax.axvline(x=len(keep_indices)-1, color='r', linestyle='--')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
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

        # 4. Van der Pol trajectory comparison - follow van_der_pol.py approach
        ax = axes[1, 0]
        test_sample = next(iter(DataLoader(dataset, batch_size=1)))
        mu, y0, dt, y1, y0_example, dt_example, y1_example = test_sample

        # Generate a single trajectory for visualization
        mu_val = mu[0].item()
        _y0 = torch.empty(1, 2, device=self.device).uniform_(-2.0, 2.0)  # Smaller range to prevent instability
        s = 0.1  # Time step for simulation like van_der_pol.py
        n = min(int(10 / s), 100)  # Reduced from 25/s to 10/s, max 100 steps to prevent crashes
        _dt = torch.tensor([s], device=self.device)

        original_model.eval()
        pruned_model.eval()
        pruned_model_refined.eval()

        with torch.no_grad():
            # Get coefficients from example data - use first sample
            coeffs_orig, _ = original_model.compute_coefficients((y0_example.to(self.device), dt_example.to(self.device)), y1_example.to(self.device))
            coeffs_pruned, _ = pruned_model.compute_coefficients((y0_example.to(self.device), dt_example.to(self.device)), y1_example.to(self.device))
            coeffs_pruned_refined, _ = pruned_model_refined.compute_coefficients((y0_example.to(self.device), dt_example.to(self.device)), y1_example.to(self.device))

            # Use first sample coefficients
            _c_orig = coeffs_orig[0].unsqueeze(0)
            _c_pruned = coeffs_pruned[0].unsqueeze(0)
            _c_refined = coeffs_pruned_refined[0].unsqueeze(0)

            # Integrate the TRUE trajectory with bounds checking
            x = _y0.clone()
            y_true = [x]
            for k in range(n):
                x_new = rk4_step(van_der_pol, x, _dt, mu=mu_val) + x
                # Check for NaN or explosion
                if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                    print(f"Warning: True trajectory became unstable at step {k}")
                    break
                x = x_new
                y_true.append(x)
            y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()

            # Integrate the ORIGINAL model trajectory with bounds checking
            x = _y0.clone()
            x = x.unsqueeze(1)
            _dt_model = _dt.unsqueeze(0)
            y_orig = [x]
            for k in range(n):
                x_new = original_model((x, _dt_model), coefficients=_c_orig) + x
                # Check for NaN or explosion
                if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                    print(f"Warning: Original model trajectory became unstable at step {k}")
                    break
                x = x_new
                y_orig.append(x)
            y_orig = torch.cat(y_orig, dim=1)[0].detach().cpu().numpy()

            # Integrate the PRUNED model trajectory with bounds checking
            x = _y0.clone()
            x = x.unsqueeze(1)
            y_pruned = [x]
            for k in range(n):
                x_new = pruned_model((x, _dt_model), coefficients=_c_pruned) + x
                # Check for NaN or explosion
                if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                    print(f"Warning: Pruned model trajectory became unstable at step {k}")
                    break
                x = x_new
                y_pruned.append(x)
            y_pruned = torch.cat(y_pruned, dim=1)[0].detach().cpu().numpy()

            # Integrate the REFINED model trajectory with bounds checking
            x = _y0.clone()
            x = x.unsqueeze(1)
            y_refined = [x]
            for k in range(n):
                x_new = pruned_model_refined((x, _dt_model), coefficients=_c_refined) + x
                # Check for NaN or explosion
                if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                    print(f"Warning: Refined model trajectory became unstable at step {k}")
                    break
                x = x_new
                y_refined.append(x)
            y_refined = torch.cat(y_refined, dim=1)[0].detach().cpu().numpy()

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.plot(y_true[:, 0], y_true[:, 1], 'k-', label='True', linewidth=2)
        ax.plot(y_orig[:, 0], y_orig[:, 1], 'b--', label='Original', alpha=0.8, linewidth=2)
        ax.plot(y_pruned[:, 0], y_pruned[:, 1], 'g--', label='Pruned', alpha=0.8, linewidth=2)
        ax.plot(y_refined[:, 0], y_refined[:, 1], 'r:', label='Pruned & Refined', linewidth=2)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        ax.grid(True)

        # 5. Coefficient comparison - exactly like polynomial_prune.py
        ax = axes[1, 1]
        coeffs_orig = comparison_results['coeffs_original'][0].cpu().numpy()
        coeffs_pruned = comparison_results['coeffs_pruned'][0].cpu().numpy()
        coeffs_pruned_refined = comparison_results['coeffs_pruned_refined'][0].cpu().numpy()

        # Define the x-positions for all original basis functions
        all_indices = np.arange(len(coeffs_orig))
        
        # Plot the original model's coefficients
        ax.bar(all_indices - 0.2, coeffs_orig, width=0.2, label='Original', alpha=0.8)

        # Use keep_indices for the x-positions of the pruned models' bars
        keep_indices_np = np.array(keep_indices)
        # Plot the pruned model's coefficients at their original indices
        ax.bar(keep_indices_np, coeffs_pruned, width=0.2, label='Pruned', alpha=0.8)
        # Plot the refined model's coefficients at their original indices
        ax.bar(keep_indices_np + 0.2, coeffs_pruned_refined, width=0.2, label='Pruned & Refined', alpha=0.8)

        ax.set_xlabel('Basis Index')
        ax.set_ylabel('Coefficient Value')
        
        # Ensure all original basis indices are shown on the x-axis
        ax.set_xticks(all_indices)
        ax.legend()
        ax.grid(True, alpha=0.3)

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
        plt.savefig('plots/van_der_pol_prune.png', dpi=300, bbox_inches='tight')
        plt.show()


# ============================== Main ===============================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize
    analyzer = VanDerPolPruneAnalyzer()

    # Create Van der Pol dataset
    dataset = VanDerPolDataset(integrator=rk4_step, n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))

    # Step 1: Train full model
    num_basis = 10  # Start with many basis functions (reduced for faster training)
    all_basis_functions, full_model, train_losses = analyzer.train_full_model(num_basis, dataset, num_epochs=1000)

    # Create a deep copy of the original model to preserve it for comparison
    import copy
    original_full_model = copy.deepcopy(full_model)

    # Step 2: Analyze basis importance
    eigenvalues, eigenvectors, explained_var = analyzer.analyze_basis_importance(full_model, dataset)

    # Step 3: Identify which basis to keep
    keep_indices = analyzer.identify_redundant_basis(eigenvalues, eigenvectors, explained_var, all_basis_functions, full_model, variance_threshold=0.99)
    print(f"\nKeeping basis functions at indices: {keep_indices}")

    # Step 4: Create pruned model
    pruned_model = analyzer.prune_model(full_model, keep_indices)

    # Step 5: Fine-tune pruned model
    pruned_model_refined, finetune_losses = analyzer.fine_tune_pruned_model(pruned_model, dataset, num_epochs=1000)

    # Step 6: Compare performance
    comparison_results = analyzer.compare_models(original_full_model, pruned_model, pruned_model_refined, dataset)

    # Step 7: Visualize results
    analyzer.visualize_results(original_full_model, pruned_model, pruned_model_refined, eigenvalues, explained_var,
                              keep_indices, comparison_results, dataset)

    # Save model
    torch.save(full_model.state_dict(), "van_der_pol_full_model.pth")
    torch.save(pruned_model_refined.state_dict(), "van_der_pol_pruned_model.pth")

    # Save experiment data
    saver = ExperimentSaver()

    # Regenerate trajectory data for visualization
    test_sample = next(iter(DataLoader(dataset, batch_size=1)))
    mu, y0, dt, y1, y0_example, dt_example, y1_example = test_sample

    mu_val = mu[0].item()
    _y0 = torch.empty(1, 2, device=analyzer.device).uniform_(-2.0, 2.0)
    s = 0.1
    n = min(int(10 / s), 100)
    _dt = torch.tensor([s], device=analyzer.device)

    original_full_model.eval()
    pruned_model.eval()
    pruned_model_refined.eval()

    with torch.no_grad():
        # Get coefficients
        coeffs_orig, _ = original_full_model.compute_coefficients((y0_example.to(analyzer.device), dt_example.to(analyzer.device)), y1_example.to(analyzer.device))
        coeffs_pruned, _ = pruned_model.compute_coefficients((y0_example.to(analyzer.device), dt_example.to(analyzer.device)), y1_example.to(analyzer.device))
        coeffs_pruned_refined, _ = pruned_model_refined.compute_coefficients((y0_example.to(analyzer.device), dt_example.to(analyzer.device)), y1_example.to(analyzer.device))

        _c_orig = coeffs_orig[0].unsqueeze(0)
        _c_pruned = coeffs_pruned[0].unsqueeze(0)
        _c_refined = coeffs_pruned_refined[0].unsqueeze(0)

        # True trajectory
        x = _y0.clone()
        y_true = [x]
        for k in range(n):
            x_new = rk4_step(van_der_pol, x, _dt, mu=mu_val) + x
            if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                break
            x = x_new
            y_true.append(x)
        y_true_traj = torch.cat(y_true, dim=0).detach().cpu().numpy()

        # Original model trajectory
        x = _y0.clone().unsqueeze(1)
        _dt_model = _dt.unsqueeze(0)
        y_orig = [x]
        for k in range(n):
            x_new = original_full_model((x, _dt_model), coefficients=_c_orig) + x
            if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                break
            x = x_new
            y_orig.append(x)
        y_orig_traj = torch.cat(y_orig, dim=1)[0].detach().cpu().numpy()

        # Pruned model trajectory
        x = _y0.clone().unsqueeze(1)
        y_pruned = [x]
        for k in range(n):
            x_new = pruned_model((x, _dt_model), coefficients=_c_pruned) + x
            if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                break
            x = x_new
            y_pruned.append(x)
        y_pruned_traj = torch.cat(y_pruned, dim=1)[0].detach().cpu().numpy()

        # Refined model trajectory
        x = _y0.clone().unsqueeze(1)
        y_refined = [x]
        for k in range(n):
            x_new = pruned_model_refined((x, _dt_model), coefficients=_c_refined) + x
            if torch.isnan(x_new).any() or torch.abs(x_new).max() > 50:
                break
            x = x_new
            y_refined.append(x)
        y_refined_traj = torch.cat(y_refined, dim=1)[0].detach().cpu().numpy()

    # Create visualization data
    viz_data = create_visualization_data_dynamics(
        trajectories_true=[y_true_traj],
        trajectories_pred=[y_orig_traj, y_pruned_traj, y_refined_traj],
        initial_conditions=[_y0[0].cpu().numpy()],
        system_params=[mu_val]
    )

    # Prepare experiment data
    experiment_data = saver.prepare_prune_data(
        problem_type="van_der_pol",
        num_basis_original=len(original_full_model.basis_functions.basis_functions),
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
            "name": "vanderpol_dt01",
            "n_points": 1000,
            "n_example_points": 100,
            "dt_range": (0.1, 0.1)
        },
        training_params={
            "num_epochs_initial": 1000,
            "num_epochs_finetune": 1000,
            "learning_rate": 1e-3,
            "batch_size": 50
        }
    )

    saver.save_experiment("vdp","prune", experiment_data, dataset_name="dt01")

    print(f"\nExperiment completed!")
    print(f"Original model: {len(original_full_model.basis_functions.basis_functions)} basis functions")
    print(f"Pruned model: {len(pruned_model_refined.basis_functions.basis_functions)} basis functions")
    print(f"Compression ratio: {len(pruned_model_refined.basis_functions.basis_functions)/len(original_full_model.basis_functions.basis_functions):.2%}")
    print(f"Performance ratio: {comparison_results['mse_pruned_refined']/comparison_results['mse_original']:.3f}")