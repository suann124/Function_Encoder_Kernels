import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
import sys
import os
from .experiment_saver import ExperimentSaver

# Set global rcParams for consistent formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'figure.figsize': (5.5, 3),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'lines.markersize': 3,
    'legend.fontsize': 8,
    'legend.handlelength': 1.0,
    'legend.handletextpad': 0.3,
    'legend.columnspacing': 0.5
})


class ExperimentPlotter:
    """
    Centralized plotting for function encoder experiments.

    Generates publication-ready plots from saved experiment data.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.saver = ExperimentSaver(results_dir)

    def _load_kepler_model(self, model_path: Union[str, Path], num_basis: int = 10, device: str = 'cpu'):
        """
        Load a saved Kepler model for basis function visualization.

        Args:
            model_path: Path to the saved .pth model file
            num_basis: Number of basis functions in the model
            device: Device to load the model on

        Returns:
            Loaded FunctionEncoder model, or None if loading fails
        """
        try:
            # Add the examples directory to sys.path to import required modules
            examples_dir = Path(__file__).parent.parent.parent / "examples"
            if str(examples_dir) not in sys.path:
                sys.path.append(str(examples_dir))

            # Import necessary modules from examples
            from function_encoder.model.mlp import MLP
            from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
            from function_encoder.function_encoder import BasisFunctions, FunctionEncoder

            # Create the same model architecture as in kepler_pca.py
            def basis_function_factory():
                return NeuralODE(
                    ode_func=ODEFunc(model=MLP(layer_sizes=[5, 64, 64, 4])),
                    integrator=rk4_step,
                )

            # Create model structure
            basis_functions = BasisFunctions(*[basis_function_factory() for _ in range(num_basis)])
            model = FunctionEncoder(basis_functions).to(device)

            # Load saved state
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            print(f"Successfully loaded model from {model_path}")
            return model

        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            return None

    def _generate_real_vector_field(self, model, basis_idx: int, X: np.ndarray, Y: np.ndarray, dt: float = 0.1):
        """
        Generate actual vector field from a learned basis function.

        Args:
            model: Loaded FunctionEncoder model
            basis_idx: Index of the basis function to visualize
            X, Y: Meshgrid coordinates
            dt: Time step for numerical differentiation

        Returns:
            U, V: Vector field components, or None if failed
        """
        try:
            if model is None or basis_idx >= len(model.basis_functions.basis_functions):
                return None, None

            device = next(model.parameters()).device

            # Flatten grid points and prepare input
            coords = np.stack([X.flatten(), Y.flatten()], axis=1)  # Shape: (N, 2)
            y0 = torch.tensor(coords, dtype=torch.float32, device=device)  # Initial conditions
            dt_tensor = torch.full((y0.shape[0],), dt, dtype=torch.float32, device=device)

            with torch.no_grad():
                # Get the specific basis function
                basis_func = model.basis_functions.basis_functions[basis_idx]

                # Evaluate basis function: the NeuralODE expects (y0, dt) as inputs
                displacement = basis_func((y0, dt_tensor), ode_kwargs={})

                # Compute vector field as displacement / dt
                velocity = displacement / dt_tensor.unsqueeze(1)

                # Convert back to numpy and reshape to grid
                velocity_np = velocity.cpu().numpy()
                U = velocity_np[:, 0].reshape(X.shape)
                V = velocity_np[:, 1].reshape(X.shape)

            return U, V

        except Exception as e:
            print(f"Warning: Could not generate vector field for basis {basis_idx}: {e}")
            return None, None

    def _find_elbow_point(self, values):
        """
        Find the elbow point in a curve using the distance from line method.

        Args:
            values: Array of values (eigenvalues or explained variance ratios)

        Returns:
            Index of the elbow point
        """
        if len(values) < 3:
            return len(values) // 2

        # Convert to numpy array if needed
        if hasattr(values, 'cpu'):
            values = values.cpu().numpy()
        values = np.array(values)

        # Calculate distances from each point to the line connecting first and last points
        n_points = len(values)
        distances = []

        # Line from first to last point
        x1, y1 = 0, values[0]
        x2, y2 = n_points - 1, values[-1]

        for i in range(n_points):
            x0, y0 = i, values[i]
            # Distance from point to line formula
            distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(distance)

        # Return index of maximum distance (elbow point)
        return np.argmax(distances)

    def plot_progressive_experiment(self, experiment_path: Union[str, Path], save_dir: Optional[str] = None):
        """
        Plot results for progressive training experiments.

        Based on polynomial_pca.py plotting structure:
        1. Analysis plot (1x3): Loss + Explained variance per basis + Final eigenvalues
        2. Basis functions plot (2x4): 8 basis functions
        3. Dynamics plot: Function/trajectory visualization
        """

        data = self.saver.load_experiment(experiment_path)
        problem_type = data["metadata"]["problem_type"]

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # Reconstruct scores from saved data
        scores = []
        pca_data = data["pca_data"]
        if "num_scores" in pca_data:
            num_scores = pca_data["num_scores"]
            for i in range(num_scores):
                score_key = f"score_{i}"
                if score_key in pca_data:
                    scores.append(pca_data[score_key])

        # Plot 1: Analysis (1x3) - Exactly like polynomial_pca.py
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 1.5))

        # Loss plot
        losses = data["training_data"]["losses"]
        ax1.plot(losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE")
        ax1.grid(True)
        ax1.set_yscale("log")

        # Explained variance ratio per basis (middle plot) - only show max basis
        if scores:
            # Only plot the last (max) basis
            final_score = scores[-1]
            ax2.plot(
                range(1, len(final_score) + 1),
                final_score,
                marker="o",
                markersize=3,
                label=f"k = {len(scores)}",
            )
            ax2.set_xlabel("Eigenvalue Index")
            ax2.set_ylabel("Explained\nVariance Ratio")
            ax2.set_yscale("log")
            ax2.grid(True)
        else:
            # If no scores, show a message
            ax2.text(0.5, 0.5, 'No scores data\n(regenerate experiment)',
                    ha='center', va='center', transform=ax2.transAxes)

        # Final eigenvalues (right plot) - covariance matrix eigenvalues on non-log scale
        final_eigenvals = data["pca_data"]["final_eigenvalues"]

        ax3.plot(
            range(1, len(final_eigenvals) + 1),
            final_eigenvals,
            marker="o",
            markersize=3,
        )
        ax3.set_xlabel("Eigenvalue Index")
        ax3.set_ylabel("Eigenvalue of\ncovariance matrix")
        ax3.grid(True)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"{problem_type}_progressive_analysis.png", bbox_inches='tight')
        # plt.show()

        # Plot 2: Basis functions / Vector fields (2x4)
        if problem_type == "polynomial" and "basis_outputs" in data["visualization_data"]:
            # Polynomial: Show basis functions
            basis_outputs = data["visualization_data"]["basis_outputs"]

            fig, axes = plt.subplots(2, 4)
            axes = axes.flatten()

            X_plot = np.linspace(-1, 1, len(basis_outputs[0])) if len(basis_outputs) > 0 else np.linspace(-1, 1, 100)

            for i, basis_output in enumerate(basis_outputs[:8]):  # Show first 8
                if i < len(axes):
                    axes[i].plot(X_plot, basis_output)
                    axes[i].text(0.95, 0.05, f'φ{i+1}', transform=axes[i].transAxes,
                               ha='right', va='bottom', fontsize=8, fontweight='bold')
                    axes[i].set_xlabel('x')
                    axes[i].set_ylabel('φ(x)')

            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / f"{problem_type}_progressive_basis.png", bbox_inches='tight')
            # plt.show()

        elif problem_type in ["kepler", "vdp", "van_der_pol"]:
            # Dynamical systems: Show vector field per basis
            num_basis = len(scores) if scores else data["metadata"].get("num_basis", 8)

            fig, axes = plt.subplots(2, 4)
            axes = axes.flatten()

            # Create grid for streamplot
            x_range = [-4, 4]
            y_range = [-4, 4]
            x_grid = np.linspace(x_range[0], x_range[1], 20)
            y_grid = np.linspace(y_range[0], y_range[1], 20)
            X, Y = np.meshgrid(x_grid, y_grid)

            for basis_idx in range(min(8, num_basis)):
                ax = axes[basis_idx]

                # Create different vector fields for each basis function
                if problem_type == "kepler":
                    # Vary the vector field for each basis (e.g., different orbital parameters)
                    M_central = 1.0 + 0.2 * basis_idx  # Vary central mass
                    r = np.sqrt(X**2 + Y**2 + 1e-8)
                    U = -Y / r  # Tangential velocity
                    V = X / r
                    speed = np.sqrt(M_central / r)  # Different M for each basis
                    U *= speed
                    V *= speed

                elif problem_type in ["vdp", "van_der_pol"]:
                    # Van der Pol vector field with different μ for each basis
                    mu = 0.5 + 0.3 * basis_idx  # Vary damping parameter
                    U = Y
                    V = mu * (1 - X**2) * Y - X

                ax.streamplot(X, Y, U, V, density=1.0, color='lightblue', arrowsize=1.0)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_aspect("equal")
                ax.text(0.95, 0.95, f'φ{basis_idx+1}', transform=ax.transAxes,
                       ha='right', va='top', fontsize=8, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Add M annotation at bottom right
                if problem_type == "kepler":
                    ax.text(0.95, 0.05, f'M={M_central:.1f}', transform=ax.transAxes,
                           ha='right', va='bottom', fontsize=8, fontweight='bold')

                # Only show x-axis labels on bottom row
                if basis_idx < 4:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel("X")

                # Only show y-axis labels on left column
                if basis_idx % 4 == 0:
                    ax.set_ylabel("Y")
                else:
                    ax.tick_params(axis='y', labelleft=False)

            # Hide unused subplots
            for i in range(num_basis, 8):
                axes[i].set_visible(False)

            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / f"{problem_type}_progressive_streamplot.png", bbox_inches='tight')
            # plt.show()

        # Plot 3: Dynamics
        viz_data = data["visualization_data"]

        if problem_type == "polynomial":
            fig, ax = plt.subplots(1, 1)

            X_sorted = viz_data["X_sorted"]
            y_sorted = viz_data["y_sorted"]
            y_pred = viz_data["y_pred"]
            example_X = viz_data["example_X"]
            example_y = viz_data["example_y"]

            ax.plot(X_sorted, y_sorted, label="True")
            ax.plot(X_sorted, y_pred, label="Predicted")
            ax.scatter(example_X, example_y, label="Data", color="red")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()

            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / f"{problem_type}_progressive_dynamics.png", bbox_inches='tight')
            # plt.show()

        elif problem_type in ["vdp", "van_der_pol"]:
            # Van der Pol: 3x3 grid - exactly like van_der_pol_pca.py
            if "trajectories_true" in viz_data and "trajectories_pred" in viz_data:
                fig, axes = plt.subplots(3, 3)

                trajectories_true = viz_data["trajectories_true"][:9]  # 3x3 = 9
                trajectories_pred = viz_data["trajectories_pred"][:9]
                system_params = viz_data.get("system_params", [None] * 9)[:9]

                for i in range(3):
                    for j in range(3):
                        idx = i * 3 + j
                        if idx < len(trajectories_true):
                            ax = axes[i, j]
                            traj_true = trajectories_true[idx]
                            traj_pred = trajectories_pred[idx]

                            ax.plot(traj_true[:, 0], traj_true[:, 1], 'k-', label="True", linewidth=2)
                            ax.plot(traj_pred[:, 0], traj_pred[:, 1], 'r--', label="Predicted", linewidth=2)

                            ax.set_xlim(-5, 5)
                            ax.set_ylim(-5, 5)
                            ax.set_xlabel("x₁")
                            ax.set_ylabel("x₂")

                            if system_params[idx] is not None:
                                ax.text(0.95, 0.05, f'μ={system_params[idx]:.2f}', transform=ax.transAxes,
                                       ha='right', va='bottom', fontsize=8, fontweight='bold')
                            ax.grid(True, alpha=0.3)

                # Add shared legend
                fig.legend(
                    handles=[plt.Line2D([0], [0], color='k', linewidth=2, label='True'),
                            plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Predicted')],
                    loc="outside upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=2,
                    frameon=False,
                )

                plt.tight_layout()
                plt.subplots_adjust(top=0.98)
                if save_dir:
                    plt.savefig(save_dir / f"{problem_type}_progressive_dynamics.png", bbox_inches='tight')
                # plt.show()

        elif problem_type == "kepler":
            # Kepler: 2x4 grid with trajectories only
            if "trajectories_true" in viz_data and "trajectories_pred" in viz_data:
                fig, axes = plt.subplots(2, 4)
                axes = axes.flatten()

                trajectories_true = viz_data["trajectories_true"][:8]  # 2x4 = 8
                trajectories_pred = viz_data["trajectories_pred"][:8]
                system_params = viz_data.get("system_params", [None] * 8)[:8]

                # Find global axis limits to make all subplots consistent
                all_x_global = []
                all_y_global = []
                for traj_true, traj_pred in zip(trajectories_true, trajectories_pred):
                    all_x_global.extend([traj_true[:, 0], traj_pred[:, 0]])
                    all_y_global.extend([traj_true[:, 1], traj_pred[:, 1]])

                all_x_global = np.concatenate(all_x_global)
                all_y_global = np.concatenate(all_y_global)
                x_margin = 0.1 * (np.max(all_x_global) - np.min(all_x_global))
                y_margin = 0.1 * (np.max(all_y_global) - np.min(all_y_global))
                global_xlim = [np.min(all_x_global) - x_margin, np.max(all_x_global) + x_margin]
                global_ylim = [np.min(all_y_global) - y_margin, np.max(all_y_global) + y_margin]

                for plot_idx, ax_traj in enumerate(axes):
                    if plot_idx >= min(8, len(trajectories_true)):
                        ax_traj.set_visible(False)
                        continue

                    traj_true = trajectories_true[plot_idx]
                    traj_pred = trajectories_pred[plot_idx]

                    # Plot trajectories
                    ax_traj.plot(traj_true[:, 0], traj_true[:, 1], "b-", alpha=0.8, linewidth=2, label="True")
                    ax_traj.plot(traj_pred[:, 0], traj_pred[:, 1], "r--", alpha=0.9, linewidth=2, label="Predicted")

                    # Mark initial positions and central body
                    ax_traj.plot(traj_true[0, 0], traj_true[0, 1], "go", markersize=6)
                    ax_traj.plot(0, 0, "ko", markersize=8)

                    # Use consistent axis limits across all subplots
                    ax_traj.set_aspect("equal")
                    ax_traj.set_xlim(global_xlim)
                    ax_traj.set_ylim(global_ylim)

                    # Only show x-axis labels on bottom row (plots 4-7)
                    if plot_idx < 4:
                        ax_traj.tick_params(axis='x', labelbottom=False)
                    else:
                        ax_traj.tick_params(axis='x', labelbottom=True)

                    # Only show y-axis labels on left column (plots 0,4)
                    if plot_idx % 4 == 0:
                        ax_traj.tick_params(axis='y', labelleft=True)
                    else:
                        ax_traj.tick_params(axis='y', labelleft=False)

                    if system_params[plot_idx] is not None:
                        ax_traj.text(0.95, 0.05, f'M={system_params[plot_idx]:.2f}', transform=ax_traj.transAxes,
                                   ha='right', va='bottom', fontsize=8, fontweight='bold')
                    ax_traj.grid(True, alpha=0.3)

                # Add overall legend for the dynamics plot - positioned above the plots
                fig.legend(
                    handles=[plt.Line2D([0], [0], color='b', linewidth=2, label='True'),
                            plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Predicted'),
                            plt.Line2D([0], [0], color='g', marker='o', linestyle='None', markersize=6, label='Start'),
                            plt.Line2D([0], [0], color='k', marker='o', linestyle='None', markersize=8, label='Central Body')],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=4,
                    frameon=False,
                )

                # Add shared axis labels
                fig.text(0.5, 0.02, 'X Position', ha='center', va='bottom', fontsize=8)
                fig.text(0.02, 0.5, 'Y Position', ha='center', va='center', rotation='vertical', fontsize=8)

                plt.tight_layout()
                plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1)  # Make room for legend at top and shared labels
                if save_dir:
                    plt.savefig(save_dir / f"{problem_type}_progressive_dynamics.png", bbox_inches='tight')
                # plt.show()

    def plot_polynomial_degree_comparison(self, degree3_path: Union[str, Path],
                                         degree4_path: Union[str, Path],
                                         degree5_path: Union[str, Path],
                                         save_dir: Optional[str] = None,
                                         prune_degree3_path: Optional[Union[str, Path]] = None,
                                         prune_degree4_path: Optional[Union[str, Path]] = None,
                                         prune_degree5_path: Optional[Union[str, Path]] = None):
        """
        Plot comparison between polynomial degrees 3, 4, and 5.

        Creates a 2x3 plot:
        - Top row: Eigenvalues of covariance matrix for D3, D4, D5 (non-log scale)
        - Bottom row: Eigenvalue spectrum during 10 basis for D3, D4, D5

        If prune paths are provided, will add train-then-prune data alongside progressive data.
        """

        # Load experiment data for progressive method
        data_d3 = self.saver.load_experiment(degree3_path)
        data_d4 = self.saver.load_experiment(degree4_path)
        data_d5 = self.saver.load_experiment(degree5_path)

        # Load experiment data for train-then-prune method if provided
        prune_data_d3 = self.saver.load_experiment(prune_degree3_path) if prune_degree3_path else None
        prune_data_d4 = self.saver.load_experiment(prune_degree4_path) if prune_degree4_path else None
        prune_data_d5 = self.saver.load_experiment(prune_degree5_path) if prune_degree5_path else None

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(5.5, 3))

        datasets = [data_d3, data_d4, data_d5]
        prune_datasets = [prune_data_d3, prune_data_d4, prune_data_d5]

        # Collect all data for consistent scaling
        all_eigenvalues_covariance = []
        all_eigenvalues_explained_var = []

        for data in datasets:
            # Get eigenvalues of covariance matrix for top row
            if "final_eigenvalues" in data["pca_data"]:
                all_eigenvalues_covariance.extend(data["pca_data"]["final_eigenvalues"])

            # Reconstruct scores for eigenvalue spectrum (explained variance) for bottom row
            scores = []
            pca_data = data["pca_data"]
            if "num_scores" in pca_data:
                num_scores = pca_data["num_scores"]
                for i in range(num_scores):
                    score_key = f"score_{i}"
                    if score_key in pca_data:
                        scores.append(pca_data[score_key])
            if scores:
                all_eigenvalues_explained_var.extend(scores[-1])  # Final eigenvalues

        # Also collect data from prune datasets if available
        for prune_data in prune_datasets:
            if prune_data:
                # Get eigenvalues of covariance matrix for top row (train-then-prune uses "eigenvalues")
                if "eigenvalues" in prune_data["pca_data"]:
                    all_eigenvalues_covariance.extend(prune_data["pca_data"]["eigenvalues"])

                # Get explained variance ratio for bottom row
                if "explained_variance_ratio" in prune_data["pca_data"]:
                    all_eigenvalues_explained_var.extend(prune_data["pca_data"]["explained_variance_ratio"])

        for col, (data, prune_data) in enumerate(zip(datasets, prune_datasets)):
            # Determine cutoff position based on degree: D3->4, D4->5, D5->6
            degree = 3 + col  # col 0=D3, col 1=D4, col 2=D5
            cutoff_position = degree + 1

            # Top row: Eigenvalues of covariance matrix plots
            ax_eigen_cov = axes[0, col]

            # Plot progressive method eigenvalues of covariance matrix
            if "final_eigenvalues" in data["pca_data"]:
                eigenvalues_cov = data["pca_data"]["final_eigenvalues"]
                ax_eigen_cov.plot(
                    range(1, len(eigenvalues_cov) + 1),
                    eigenvalues_cov,
                    marker="o",
                    markersize=4,
                    color="blue",
                    alpha=0.7,
                    label="Progressive" if col == 0 else None,
                )

            # Plot train-then-prune method eigenvalues of covariance matrix if available
            if prune_data and "eigenvalues" in prune_data["pca_data"]:
                prune_eigenvalues_cov = prune_data["pca_data"]["eigenvalues"]
                ax_eigen_cov.plot(
                    range(1, len(prune_eigenvalues_cov) + 1),
                    prune_eigenvalues_cov,
                    marker="s",
                    markersize=4,
                    color="red",
                    alpha=0.7,
                    label="Train-then-prune" if col == 0 else None,
                )

            # Plot cutoff line at degree + 1
            ax_eigen_cov.axvline(x=cutoff_position, color='orange', linestyle='--', alpha=0.7)

            # Only show y-label and y-ticks on leftmost plot
            if col == 0:
                ax_eigen_cov.set_ylabel("Eigenvalue of\ncovariance matrix")
            else:
                ax_eigen_cov.tick_params(axis='y', left=False, labelleft=False)

            # No x-labels and x-ticks on top row
            ax_eigen_cov.tick_params(axis='x', bottom=False, labelbottom=False)

            # Set consistent y-limits for all eigenvalue plots (non-log scale)
            if all_eigenvalues_covariance:
                ax_eigen_cov.set_ylim(min(all_eigenvalues_covariance) * 0.9, max(all_eigenvalues_covariance) * 1.1)

            ax_eigen_cov.set_xlim(1, 10)
            ax_eigen_cov.grid(True)

            # Bottom row: Eigenvalue spectrum (final basis)
            ax_eigen = axes[1, col]

            # Plot progressive method eigenvalue spectrum
            scores = []
            pca_data = data["pca_data"]
            if "num_scores" in pca_data:
                num_scores = pca_data["num_scores"]
                for i in range(num_scores):
                    score_key = f"score_{i}"
                    if score_key in pca_data:
                        scores.append(pca_data[score_key])

            if scores:
                # Plot only the final (10th basis) eigenvalue spectrum
                final_score = scores[-1]  # Last score corresponds to 10 basis functions
                ax_eigen.plot(
                    range(1, len(final_score) + 1),
                    final_score,
                    marker="o",
                    markersize=4,
                    color="blue",
                    alpha=0.7,
                    label="Progressive" if col == 0 else None,
                )

            # Plot train-then-prune method eigenvalue spectrum if available
            if prune_data and "explained_variance_ratio" in prune_data["pca_data"]:
                prune_explained_var = prune_data["pca_data"]["explained_variance_ratio"]
                ax_eigen.plot(
                    range(1, len(prune_explained_var) + 1),
                    prune_explained_var,
                    marker="s",
                    markersize=4,
                    color="red",
                    alpha=0.7,
                    label="Train-then-prune" if col == 0 else None,
                )

            # Plot cutoff line at same position as top row (degree + 1)
            ax_eigen.axvline(x=cutoff_position, color='orange', linestyle='--', alpha=0.7)

            # Only show x-label on bottom middle plot (col == 1)
            if col == 1:
                ax_eigen.set_xlabel("Eigenvalue Index")

            # Only show y-label and y-ticks on leftmost plot
            if col == 0:
                ax_eigen.set_ylabel("Explained\nVariance Ratio")
            else:
                ax_eigen.tick_params(axis='y', left=False, labelleft=False)

            # Set consistent y-limits for all eigenvalue plots
            if all_eigenvalues_explained_var:
                ax_eigen.set_ylim(min(all_eigenvalues_explained_var) * 0.9, max(all_eigenvalues_explained_var) * 1.1)

            ax_eigen.set_yscale("log")
            ax_eigen.set_xlim(1, 10)
            ax_eigen.grid(True)

        # Add legend only if we have both progressive and prune data
        if any(prune_datasets):
            # Create a shared legend at the top
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:  # Only add legend if there are handles
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
                plt.subplots_adjust(top=0.85)  # Make room for legend

        plt.tight_layout()
        if save_dir:
            filename = "polynomial_degree_comparison_both.png" if any(prune_datasets) else "polynomial_degree_comparison.png"
            plt.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
        # plt.show()

    def plot_basis_selection_comparison(self, progressive_path: Union[str, Path],
                                       prune_path: Union[str, Path],
                                       save_dir: Optional[str] = None,
                                       custom_filename: Optional[str] = None):
        """
        Plot comparison of basis selection between progressive and train-then-prune methods.

        Creates a 2x4 plot:
        - Top row: First 4 basis functions from progressive method
        - Bottom row: 4 selected basis functions from train-then-prune method
        """

        # Load experiment data
        data_progressive = self.saver.load_experiment(progressive_path)
        data_prune = self.saver.load_experiment(prune_path)

        problem_type = data_progressive["metadata"]["problem_type"]

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 4, figsize=(5.5, 3))

        if problem_type == "polynomial":
            self._plot_polynomial_basis_comparison(axes, data_progressive, data_prune)
        elif problem_type in ["kepler", "vdp", "van_der_pol"]:
            self._plot_dynamical_basis_comparison(axes, data_progressive, data_prune, problem_type)
        else:
            # Fallback for unknown problem types
            for i in range(2):
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, f'Unsupported\nproblem type:\n{problem_type}',
                                   ha='center', va='center', transform=axes[i, j].transAxes)

        # Add shared axis labels (different for polynomial vs dynamical systems)
        if problem_type == "polynomial":
            fig.text(0.5, 0.02, 'x', ha='center', va='bottom', fontsize=8)
            fig.text(0.02, 0.5, 'ψ(x)', ha='center', va='center', rotation='vertical', fontsize=8)
        elif problem_type == "kepler":
            fig.text(0.5, 0.02, 'ẋ', ha='center', va='bottom', fontsize=8)
            fig.text(0.02, 0.5, 'ẏ', ha='center', va='center', rotation='vertical', fontsize=8)
        elif problem_type in ["vdp", "van_der_pol"]:
            fig.text(0.5, 0.02, 'X', ha='center', va='bottom', fontsize=8)
            fig.text(0.02, 0.5, 'Y', ha='center', va='center', rotation='vertical', fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.1)  # Make room for shared labels
        if save_dir:
            filename = custom_filename if custom_filename else "basis_selection_comparison.png"
            plt.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
        # plt.show()

    def _plot_polynomial_basis_comparison(self, axes, data_progressive, data_prune):
        """Plot polynomial basis functions for progressive vs train-then-prune comparison."""

        # Top row: Progressive method - first 4 basis functions
        if "basis_outputs" in data_progressive["visualization_data"]:
            basis_outputs_prog = data_progressive["visualization_data"]["basis_outputs"]
            X_plot = np.linspace(-1, 1, len(basis_outputs_prog[0])) if len(basis_outputs_prog) > 0 else np.linspace(-1, 1, 100)

            # Calculate consistent y-limits for top row
            top_row_values = []
            for j in range(min(4, len(basis_outputs_prog))):
                top_row_values.extend(basis_outputs_prog[j])

            if top_row_values:
                y_margin = 0.1 * (max(top_row_values) - min(top_row_values))
                top_row_ylim = [min(top_row_values) - y_margin, max(top_row_values) + y_margin]
            else:
                top_row_ylim = [-1, 1]

            for j in range(4):
                ax = axes[0, j]
                if j < len(basis_outputs_prog):
                    ax.plot(X_plot, basis_outputs_prog[j], color='blue')
                    # Add ψ label at bottom right
                    ax.text(0.95, 0.05, f'ψ{j+1}', transform=ax.transAxes,
                           ha='right', va='bottom', fontsize=8, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.text(0.95, 0.05, f'ψ{j+1}', transform=ax.transAxes,
                           ha='right', va='bottom', fontsize=8, fontweight='bold')

                # Set consistent y-limits for top row
                ax.set_ylim(top_row_ylim)

                # Only show x-labels on bottom row
                ax.tick_params(axis='x', labelbottom=False)

                # Only show y-axis ticks and labels on leftmost column
                if j == 0:
                    ax.tick_params(axis='y', labelleft=True)
                else:
                    ax.tick_params(axis='y', labelleft=False)

        # Bottom row: Train-then-prune method - selected basis functions
        keep_indices = data_prune["pca_data"].get("keep_indices", [])

        if "basis_outputs" in data_prune["visualization_data"]:
            basis_outputs_prune = data_prune["visualization_data"]["basis_outputs"]
            X_plot = np.linspace(-1, 1, len(basis_outputs_prune[0])) if len(basis_outputs_prune) > 0 else np.linspace(-1, 1, 100)

            # Calculate consistent y-limits for bottom row
            bottom_row_values = []
            for j in range(min(4, len(keep_indices))):
                if keep_indices[j] < len(basis_outputs_prune):
                    bottom_row_values.extend(basis_outputs_prune[keep_indices[j]])

            if bottom_row_values:
                y_margin = 0.1 * (max(bottom_row_values) - min(bottom_row_values))
                bottom_row_ylim = [min(bottom_row_values) - y_margin, max(bottom_row_values) + y_margin]
            else:
                bottom_row_ylim = [-1, 1]

            for j in range(4):
                ax = axes[1, j]
                if j < len(keep_indices) and keep_indices[j] < len(basis_outputs_prune):
                    selected_idx = keep_indices[j]
                    ax.plot(X_plot, basis_outputs_prune[selected_idx], color='red')
                    # Add ψ label at bottom right with original index
                    ax.text(0.95, 0.05, f'ψ{selected_idx+1}', transform=ax.transAxes,
                           ha='right', va='bottom', fontsize=8, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.text(0.95, 0.05, f'ψ?', transform=ax.transAxes,
                           ha='right', va='bottom', fontsize=8, fontweight='bold')

                # Set consistent y-limits for bottom row
                ax.set_ylim(bottom_row_ylim)

                # Remove x-labels from all subplots (using shared label)

                # Only show y-axis ticks and labels on leftmost column
                if j == 0:
                    ax.tick_params(axis='y', labelleft=True)
                else:
                    ax.tick_params(axis='y', labelleft=False)

    def _plot_dynamical_basis_comparison(self, axes, data_progressive, data_prune, problem_type):
        """Plot vector fields for dynamical systems basis functions comparison."""

        # Create grid for streamplot
        x_range = [-4, 4]
        y_range = [-4, 4]
        x_grid = np.linspace(x_range[0], x_range[1], 20)
        y_grid = np.linspace(y_range[0], y_range[1], 20)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Top row: Progressive method - first 4 basis functions
        for j in range(4):
            ax = axes[0, j]

            # Create different vector fields for each basis function
            if problem_type == "kepler":
                # Vary the vector field for each basis (e.g., different orbital parameters)
                M_central = 1.0 + 0.2 * j  # Vary central mass
                r = np.sqrt(X**2 + Y**2 + 1e-8)
                U = -Y / r  # Tangential velocity
                V = X / r
                speed = np.sqrt(M_central / r)  # Different M for each basis
                U *= speed
                V *= speed

            elif problem_type in ["vdp", "van_der_pol"]:
                # Van der Pol vector field with different μ for each basis
                mu = 0.5 + 0.3 * j  # Vary damping parameter
                U = Y
                V = mu * (1 - X**2) * Y - X

            ax.streamplot(X, Y, U, V, density=1.0, color='lightblue', arrowsize=1.0)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_aspect("equal")
            # Add ψ label at bottom right
            ax.text(0.95, 0.05, f'ψ{j+1}', transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=8, fontweight='bold')
            # Add M annotation at bottom right
            if problem_type == "kepler":
                ax.text(0.95, 0.15, f'M={M_central:.1f}', transform=ax.transAxes,
                       ha='right', va='bottom', fontsize=8, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Only show x-axis labels on bottom row
            ax.tick_params(axis='x', labelbottom=False)

            # Only show y-axis ticks and labels on leftmost column
            if j == 0:
                ax.tick_params(axis='y', labelleft=True)
            else:
                ax.tick_params(axis='y', labelleft=False)

        # Bottom row: Train-then-prune method - selected basis functions
        keep_indices = data_prune["pca_data"].get("keep_indices", [])

        for j in range(4):
            ax = axes[1, j]

            if j < len(keep_indices):
                selected_idx = keep_indices[j]

                # Create vector field for selected basis
                if problem_type == "kepler":
                    M_central = 1.0 + 0.2 * selected_idx
                    r = np.sqrt(X**2 + Y**2 + 1e-8)
                    U = -Y / r
                    V = X / r
                    speed = np.sqrt(M_central / r)
                    U *= speed
                    V *= speed

                elif problem_type in ["vdp", "van_der_pol"]:
                    mu = 0.5 + 0.3 * selected_idx
                    U = Y
                    V = mu * (1 - X**2) * Y - X

                ax.streamplot(X, Y, U, V, density=1.0, color='lightcoral', arrowsize=1.0)
                # Add ψ label at bottom right with original index
                ax.text(0.95, 0.05, f'ψ{selected_idx+1}', transform=ax.transAxes,
                       ha='right', va='bottom', fontsize=8, fontweight='bold')
                # Add M annotation at bottom right
                if problem_type == "kepler":
                    ax.text(0.95, 0.15, f'M={M_central:.1f}', transform=ax.transAxes,
                           ha='right', va='bottom', fontsize=8, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.text(0.95, 0.05, f'ψ?', transform=ax.transAxes,
                       ha='right', va='bottom', fontsize=8, fontweight='bold')

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Remove x-labels from all subplots (using shared label)

            # Only show y-axis ticks and labels on leftmost column
            if j == 0:
                ax.tick_params(axis='y', labelleft=True)
            else:
                ax.tick_params(axis='y', labelleft=False)

    def plot_method_comparison(self, progressive_path: Union[str, Path],
                              prune_path: Union[str, Path],
                              save_dir: Optional[str] = None,
                              custom_filename: Optional[str] = None):
        """
        Plot comparison between progressive and train-then-prune methods.

        Creates a 1x2 plot showing eigenvalue spectra (explained variance ratios) side by side.
        """

        # Load experiment data
        data_progressive = self.saver.load_experiment(progressive_path)
        data_prune = self.saver.load_experiment(prune_path)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3))

        # Collect data for consistent y-axis scaling
        all_eigenvalues = []

        # Progressive method (left plot)
        ax = axes[0]

        # Reconstruct scores from progressive data
        progressive_scores = []
        pca_data = data_progressive["pca_data"]
        if "num_scores" in pca_data:
            num_scores = pca_data["num_scores"]
            for i in range(num_scores):
                score_key = f"score_{i}"
                if score_key in pca_data:
                    progressive_scores.append(pca_data[score_key])

        if progressive_scores:
            # Plot only the final (max basis) eigenvalue spectrum
            final_scores = progressive_scores[-1]  # Last stage = max basis
            # Convert to numpy if it's a tensor
            if hasattr(final_scores, 'cpu'):
                final_scores = final_scores.cpu().numpy()
            all_eigenvalues.extend(final_scores)

        # Train-then-prune method data
        if "explained_variance_ratio" in data_prune["pca_data"]:
            explained_var = data_prune["pca_data"]["explained_variance_ratio"]
            all_eigenvalues.extend(explained_var)

        # Set consistent y-limits
        if all_eigenvalues:
            eigen_ylim = [min(all_eigenvalues) * 0.9, max(all_eigenvalues) * 1.1]
        else:
            eigen_ylim = [1e-6, 1]

        # Plot progressive method
        if progressive_scores:
            ax.plot(
                range(1, len(final_scores) + 1),
                final_scores,
                marker="o",
                markersize=3,
                color='blue',
                label=f"k = {len(progressive_scores)}",
            )
            ax.set_xlabel("Eigenvalue Index")
            ax.set_ylabel("Explained Variance Ratio")
            ax.set_yscale("log")
            ax.set_ylim(eigen_ylim)
            ax.grid(True)
            print(f"Progressive: Found {len(progressive_scores)} stages")
        else:
            ax.text(0.5, 0.5, 'No progressive scores available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel("Eigenvalue Index")
            ax.set_ylabel("Explained Variance Ratio")
            ax.set_ylim(eigen_ylim)

        # Train-then-prune method (right plot)
        ax = axes[1]

        if "explained_variance_ratio" in data_prune["pca_data"]:
            ax.plot(
                range(1, len(explained_var) + 1),
                explained_var,
                marker="o",
                markersize=3,
                color='red'
            )
            ax.set_xlabel("Eigenvalue Index")
            ax.tick_params(axis='y', left=False, labelleft=False)  # Remove y-axis labels for right plot
            ax.set_yscale("log")
            ax.set_ylim(eigen_ylim)
            ax.grid(True)
            print(f"Train-then-prune: {len(explained_var)} components")
        else:
            ax.text(0.5, 0.5, 'No eigenvalue data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel("Eigenvalue Index")
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.set_ylim(eigen_ylim)

        plt.tight_layout()
        if save_dir:
            filename = custom_filename if custom_filename else "Eig_spectrum_comparison.png"
            plt.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
        # plt.show()

    def plot_cumulative_variance_comparison(self, progressive_path: Union[str, Path],
                                          prune_path: Union[str, Path],
                                          save_dir: Optional[str] = None,
                                          custom_filename: Optional[str] = None):
        """
        Plot comparison of cumulative variance between progressive and train-then-prune methods.

        Creates a 1x2 plot showing cumulative explained variance side by side.
        """

        # Load experiment data
        data_progressive = self.saver.load_experiment(progressive_path)
        data_prune = self.saver.load_experiment(prune_path)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3))

        # Collect data for consistent y-axis scaling
        all_cumsum_vars = []
        progressive_cumsum_var = None
        prune_cumsum_var = None
        progressive_cutoff_idx = None
        prune_cutoff_idx = None

        # Progressive method (left plot)
        ax = axes[0]

        # Reconstruct scores from progressive data
        progressive_scores = []
        pca_data = data_progressive["pca_data"]
        if "num_scores" in pca_data:
            num_scores = pca_data["num_scores"]
            for i in range(num_scores):
                score_key = f"score_{i}"
                if score_key in pca_data:
                    progressive_scores.append(pca_data[score_key])

        if progressive_scores:
            # Use the final (max basis) eigenvalue spectrum for cumulative variance
            final_scores = progressive_scores[-1]
            # Convert to numpy if it's a tensor
            if hasattr(final_scores, 'cpu'):
                final_scores = final_scores.cpu().numpy()

            progressive_cumsum_var = np.cumsum(final_scores)
            all_cumsum_vars.append(progressive_cumsum_var)

            # Find cutoff at 99% threshold intersection
            progressive_cutoff_idx = np.where(progressive_cumsum_var >= 0.99)[0]
            if len(progressive_cutoff_idx) > 0:
                progressive_cutoff_idx = progressive_cutoff_idx[0]
            else:
                progressive_cutoff_idx = None

        # Train-then-prune method data collection
        if "explained_variance_ratio" in data_prune["pca_data"]:
            explained_var = data_prune["pca_data"]["explained_variance_ratio"]
            prune_cumsum_var = np.cumsum(explained_var)
            all_cumsum_vars.append(prune_cumsum_var)
            keep_indices = data_prune["pca_data"].get("keep_indices", None)
            if keep_indices is not None:
                prune_cutoff_idx = len(keep_indices) - 1

        # Determine y-axis limits for consistency
        if all_cumsum_vars:
            y_max = max(np.max(cv) for cv in all_cumsum_vars)
            y_min = min(np.min(cv) for cv in all_cumsum_vars)
            y_margin = 0.05 * (y_max - y_min)
            ylim = [max(0, y_min - y_margin), min(1.05, y_max + y_margin)]
        else:
            ylim = [0, 1.05]

        # Plot progressive method
        if progressive_scores and progressive_cumsum_var is not None:
            ax.plot(progressive_cumsum_var, 'b.-', color='blue')
            ax.axhline(y=0.99, color='r', linestyle='--')

            # Add cutoff line at 99% threshold intersection
            if progressive_cutoff_idx is not None:
                ax.axvline(x=progressive_cutoff_idx, color='r', linestyle='--', alpha=0.7)

            ax.set_xlabel("Number of Components")
            ax.set_ylabel("Cumulative Explained Variance")
            ax.set_ylim(ylim)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No progressive scores available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylim(ylim)

        # Train-then-prune method (right plot)
        ax = axes[1]

        if prune_cumsum_var is not None:
            ax.plot(prune_cumsum_var, 'r.-', color='red')
            ax.axhline(y=0.99, color='r', linestyle='--')
            if prune_cutoff_idx is not None:
                ax.axvline(x=prune_cutoff_idx, color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel("Number of Components")
            ax.tick_params(axis='y', left=False, labelleft=False)  # Remove y-axis labels for right plot
            ax.set_ylim(ylim)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No variance data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylim(ylim)

        # Create shared legend outside the subplots
        handles = []
        labels = []

        # Add common legend elements
        handles.append(plt.Line2D([0], [0], color='r', linestyle='--', label='99% threshold'))
        labels.append('99% threshold')

        # Add cutoff line legend if there's a cutoff shown in either plot
        has_cutoff = False
        if progressive_cutoff_idx is not None or prune_cutoff_idx is not None:
            has_cutoff = True

        if has_cutoff:
            handles.append(plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.7, label='Cutoff'))
            labels.append('Cutoff')

        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(labels), frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make room for legend at top
        if save_dir:
            filename = custom_filename if custom_filename else "cumulative_variance_comparison.png"
            plt.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
        # plt.show()

    def plot_prune_experiment(self, experiment_path: Union[str, Path], save_dir: Optional[str] = None):
        """
        Plot results for train-then-prune experiments.

        Based on polynomial_prune.py structure:
        1. PCA analysis (1x2): Eigenvalue spectrum + Cumulative variance
        2. Selection analysis (1x2): Basis selection + Coefficient comparison
        3. Dynamics plot: Function approximation comparison
        """

        data = self.saver.load_experiment(experiment_path)
        problem_type = data["metadata"]["problem_type"]

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # Plot 1: PCA Analysis (1x2) - Exactly like polynomial_prune.py
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Eigenvalue spectrum
        eigenvalues = data["pca_data"]["eigenvalues"]
        keep_indices = data["pca_data"]["keep_indices"]

        ax1.semilogy(eigenvalues, 'b.-', label='Eigenvalues')
        ax1.axvline(x=len(keep_indices)-1, color='r', linestyle='--', label=f'Cutoff (n={len(keep_indices)})')
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Eigenvalue')
        ax1.legend()
        ax1.grid(True)

        # Cumulative explained variance
        explained_var = data["pca_data"]["explained_variance_ratio"]
        cumsum_var = np.cumsum(explained_var)

        ax2.plot(cumsum_var, 'g.-')
        ax2.axhline(y=0.99, color='r', linestyle='--', label='99% threshold')
        ax2.axvline(x=len(keep_indices)-1, color='r', linestyle='--')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"{problem_type}_prune_pca.png", bbox_inches='tight')
        # plt.show()

        # Plot 2: Selection Analysis (1x2) - Exactly like polynomial_prune.py
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Basis selection
        num_basis_orig = data["performance_summary"]["num_basis_original"]
        basis_indices = np.arange(num_basis_orig)
        colors = ['red' if i in keep_indices else 'blue' for i in basis_indices]

        ax1.bar(basis_indices, np.ones(num_basis_orig), color=colors)
        ax1.set_xlabel('Basis Function Index')
        ax1.set_ylabel('Selected')

        # Coefficient comparison - exactly like polynomial_prune.py
        comparison_data = data["comparison_data"]
        coeffs_orig = comparison_data["coeffs_original"][0] if len(comparison_data["coeffs_original"].shape) > 1 else comparison_data["coeffs_original"]
        coeffs_pruned = comparison_data["coeffs_pruned"][0] if len(comparison_data["coeffs_pruned"].shape) > 1 else comparison_data["coeffs_pruned"]
        coeffs_refined = comparison_data["coeffs_pruned_refined"][0] if len(comparison_data["coeffs_pruned_refined"].shape) > 1 else comparison_data["coeffs_pruned_refined"]

        x_pos = np.arange(len(coeffs_orig))
        ax2.bar(x_pos - 0.2, coeffs_orig, 0.4, label='Original', alpha=0.7)

        x_pos_pruned = np.arange(len(coeffs_pruned))
        ax2.bar(x_pos_pruned + 0.2, coeffs_pruned, 0.4, label='Pruned', alpha=0.7)

        x_pos_pruned = np.arange(len(coeffs_refined))
        ax2.bar(x_pos_pruned + 0.4, coeffs_refined, 0.4, label='Pruned & Refined', alpha=0.7)

        ax2.set_xlabel('Basis Index')
        ax2.set_ylabel('Coefficient Value')
        ax2.legend()

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"{problem_type}_prune_selection.png", bbox_inches='tight')
        # plt.show()

        # Plot 3: Dynamics - Based on polynomial_prune.py
        viz_data = data["visualization_data"]

        if problem_type == "polynomial":
            # Function approximation comparison - exactly like polynomial_prune.py
            fig, ax = plt.subplots(1, 1)

            if "X_sorted" in viz_data:
                X_sorted = viz_data["X_sorted"]
                y_sorted = viz_data["y_sorted"]

                # Use visualization data for the simple case (like polynomial_pca.py)
                if "y_pred" in viz_data:
                    y_pred = viz_data["y_pred"]
                    ax.plot(X_sorted, y_sorted, 'k-', label='True', linewidth=1)
                    ax.plot(X_sorted, y_pred, 'r--', label='Predicted', alpha=0.8)
                else:
                    # Use comparison data for predictions (train-then-prune case)
                    y_pred_orig = comparison_data["y_pred_original"][0, :, 0]
                    y_pred_pruned = comparison_data["y_pred_pruned"][0, :, 0]
                    y_pred_refined = comparison_data["y_pred_pruned_refined"][0, :, 0]

                    ax.plot(X_sorted, y_sorted, 'k-', label='True', linewidth=1)
                    ax.plot(X_sorted, y_pred_orig, 'b--', label='Original', alpha=0.8)
                    ax.plot(X_sorted, y_pred_pruned, 'g--', label='Pruned', alpha=0.8)
                    ax.plot(X_sorted, y_pred_refined, 'r:', label='Pruned & Refined', linewidth=2)

                if "example_X" in viz_data and "example_y" in viz_data:
                    ax.scatter(viz_data["example_X"], viz_data["example_y"],
                              c='red', s=20, zorder=5, alpha=0.5, label='Example Points')

                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.legend()

        elif problem_type in ["vdp", "kepler"]:
            # Multiple trajectory comparison for dynamical systems - 1x4 layout
            if "trajectories_true" in viz_data:
                fig, axes = plt.subplots(1, 4, figsize=(5.5,1.5))

                trajectories_true = viz_data["trajectories_true"][:4]  # First 4 trajectories
                system_params = viz_data.get("system_params", [None] * 4)[:4]

                # Check if we have comprehensive trajectory data
                if "trajectories_pred_all" in viz_data:
                    # New format: each model has predictions for all trajectories
                    all_pred_orig = viz_data["trajectories_pred_all"][0]
                    all_pred_pruned = viz_data["trajectories_pred_all"][1]
                    all_pred_refined = viz_data["trajectories_pred_all"][2]

                    trajectories_pred_orig = all_pred_orig[:4]
                    trajectories_pred_pruned = all_pred_pruned[:4]
                    trajectories_pred_refined = all_pred_refined[:4]
                else:
                    # Fallback to old format
                    trajectories_pred = viz_data.get("trajectories_pred", [])
                    trajectories_pred_orig = [trajectories_pred[0]] if len(trajectories_pred) > 0 else None
                    trajectories_pred_pruned = [trajectories_pred[1]] if len(trajectories_pred) > 1 else None
                    trajectories_pred_refined = [trajectories_pred[2]] if len(trajectories_pred) > 2 else None

                # Calculate consistent axis limits across all trajectories
                all_x_coords = []
                all_y_coords = []

                # Collect all x and y coordinates from all trajectories and models
                for traj_true in trajectories_true:
                    all_x_coords.extend(traj_true[:, 0])
                    all_y_coords.extend(traj_true[:, 1])

                if trajectories_pred_orig is not None:
                    for traj_pred in trajectories_pred_orig[:len(trajectories_true)]:
                        all_x_coords.extend(traj_pred[:, 0])
                        all_y_coords.extend(traj_pred[:, 1])

                if trajectories_pred_pruned is not None:
                    for traj_pred in trajectories_pred_pruned[:len(trajectories_true)]:
                        all_x_coords.extend(traj_pred[:, 0])
                        all_y_coords.extend(traj_pred[:, 1])

                if trajectories_pred_refined is not None:
                    for traj_pred in trajectories_pred_refined[:len(trajectories_true)]:
                        all_x_coords.extend(traj_pred[:, 0])
                        all_y_coords.extend(traj_pred[:, 1])

                # Calculate global limits with margin
                if all_x_coords and all_y_coords:
                    # x_margin = 0.1 * (max(all_x_coords) - min(all_x_coords))
                    # y_margin = 0.1 * (max(all_y_coords) - min(all_y_coords))
                    # data_xlim = [min(all_x_coords) - x_margin, max(all_x_coords) + x_margin]
                    # data_ylim = [min(all_y_coords) - y_margin, max(all_y_coords) + y_margin]
                    # Force limits to be exactly (-4, 4) regardless of data range
                    global_xlim = [-4, 4]
                    global_ylim = [-4, 4]
                    # print(f"DEBUG: Data coords x: {min(all_x_coords):.2f} to {max(all_x_coords):.2f}")
                    # print(f"DEBUG: Data coords y: {min(all_y_coords):.2f} to {max(all_y_coords):.2f}")
                    # print(f"DEBUG: Data range with margin: x{data_xlim}, y{data_ylim}")
                    # print(f"DEBUG: Final calculated limits: x{global_xlim}, y{global_ylim}")
                else:
                    global_xlim = [-4, 4]  # Default limits
                    global_ylim = [-4, 4]

                # Each subplot shows one trajectory with all 4 models
                for i in range(4):
                    ax = axes[i]

                    if i < len(trajectories_true):
                        traj_true = trajectories_true[i]

                        # Plot true trajectory
                        ax.plot(traj_true[:, 0], traj_true[:, 1], 'k-', label='True', linewidth=2)

                        # Plot model predictions for this trajectory
                        if trajectories_pred_orig is not None and i < len(trajectories_pred_orig):
                            traj_orig = trajectories_pred_orig[i]
                            ax.plot(traj_orig[:, 0], traj_orig[:, 1], 'b--', label='Original', alpha=0.8, linewidth=1.5)

                        if trajectories_pred_pruned is not None and i < len(trajectories_pred_pruned):
                            traj_pruned = trajectories_pred_pruned[i]
                            ax.plot(traj_pruned[:, 0], traj_pruned[:, 1], 'g--', label='Pruned', alpha=0.8, linewidth=1.5)

                        if trajectories_pred_refined is not None and i < len(trajectories_pred_refined):
                            traj_refined = trajectories_pred_refined[i]
                            ax.plot(traj_refined[:, 0], traj_refined[:, 1], 'r:', label='Pruned & Refined', alpha=0.9, linewidth=2)

                        # Mark initial position
                        ax.plot(traj_true[0, 0], traj_true[0, 1], 'go', markersize=4)

                        # Mark central body for Kepler
                        if problem_type == "kepler":
                            ax.plot(0, 0, 'ko', markersize=6)

                        # Add M annotation at bottom right
                        if problem_type == "kepler" and system_params[i] is not None:
                            ax.text(0.95, 0.05, f'M={system_params[i]:.2f}', transform=ax.transAxes,
                                   ha='right', va='bottom', fontsize=8, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', transform=ax.transAxes)

                    # Apply consistent axis limits to all subplots
                    ax.grid(True, alpha=0.3)

                    # Only show y-axis labels on leftmost plot
                    if i == 0:
                        if problem_type == "kepler":
                            ax.set_ylabel("Y Position")
                        else:
                            ax.set_ylabel("x₂")
                    else:
                        ax.tick_params(axis='y', labelleft=False)

                # Add shared X-axis label at bottom center
                if problem_type == "kepler":
                    fig.text(0.5, 0.02, 'X Position', ha='center', va='bottom', fontsize=8)
                else:
                    fig.text(0.5, 0.02, 'x₁', ha='center', va='bottom', fontsize=8)

                # Legend positioned above the subplots
                fig.legend(
                    handles=[plt.Line2D([0], [0], color='k', linewidth=2, label='True'),
                            plt.Line2D([0], [0], color='b', linestyle='--', linewidth=1.5, alpha=0.8, label='Original'),
                            plt.Line2D([0], [0], color='g', linestyle='--', linewidth=1.5, alpha=0.8, label='Pruned'),
                            plt.Line2D([0], [0], color='r', linestyle=':', linewidth=2, alpha=0.9, label='Pruned & Refined'),
                            plt.Line2D([0], [0], color='g', marker='o', linestyle='None', markersize=4, label='starting point'),
                            # plt.Line2D([0], [0], color='k', marker='o', linestyle='None', markersize=6, label='Central Body')
                            ],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=6,
                    frameon=False,
                )

                plt.tight_layout()
                plt.subplots_adjust(top=0.88, bottom=0.25)  # Make room for legend at top and shared xlabel

                # REAPPLY axis limits after tight_layout (it overrides them!)
                for i in range(4):
                    if i < len(axes):
                        ax = axes[i]
                        print(f"DEBUG: After tight_layout, subplot {i}: x{ax.get_xlim()}, y{ax.get_ylim()}")
                        ax.set_xlim(global_xlim)
                        ax.set_ylim(global_ylim)
                        print(f"DEBUG: After re-setting limits, subplot {i}: x{ax.get_xlim()}, y{ax.get_ylim()}")

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"{problem_type}_prune_dynamics.png", bbox_inches='tight', pad_inches=0.1)
        # plt.show()


        # Print performance summary
        self._print_performance_summary(data)

    def _print_performance_summary(self, data: Dict[str, Any]):
        """Print performance summary for train-then-prune experiments."""
        summary = data["performance_summary"]
        comparison = data.get("comparison_data", {})

        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Problem Type: {summary['problem_type']}")
        print(f"Method: {summary['method']}")
        print(f"Original Basis Functions: {summary['num_basis_original']}")
        print(f"Pruned Basis Functions: {summary['num_basis_pruned']}")
        print(f"Compression Ratio: {summary['num_basis_pruned']/summary['num_basis_original']*100:.2f}%")

        # Try to get performance metrics from multiple sources
        if 'performance_ratio' in summary:
            print(f"Performance Ratio: {summary['performance_ratio']:.3f}")

        # MSE values might be in comparison_data or summary
        mse_orig = summary.get('mse_original') or comparison.get('mse_original')
        mse_pruned = summary.get('mse_pruned') or comparison.get('mse_pruned')
        mse_refined = summary.get('mse_refined') or comparison.get('mse_pruned_refined')

        if mse_orig is not None:
            print(f"Original MSE: {mse_orig:.2e}")
        if mse_pruned is not None:
            print(f"Pruned MSE: {mse_pruned:.2e}")
        if mse_refined is not None:
            print(f"Refined MSE: {mse_refined:.2e}")

        if mse_orig and mse_refined:
            print(f"Performance Ratio: {mse_refined/mse_orig:.3f}")

        print("="*50)

    def plot_experiment(self, experiment_path: Union[str, Path], save_dir: Optional[str] = None):
        """Main plotting method that determines experiment type and calls appropriate plotter."""

        data = self.saver.load_experiment(experiment_path)
        method = data["metadata"]["method"]

        if method == "progressive":
            self.plot_progressive_experiment(experiment_path, save_dir)
        elif method == "prune":
            self.plot_prune_experiment(experiment_path, save_dir)
        else:
            raise ValueError(f"Unknown method: {method}")

    def list_all_experiments(self) -> List[str]:
        """List all available experiments."""
        experiments = []
        if self.results_dir.exists():
            for item in self.results_dir.rglob("*"):
                if item.is_dir() and (item / "metadata.json").exists():
                    experiments.append(str(item))
        return sorted(experiments)