import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
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
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

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
            ax2.set_ylabel("Explained Variance Ratio")
            ax2.set_yscale("log")
            ax2.grid(True)
        else:
            # If no scores, show a message
            ax2.text(0.5, 0.5, 'No scores data\n(regenerate experiment)',
                    ha='center', va='center', transform=ax2.transAxes)

        # Final eigenvalues (right plot)
        final_eigenvals = data["pca_data"]["final_eigenvalues"]

        ax3.plot(
            range(1, len(final_eigenvals) + 1),
            final_eigenvals,
            marker="o",
            markersize=3,
        )
        ax3.set_xlabel("Eigenvalue Index")
        ax3.set_ylabel("Eigenvalue of covariance matrix")
        ax3.set_yscale("log")
        ax3.grid(True)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"{problem_type}_progressive_analysis.png", bbox_inches='tight')
        plt.show()

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
                    axes[i].set_title(f"φ{i+1}")
                    axes[i].set_xlabel('x')
                    axes[i].set_ylabel('φ(x)')

            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / f"{problem_type}_progressive_basis.png", bbox_inches='tight')
            plt.show()

        elif problem_type in ["kepler", "vdp"]:
            # Dynamical systems: Show vector field per basis
            num_basis = len(scores) if scores else data["metadata"].get("num_basis", 8)

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            # Create grid for streamplot
            x_range = [-4, 4]
            y_range = [-4, 4]
            x_grid = np.linspace(x_range[0], x_range[1], 20)
            y_grid = np.linspace(y_range[0], y_range[1], 20)
            X, Y = np.meshgrid(x_grid, y_grid)

            for basis_idx in range(min(8, num_basis)):
                ax = axes[basis_idx]

                if problem_type == "kepler":
                    # Kepler vector field
                    r = np.sqrt(X**2 + Y**2 + 1e-8)
                    U = -Y / r  # Tangential velocity
                    V = X / r
                    speed = np.sqrt(1.0 / r)  # Use default M=1.0
                    U *= speed
                    V *= speed

                elif problem_type == "vdp":
                    # Van der Pol vector field: dx/dt = y, dy/dt = μ(1-x²)y - x
                    mu = 1.0  # Default parameter
                    U = Y
                    V = mu * (1 - X**2) * Y - X

                ax.streamplot(X, Y, U, V, density=1.0, color='lightblue', arrowsize=1.0)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_aspect("equal")
                ax.set_title(f"φ{basis_idx+1}")
                ax.grid(True, alpha=0.3)

                # Remove x-axis labels from top row
                if basis_idx < 4:
                    ax.set_xlabel("")
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel("X")
                ax.set_ylabel("Y")

            # Hide unused subplots
            for i in range(num_basis, 8):
                axes[i].set_visible(False)

            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / f"{problem_type}_progressive_streamplot.png", bbox_inches='tight')
            plt.show()

        # Plot 3: Dynamics - Based on original polynomial_pca.py structure
        viz_data = data["visualization_data"]

        if problem_type == "polynomial":
            # Single function plot - exactly like polynomial_pca.py
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
            plt.show()

        elif problem_type == "vdp":
            # Van der Pol: 3x3 grid - exactly like van_der_pol_pca.py
            if "trajectories_true" in viz_data and "trajectories_pred" in viz_data:
                fig, axes = plt.subplots(3, 3, figsize=(10, 10))

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
                                ax.set_title(f"μ={system_params[idx]:.2f}")
                            ax.grid(True, alpha=0.3)

                # Add shared legend
                fig.legend(
                    handles=[plt.Line2D([0], [0], color='k', linewidth=2, label='True'),
                            plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Predicted')],
                    loc="outside upper center",
                    bbox_to_anchor=(0.5, 0.95),
                    ncol=2,
                    frameon=False,
                )

                plt.tight_layout()
                if save_dir:
                    plt.savefig(save_dir / f"{problem_type}_progressive_dynamics.png", bbox_inches='tight')
                plt.show()

        elif problem_type == "kepler":
            # Kepler: 2x4 grid with trajectories only
            if "trajectories_true" in viz_data and "trajectories_pred" in viz_data:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
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
                    ax_traj.set_xlim(global_xlim)
                    ax_traj.set_ylim(global_ylim)
                    ax_traj.set_aspect("equal")

                    # Remove x-axis labels from top row (plots 0-3)
                    if plot_idx < 4:
                        ax_traj.set_xlabel("")
                        ax_traj.tick_params(axis='x', labelbottom=False)
                    else:
                        ax_traj.set_xlabel("X Position")

                    ax_traj.set_ylabel("Y Position")

                    if system_params[plot_idx] is not None:
                        ax_traj.set_title(f"M={system_params[plot_idx]:.2f}")
                    ax_traj.grid(True, alpha=0.3)

                # Add overall legend for the dynamics plot
                fig.legend(
                    handles=[plt.Line2D([0], [0], color='b', linewidth=2, label='True'),
                            plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Predicted'),
                            plt.Line2D([0], [0], color='g', marker='o', linestyle='None', markersize=6, label='Start'),
                            plt.Line2D([0], [0], color='k', marker='o', linestyle='None', markersize=8, label='Central Body')],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.95),
                    ncol=4,
                    frameon=False,
                )

                plt.tight_layout()
                if save_dir:
                    plt.savefig(save_dir / f"{problem_type}_progressive_dynamics.png", bbox_inches='tight')
                plt.show()

    def plot_polynomial_degree_comparison(self, degree3_path: Union[str, Path],
                                         degree4_path: Union[str, Path],
                                         degree5_path: Union[str, Path],
                                         save_dir: Optional[str] = None):
        """
        Plot comparison between polynomial degrees 3, 4, and 5.

        Creates a 2x3 plot:
        - Top row: MSE plots for D3, D4, D5
        - Bottom row: Eigenvalue spectrum during 10 basis for D3, D4, D5
        """

        # Load experiment data
        data_d3 = self.saver.load_experiment(degree3_path)
        data_d4 = self.saver.load_experiment(degree4_path)
        data_d5 = self.saver.load_experiment(degree5_path)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(5.5, 3))

        datasets = [data_d3, data_d4, data_d5]

        # Collect all data for consistent scaling
        all_losses = []
        all_eigenvalues = []

        for data in datasets:
            all_losses.extend(data["training_data"]["losses"])

            # Reconstruct scores for eigenvalue spectrum
            scores = []
            pca_data = data["pca_data"]
            if "num_scores" in pca_data:
                num_scores = pca_data["num_scores"]
                for i in range(num_scores):
                    score_key = f"score_{i}"
                    if score_key in pca_data:
                        scores.append(pca_data[score_key])
            if scores:
                all_eigenvalues.extend(scores[-1])  # Final eigenvalues

        for col, data in enumerate(datasets):
            # Top row: MSE loss plots
            ax_loss = axes[0, col]
            losses = data["training_data"]["losses"]
            ax_loss.plot(losses)

            # Only show y-label and y-ticks on leftmost plot
            if col == 0:
                ax_loss.set_ylabel("MSE")
            else:
                ax_loss.tick_params(axis='y', left=False, labelleft=False)

            # No x-labels and x-ticks on top row
            ax_loss.tick_params(axis='x', bottom=False, labelbottom=False)

            # Set consistent y-limits for all loss plots
            ax_loss.set_ylim(min(all_losses) * 0.9, max(all_losses) * 1.1)
            ax_loss.set_yscale("log")
            ax_loss.grid(True)

            # Bottom row: Eigenvalue spectrum (final basis)
            ax_eigen = axes[1, col]

            # Reconstruct scores for eigenvalue spectrum
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
                )

                # Only show x-label on bottom middle plot (col == 1)
                if col == 1:
                    ax_eigen.set_xlabel("Eigenvalue Index")

                # Only show y-label and y-ticks on leftmost plot
                if col == 0:
                    ax_eigen.set_ylabel("Explained\nVariance Ratio")
                else:
                    ax_eigen.tick_params(axis='y', left=False, labelleft=False)

                # Set consistent y-limits for all eigenvalue plots
                if all_eigenvalues:
                    ax_eigen.set_ylim(min(all_eigenvalues) * 0.9, max(all_eigenvalues) * 1.1)

                ax_eigen.set_yscale("log")
                ax_eigen.grid(True)
            else:
                ax_eigen.text(0.5, 0.5, 'No scores data available',
                            ha='center', va='center', transform=ax_eigen.transAxes)

                # Only show x-label on bottom middle plot (col == 1)
                if col == 1:
                    ax_eigen.set_xlabel("Eigenvalue Index")

                # Only show y-label and y-ticks on leftmost plot
                if col == 0:
                    ax_eigen.set_ylabel("Explained\nVariance Ratio")
                else:
                    ax_eigen.tick_params(axis='y', left=False, labelleft=False)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "polynomial_degree_comparison.png", bbox_inches='tight', dpi=300)
        plt.show()

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
        plt.show()

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
        plt.show()

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
        plt.show()

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
        plt.show()

        # Plot 3: Dynamics - Based on polynomial_prune.py
        viz_data = data["visualization_data"]

        if problem_type == "polynomial":
            # Function approximation comparison - exactly like polynomial_prune.py
            fig, ax = plt.subplots(1, 1)

            if "X_sorted" in viz_data:
                X_sorted = viz_data["X_sorted"]
                y_sorted = viz_data["y_sorted"]

                # Use comparison data for predictions
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
            # Multiple trajectory comparison for dynamical systems
            if "trajectories_true" in viz_data and "trajectories_pred" in viz_data:
                fig, axes = plt.subplots(2, 2)
                axes = axes.flatten()

                trajectories_true = viz_data["trajectories_true"][:1]  # Just first trajectory
                trajectories_pred = viz_data["trajectories_pred"][:3]  # Original, pruned, refined

                # Show comparison: True vs Original, True vs Pruned, True vs Refined, All together
                labels = ["Original", "Pruned", "Refined", "All Models"]
                colors = ["b--", "g--", "r:", ["b--", "g--", "r:"]]

                for i in range(4):
                    ax = axes[i]
                    ax.plot(trajectories_true[0][:, 0], trajectories_true[0][:, 1], 'k-', label='True', linewidth=1)

                    if i < 3:  # Individual comparisons
                        ax.plot(trajectories_pred[i][:, 0], trajectories_pred[i][:, 1],
                               colors[i], label=labels[i], alpha=0.8)
                    else:  # All together
                        for j, (color, label) in enumerate(zip(colors[3], labels[:3])):
                            ax.plot(trajectories_pred[j][:, 0], trajectories_pred[j][:, 1],
                                   color, label=label, alpha=0.8)

                    ax.plot(trajectories_true[0][0, 0], trajectories_true[0][0, 1], 'go', markersize=4)
                    if problem_type == "kepler":
                        ax.plot(0, 0, 'ko', markersize=6)

                    ax.set_aspect('equal')
                    ax.set_title(labels[i])
                    ax.legend(fontsize=6)
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"{problem_type}_prune_dynamics.png", bbox_inches='tight')
        plt.show()

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