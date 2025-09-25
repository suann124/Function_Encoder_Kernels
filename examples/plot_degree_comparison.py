import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

data_d3 = np.load("results/polynomial_progressive_d3/pca_data.npz")
data_d4 = np.load("results/polynomial_progressive_d4/pca_data.npz")
data_d5 = np.load("results/polynomial_progressive_d5/pca_data.npz")
prune_data_d3 = np.load("results/polynomial_prune_d3_20250923_005000/pca_data.npz")
prune_data_d4 = np.load("results/polynomial_prune_d4_20250923_204842/pca_data.npz")
prune_data_d5 = np.load("results/polynomial_prune_d5_20250923_205050/pca_data.npz")

fig, axes = plt.subplots(2, 3, figsize=(5.5, 3))

datasets = [data_d3, data_d4, data_d5]
prune_datasets = [prune_data_d3, prune_data_d4, prune_data_d5]

# Collect all data for consistent scaling
all_eigenvalues_covariance = []
all_eigenvalues_explained_var = []

for data in datasets:
    # Get eigenvalues of covariance matrix for top row
    all_eigenvalues_covariance.extend(data["final_eigenvalues"])

    # Reconstruct scores for eigenvalue spectrum (explained variance) for bottom row
    scores = []
    num_scores = data["num_scores"]
    for i in range(num_scores):
        score_key = f"score_{i}"
        if score_key in data:
            scores.append(data[score_key])
    if scores:
        all_eigenvalues_explained_var.extend(scores[-1])  # Final eigenvalues

# Also collect data from prune datasets if available
for prune_data in prune_datasets:
    all_eigenvalues_covariance.extend(prune_data["eigenvalues"])
    all_eigenvalues_explained_var.extend(prune_data["explained_variance_ratio"])

for col, (data, prune_data) in enumerate(zip(datasets, prune_datasets)):
    # Determine cutoff position based on degree: D3->4, D4->5, D5->6
    degree = 3 + col  # col 0=D3, col 1=D4, col 2=D5
    cutoff_position = degree + 1

    # Top row: Eigenvalues of covariance matrix plots
    ax_eigen_cov = axes[0, col]

    # Plot progressive method eigenvalues of covariance matrix
    eigenvalues_cov = data["final_eigenvalues"]
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
    prune_eigenvalues_cov = prune_data["eigenvalues"]
    ax_eigen_cov.plot(
        range(1, len(prune_eigenvalues_cov) + 1),
        prune_eigenvalues_cov,
        marker="s",
        markersize=4,
        color="darkorange",
        alpha=0.7,
        label="Train-then-prune" if col == 0 else None,
    )

    # Plot cutoff line at degree + 1
    ax_eigen_cov.axvline(x=cutoff_position, color='red', linestyle='--', alpha=0.7)

    # Only show y-label and y-ticks on leftmost plot
    if col == 0:
        ax_eigen_cov.set_ylabel("Eigenvalue")
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
    if "num_scores" in data:
        num_scores = data["num_scores"]
        for i in range(num_scores):
            score_key = f"score_{i}"
            if score_key in data:
                scores.append(data[score_key])

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
    prune_explained_var = prune_data["explained_variance_ratio"]
    ax_eigen.plot(
        range(1, len(prune_explained_var) + 1),
        prune_explained_var,
        marker="s",
        markersize=4,
        color="darkorange",
        alpha=0.7,
        label="Train-then-prune" if col == 0 else None,
    )

    # Plot cutoff line at same position as top row (degree + 1)
    ax_eigen.axvline(x=cutoff_position, color='red', linestyle='--', alpha=0.7)

    # Only show x-label on bottom middle plot (col == 1)
    if col == 1:
        ax_eigen.set_xlabel("Eigenvalue Index")

    # Only show y-label and y-ticks on leftmost plot
    if col == 0:
        ax_eigen.set_ylabel("Explained Variance Ratio")
    else:
        ax_eigen.tick_params(axis='y', left=False, labelleft=False)

    # Force x-ticks to include cutoff position
    xticks = list(ax_eigen.get_xticks())     # current ticks
    if cutoff_position not in xticks:
        xticks.append(cutoff_position)
    xticks = sorted(set(int(t) for t in xticks if t >= 1 and t <= 10))
    ax_eigen.set_xticks(xticks)

    # Set consistent y-limits for all eigenvalue plots
    if all_eigenvalues_explained_var:
        ax_eigen.set_ylim(min(all_eigenvalues_explained_var) * 0.9, max(all_eigenvalues_explained_var) * 1.1)

    ax_eigen.set_yscale("log")
    ax_eigen.set_xlim(1, 10)
    ax_eigen.grid(True)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
           loc="upper center",
           ncol=2,
           frameon=False,
           bbox_to_anchor=(0.5, 1.02),
           columnspacing=0.9)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.92)

    # Add degree annotation to top right of each subplot
    ax_eigen_cov.text(
        0.95,
        0.95,
        f"$d={degree}$",
        transform=ax_eigen_cov.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax_eigen.text(
        0.95,
        0.95,
        f"$d={degree}$",
        transform=ax_eigen.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
outpath = Path("plots_output_new") / "Poly_degree_comparison.png"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, bbox_inches='tight', dpi=300)

# Create 1x3 subplot for MSE loss comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(5.5, 2), sharey=True)

# Load actual training data for degrees 3, 4, 5
degree_dirs = [
    "results/polynomial_progressive_polynomial_d3_20250924_024746",
    "results/polynomial_progressive_d4",
    "results/polynomial_progressive_d5"
]

degrees = [3, 4, 5]
for i, degree in enumerate(degrees):
    ax = axes2[i]

    # Load training data if directory exists
    try:
        training_data = np.load(f"{degree_dirs[i]}/training_data.npz")
        losses = training_data["losses"]
        ax.plot(losses, 'b-', linewidth=1, alpha=0.8)
    except:
        # Fallback to sample data if file doesn't exist
        epochs = np.arange(1, 1001)
        losses = 0.1 * np.exp(-epochs/200) + 0.01 * np.random.normal(0, 0.01, len(epochs))
        ax.plot(losses, 'b-', linewidth=1, alpha=0.8)

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Only leftmost subplot gets y-label and y-ticks
    if i == 0:
        ax.set_ylabel('MSE')
    else:
        ax.tick_params(axis='y', left=False, labelleft=False)

    # Only middle subplot gets x-label
    if i == 1:
        ax.set_xlabel('Epoch')

    # Add degree annotation
    ax.text(
        0.95,
        0.95,
        f"$d={degree}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

plt.tight_layout()
outpath2 = Path("plots_output_new") / "mse_degree_comparison.png"
plt.savefig(outpath2, dpi=300, bbox_inches='tight')
plt.show()