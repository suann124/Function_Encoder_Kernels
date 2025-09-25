import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set global rcParams for consistent formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'figure.figsize': (5.5, 2),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png'
})

# System configurations
systems = {
    'kepler': {
        'progressive_dir': 'results/kepler_progressive_64_20250924_045100',
        'prune_dir': 'results/kepler_prune_10tu_20250924_210635',
        'name': 'Kepler'
    },
    'vanderpol': {
        'progressive_dir': 'results/vdp_progressive_dt01_20250923_200711',
        'prune_dir': 'results/vdp_prune_dt01_20250923_202416',
        'name': 'Van der Pol'
    }
}

for system_key, system_info in systems.items():
    # Create 1x2 subplot for each system
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2),sharey=True)

    methods = ['Progressive', 'Train-then-prune']
    dirs = [system_info['progressive_dir'], system_info['prune_dir']]

    for i, (method, dir_path) in enumerate(zip(methods, dirs)):
        ax = axes[i]

        # Load training data
        training_data = np.load(f"{dir_path}/training_data.npz")

        # For progressive methods: use "losses"
        # For prune methods: use "train_losses" (initial training phase)
        if method == "Progressive" and "losses" in training_data:
            losses = training_data["losses"]
        elif method == "Train-then-prune" and "train_losses" in training_data and len(training_data["train_losses"]) > 0:
            losses = training_data["train_losses"]
        elif "losses" in training_data:  # Fallback
            losses = training_data["losses"]
        else:
            raise KeyError("No appropriate losses found")

        # Check if losses array is valid
        if len(losses) > 0 and not np.all(losses == 0):
            ax.plot(losses, ["b-", "r-"][i], linewidth=1, alpha=0.8)
        else:
            raise ValueError("Empty or invalid losses")

        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Only leftmost subplot gets y-label and y-ticks
        if i == 0:
            ax.set_ylabel('MSE')
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)

        # # Add method annotation
        # ax.text(
        #     0.05,
        #     0.95,
        #     method,
        #     transform=ax.transAxes,
        #     va='top',
        #     ha='right',
        #     fontsize=8,
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        # )

    fig.supxlabel("Epoch",y=0) 
    plt.tight_layout()
    outpath = Path("plots_output_new") / f"{system_key}_mse_comparison.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved {system_info['name']} MSE comparison: {outpath}")
    plt.show()