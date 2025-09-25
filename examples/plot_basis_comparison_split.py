import os
from pathlib import Path

# Always set cwd to the script's directory
os.chdir(Path(__file__).resolve().parent)
print("CWD forced to:", os.getcwd())

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import json

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "figure.figsize": (5.5, 3.0),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    'savefig.format': 'png',

})

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
PROGRESSIVE_DIR = "results/polynomial_progressive_polynomial_d3_20250924_024746" 
PRUNE_DIR       = "results/polynomial_prune_d3_20250923_005000"
SAVE_DIR        = "plots_output_new"
FILENAME_PROG   = "basis_progressive_only.png"
FILENAME_PRUNE  = "basis_prune_only.png"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def load_experiment(dir_path):
    """
    Load experiment data from a directory containing
    metadata.json, visualization_data.npz, pca_data.npz
    """
    dir_path = Path(dir_path)

    # metadata
    with open(dir_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    # visualization_data
    viz = np.load(dir_path / "visualization_data.npz", allow_pickle=True)
    visualization_data = {k: viz[k].tolist() for k in viz.files}

    # pca_data (may not exist for progressive runs)
    pca_file = dir_path / "pca_data.npz"
    pca_data = {}
    if pca_file.exists():
        pca = np.load(pca_file, allow_pickle=True)
        pca_data = {k: pca[k].tolist() for k in pca.files}

    return {
        "metadata": metadata,
        "visualization_data": visualization_data,
        "pca_data": pca_data
    }

def plot_progressive_polynomial(axes, prog):
    basis_prog = [np.asarray(b) for b in prog["visualization_data"]["basis_outputs"]]
    Xp = np.linspace(-1, 1, len(basis_prog[0]))

    top_vals = np.concatenate(basis_prog[:4]) if basis_prog else np.array([-1, 1])
    m = 0.5 * (top_vals.max() - top_vals.min())
    ylim_top = [top_vals.min() - m, top_vals.max() + m]

    # Progressive (blue)
    for j in range(4):
        ax = axes[j]
        if j < len(basis_prog):
            ax.plot(Xp, basis_prog[j], color="blue")
            ax.text(0.95, 0.95, f"ψ{j+1}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(ylim_top)
        ax.tick_params(axis="y", labelleft=(j == 0))

def plot_prune_polynomial(axes, prn):
    keep = prn["pca_data"].get("keep_indices", [])
    basis_prn = [np.asarray(b) for b in prn["visualization_data"]["basis_outputs"]]
    Xq = np.linspace(-1, 1, len(basis_prn[0]))

    bottom_vals = []
    for idx in keep[:4]:
        if 0 <= idx < len(basis_prn):
            bottom_vals.extend(basis_prn[idx])
    if bottom_vals:
        bottom_vals = np.array(bottom_vals)
        m = 0.1 * (bottom_vals.max() - bottom_vals.min())
        ylim_bottom = [bottom_vals.min() - m, bottom_vals.max() + m]
    else:
        ylim_bottom = [-1, 1]

    # Prune (red)
    for j in range(4):
        ax = axes[j]
        if j < len(keep) and 0 <= keep[j] < len(basis_prn):
            idx = keep[j]
            ax.plot(Xq, basis_prn[idx], color="red")
            ax.text(0.95, 0.95, f"ψ{idx+1}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(ylim_bottom)
        ax.tick_params(axis="y", labelleft=(j == 0))

def main():
    prog = load_experiment(PROGRESSIVE_DIR)
    prn  = load_experiment(PRUNE_DIR)

    problem_type = prog["metadata"]["problem_type"]

    if problem_type == "polynomial":
        # Create Progressive plot
        fig_prog, axes_prog = plt.subplots(1, 4, figsize=(5.5, 1.5))
        plot_progressive_polynomial(axes_prog, prog)
        fig_prog.text(0.5, 0.002, "x", ha="center", va="bottom", fontsize=8)
        fig_prog.text(0.01, 0.5, "ψ(x)", ha="center", va="center", rotation="vertical", fontsize=8)

        # prog_line = mlines.Line2D([], [], color="blue", label="Progressive")
        # fig_prog.legend(
        #     handles=[prog_line],
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 1.15),
        #     fontsize=8,
        #     frameon=False
        # )
        # plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.85)

        fig_prog.tight_layout()
        outdir = Path(SAVE_DIR); outdir.mkdir(parents=True, exist_ok=True)
        outpath_prog = outdir / FILENAME_PROG
        fig_prog.savefig(outpath_prog, bbox_inches="tight")
        print(f"Saved: {outpath_prog}")
        # plt.close(fig_prog)

        # Create Prune plot
        fig_prune, axes_prune = plt.subplots(1, 4, figsize=(5.5, 1.5))
        plot_prune_polynomial(axes_prune, prn)
        for ax in axes_prune:
            ax.set_ylim(-2, 2)
        fig_prune.text(0.5, 0.002, "x", ha="center", va="bottom", fontsize=8)
        fig_prune.text(0.01, 0.5, "ψ(x)", ha="center", va="center", rotation="vertical", fontsize=8)

        # prune_line = mlines.Line2D([], [], color="red", label="Train-then-prune")
        # fig_prune.legend(
        #     handles=[prune_line],
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 1.15),
        #     fontsize=8,
        #     frameon=False
        # )
        # plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.85)
        fig_prune.tight_layout()
        outpath_prune = outdir / FILENAME_PRUNE
        fig_prune.savefig(outpath_prune, bbox_inches="tight")
        print(f"Saved: {outpath_prune}")
        # plt.close(fig_prune)

    else:
        print(f"Unsupported problem type: {problem_type}")

if __name__ == "__main__":
    main()