#!/usr/bin/env python3
# kepler_dynamics_split.py
#
# Creates separate single-row plots for Progressive and Train-then-prune methods
# Shows 4 Kepler trajectories in each plot
#
# Expects .npz keys:
#   trajectories_true : array [N, T, 2]
#   trajectories_pred : array [N, T, 2]
#   system_params     : optional array/list length N
import os
from pathlib import Path

# Always set cwd to the script's directory
os.chdir(Path(__file__).resolve().parent)
print("CWD forced to:", os.getcwd())

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "figure.figsize": (5.5, 3.0),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    'savefig.format': 'png',

})

# --- inputs ---
SAVE_DIR = "plots_output_new"

# System data definitions
SYSTEMS = {
    "kepler": {
        "prog_file": "results/kepler_progressive_64_20250922_030233/visualization_data.npz",
        "prune_file": "results/kepler_prune_15tu_20250924_025630/visualization_data.npz",
        "prog_filename": "kepler_progressive.png",
        "prune_filename": "kepler_prune.png",
        "xlim": [-4, 4],
        "ylim": [-4, 4],
        "xlabel": "X Position",
        "ylabel": "Y Position",
        "traj_idx": 3,  # trajectory index to plot
        "has_mass_params": True
    },
    "van_der_pol": {
        "prog_file": "results/vdp_progressive_dt01_20250923_200711/visualization_data.npz",
        "prune_file": "results/vdp_prune_dt01_20250923_202416/visualization_data.npz",
        "prog_filename": "vdp_progressive.png",
        "prune_filename": "vdp_prune.png",
        "xlim": [-4, 4],
        "ylim": [-4, 4],
        "xlabel": "Position",
        "ylabel": "Velocity",
        "traj_idx": 4,  # trajectory index to plot
        "has_mass_params": False
    }
}

def plot_progressive_trajectories(system_name, system_config, outdir):
    """Create progressive plot for a given system"""
    Dp = np.load(system_config["prog_file"])

    # Progressive data
    n_traj = min(4, Dp["trajectories_true"].shape[0])  # Use available trajectories, max 4
    Pt = Dp["trajectories_true"][:n_traj, :, :2]
    Pp = Dp["trajectories_pred"][:n_traj, :, :2]
    Pm = Dp["system_params"][:n_traj] if "system_params" in Dp else [None]*n_traj

    fig_prog, axes_prog = plt.subplots(1, 4, figsize=(5.5, 1.5))

    for j in range(4):
        ax = axes_prog[j]
        if j < n_traj:
            tt, tp = Pt[j], Pp[j]
            ax.plot(tt[:,0], tt[:,1], "k-",  lw=1.6)
            ax.plot(tp[:,0], tp[:,1], "r--", lw=1.4, alpha=0.9)

            if system_name == "kepler":
                ax.plot(tp[-1,0], tp[-1,1], "ro", ms=4)
                ax.plot(0,0,"ko", ms=6)
                if system_config["has_mass_params"] and Pm[j] is not None:
                    ax.text(0.95, 0.95, f"M={float(Pm[j]):.2f}",
                           transform=ax.transAxes, ha="right", va="top", fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            # Hide empty subplots
            ax.axis('off')
            continue

        ax.set_aspect("equal")
        ax.set_xlim(system_config["xlim"])
        ax.set_ylim(system_config["ylim"])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelleft=(j == 0))

    # Labels and legend
    fig_prog.text(0.5, -0.05, system_config["xlabel"], ha="center", va="bottom", fontsize=8)
    fig_prog.text(0.01, 0.5, system_config["ylabel"], ha="center", va="center", rotation="vertical", fontsize=8)

    handles_prog = [
        plt.Line2D([0],[0], color="k", lw=1, label="True"),
        plt.Line2D([0],[0], color="r", ls="--", lw=1, label="Predicted"),
    ]
    fig_prog.legend(handles=handles_prog, loc="upper center", bbox_to_anchor=(0.5, 1.08),
                   ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8, fontsize=8)

    plt.subplots_adjust(bottom=0.25)
    fig_prog.tight_layout()

    outpath_prog = outdir / system_config["prog_filename"]
    fig_prog.savefig(outpath_prog, bbox_inches="tight")
    print(f"Saved: {outpath_prog}")
    # plt.close(fig_prog)

def plot_prune_trajectories(system_name, system_config, outdir):
    """Create prune plot for a given system"""
    Dq = np.load(system_config["prune_file"])

    # Prune data
    n_traj = min(4, Dq["trajectories_true"].shape[0])  # Use available trajectories, max 4
    Qt = Dq["trajectories_true"][:n_traj, :, :2]
    Qm = Dq["system_params"][:n_traj] if "system_params" in Dq else [None]*n_traj

    if "trajectories_pred_all" in Dq.files:
        TALL = Dq["trajectories_pred_all"]
        Orig = TALL[0, :n_traj, :, :2]
        Pruned = TALL[1, :n_traj, :, :2]
        Refined = TALL[2, :n_traj, :, :2]
    else:
        # Fallback if trajectories_pred_all doesn't exist
        pred = Dq["trajectories_pred"][:n_traj, :, :2]
        Orig = Refined = pred
        Pruned = pred

    fig_prune, axes_prune = plt.subplots(1, 4, figsize=(5.5, 1.5))

    for j in range(4):
        ax = axes_prune[j]
        if j < n_traj:
            tt = Qt[j]
            ax.plot(tt[:,0], tt[:,1], "k-",  lw=1.6)
            ax.plot(Orig[j,   :, 0], Orig[j,   :, 1], "b--", lw=1.2, alpha=0.9)
            ax.plot(Pruned[j, :, 0], Pruned[j, :, 1], "g--", lw=1.2, alpha=0.9)
            ax.plot(Refined[j,:, 0], Refined[j,:, 1], "r:",  lw=1.6, alpha=0.95)

            if system_name == "kepler":
                ax.plot(Orig[j,-1, 0], Orig[j,-1, 1], "bo", ms=4)
                ax.plot(Refined[j,-1, 0], Refined[j,-1, 1], "ro", ms=4)
                ax.plot(0,0,"ko", ms=6)
                if system_config["has_mass_params"] and Qm[j] is not None:
                    ax.text(0.95, 0.95, f"M={float(Qm[j]):.2f}",
                           transform=ax.transAxes, ha="right", va="top", fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            # Hide empty subplots
            ax.axis('off')
            continue

        ax.set_aspect("equal")
        ax.set_xlim(system_config["xlim"])
        ax.set_ylim(system_config["ylim"])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelleft=(j == 0))

    # Labels and legend
    fig_prune.text(0.5, -0.05, system_config["xlabel"], ha="center", va="bottom", fontsize=8)
    fig_prune.text(0.01, 0.5, system_config["ylabel"], ha="center", va="center", rotation="vertical", fontsize=8)

    handles_prune = [
        plt.Line2D([0],[0], color="k", lw=1, label="True"),
        plt.Line2D([0],[0], color="b", ls="--", lw=1, label="Original"),
        plt.Line2D([0],[0], color="g", ls="--", lw=1, label="Pruned"),
        plt.Line2D([0],[0], color="r", ls=":",  lw=1, label="Refined"),
    ]
    fig_prune.legend(handles=handles_prune, loc="upper center", bbox_to_anchor=(0.5, 1.08),
                    ncol=4, frameon=False, handletextpad=0.3, columnspacing=0.8, fontsize=8)

    plt.subplots_adjust(bottom=0.25)
    fig_prune.tight_layout()

    outpath_prune = outdir / system_config["prune_filename"]
    fig_prune.savefig(outpath_prune, bbox_inches="tight")
    print(f"Saved: {outpath_prune}")
    # plt.close(fig_prune)

def main():
    outdir = Path(SAVE_DIR); outdir.mkdir(parents=True, exist_ok=True)

    print("Creating trajectory plots for all systems...")
    for system_name, system_config in SYSTEMS.items():
        print(f"Processing {system_name}...")
        plot_progressive_trajectories(system_name, system_config, outdir)
        plot_prune_trajectories(system_name, system_config, outdir)

    print("Plot creation complete!")

if __name__ == "__main__":
    main()
