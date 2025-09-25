
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
out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("kepler_2x4.png")
Dp = np.load("results/kepler_progressive_64_20250922_030233/visualization_data.npz")
Dq = np.load("results/kepler_prune_15tu_20250924_025630/visualization_data.npz")

# --- Top row (Progressive): take first 4 trajectories, keep XY only ---
# --- Progressive (take first 4 of 8) ---
Pt = Dp["trajectories_true"][:4, :, :2]    # (4,101,2)
Pp = Dp["trajectories_pred"][:4, :, :2]    # (4,101,2)
Pm = Dp["system_params"][:4] if "system_params" in Dp else [None]*4

# --- Prune truth ---
Qt = Dq["trajectories_true"][:, :, :2]     # (4,101,2)
Qm = Dq["system_params"][:4] if "system_params" in Dq else [None]*4

# --- REAL prune predictions (3 models × 4 traj × T × 4) ---
TALL   = Dq["trajectories_pred_all"]       # (3,4,101,4)
Orig   = TALL[0, :, :, :2]                  # (4,101,2)
Pruned = TALL[1, :, :, :2]                  # (4,101,2)
Refined= TALL[2, :, :, :2]                  # (4,101,2)


# --- Plot 2x4 ---
xlim = [-4, 4]
ylim = [-4, 4]
fig, axes = plt.subplots(2, 4)

# Top row: Progressive (True + Pred)
for j in range(4):
    ax = axes[0, j]
    tt, tp = Pt[j], Pp[j]
    ax.plot(tt[:,0], tt[:,1], "k-",  lw=1.6)
    ax.plot(tp[:,0], tp[:,1], "r--", lw=1.4, alpha=0.9)
    ax.plot(tp[-1,0], tp[-1,1], "ro", ms=4)
    ax.plot(0,0,"ko", ms=6)
    if Pm[j] is not None:
        ax.text(
        0.95,
        0.95,
        f"M={float(Pm[j]):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    ax.set_aspect("equal"); ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(True, alpha=0.3)

# Bottom row: Train-then-Prune (True + Original + Pruned + Refined)
for j in range(4):
    ax = axes[1, j]
    tt = Qt[j]
    ax.plot(tt[:,0], tt[:,1], "k-",  lw=1.6)
    ax.plot(Orig[j,   :, 0], Orig[j,   :, 1], "b--", lw=1.2, alpha=0.9)
    ax.plot(Pruned[j, :, 0], Pruned[j, :, 1], "g--", lw=1.2, alpha=0.9)
    ax.plot(Refined[j,:, 0], Refined[j,:, 1], "r:",  lw=1.6, alpha=0.95)
    ax.plot(Orig[j,-1, 0], Orig[j,-1, 1], "bo", ms=4)
    ax.plot(Refined[j,-1, 0], Refined[j,-1, 1], "ro", ms=4)
    ax.plot(0,0,"ko", ms=6)
    if Qm[j] is not None:
        ax.text(
        0.95,
        0.95,
        f"M={float(Qm[j]):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_aspect("equal"); ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(True, alpha=0.3)

# Hide inner tick labels, keep only leftmost Y and bottom X
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        if j != 0:   # not first column → hide y labels
            ax.set_yticklabels([])
        if i != len(axes)-1:  # not last row → hide x labels
            ax.set_xticklabels([])

# Shared labels + legend
fig.text(0.5, 0.02, "X Position", ha="center")
fig.text(0.02, 0.5, "Y Position", va="center", rotation="vertical")
handles = [
    plt.Line2D([0],[0], color="k", lw=1, label="True"),
    plt.Line2D([0],[0], color="r", ls="--", lw=1, label="Progressive Pred"),
    plt.Line2D([0],[0], color="b", ls="--", lw=1, label="Original"),
    plt.Line2D([0],[0], color="g", ls="--", lw=1, label="Pruned"),
    plt.Line2D([0],[0], color="r", ls=":",  lw=1, label="Refined"),
    # plt.Line2D([0],[0], color="g", marker="o", ls="None", ms=4, label="Start"),
    # plt.Line2D([0],[0], color="k", marker="o", ls="None", ms=6, label="Central body"),
]
fig.legend(handles=handles, 
           loc="upper center", 
           bbox_to_anchor=(0.5, 1.02), 
           ncol=6, 
           frameon=False,
           handletextpad=0.3,   # tighten space between line and text
           columnspacing=0.8,   # tighten space between legend entries
           fontsize=8  )

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.12, left=0.09, right=0.99)


plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path.resolve()}")