from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------- Matplotlib style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "figure.figsize": (5.5, 3.0),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "png",
})

# ---------- Run directories ----------
BASE_DIR = Path("results")

RUNS = {
    "van_der_pol": {
        "progressive": "vdp_progressive_dt01_20250923_200711",
        "prune":       "vdp_prune_dt01_20250923_202416",
    },
    "kepler": {
        "progressive": "kepler_progressive_64_20250924_045100",
        "prune":       "kepler_prune_15tu_20250924_025630",
    },
}

# ---------- Colors / markers ----------
COLORS = {
    "Progressive": "#1f77b4",
    "Train-then-prune": "#ff7f0e",
    "true": "k",
    "before": "b",
    "after": "#d62728",
}
MARKERS = {"Progressive": "o", "Train-then-prune": "s"}


# ---------- Data loading ----------
def load_dynamical_systems_data(base_dir: Path = BASE_DIR):
    """Load trajectory data for van der Pol and Kepler systems."""
    data = {}

    for system, dirs in RUNS.items():
        prog_dir = base_dir / dirs["progressive"]
        prune_dir = base_dir / dirs["prune"]

        prog_viz = prog_dir / "visualization_data.npz"
        prune_viz = prune_dir / "visualization_data.npz"
        cmp_file  = prune_dir / "comparison_data.npz"

        if prog_viz.exists() and prune_viz.exists():
            prog = np.load(prog_viz)
            prun = np.load(prune_viz)
            cmpd = np.load(cmp_file) if cmp_file.exists() else None

            traj_pred_all = (
                prun["trajectories_pred_all"] if "trajectories_pred_all" in prun.files else None
            )

            data[system] = {
                "progressive": {
                    "trajectories_true": prog["trajectories_true"],
                    "trajectories_pred": prog["trajectories_pred"],
                    "initial_conditions": prog["initial_conditions"],
                },
                "prune": {
                    "trajectories_true": prun["trajectories_true"],
                    "trajectories_pred": prun["trajectories_pred"],
                    "trajectories_pred_all": traj_pred_all,
                    "initial_conditions": prun["initial_conditions"],
                    "comparison": cmpd,
                },
            }

    return data


def load_pca_data(base_dir: Path = BASE_DIR):
    """Load PCA explained variance data from the same dirs."""
    pca = {}

    for system, dirs in RUNS.items():
        prog_pca = base_dir / dirs["progressive"] / "pca_data.npz"
        prune_pca = base_dir / dirs["prune"] / "pca_data.npz"

        if prog_pca.exists() and prune_pca.exists():
            a, b = np.load(prog_pca), np.load(prune_pca)
            cutoff = 2 if system == "van_der_pol" else 5
            pca[system] = {
                "progressive": {"explained_variance": a[f"score_{int(a['num_scores'])-1}"], "num_components": cutoff},
                "prune":       {"explained_variance": b["explained_variance_ratio"], "num_components_kept": cutoff},
            }

    return pca


# ---------- Plotting ----------
def create_trajectory_plots(data_dict, system="van_der_pol", save_path=None):
    """
    Make the figure with:
      - PCA legend on top-left
      - Shared legend (for right two) on top-right
      - Right two axes share Y axis (ticks on middle only) and a shared group X title
      - Wider gap between left/middle; tighter gap between middle/right
    """
    plt.rcParams.update({"font.size": 8})

    # GridSpec with a skinny spacer column between left & middle
    # col0: PCA, col1: spacer, col2: Progressive, col3: Prune
    fig = plt.figure(figsize=(5.5, 2))
    gs = GridSpec(
        nrows=1, ncols=4, figure=fig,
        width_ratios=[1.0, 0.2, 1.0, 1.0],  # spacer makes left↔middle farther
        wspace=0.08  # tight between middle↔right
    )

    ax_scree = fig.add_subplot(gs[0, 0])
    # skip gs[0,1] (spacer)
    ax_prog  = fig.add_subplot(gs[0, 2])
    ax_prune = fig.add_subplot(gs[0, 3], sharey=ax_prog)  # share Y with middle

    if system not in data_dict:
        print(f"Warning: No data found for {system}")
        plt.show()
        return fig

    system_data = data_dict[system]
    pca_data = load_pca_data()
    system_pca = pca_data.get(system, {})

    # ---- (1) PCA scree + legend handles ----
    pca_handles = []
    for algorithm in ["progressive", "prune"]:
        if algorithm in system_pca:
            info = system_pca[algorithm]
            ev = info["explained_variance"]
            x = range(1, len(ev) + 1)
            label = "Progressive" if algorithm == "progressive" else "Train-then-prune"
            h, = ax_scree.plot(
                x, ev, "-", label=label,
                color=COLORS[label], marker=MARKERS[label],
                linewidth=1, markersize=3, zorder=3
            )
            cutoff = info["num_components"] if algorithm == "progressive" else info["num_components_kept"]
            ax_scree.axvline(x=cutoff, color="red", linestyle="--", linewidth=1, zorder=2)
            pca_handles.append(h)

    if system == "van_der_pol":
        ax_scree.set_xlim(1, 5);  ax_scree.set_xticks([1, 2, 5])
        ax_scree.set_xlabel("Eigenvalue Index")
        ax_scree.set_ylabel("Eigenvalues")
    else:
        ax_scree.set_xlim(1, 10); ax_scree.set_xticks([1, 5, 10])
        ax_scree.set_xlabel("Eigenvalue Index")
        ax_scree.set_ylabel("Eigenvalues")

    ax_scree.grid(True, alpha=0.3)
    ax_scree.set_title("PCA Analysis")

    # ---- (2) Progressive (middle; will carry shared Y label/ticks) ----
    prog_data = system_data["progressive"]
    traj_handles = []

    if system == "van_der_pol":
        i = 4
        t_true, t_pred = prog_data["trajectories_true"][i], prog_data["trajectories_pred"][i]
        h_true, = ax_prog.plot(t_true[:,0], t_true[:,1], color=COLORS["true"], linewidth=1, label="True")
        h_prog, = ax_prog.plot(t_pred[:,0], t_pred[:,1], color=COLORS["Progressive"], linewidth=1.5, linestyle="--", label="Progressive")
        ax_prog.set_ylabel("Velocity")  # shared Y label shown on middle
        ax_prog.yaxis.set_label_coords(-0.15, 0.5)
        ax_prog.set_xticks([-4,-2,0,2,4]); ax_prog.set_yticks([-4,-2,0,2,4])
        traj_handles = [h_true, h_prog]
    else:
        i = 3
        t_true, t_pred = prog_data["trajectories_true"][i], prog_data["trajectories_pred"][i]
        h_true, = ax_prog.plot(t_true[:,0], t_true[:,1], color=COLORS["true"], linewidth=1, label="True")
        h_prog, = ax_prog.plot(t_pred[:,0], t_pred[:,1], color=COLORS["Progressive"], linewidth=1.5, linestyle="--", label="Progressive")
        # h_mb,   = ax_prog.plot(0,0,"ko",markersize=6,label="Main Body")
        ax_prog.plot(t_pred[-1,0], t_pred[-1,1], "o", color=COLORS["Progressive"], markersize=4)
        ax_prog.set_ylabel("Y")  # shared Y label shown on middle
        ax_prog.yaxis.set_label_coords(-0.15, 0.5)
        traj_handles = [h_true, h_prog]

    ax_prog.set_aspect("equal", adjustable="datalim")
    ax_prog.set_xlim(-4.5,4.5); ax_prog.set_ylim(-4.5,4.5)
    ax_prog.grid(True, alpha=0.3)
    ax_prog.set_title("Progressive")

    # ---- (3) Prune (right; hide Y tick labels; keep X ticks) ----
    pr_data = system_data["prune"]
    t_true = pr_data["trajectories_true"][0]

    if system=="kepler" and pr_data.get("trajectories_pred_all") is not None:
        pred0, pred2 = pr_data["trajectories_pred_all"][0,0], pr_data["trajectories_pred_all"][2,0]
    else:
        pred0, pred2 = pr_data["trajectories_pred"][0], pr_data["trajectories_pred"][2]

    h_true2, = ax_prune.plot(t_true[:,0], t_true[:,1], color=COLORS["true"], linewidth=1, label="True")
    h_bef,   = ax_prune.plot(pred0[:,0], pred0[:,1], color=COLORS["before"], linewidth=1.5, linestyle="--", label="Before Prune")
    h_aft,   = ax_prune.plot(pred2[:,0], pred2[:,1], color=COLORS["after"],  linewidth=1.5, linestyle=":",  label="After Retrain")

    if system=="kepler":
        h_mb2, = ax_prune.plot(0,0,"ko",markersize=6)
        ax_prune.plot(pred0[-1,0], pred0[-1,1], "o", color=COLORS["before"], markersize=4)
        ax_prune.plot(pred2[-1,0], pred2[-1,1], "o", color=COLORS["after"], markersize=4)
        traj_handles += [h_bef, h_aft, h_mb2]
    else:
        ax_prune.set_xticks([-4,-2,0,2,4])
        traj_handles += [h_bef, h_aft]

    # Hide right panel's Y tick labels (shared with middle)
    ax_prune.tick_params(labelleft=False)
    ax_prune.set_aspect("equal", adjustable="datalim")
    ax_prune.set_xlim(-4.5,4.5); ax_prune.set_ylim(-4.5,4.5)
    ax_prune.grid(True, alpha=0.3)
    ax_prune.set_title("Train-then-prune")

    # ---- Shared legends (inline on top) ----
    # PCA legend block (left)
    if pca_handles:
        pca_labels = [h.get_label() for h in pca_handles]
        fig.legend(pca_handles, pca_labels,
                   loc="upper center", bbox_to_anchor=(0.25, 1.15),
                   ncol=1, frameon=True, fontsize=8)

    # Trajectory legend block (right two)
    traj_labels = []
    uniq_traj_handles = []
    for h in traj_handles:
        lab = h.get_label()
        if lab not in traj_labels and not lab.startswith("_"):
            traj_labels.append(lab); uniq_traj_handles.append(h)

    fig.legend(uniq_traj_handles, traj_labels,
               loc="upper center", bbox_to_anchor=(0.65, 1.15),
               ncol=2, frameon=True, fontsize=8)

    # ---- Layout: x-axis titles ----
    # Left plot keeps its own xlabel
    ax_scree.set_xlabel("Eigenvalue Index")

    # Right-two shared X title centered under the two right axes
    fig.canvas.draw()  # ensure positions are updated
    bbox_mid = ax_prog.get_position()
    bbox_rgt = ax_prune.get_position()
    group_center_x = 0.5 * (bbox_mid.x0 + bbox_rgt.x1)
    # Choose the shared X label text based on system
    shared_xlabel = "Position" if system == "van_der_pol" else "X"
    fig.text(group_center_x, 0.01, shared_xlabel, ha="center", va="center", fontsize=8)

    # Final spacing tweaks
    plt.subplots_adjust(bottom=0.15, top=0.78, wspace=0.6)

    # Save + show
    if save_path:
        outdir = Path(save_path).parent
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


# ---------- Main ----------
def main():
    print("Loading dynamical systems data...")
    data = load_dynamical_systems_data()
    print(f"Found data for systems: {list(data.keys())}")

    outdir = Path("plots_output_new"); outdir.mkdir(parents=True, exist_ok=True)

    for system in data.keys():
        print(f"Creating plot for {system}...")
        save_path = outdir / f"{system}_dynamics_comparison_5.png"
        create_trajectory_plots(data, system, save_path)

    print("Plot creation complete!")


if __name__ == "__main__":
    main()