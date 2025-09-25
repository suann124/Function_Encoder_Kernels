# dynamics_basis_streamplot.py
# -------------------------------------------------
# Top row: first 6 bases from progressive (blue)
# Bottom row: kept 6 from pruned-refined (red)
# No metadata reads. Calls each basis's internal MLP directly.

import os, sys, re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Always set cwd to the script's directory
os.chdir(Path(__file__).resolve().parent)
print("CWD forced to:", os.getcwd())

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (5.5, 3.0),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "png",
})

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# EDIT THESE:
PROGRESSIVE_DIR  = "results/kepler_progressive_64_20250924_045100"
PRUNE_DIR        = "results/kepler_prune_15tu_20250924_025630"
PROG_MODEL_FILE  = "model_full.pth"            # inside progressive dir
PRUNE_MODEL_FILE = "kepler_refined_model.pth"  # inside prune dir
SAVE_DIR         = "plots_output"
FILENAME         = "basis_selection_dynamics_stream.png"

GRID_MIN, GRID_MAX = -1.0, 1.0
GRID_N = 25
DT_CONST = 0.1        # constant dt used for visualization input
VX0, VY0 = 0.0, 0.0   # set velocities to zero for 2D streamplot
X_LABEL = "X"         # axis label
Y_LABEL = "Y"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# repo imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder

# --------- helpers to load models (state_dict or full) ----------

def _num_basis_from_state_dict(sd) -> int | None:
    idxs = []
    for k in sd.keys():
        m = re.match(r"^basis_functions\.basis_functions\.(\d+)\.", k)
        if m:
            idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else None

def _basis_factory():
    # same architecture as training: MLP 5->64->64->4, wrapped in ODEFunc/NeuralODE
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[5, 64, 64, 4])),
        integrator=rk4_step,  # harmless here; we won't call forward()
    )

def _build_model(n_basis: int) -> FunctionEncoder:
    bfs = BasisFunctions(*[_basis_factory() for _ in range(n_basis)])
    return FunctionEncoder(bfs)

def load_model(model_path: Path) -> FunctionEncoder:
    ckpt = torch.load(model_path, map_location="cpu")
    # Full pickled model?
    if hasattr(ckpt, "eval") and hasattr(ckpt, "basis_functions"):
        m = ckpt
        m.eval()
        return m
    # state_dict
    if isinstance(ckpt, dict):
        n = _num_basis_from_state_dict(ckpt)
        if n is None:
            raise RuntimeError(f"Couldn't infer #bases from {model_path}")
        m = _build_model(n)
        m.load_state_dict(ckpt, strict=True)
        m.eval()
        return m
    raise TypeError(f"Unrecognized checkpoint format at {model_path}")

def load_keep_indices(prune_dir: Path):
    arr = np.load(prune_dir / "pca_data.npz", allow_pickle=True)
    return [int(i) for i in np.asarray(arr["keep_indices"]).ravel().tolist()]

# --------------- evaluate vector fields via internal MLP ----------------
# We bypass NeuralODE.forward()/integrator and call the basis MLP directly.
# Input per point is 5D: [x, y, vx, vy, dt] -> output 4D; we streamplot [:, :, 0:2].

def eval_bases_internal_mlp(model: FunctionEncoder, x_min, x_max, n, vx0=0.0, vy0=0.0, dt=0.1):
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(x_min, x_max, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")                            # [n, n]
    pts5 = np.stack([X.ravel(),
                     Y.ravel(),
                     np.full(X.size, vx0),
                     np.full(X.size, vy0),
                     np.full(X.size, dt)], axis=1)                        # [n*n, 5]
    inp = torch.from_numpy(pts5).float().unsqueeze(0)                     # [1, n*n, 5]

    outs = []
    with torch.no_grad():
        for b in model.basis_functions.basis_functions:
            # Call the internal MLP directly:
            o = b.ode_func.model(inp)   # shape [1, n*n, 4]
            if o.dim() == 3 and o.shape[0] == 1:
                o = o[0]                # [n*n, 4]
            outs.append(o[:, :2])       # take first two components for U,V

    # Stack -> [n*n, 2, n_basis] -> reshape to [n, n, 2, n_basis]
    out = torch.stack(outs, dim=-1)                                     # [n*n, 2, n_basis]
    B = out.cpu().numpy().reshape(n, n, 2, -1)
    return X, Y, B

# ------------------------ plotting ------------------------

def plot_stream_basis_row(ax_row, X, Y, B, color, indices, labels=None, row_is_bottom=False):
    n_basis = B.shape[-1]
    m = min(len(ax_row), len(indices))
    for j in range(m):
        ax = ax_row[j]
        idx = indices[j]
        lbl = labels[j] if (labels is not None and j < len(labels)) else idx
        if 0 <= idx < n_basis:
            U = B[:, :, 0, idx]
            V = B[:, :, 1, idx]
            ax.streamplot(X, Y, U, V, color=color, density=1.0, linewidth=0.8, arrowsize=0.5)
            ax.text(0.95, 0.95, f"ψ{lbl+1}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "Index OOB", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim([GRID_MIN, GRID_MAX]); ax.set_ylim([GRID_MIN, GRID_MAX])
        ax.tick_params(axis="x", labelbottom=row_is_bottom)
        ax.tick_params(axis="y", labelleft=(j == 0))
    for j in range(m, len(ax_row)):
        ax_row[j].axis("off")

def main():
    prog_path  = Path(PROGRESSIVE_DIR) / PROG_MODEL_FILE
    prune_path = Path(PRUNE_DIR)        / PRUNE_MODEL_FILE

    model_prog  = load_model(prog_path)
    model_prune = load_model(prune_path)

    # Evaluate basis vector fields via internal MLP (no integrator)
    Xp, Yp, Bp = eval_bases_internal_mlp(model_prog,  GRID_MIN, GRID_MAX, GRID_N, VX0, VY0, DT_CONST)
    Xq, Yq, Bq = eval_bases_internal_mlp(model_prune, GRID_MIN, GRID_MAX, GRID_N, VX0, VY0, DT_CONST)

    # Choose indices: first 6 progressive; first 6 kept from prune
    keep_indices = load_keep_indices(Path(PRUNE_DIR))
    n_prog  = min(6, Bp.shape[-1])
    n_prune = min(6, len(keep_indices))
    cols = max(n_prog, n_prune)

    fig, axes = plt.subplots(2, cols, figsize=(5.5, 3.0))
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Top row: Progressive (blue)
    prog_indices = list(range(n_prog))
    plot_stream_basis_row(axes[0, :], Xp, Yp, Bp, color="blue", indices=prog_indices, row_is_bottom=False)

    # Bottom row: index pruned model with 0..k-1, but label with original keep indices
    prune_data_idx  = list(range(n_prune))                 # 0..k-1 for the pruned model
    prune_label_idx = keep_indices[:n_prune]               # original indices for display
    plot_stream_basis_row(axes[1, :], Xq, Yq, Bq,
                        color="red",
                        indices=prune_data_idx,
                        labels=prune_label_idx,
                        row_is_bottom=True)
    # Shared labels + legend
    fig.text(0.5, 0.02, X_LABEL, ha="center", va="bottom", fontsize=8)
    fig.text(0.02, 0.5, Y_LABEL, ha="center", va="center", rotation="vertical", fontsize=8)

    prog_line  = mlines.Line2D([], [], color="blue", label="Progressive (first 6)")
    prune_line = mlines.Line2D([], [], color="red",  label="Pruned (kept 6)")
    fig.legend(
        handles=[prog_line, prune_line],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=8,
        frameon=False
    )

    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.92)

    outdir = Path(SAVE_DIR); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / FILENAME
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()