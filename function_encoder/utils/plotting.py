# plotting_utils.py
from pathlib import Path
import matplotlib.pyplot as plt

def save_figure(fig, filepath, width=5.5, height=None, font_size=8, dpi=300):
    if height is None:
        height = width * 0.65
    fig.set_size_inches(width, height)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
    })

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")