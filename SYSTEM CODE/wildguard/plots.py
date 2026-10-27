"""Shared figure style for report / slide charts (static PNGs).

Palette: the first three categorical slots of a colour-vision-deficiency-validated
reference palette (blue, orange, aqua) in fixed order; neutral grey for baselines;
status colours (good / warning / critical) only where the colour means a triage state,
always paired with a text label.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BASELINE = "#a8a7a0"
STATUS = {"auto-clear": "#0ca30c", "review": "#fab219", "priority": "#d03b3b"}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "axes.titleweight": "semibold", "axes.titlesize": 12,
        "axes.titlelocation": "left", "axes.titlecolor": INK,
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
