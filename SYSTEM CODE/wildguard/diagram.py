"""System architecture diagram (report / slides).   python -m wildguard.diagram"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import config  # noqa: E402

FOREST, MOSS, DARK, INK, PALE, LANE = "#2C5F2D", "#97BC62", "#1A2E1B", "#445645", "#F3F8EF", "#E7F0E1"
ORANGE = "#B5651D"


def box(ax, x, y, w, h, title, sub, fill=PALE, edge=FOREST, tcolor=DARK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.7,
            fontweight="bold", color=tcolor)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.4, color=INK,
            linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=FOREST, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 color=color, lw=1.3, linestyle=ls))


def main():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for y0, label in ((3.0, "LEMIS  ·  labelled  ·  train & evaluate"),
                      (0.25, "CITES  ·  unlabelled  ·  deploy")):
        ax.add_patch(FancyBboxPatch((0.1, y0), 12.8, 2.25, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=LANE, ec="none"))
        ax.text(0.3, y0 + 2.05, label, fontsize=9, fontweight="bold", color=FOREST, va="center")

    w, h = 1.8, 1.35
    xs = [0.3 + i * (w + 0.33) for i in range(6)]
    top = [("LEMIS 2000-2014", "5.5M US import\nline items with\nenforcement outcome"),
           ("Schema alignment", "knowledge model:\nproduct, unit, source,\npurpose, country, appendix"),
           ("Relative features", "peer-group norms\nlearned on 2000-2010\n(quantity, route, rarity)"),
           ("Pattern models", "supervised XGBoost on\noutcomes + anomaly: peer z,\nIsolation Forest, K-Means"),
           ("Hybrid stacker", "meta-learner fitted\non 2011-2012"),
           ("Evaluation", "held-out 2013-2014:\nPR-AUC, recall @ budget,\nconfidence bands")]
    bottom = [("CITES 2015-2024", "8.25M animal trade\nrecords, no labels"),
              ("Schema alignment", "same crosswalk\n(shared fields only)"),
              ("Re-normalised features", "peer norms re-learned\nfrom CITES itself\n(no labels needed)"),
              ("Transfer + re-fitted", "LEMIS-trained XGBoost\n+ detectors re-fitted\non CITES"),
              ("Hybrid score + tiers", "LEMIS-fitted combiner;\npriority = top 1 % / year,\nreview = next 4 %"),
              ("Review queue (MVP)", "flag cards with\nplain-language reasons\n(TreeSHAP)")]
    for i, (x, (t, s)) in enumerate(zip(xs, top)):
        box(ax, x, 3.25, w, h, t, s)
        if i:
            arrow(ax, xs[i - 1] + w + 0.02, 3.25 + h / 2, x - 0.02, 3.25 + h / 2)
    for i, (x, (t, s)) in enumerate(zip(xs, bottom)):
        fill = FOREST if i == 5 else PALE
        box(ax, x, 0.45, w, h, t, s, fill=fill, tcolor="white" if i == 5 else DARK)
        if i == 5:
            for txt in ax.texts[-1:]:
                txt.set_color("#E7F0E1")
        if i:
            arrow(ax, xs[i - 1] + w + 0.02, 0.45 + h / 2, x - 0.02, 0.45 + h / 2)
    # what transfers from the LEMIS lane to the CITES lane
    for i, label in ((1, "same mapping"), (3, "trained model"), (4, "combiner weights")):
        x = xs[i] + w / 2
        arrow(ax, x, 3.23, x, 1.82, color=ORANGE, ls="--")
        ax.text(x + 0.08, 2.52, label, fontsize=7.6, color=ORANGE, fontweight="bold", va="center")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "system_architecture.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
