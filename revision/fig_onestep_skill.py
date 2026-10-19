"""One-step CRPS skill vs M0, as % less error than history alone.

Numbers: M0-M2 from paper/agriai/supplement.tex tab:onestep (runs/20260914_1708_pipeline_h100_fairfight);
GBM and GRU-M2 from runs/20261006_1759_baselines (revision/REVISION.md section 2).
Run from the repo root: .venv/bin/python revision/fig_onestep_skill.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "OpenSans-*.ttf")):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Open Sans", "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# (label, Box A 2019-25, Box B all years), CRPS skill vs M0; table order, top to bottom
ROWS = [
    ("M0, history only", 0.000, 0.000),
    ("M1, linear rain", 0.013, 0.052),
    ("M1.5, S-curve rain", 0.009, 0.052),
    ("Gradient-boosted trees", 0.046, 0.090),
    ("M2, rain removed", 0.016, 0.023),
    ("M2, learned (ours)", 0.085, 0.104),
]
SERIES = [("Unseen years (Box A, 2019–25)", "#2a78d6"), ("Unseen landscape (Box B)", "#eb6834")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e8e7e2", "#fcfcfb"


def chart(out, figsize=(6.4, 3.4), fs=8):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    y = np.arange(len(ROWS))[::-1].astype(float)
    h = 0.34  # bar thickness in row units; the rest of each row is air
    for j, (name, col) in enumerate(SERIES):
        vals = np.array([r[1 + j] for r in ROWS]) * 100
        yy = y + (h / 2 + 0.02) * (1 if j == 0 else -1)  # 2-ish px gap between the pair
        ax.barh(yy, vals, height=h, color=col, label=name, zorder=2)
        for v, yi in zip(vals, yy):
            ax.text(v + 0.15, yi, f"{v:.1f}%", va="center", ha="left", fontsize=fs - 1, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=fs, color=INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK2, lw=1.0, zorder=3)
    ax.set_xlim(0, 12)
    ax.set_xticks(range(0, 13, 2)); ax.set_xticklabels([f"{v}%" for v in range(0, 13, 2)], fontsize=fs, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_xlabel("Less one-step error (CRPS) than history alone", fontsize=fs, color=INK2)
    ax.set_title("% reduction in mean error, one step ahead (16 days)", fontsize=fs + 1, loc="left", color=INK, fontweight="bold")
    ax.legend(loc="upper right", fontsize=fs - 1, frameon=False)
    fig.savefig(out, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    chart("paper/figs/fig_onestep_skill.pdf")
    chart("paper/figs/fig_onestep_skill.png")
