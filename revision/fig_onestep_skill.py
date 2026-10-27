"""One-step CRPS skill vs M0, as % less error than history alone.

Numbers: M0-M2 from paper/agriai/supplement.tex tab:onestep (runs/20260914_1708_pipeline_h100_fairfight);
GBM and GRU-M2 from runs/20261006_1759_baselines (revision/REVISION.md section 2).
M2 (refit), M3 (+ topography), M4 (+ evaporative demand): 5-seed means of single fits, and the M4
5-seed ensemble (its 1-step rollout skill), runs/20261021_0147_m3/gpu.
Box B column is 2019-25 only (unseen landscape and unseen years): M1, M1.5, M2 rain removed re-scored from the
fair-fight fits (runs/20261021_0147_m3/onestep_B_test_paper_models.json). Trees, both columns and both horizons:
given M4's inputs (rain, ET0, topography), runs/20261027_1637_gbm_full/RESULT.md.
Chronos-2 row (zero-shot, given rain and ET0): runs/20261022_1639_suite/tables.md, scored on the suite's
4,000-pixel samples (M4 ensemble on the same pixels: 0.109 / 0.130).
3-month version (chronos2_3m), rows with both boxes saved: M1 Box A from the fair-fight metrics.json, Box B from
runs/20261022_1639_suite/tables.md (same M1 fit); M2-M4 and the ensemble from runs/20261021_0147_m3/RESULT.md;
Chronos-2 as above (M4 ensemble on its pixels: 0.268 / 0.274). M2 rain removed: runs/20261022_1639_suite/skill_m2nd.json
(revision/skill_m2nd.py). M1.5: no Box B 2019-25 3-month score.
Run from the repo root: .venv/bin/python revision/fig_onestep_skill.py [chronos2 | chronos2_3m]
  -> paper/figs/fig_onestep_skill.png, or paper/figs/chronos2/fig_{onestep,3month}_skill_c2.png
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

# (label, Box A 2019-25, Box B 2019-25), CRPS skill vs M0; table order, top to bottom
ROWS = [
    ("M0, history only", 0.000, 0.000),
    ("M1, linear rain", 0.013, 0.040),
    ("M1.5, S-curve rain", 0.009, 0.042),
    ("Gradient-boosted trees", 0.083, 0.047),
    ("M2, rain removed", 0.016, 0.012),
    ("M2, learned", 0.092, 0.075),
    ("M3 (inc. topography)", 0.099, 0.083),
    ("M4 (inc. evaporative demand)", 0.103, 0.112),
    ("M4 5-seed ensemble (ours)", 0.111, 0.131),
]
C2 = [("Chronos-2 (inc. rain & evap.)", 0.058, 0.071)]
ROWS_C2 = ROWS[:4] + C2 + ROWS[4:]  # the zero-shot model beside the other outside baselines
ROWS_3M = [
    ("M0, history only", 0.000, 0.000),
    ("M1, linear rain", 0.107, 0.099),
    ("Gradient-boosted trees", 0.246, 0.209),
    ("Chronos-2 (inc. rain & evap.)", 0.271, 0.284),
    ("M2, rain removed", -0.005, -0.048),
    ("M2, learned", 0.241, 0.177),
    ("M3 (inc. topography)", 0.254, 0.204),
    ("M4 (inc. evaporative demand)", 0.260, 0.250),
    ("M4 5-seed ensemble (ours)", 0.273, 0.274),
]
ONE = ("% reduction in mean error, one step ahead (16 days)", "Less one-step error (CRPS) than history alone", 16, 2)
THREE = ("% reduction in mean error, three months ahead (6 steps)", "Less 3-month error (CRPS) than history alone", 30, 5)
SERIES = [("Unseen years (Box A, 2019–25)", "#2a78d6"), ("Unseen landscape and years (Box B, 2019–25)", "#eb6834")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e8e7e2", "#fcfcfb"


def chart(out, rows=ROWS, x0=0, figsize=(6.4, 4.6), fs=8, text=ONE):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.34  # bar thickness in row units; the rest of each row is air
    for j, (name, col) in enumerate(SERIES):
        vals = np.array([r[1 + j] for r in rows]) * 100
        yy = y + (h / 2 + 0.02) * (1 if j == 0 else -1)  # 2-ish px gap between the pair
        ax.barh(yy, vals, height=h, color=col, label=name, zorder=2)
        for v, yi in zip(vals, yy):
            ax.text(v + (0.15 if v >= 0 else -0.15), yi, f"{v:.1f}%", va="center", ha="left" if v >= 0 else "right",
                    fontsize=fs - 1, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=fs, color=INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK2, lw=1.0, zorder=3)
    title, xlabel, x1, dx = text
    ax.set_xlim(x0, x1)
    ticks = range(x0, x1 + 1, dx)
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v}%" for v in ticks], fontsize=fs, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_xlabel(xlabel, fontsize=fs, color=INK2)
    ax.set_title(title, fontsize=fs + 1, loc="left", color=INK, fontweight="bold")
    ax.legend(loc="upper right", fontsize=fs - 1, frameon=False)
    fig.savefig(out, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    import sys
    os.makedirs("paper/figs/chronos2", exist_ok=True)
    if "chronos2_3m" in sys.argv:
        chart("paper/figs/chronos2/fig_3month_skill_c2.png", ROWS_3M, x0=-10, figsize=(6.4, 4.6), text=THREE)
    elif "chronos2" in sys.argv:
        chart("paper/figs/chronos2/fig_onestep_skill_c2.png", ROWS_C2, figsize=(6.4, 5.1))
    else:
        chart("paper/figs/fig_onestep_skill.png")
