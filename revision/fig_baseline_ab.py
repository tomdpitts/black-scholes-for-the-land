"""Explainer: the two ways to split the 2022 season-end anomaly (Box A, M4 5-seed ensemble, box means), as steps
from 0 (normal) down to observed.

  A  weather = observed - N_eps (normal weather, surprises kept), residual = observed - F, baseline = leftover
  B  baseline = N0 (forecast, normal weather), weather = F - N0, residual = observed - F
Run end values from revision/fig_maps_split_m4.py (old and new split) and the direct N0 check, 2026-10-26.

  .venv/bin/python revision/fig_baseline_ab.py   -> paper/figs/candidates/fig_baseline_ab.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_maps_split import INK, MUTED  # noqa: E402  (also sets the Open Sans fonts)

OBS, F, N0, NEPS = -0.168, -0.113, -0.007, -0.084
BASE, WEATHER, RESID, TOTAL = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
OUT = "paper/figs/candidates/fig_baseline_ab"


def num(v):
    return f"{v:+.2f}".replace("-", "−")


def panel(ax, title, steps, note):
    """steps: (name, size, colour, hatch), each starting where the last ended; then the observed total."""
    level = 0.0
    for i, (name, size, c, hatch) in enumerate(steps):
        ax.bar(i, size, bottom=level, width=0.6, color=c, hatch=hatch, edgecolor="white", linewidth=1.5)
        ax.text(i, level + size - 0.004, num(size), ha="center", va="top", color=INK, fontsize=11)
        ax.plot([i + 0.3, i + 0.7], [level + size] * 2, color=MUTED, lw=1)
        level += size
    n = len(steps)
    ax.bar(n, OBS, width=0.6, color=TOTAL)
    ax.text(n, OBS - 0.004, num(OBS), ha="center", va="top", color=INK, fontsize=11)
    ax.set_xticks(range(n + 1), [s[0] for s in steps] + ["= observed"], fontsize=11)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left")
    ax.text(-0.4, -0.22, note, color=INK, fontsize=10, va="top")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    leftover = OBS - (OBS - NEPS) - (OBS - F)
    panel(a, "A  (paper text, old figures)",
          [("baseline", leftover, BASE, "//"), ("+ weather", OBS - NEPS, WEATHER, None), ("+ residual", OBS - F, RESID, None)],
          f"baseline = whatever is left: {num(N0)} real + {num(leftover - N0)} mismatch\n"
          "weather = observed − replay of the season in a normal year\n"
          "residual = observed − forecast for the actual weather")
    panel(b, "B  (new figures)",
          [("baseline", N0, BASE, None), ("+ weather", F - N0, WEATHER, None), ("+ residual", OBS - F, RESID, None)],
          "baseline = forecast for a normal year\n"
          "weather = forecast for the actual weather − baseline\n"
          "residual = observed − forecast for the actual weather")
    a.set_ylabel("NDVI anomaly, end of May 2022\n(0 = normal)", color=INK, fontsize=11)
    a.set_ylim(-0.19, 0.01)
    fig.suptitle("Box A, M4 5-seed ensemble. The residual is the same in both.", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=150)
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    main()
