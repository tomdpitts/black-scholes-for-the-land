"""Rain sensitivity maps, Box A 2019-25: observed vs M4 5-seed ensemble vs Chronos-2, same colour scale.

Per 8x8-pixel block (~2 km), NDVI per 100 mm of extra rain:
  observed    slope of a_t+2 - a_t on the rain departure in window t+1, controlling for a_t, over the block's gap-free
              test-4 starts (greekstest.outcomes; data only)
  M4 ensemble mean Delta over the same starts (suite.deltas, mean of the 5 fits; reproduces sens.json's sample mean)
  Chronos-2   mean Delta over the same starts on its 4,000 sample pixels (c2_sens.npz), averaged per block
Foot: box mean, and the typical error: mean absolute gap to the observed block map, as % of the observed map's mean
absolute value (0% = perfect; zero everywhere = 100%). For scale (Box A, NDVI per 100 mm): noise floor ~0.013, ~19%
(odd- vs even-year observed maps differ by 0.026, halved for the full data), flat map at the box mean 0.026, 37%. Pattern match r (M4 0.81, Chronos-2 0.63; ceiling ~0.86) is printed, not drawn.
The pre-registered test 4 (within each pixel, over time) is in sens.json; this figure shows where, not when.

  .venv/bin/python revision/fig_sens_c2.py   -> paper/figs/chronos2/fig_sens_c2.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aoi  # noqa: E402
import greekstest as G  # noqa: E402
from run import load  # noqa: E402
import suite as S  # noqa: E402
from fig_maps_split import INK, MUTED, SURF  # noqa: E402  (also sets the Open Sans fonts)
import matplotlib.pyplot as plt  # noqa: E402

RUN = "runs/20261022_1639_suite"
OUT = "paper/figs/chronos2/fig_sens_c2.png"
B, N = 8, 160
n = N // B


def observed(y, x, a0, ok):
    """Per-block slope of y on x controlling for a0, pooled over the block's pixels and starts. [n, n]"""
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            blk = lambda v: v[:, i * B:(i + 1) * B, j * B:(j + 1) * B].ravel()
            o = blk(ok)
            out[i, j] = np.linalg.lstsq(np.stack([np.ones(o.sum()), blk(x)[o], blk(a0)[o]], 1), blk(y)[o], rcond=None)[0][1]
    return out


def main(box="A"):
    P = np.load(f"{RUN}/plan.npz")
    st = P[f"{box}_sens_starts"].tolist()
    ng = P[f"{box}_sens_nogap"].astype(bool)
    y, x, a0, ok, yr = G.outcomes(load(box), st)
    y, x, a0, ok, yr = y[ng], x[ng], a0[ng], ok[ng], yr[ng]
    obs = observed(y, x, a0, ok)
    odd = yr % 2 == 1  # noise floor: odd- vs even-year maps; their gap holds 2x a half's noise variance, the full map half
    floor = np.abs(observed(y[odd], x[odd], a0[odd], ok[odd]) - observed(y[~odd], x[~odd], a0[~odd], ok[~odd])).mean() / 2
    m4 = np.mean([S.deltas("M4", box, st, s) for s in S.SEEDS], 0)[ng].mean(0).reshape(n, B, n, B).mean((1, 3))
    c2 = np.load(f"{RUN}/c2_sens.npz")[f"{box}_delta"][ng].mean(0)
    px = P["px"]
    tot, cnt = np.zeros((n, n)), np.zeros((n, n))
    np.add.at(tot, (px // N // B, px % N // B), c2)
    np.add.at(cnt, (px // N // B, px % N // B), 1)
    c2 = tot / cnt
    r = lambda m: float(np.corrcoef(m.ravel(), obs.ravel())[0, 1])
    err = lambda m: float(np.abs(m - obs).mean() / np.abs(obs).mean() * 100)
    floor = float(floor / np.abs(obs).mean() * 100)
    print(f"box means: observed {obs.mean():.3f}  M4 ensemble {m4.mean():.3f}  Chronos-2 {c2.mean():.3f};  "
          f"typical error: M4 {err(m4):.0f}%  Chronos-2 {err(c2):.0f}%  perfect model ~{floor:.0f}%;  r with observed: M4 {r(m4):.2f}  Chronos-2 {r(c2):.2f}")

    fig = plt.figure(figsize=(6.6, 3.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, .055], wspace=0.08, hspace=0.75)
    ext = (0, N, N, 0)
    panels = [(obs, "Observed", "what the land did, 2019–25", f"average {obs.mean():.2f}"),
              (m4, "M4 5-seed ensemble (ours)", "predicted", f"average {m4.mean():.2f};  typical error {err(m4):.0f}%"),
              (c2, "Chronos-2 (inc. rain & evap.)", "predicted", f"average {c2.mean():.2f};  typical error {err(c2):.0f}%")]
    for k, (m, title, sub, foot) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(m, cmap="YlGn", vmin=0, vmax=0.14, interpolation="nearest", extent=ext)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7"); sp.set_linewidth(0.6)
        ax.set_title(title, fontsize=8.5, fontweight="bold" if k == 1 else "normal", color=INK, pad=13)
        ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.text(0.5, -0.05, foot, transform=ax.transAxes, ha="center", va="top", fontsize=7, color=INK)
        if k == 0:
            km = 10 / (aoi.RES * 111.32)
            ax.plot([8, 8 + km], [150, 150], color=INK, lw=1.6, solid_capstyle="butt")
            ax.text(8 + km / 2, 146, "10 km", ha="center", va="bottom", fontsize=6.5, color=INK)
    cax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("Extra greenness (NDVI) per 100 mm of extra rain, 2 km blocks", fontsize=7, color=INK)
    cax.text(0.5, 3.0, f"Even the true sensitivities would show ~{floor:.0f}% typical error, from year-to-year noise", transform=cax.transAxes, ha="center", va="center", fontsize=7, color=MUTED)
    fig.suptitle("How much greener does each place get per 100 mm of extra rain?  Box A", x=0.125, ha="left", y=1.04,
                 fontsize=9.5, fontweight="bold", color=INK)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor=SURF)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
