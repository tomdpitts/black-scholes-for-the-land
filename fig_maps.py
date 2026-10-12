"""Candidate figure: Box A, March-May 2022. State against history vs surprise given the rain.

(a) z-score of the anomaly, averaged over the season's six composites (a persistent level).
(b) the residual eps-hat summed over the same six steps and divided by sqrt(6): the season-standardised
    surprise. Six near-independent per-step residuals average to a spread ~1/sqrt(6), so the sum / sqrt(6)
    puts (b) on the same footing as (a): -1 means an unusual season in both panels.

  .venv/bin/python fig_maps.py  -> paper/figs/candidates/fig_maps_2022.pdf/.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import load, H

CURVES = "runs/20260916_0928_residual/curves.npz"
T0 = 460                                              # MAM2022 season start in that run


def main():
    A = load("A")
    ts = range(T0 + 1, T0 + 1 + H)
    z = np.stack([A["a"][t] / np.maximum(A["cstd"][A["idx"][t]], 1e-3) for t in ts])
    ok = np.stack([A["obs"][t] for t in ts]).astype(bool)
    zmap = np.nanmean(np.where(ok, z, np.nan), 0)
    emap = np.load(CURVES)["MAM2022_eps_map"] * np.sqrt(H)   # season mean x sqrt(6) = sum / sqrt(6)
    stats = {}
    for name, m in (("z", zmap), ("eps", emap)):
        stats[name] = (np.nanmean(m), np.nanmean(m < -1), np.nanmean(m < -2))
        print(f"{name}: mean {stats[name][0]:+.2f}  below -1: {stats[name][1]:.0%}  below -2: {stats[name][2]:.0%}")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    kw = dict(cmap="RdBu", vmin=-3, vmax=3, interpolation="nearest")
    panels = ((ax[0], zmap, "z", "(a) state: z-score against the pixel's own history"),
              (ax[1], emap, "eps", "(b) surprise: residual given the rain that fell"))
    for a, m, k, t in panels:
        im = a.imshow(m, **kw)
        a.set_title(t, fontsize=8)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        mean, b1, b2 = stats[k]
        a.text(0.01, -0.04, f"mean {mean:+.2f} · {b1:.0%} of pixels below −1 · {b2:.0%} below −2",
               transform=a.transAxes, fontsize=7, va="top", color="#52514e")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("standard deviations, season", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Box A, Mar–May 2022 (long-rains season, rain 124 mm below normal). "
                 "(a) six-composite mean; (b) six-step sum ÷ √6", fontsize=8, x=0.01, ha="left")
    fig.savefig("paper/figs/candidates/fig_maps_2022.pdf")
    fig.savefig("paper/figs/candidates/fig_maps_2022.png", dpi=200)
    print("wrote paper/figs/candidates/fig_maps_2022.pdf/.png")


if __name__ == "__main__":
    main()
