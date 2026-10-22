"""Box A, March-May 2022 with M4 (5-seed ensemble): the season-end anomaly split into maps that add up,
next to the terrain that shapes them.

Per pixel at season end, NDVI units, averaged over the fits:
  rain, evaporative demand   Shapley over the two forcings, residual held fixed (m3_attribution.py)
  residual                   observed - expected (the weather that fell, residual zero)
  baseline                   observed - rain - demand - residual (the late-Feb state run on under normal weather)
Averaged to 8x8-pixel blocks (~2.2 km), as fig_maps_split.py. Residual blocks inside their own 5-95% range
over the 2003-17 long-rains seasons are faded. Elevation (Copernicus DEM GLO-30) is an input, not a share.

  .venv/bin/python revision/fig_maps_split_m4.py   -> paper/figs/candidates/fig_maps_split_2022_m4.pdf/.png
"""
import os
import sys

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aoi  # noqa: E402
from run import H  # noqa: E402
from residual import season_starts  # noqa: E402
from fig_maps_split import blocks, INK, MUTED, SURF  # noqa: E402  (also sets the Open Sans fonts)
import m3  # noqa: E402
from m3_attribution import rerun, implied  # noqa: E402

R, CONFIG, SEEDS = "runs/20261021_0147_m3/gpu", "topo_pet", range(5)
OUT = "paper/figs/candidates/fig_maps_split_2022_m4"


def main():
    d = m3.load_box("A", True, "pet")
    D = m3.jaxify(d)
    s = D["s"]
    t0 = [t for (y, sn, t) in season_starts(d, [2022]) if sn == "MAM"][0]
    train = [t for (y, sn, t) in season_starts(d, aoi.TRAIN_YEARS) if sn == "MAM"]

    def season(p, w, t):
        """Season-end maps: observed, rain, demand, residual."""
        sl = slice(t + 1, t + 1 + H)
        a0, y = D["a"][t], D["a"][sl]
        r, rc, ex, ph = D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]
        eps, _ = implied(p, s, a0, w[t], y, r, ex, ph)
        end = lambda rr, ee, e: np.asarray(rerun(p, s, a0, w[t], rr, ee, ph, e)[-1])
        ex0 = jnp.zeros_like(ex)
        obs, cf_r, cf_t, cf_rt = np.asarray(y[-1]), end(rc, ex, eps), end(r, ex0, eps), end(rc, ex0, eps)
        rain = 0.5 * ((obs - cf_r) + (cf_t - cf_rt))
        demand = 0.5 * ((obs - cf_t) + (cf_r - cf_rt))
        resid = obs - end(r, ex, jnp.zeros_like(eps))
        return obs, rain, demand, resid

    ok = d["obs"][t0 + H].astype(bool)
    per_fit, ordinary = [], []
    for k in SEEDS:
        p = {kk: jnp.asarray(v) for kk, v in np.load(f"{R}/params_{CONFIG}_s{k}.npy", allow_pickle=True).item().items()}
        w = m3.w_path(p, D)
        per_fit.append(season(p, w, t0))
        ordinary.append([blocks(season(p, w, t)[3], d["obs"][t + H].astype(bool)) for t in train])
    obs, rain, demand, resid = (np.mean(z, 0) for z in zip(*per_fit))
    carried = obs - rain - demand - resid
    maps = [blocks(m, ok) for m in (obs, carried, rain, demand, resid)]
    Rr = np.mean(ordinary, 0)                                  # [season, block, block], fit-averaged
    lo, hi = np.nanpercentile(Rr, 5, 0), np.nanpercentile(Rr, 95, 0)
    unusual = (maps[4] < lo) | (maps[4] > hi)
    elev = blocks(np.load("data/A/topo.npz")["elev"])
    means = [float(np.nanmean(m)) for m in maps]
    sl = slice(t0 + 1, t0 + 1 + H)
    rain_mm = float((D["r"][sl] - D["rclim"][sl]).mean((1, 2)).sum() * 100)
    pet_mm = float(D["ex"][sl].mean((1, 2, 3)).sum() * 100)
    corr = lambda m: float(np.corrcoef(elev.ravel(), m.ravel())[0, 1])
    print("box means obs/baseline/rain/demand/residual:", " ".join(f"{v:+.3f}" for v in means))
    print(f"corr with elevation (2 km blocks): rain {corr(maps[2]):+.2f}  demand {corr(maps[3]):+.2f}  "
          f"residual {corr(maps[4]):+.2f};  unusual residual blocks {unusual.mean():.0%}")

    fig = plt.figure(figsize=(8.6, 2.6))
    gs = fig.add_gridspec(2, 11, width_ratios=[1, .3, 1, .16, 1, .16, 1, .16, 1, .16, 1], height_ratios=[1, .06],
                          wspace=0.02, hspace=0.32)  # terrain | gap | observed = baseline + rain + demand + residual
    ext = (0, 160, 160, 0)

    def frame(ax, title, sub, foot):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7"); sp.set_linewidth(0.6)
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=INK, pad=13)
        ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.text(0.5, -0.05, foot, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)

    ax = fig.add_subplot(gs[0, 0])
    ime = ax.imshow(elev * 1000, cmap="Greys", interpolation="nearest", extent=ext)
    frame(ax, "Terrain", "input, not a share", f"{elev.min() * 1000:.0f}–{elev.max() * 1000:.0f} m")
    px = 10 / (aoi.RES * 111.32)
    ax.plot([8, 8 + px], [150, 150], color="white", lw=1.6, solid_capstyle="butt")
    ax.text(8 + px / 2, 146, "10 km", ha="center", va="bottom", fontsize=6.5, color="white")
    cax = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(ime, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.5, length=2); cb.set_label("elevation, m", fontsize=6.5); cb.outline.set_linewidth(0.4)

    titles = ["Observed", "Baseline", "Rain", "Evap. demand", "Residual"]
    subs = ["end of May 2022", "under normal weather", f"{abs(rain_mm):.0f} mm below normal", f"{pet_mm:.0f} mm above normal", ""]
    kw = dict(cmap="BrBG", vmin=-0.3, vmax=0.3, interpolation="nearest", extent=ext)
    n = maps[0].shape[0]
    for i, (m, t, sub, mean) in enumerate(zip(maps, titles, subs, means)):
        ax = fig.add_subplot(gs[0, 2 + 2 * i])
        im = ax.imshow(m, **kw)
        if i == 4:
            veil = np.zeros((n, n, 4)); veil[..., :3] = 1; veil[..., 3] = np.where(unusual, 0, 0.72)
            ax.imshow(veil, interpolation="nearest", extent=ext)
        frame(ax, t, sub, f"box mean {mean:+.2f}".replace("-", "−"))
        if i < 4:
            op = fig.add_subplot(gs[0, 3 + 2 * i]); op.axis("off")
            op.text(0.5, 0.5, "=" if i == 0 else "+", ha="center", va="center", fontsize=15, color=MUTED)
    cax = fig.add_subplot(gs[1, 3:10])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("NDVI anomaly at season end, 2 km blocks; M4, mean of 5 fits", fontsize=7, color=INK)
    cax.text(-0.02, 0.5, "browner", transform=cax.transAxes, ha="right", va="center", fontsize=7, color=INK)
    cax.text(1.02, 0.5, "greener", transform=cax.transAxes, ha="left", va="center", fontsize=7, color=INK)
    cb.outline.set_linewidth(0.4)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", facecolor=SURF)
    fig.savefig(f"{OUT}.png", dpi=220, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
