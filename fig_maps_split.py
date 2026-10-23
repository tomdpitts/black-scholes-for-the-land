"""Candidate figure: Box A, March-May 2022, the season-end anomaly split into three maps that add up.

Per pixel at the end of the season (h = 6), in NDVI units:
  rain's share     = observed - counterfactual (normal rain, same residual)        as in fig_attribution.py
  residual         = observed - expected (the rain that fell, residual zero)
  baseline         = observed - rain's share - residual   (the late-Feb state run on under normal rain)
Averaged to 8x8-pixel blocks (~2.2 km): attribution is a project-scale product (REVISION.md section 1).
Blocks whose residual sits inside their own range over the 2003-17 long-rains seasons are faded:
that much happens in an ordinary year.

  .venv/bin/python fig_maps_split.py [run dir]  -> paper/figs/candidates/fig_maps_split_2022.png
"""
import glob
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import aoi
import model as M
from run import load, H
from residual import season_starts

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/20260914_1708_pipeline_h100_fairfight"
K = 8                                                      # block side in pixels
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#ffffff"
for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "OpenSans-*.ttf")):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Open Sans", "pdf.fonttype": 42, "font.size": 8})


def blocks(m, ok=None):
    """Mean over KxK blocks, ignoring unobserved pixels."""
    m = np.where(ok, m, np.nan) if ok is not None else m
    n = m.shape[0] // K
    return np.nanmean(m[:n * K, :n * K].reshape(n, K, n, K), axis=(1, 3))


def main():
    A = load("A")
    params = np.load(f"{RUN}/params.npy", allow_pickle=True).item()
    p = {k: jnp.asarray(v) for k, v in params["M2"].items()}
    D = {k: jnp.asarray(A[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    w, x = M.w_path(p, D), D["x"]

    @jax.jit
    def implied(t):
        mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], x, True)
        return (D["a"][t + 1] - mu) / sig

    @jax.jit
    def roll_end(a0, w0, r_path, ph_path, eps_path):
        """Season-end map of the rollout along r_path with eps_path re-injected."""
        def body(c, inp):
            a, wv = c
            r, ph, e = inp
            mu, sig, wv = M.step(p, a, wv, r, ph, x, True)
            return (mu + sig * e, wv), None
        (a, _), _ = jax.lax.scan(body, (a0, w0), (r_path, ph_path, eps_path))
        return a

    def season(t0):
        sl = slice(t0 + 1, t0 + 1 + H)
        eps = jnp.stack([implied(t) for t in range(t0, t0 + H)])
        obs = np.asarray(D["a"][t0 + H])
        expected = np.asarray(roll_end(D["a"][t0], w[t0], D["r"][sl], D["phase"][sl], jnp.zeros_like(eps)))
        cf = np.asarray(roll_end(D["a"][t0], w[t0], D["rclim"][sl], D["phase"][sl], eps))
        rain_mm = float((D["r"][sl] - D["rclim"][sl]).mean((1, 2)).sum() * 100)
        return obs, expected, cf, A["obs"][t0 + H].astype(bool), rain_mm

    t0 = [t for (y, s, t) in season_starts(A, [2022]) if s == "MAM"][0]
    obs, expected, cf, ok, rain_mm = season(t0)
    rain, resid = obs - cf, obs - expected
    carried = obs - rain - resid
    maps = [blocks(m, ok) for m in (obs, carried, rain, resid)]

    # ordinary-year range of the season-end "not explained", per block, over the training long rains
    train = [season(t) for (y, s, t) in season_starts(A, aoi.TRAIN_YEARS) if s == "MAM"]
    R = np.stack([blocks(o - e, k) for (o, e, _, k, _) in train])
    lo, hi = np.nanpercentile(R, 5, 0), np.nanpercentile(R, 95, 0)
    unusual = (maps[3] < lo) | (maps[3] > hi)
    means = [float(np.nanmean(m)) for m in maps]
    print(f"MAM2022 t0={t0} rain {rain_mm:+.0f} mm  box means obs/carried/rain/resid:",
          " ".join(f"{v:+.3f}" for v in means))
    print(f"{len(train)} training seasons; blocks outside own 5-95% range: {unusual.mean():.0%} "
          f"(below {np.mean(maps[3] < lo):.0%}, above {np.mean(maps[3] > hi):.0%}); "
          f"training box-mean residuals {np.round(np.nanmean(R, (1, 2)), 3)}")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, .16, 1, .16, 1, .16, 1], height_ratios=[1, .06],
                          wspace=0.02, hspace=0.32)
    titles = ["Observed", "Baseline", "Rain's share", "Residual"]
    subs = ["end of May 2022", "under normal rain", "rain 124 mm below normal", ""]
    kw = dict(cmap="BrBG", vmin=-0.3, vmax=0.3, interpolation="nearest", extent=(0, 160, 160, 0))
    n = maps[0].shape[0]
    for i, (m, t, sub, mean) in enumerate(zip(maps, titles, subs, means)):
        ax = fig.add_subplot(gs[0, 2 * i])
        im = ax.imshow(m, **kw)
        if i == 3:                                          # fade the ordinary blocks
            veil = np.zeros((n, n, 4)); veil[..., :3] = 1; veil[..., 3] = np.where(unusual, 0, 0.72)
            ax.imshow(veil, interpolation="nearest", extent=(0, 160, 160, 0))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c3c2b7"); s.set_linewidth(0.6)
        ax.set_title(t, fontsize=8.5, fontweight="bold", color=INK, pad=13)
        ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.text(0.5, -0.05, f"box mean {mean:+.2f}".replace("-", "−"), transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=INK)
        if i < 3:
            op = fig.add_subplot(gs[0, 2 * i + 1]); op.axis("off")
            op.text(0.5, 0.5, "=" if i == 0 else "+", ha="center", va="center", fontsize=15, color=MUTED)
        if i == 0:                                          # 10 km scale bar: 0.0025 deg ~ 278 m per pixel
            px = 10 / (aoi.RES * 111.32)
            ax.plot([8, 8 + px], [150, 150], color="white", lw=1.6, solid_capstyle="butt")
            ax.text(8 + px / 2, 146, "10 km", ha="center", va="bottom", fontsize=6.5, color="white")
    ax.text(1.0, 1.29, "Box A, long-rains season, late Feb – end of May 2022", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=4.75, color=MUTED)    # season span, top right
    cax = fig.add_subplot(gs[1, 2:5])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("NDVI anomaly at season end, 2 km blocks", fontsize=7, color=INK)
    cax.text(-0.02, 0.5, "browner", transform=cax.transAxes, ha="right", va="center", fontsize=7, color=INK)
    cax.text(1.02, 0.5, "greener", transform=cax.transAxes, ha="left", va="center", fontsize=7, color=INK)
    cb.outline.set_linewidth(0.4)
    fig.savefig("paper/figs/candidates/fig_maps_split_2022.png", dpi=220, bbox_inches="tight", facecolor=SURF)
    print("wrote paper/figs/candidates/fig_maps_split_2022.png")


if __name__ == "__main__":
    main()
