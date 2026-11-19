"""Candidate figures: what Delta is good for.

Figure 1 (fig_greeks_rank_state):
  (a) Does Delta rank real exposure? Held-out 2019-25 rainy-season steps split into thirds by each model's
      Delta at the starting state; real response = slope of the two-step NDVI change on the window's rain
      anomaly, controlling for the starting anomaly (greekstest.py G2, pre-registered run
      runs/20260915_1228_greekstest). 95% bootstrap over the 7 test years, per third.
  (b) Exposure moves with the state: M2's Delta (h = 2, along normal rain) from two long-rains starts on
      Box A, after a wet short-rains season (2020) and after a failed one (2022).
Figure 2 (fig_greeks_scenario):
  Before the 2025 long rains (state of 18 Feb 2025), the end-of-May greenness change if the season brings
  100 mm less rain than normal, spread over its six windows in proportion to normal rain: a directional
  Delta (Jacobian-vector product), on 2 km blocks, and the land split into exposure thirds.

  .venv/bin/python fig_greeks_value.py  -> paper/figs/candidates/fig_greeks_{rank_state,scenario}.png
"""
import glob
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import model as M
from greekstest import starts, deltas, outcomes, TEST
from residual import season_starts
from run import load, H

RUN = "runs/20260914_1708_pipeline_h100_fairfight"
NBOOT, K = 2000, 8
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e8e7e2", "#ffffff"
for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "OpenSans-*.ttf")):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Open Sans", "pdf.fonttype": 42, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def thirds(G, y, x, a0, ok, yr, rng):
    """Per-third slope of y on x given a0 (greekstest.sort_test), with a per-third 95% year bootstrap."""
    q = np.quantile(G[ok], [1 / 3, 2 / 3])
    grp = np.digitize(G, q)
    yl = sorted(set(yr.tolist()))
    XtX = np.zeros((len(yl), 3, 3, 3)); Xty = np.zeros((len(yl), 3, 3))
    dsum, dn = np.zeros(3), np.zeros(3)
    for s in range(len(yr)):
        yi = yl.index(int(yr[s]))
        for g in range(3):
            mm = ok[s] & (grp[s] == g)
            X = np.stack([np.ones(mm.sum()), x[s][mm], a0[s][mm]], 1)
            XtX[yi, g] += X.T @ X; Xty[yi, g] += X.T @ y[s][mm]
            dsum[g] += G[s][mm].sum(); dn[g] += mm.sum()

    def slopes(wt):
        A = np.einsum("y,ygij->gij", wt, XtX); b = np.einsum("y,ygi->gi", wt, Xty)
        return np.array([np.linalg.solve(A[g], b[g])[1] for g in range(3)])

    bs = np.array([slopes(np.bincount(rng.integers(0, len(yl), len(yl)), minlength=len(yl)).astype(float))
                   for _ in range(NBOOT)])
    pooled = np.linalg.solve(XtX.sum((0, 1)), Xty.sum((0, 1)))[1]
    return slopes(np.ones(len(yl))), np.percentile(bs, [2.5, 97.5], 0), dsum / dn, pooled


def blocks(m):
    n = m.shape[0] // K
    return m[:n * K, :n * K].reshape(n, K, n, K).mean((1, 3))


def main():
    P = {k: {kk: jnp.asarray(v) for kk, v in d.items()}
         for k, d in np.load(f"{RUN}/params.npy", allow_pickle=True).item().items() if k in ("M2", "M15")}

    # ------------------------------------------------------------------ (a) ranking, both boxes
    rank = {}
    for box in ("A", "B"):
        d = load(box)
        D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
        st = starts(d, TEST)
        y, x, a0, ok, yr = outcomes(d, st)
        for nm in ("M2", "M15"):
            rank[box, nm] = thirds(deltas(P[nm], D, st), y, x, a0, ok, yr, np.random.default_rng(0))
            pt, ci, dm, pooled = rank[box, nm]
            print(f"{box} {nm}: real {np.round(pt, 3)}  ci lo {np.round(ci[0], 3)} hi {np.round(ci[1], 3)}  "
                  f"model {np.round(dm, 3)}  pooled {pooled:.3f}")

    # ------------------------------------------------------------------ (b) Delta maps, wet vs dry start, Box A
    A = load("A")
    D = {k: jnp.asarray(A[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    p = P["M2"]
    w = M.w_path(p, D)
    t_of = {f"{s}{y}": t for (y, s, t) in season_starts(A, range(2019, 2026))}

    def delta_map(t):
        g = M.greeks(p, D["a"][t], w[t], D["rclim"][t + 1:t + 1 + H], D["phase"][t + 1:t + 1 + H], D["x"])
        return np.asarray(g["delta"][1])

    dm = {k: delta_map(t_of[k]) for k in ("MAM2020", "MAM2022")}
    r_wd = np.corrcoef(dm["MAM2020"].ravel(), dm["MAM2022"].ravel())[0, 1]
    print(f"corr Delta map MAM2020 vs MAM2022: {r_wd:.2f}")

    # one row per model x box: real response of the least- and most-exposed third, by that model's own Delta
    fig = plt.figure(figsize=(7.2, 2.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.1, .25, 1.05, 1.05], wspace=0.1)
    ax = fig.add_subplot(gs[0])
    rows = [("M2", "A", "M2 · unseen years"), ("M2", "B", "M2 · unseen landscape"),
            ("M15", "A", "S-curve · unseen years"), ("M15", "B", "S-curve · unseen landscape")]
    for i, (nm, box, lab) in enumerate(rows):
        yv = len(rows) - 1 - i - (0.35 if nm == "M15" else 0)
        lo, hi = rank[box, nm][0][[0, 2]]
        col = BLUE if nm == "M2" else ORANGE
        ax.annotate("", xy=(hi, yv), xytext=(lo, yv),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, shrinkA=4, shrinkB=4, mutation_scale=9))
        ax.plot(lo, yv, "o", ms=6, mfc="white", mec=col, mew=1.5, zorder=3)
        ax.plot(hi, yv, "o", ms=6, color=col, mec="white", mew=0.8, zorder=3)
        ax.text(0.215, yv, f"{hi / lo:.1f}×", va="center", ha="right", fontsize=8,
                fontweight="bold" if nm == "M2" else "normal", color=INK)
        ax.text(-0.004, yv, lab, va="center", ha="right", fontsize=7.5, color=INK)
    ax.set_xlim(0, 0.22); ax.set_ylim(-0.75, 3.6)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.05, 0.1, 0.15]); ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Real greenness gain per extra 100 mm of rain (NDVI, a month on)", fontsize=7)
    ax.plot([], [], "o", ms=6, mfc="white", mec=MUTED, mew=1.5, label="least exposed third")
    ax.plot([], [], "o", ms=6, color=MUTED, label="most exposed third")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.06), fontsize=6.5, frameon=False, ncol=2,
              handletextpad=0.2, columnspacing=1.0)
    fig.text(0.0, 0.98, "(a) Land the model calls most exposed responds most", fontsize=8.5, fontweight="bold",
             color=INK, va="bottom")

    for i, (k, ttl) in enumerate((("MAM2020", "after a wet season\nFeb 2020"), ("MAM2022", "after a failed season\nFeb 2022"))):
        mx = fig.add_subplot(gs[2 + i])
        im = mx.imshow(dm[k], cmap="YlGn", vmin=0, vmax=0.14, interpolation="nearest")
        mx.set_xticks([]); mx.set_yticks([])
        for s in mx.spines.values():
            s.set_visible(True); s.set_color("#c3c2b7"); s.set_linewidth(0.6)
        mx.set_title(ttl, fontsize=7.5, color=INK)
        if i == 0:
            px = 10 / (0.0025 * 111.32)
            mx.plot([8, 8 + px], [150, 150], color=INK, lw=1.6, solid_capstyle="butt")
            mx.text(8 + px / 2, 146, "10 km", ha="center", va="bottom", fontsize=6.5, color=INK)
    fig.text(0.635, 0.98, "(b) Exposure moves with the land's state", fontsize=8.5, fontweight="bold", color=INK, va="bottom")
    cax = fig.add_axes([0.68, 0.06, 0.24, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.05, 0.1])
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("Box A, Δ: NDVI gain per extra 100 mm, a month on", fontsize=7, color=INK)
    fig.savefig("paper/figs/candidates/fig_greeks_rank_state.png", dpi=220, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)

    # ------------------------------------------------------------------ scenario: 2025 long rains 100 mm short
    t0 = t_of["MAM2025"]
    sl = slice(t0 + 1, t0 + 1 + H)
    rc, ph = D["rclim"][sl], D["phase"][sl]
    u = rc / rc.sum(0, keepdims=True)                               # +100 mm over the season, shaped like normal rain
    f = lambda r: M.rollout_mean(p, D["a"][t0], w[t0], r, ph, D["x"], True)[-1]
    base, delta = jax.jvp(f, (rc,), (u,))                           # seasonal Delta, NDVI per 100 mm
    _, gamma = jax.jvp(lambda r: jax.jvp(f, (r,), (u,))[1], (rc,), (u,))
    _, theta = jax.jvp(lambda a: M.rollout_mean(p, a, w[t0], rc, ph, D["x"], True)[-1],
                       (D["a"][t0],), (jnp.ones_like(D["a"][t0]),))
    exact = np.asarray(f(jnp.clip(rc - u, 0, None))) - np.asarray(base)
    delta, gamma, theta = (np.asarray(v) for v in (delta, gamma, theta))
    print(f"2025 scenario: box mean -Delta {-delta.mean():+.4f}, exact {exact.mean():+.4f}; "
          f"Gamma {gamma.mean():+.4f}, Theta {theta.mean():.3f}")
    G = {k: blocks(v) for k, v in (("d", delta), ("g", gamma), ("t", theta))}
    L = -G["d"]
    pick = {}                                                       # example blocks: 90th, 50th, 10th pct of exposure
    for lab, qq in (("1", 0.9), ("2", 0.5), ("3", 0.1)):
        inner = np.full_like(L, np.inf); inner[1:-1, 1:-1] = np.abs(G["d"][1:-1, 1:-1] - np.quantile(G["d"], qq))
        pick[lab] = np.unravel_index(np.argmin(inner), L.shape)

    browns = matplotlib.colors.LinearSegmentedColormap.from_list("browns", plt.get_cmap("BrBG")(np.linspace(0, 0.5, 128)))
    fig = plt.figure(figsize=(5.9, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(L, cmap=browns, vmin=-0.1, vmax=0, interpolation="nearest", extent=(0, 160, 160, 0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#c3c2b7"); s.set_linewidth(0.6)
    px = 10 / (0.0025 * 111.32)
    ax.plot([8, 8 + px], [150, 150], color=INK, lw=1.6, solid_capstyle="butt")
    ax.text(8 + px / 2, 146, "10 km", ha="center", va="bottom", fontsize=6.5, color=INK)
    for lab, (i, j) in pick.items():
        ax.text((j + 0.5) * K, (i + 0.5) * K, lab, ha="center", va="center", fontsize=6.5, fontweight="bold",
                color=INK, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=INK, lw=0.6))
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, ticks=[-0.1, -0.075, -0.05, -0.025, 0])
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("End-of-May NDVI change (−Δ × 100 mm), 2 km blocks", fontsize=7, color=INK)
    ax.set_title("Box A, example long-rains season start", fontsize=7.5, color=INK, pad=11)
    ax.text(0.5, 1.01, "(state on 18 Feb 2025, latest in our data)", transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color=MUTED)

    tx = fig.add_subplot(gs[1]); tx.axis("off")
    cols = [("Δ", "NDVI per\n100 mm"), ("Γ", "Δ change\nper 100 mm"), ("Θ", "share of Feb\nanomaly left")]
    xc, y0, dy = [0.55, 0.74, 0.93], 0.80, 0.135
    for x, (g, sub) in zip(xc, cols):
        tx.text(x, y0 + 0.09, g, ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        tx.text(x, y0 + 0.08, sub, ha="center", va="top", fontsize=5.5, color=MUTED, linespacing=1.1)
    fmt = lambda v, d=3: f"{v:+.{d}f}".replace("-", "−")
    rows = [("Whole box", delta.mean(), gamma.mean(), theta.mean(), True)]
    for lab, name in (("1", "most exposed"), ("2", "typical"), ("3", "least exposed")):
        i, j = pick[lab]
        rows.append((f"{lab}  {name}", G["d"][i, j], G["g"][i, j], G["t"][i, j], False))
    for r, (name, d_, g_, t_, bold) in enumerate(rows):
        yv = y0 - 0.12 - r * dy - (0.06 if r else 0)
        tx.text(0.02, yv, name, ha="left", va="center", fontsize=7.5, color=INK, fontweight="bold" if bold else "normal")
        for x, v in zip(xc, (fmt(d_), fmt(g_), f"{t_:.2f}")):
            tx.text(x, yv, v, ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold" if bold else "normal")
        if r == 0:                                                  # Delta as a share of normal end-of-May NDVI
            pct = 100 * d_ / A["clim"][A["idx"][t0 + H]].mean()
            tx.text(xc[0], yv - 0.065, f"({pct:.0f}%)", ha="center", va="center", fontsize=6.5, color=MUTED)
    tx.plot([0.02, 0.98], [y0 - 0.24, y0 - 0.24], color="#c3c2b7", lw=0.6, transform=tx.transAxes)
    tx.text(0.02, 0.08, "Higher Δ = more greenness lost when rain falls short.", fontsize=6.5, color=MUTED, va="top")
    tx.set_xlim(0, 1); tx.set_ylim(0, 1)
    normal_mm = float(rc.mean((1, 2)).sum() * 100)                   # box-mean normal long-rains total
    ttl = fig.text(0.06, 1.03, "Sensitivities to 100mm less rain than normal", ha="left", va="bottom", fontsize=9,
                   fontweight="bold", color=INK)
    bb = ttl.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.01, 1.03, f"(normal = {normal_mm:.0f}mm)", ha="left", va="bottom", fontsize=9, color=MUTED)
    fig.savefig("paper/figs/candidates/fig_greeks_scenario.png", dpi=220, bbox_inches="tight", facecolor=SURF)
    print("wrote paper/figs/candidates/fig_greeks_{rank_state,scenario}.png")


if __name__ == "__main__":
    main()
