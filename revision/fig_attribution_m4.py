"""The 2022 drought with M4 (rain + topography + evaporative demand), 5-seed ensemble.

As fig_attribution.py, with M4 in place of M2 and a fourth share. Lines: history only (M0), expected from
the weather that fell (ensemble mean of the fits' rollouts, residual zero), observed; box means. Band: 90%
range of the same rollout with whole residual maps re-injected, each draw taking a random fit and, per step,
that fit's residual map from the same step of a random March-May season in 2003-2017. Bar: the season-end
anomaly split into starting deficit, rain, evaporative demand (Shapley over the two forcings, residual held
fixed) and residual, averaged over the fits.

  .venv/bin/python revision/fig_attribution_m4.py   -> paper/figs/candidates/fig_2022_attribution_m4.pdf/.png
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
import model as M  # noqa: E402
from run import load, H  # noqa: E402
from residual import season_starts  # noqa: E402
import m3  # noqa: E402
from m3_attribution import rerun, implied  # noqa: E402

R, CONFIG, SEEDS = "runs/20261021_0147_m3/gpu", "topo_pet", range(5)
REF = "runs/20260914_1708_pipeline_h100_fairfight"
BLUE, ORANGE, YELLOW, GREY, INK, MUTED, LIGHT = "#2a78d6", "#eb6834", "#eda100", "#9a9891", "#0b0b0b", "#52514e", "#c3c2b7"
B, SEED = 500, 0
OUT = "paper/figs/candidates/fig_2022_attribution_m4"


def main():
    d = m3.load_box("A", True, "pet")
    D = m3.jaxify(d)
    s = D["s"]
    t0 = [t for (y, sn, t) in season_starts(d, [2022]) if sn == "MAM"][0]
    sl = slice(t0 + 1, t0 + 1 + H)
    a0, y = D["a"][t0], D["a"][sl]
    r, rc, ex, ph = D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]
    ex0 = jnp.zeros_like(ex)
    train = [t for (yy, sn, t) in season_starts(d, range(2003, 2018)) if sn == "MAM"]
    bm = lambda z: np.asarray(z).mean((1, 2))

    fits, parts = [], []
    for k in SEEDS:
        p = {kk: jnp.asarray(v) for kk, v in np.load(f"{R}/params_{CONFIG}_s{k}.npy", allow_pickle=True).item().items()}
        w = m3.w_path(p, D)
        eps, _ = implied(p, s, a0, w[t0], y, r, ex, ph)
        run = lambda rr, ee, e: bm(rerun(p, s, a0, w[t0], rr, ee, ph, e))[-1]
        obs, cf_r, cf_t, cf_rt = run(r, ex, eps), run(rc, ex, eps), run(r, ex0, eps), run(rc, ex0, eps)
        expected = bm(rerun(p, s, a0, w[t0], r, ex, ph, jnp.zeros_like(eps)))
        rain = 0.5 * ((obs - cf_r) + (cf_t - cf_rt))
        demand = 0.5 * ((obs - cf_t) + (cf_r - cf_rt))
        resid = obs - expected[-1]
        parts.append((obs - rain - demand - resid, rain, demand, resid))
        E = np.stack([np.asarray(implied(p, s, D["a"][t], w[t], D["a"][t + 1:t + 1 + H], D["r"][t + 1:t + 1 + H],
                                         D["ex"][t + 1:t + 1 + H], D["phase"][t + 1:t + 1 + H])[0]) for t in train])
        fits.append((p, w, expected, E))
    start, rain, demand, resid = np.mean(parts, 0)
    expected = np.mean([f[2] for f in fits], 0)

    rng = np.random.default_rng(SEED)
    paths = np.empty((B, H))
    for b in range(B):
        p, w, _, E = fits[rng.integers(len(fits))]
        pick = rng.integers(0, len(train), H)
        paths[b] = bm(rerun(p, s, a0, w[t0], r, ex, ph, jnp.asarray(np.stack([E[pick[k], k] for k in range(H)]))))
    lo, hi = np.percentile(paths, [5, 95], axis=0)

    P = np.load(f"{REF}/params.npy", allow_pickle=True).item()
    p0 = {k: jnp.asarray(v) for k, v in P["M0"].items()}
    A = load("A")
    D0 = {k: jnp.asarray(A[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    hist = bm(M.rollout_mean(p0, a0, M.w_path(p0, D0, False)[t0], r, ph, D0["x"], False))

    obs = bm(y)
    pre = lambda v: np.concatenate([[float(np.asarray(a0).mean())], v])
    obs, expected, hist, lo, hi = map(pre, (obs, expected, hist, lo, hi))
    rain_mm = float((r - rc).mean((1, 2)).sum() * 100)
    pet_mm = float(ex.mean((1, 2, 3)).sum() * 100)
    print(f"MAM2022: obs {obs[-1]:+.3f}  expected {expected[-1]:+.3f}  history {hist[-1]:+.3f}  rain {rain_mm:+.0f} mm  demand {pet_mm:+.0f} mm")
    print(f"split (mean of {len(fits)} fits): start {start:+.3f}  rain {rain:+.3f}  demand {demand:+.3f}  residual {resid:+.3f}")

    plt.rcParams.update({"font.size": 8})
    hh = np.arange(H + 1) * 16
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.5], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1], sharey=ax)
    for a in (ax, bx):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax.axhline(0, color=LIGHT, lw=0.8)
    ax.fill_between(hh, lo, hi, color=BLUE, alpha=0.14, lw=0, label="90% range: same weather, ordinary-year residuals")
    ax.plot(hh, hist, "--", color=GREY, lw=1.4, label="history alone")
    ax.plot(hh, expected, "-", color=BLUE, lw=2, label="expected from the weather that fell")
    ax.plot(hh, obs, "o-", color=INK, lw=2, ms=4.5, mec="white", mew=0.8, label="observed", zorder=5)
    ax.set_xlim(-3, 100); ax.set_ylim(-0.2, 0.03); ax.set_xticks(hh)
    ax.set_xlabel("days from season start"); ax.set_ylabel("NDVI anomaly, box mean (0 = normal)")
    ax.set_title(f"Box A, Mar–May 2022: rain {abs(rain_mm):.0f} mm below normal, evaporative demand {pet_mm:.0f} mm above",
                 fontsize=8.5, loc="left")
    ax.grid(axis="y", color="#ececea", lw=0.6); ax.set_axisbelow(True)
    kw = dict(fontsize=7.5, ha="right")
    ax.text(95, hist[-1] + 0.007, f"history alone {hist[-1]:+.2f}", color=MUTED, **kw)
    ax.text(95, expected[-1] + 0.007, f"expected from weather {expected[-1]:+.2f}", color=INK, **kw)
    ax.text(95, obs[-1] - 0.016, f"observed {obs[-1]:+.2f}", color=INK, fontweight="bold", **kw)
    ax.legend(loc="lower left", fontsize=6.8, frameon=False, handlelength=1.8)
    top, segs = 0.0, []
    for v, col, lab in ((start, LIGHT, "starting deficit"), (rain, BLUE, "rain"),
                        (demand, YELLOW, "evaporative\ndemand"), (resid, ORANGE, "not explained\nby weather")):
        segs.append((top, top + v, col, lab))
        top += v
    for t_, b_, col, lab in segs:
        bx.bar(0, b_ - t_, bottom=t_, width=0.55, color=col, edgecolor="white", linewidth=1.2)
        bx.text(0.36, (t_ + b_) / 2, f"{lab}\n{abs(b_ - t_):.2f}", va="center", ha="left", fontsize=7,
                color=INK, linespacing=1.15)
    bx.set_xlim(-0.4, 1.9); bx.set_xticks([]); bx.tick_params(left=False, labelleft=False)
    bx.spines["left"].set_visible(False); bx.spines["bottom"].set_visible(False)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=220, bbox_inches="tight")
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
