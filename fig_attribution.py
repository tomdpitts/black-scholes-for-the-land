"""Figure 1: the 2022 drought, what the rain explains and what it does not.

Lines: history only (M0), expected from the rain that fell (M2, residual zero), observed; box means.
Band: 90% range of the same rollout along the 2022 rain with whole residual maps re-injected, each
step's map drawn from the same step of a random March-May season in 2003-2017 (keeps the residual's
spatial correlation, so the band is honest at box scale). Bar: the season-end anomaly split into the
starting deficit, the rain's share (observed minus the normal-rain counterfactual, Eq. cf) and the
residual (observed minus expected).

  .venv/bin/python fig_attribution.py [run dir]  -> paper/figs/fig_attribution.pdf/.png
"""
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import model as M
from run import load, H
from residual import season_starts

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/20260914_1708_pipeline_h100_fairfight"
BLUE, ORANGE, GREY, INK, MUTED, LIGHT = "#2a78d6", "#eb6834", "#9a9891", "#0b0b0b", "#52514e", "#c3c2b7"
B, SEED = 500, 0


def main():
    A = load("A")
    params = np.load(f"{RUN}/params.npy", allow_pickle=True).item()
    p = {k: jnp.asarray(v) for k, v in params["M2"].items()}
    p0 = {k: jnp.asarray(v) for k, v in params["M0"].items()}
    D = {k: jnp.asarray(A[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    w, w_m0, x = M.w_path(p, D), M.w_path(p0, D), D["x"]

    @jax.jit
    def implied(t):
        mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], x, True)
        return (D["a"][t + 1] - mu) / sig

    @jax.jit
    def roll(a0, w0, r_path, ph_path, eps_path):
        """Box-mean rollout along r_path with eps_path re-injected."""
        def body(c, inp):
            a, wv = c
            r, ph, e = inp
            mu, sig, wv = M.step(p, a, wv, r, ph, x, True)
            a = mu + sig * e
            return (a, wv), a.mean()
        _, path = jax.lax.scan(body, (a0, w0), (r_path, ph_path, eps_path))
        return path

    t0 = [t for (y, s, t) in season_starts(A, [2022]) if s == "MAM"][0]
    a0, w0 = D["a"][t0], w[t0]
    ph = D["phase"][t0 + 1:t0 + 1 + H]
    r_real, r_clim = D["r"][t0 + 1:t0 + 1 + H], D["rclim"][t0 + 1:t0 + 1 + H]
    eps = jnp.stack([implied(t) for t in range(t0, t0 + H)])
    obs = np.asarray(D["a"][t0 + 1:t0 + 1 + H]).mean((1, 2))
    expected = np.asarray(roll(a0, w0, r_real, ph, jnp.zeros_like(eps)))
    cf = np.asarray(roll(a0, w0, r_clim, ph, eps))
    hist = np.asarray(M.rollout_mean(p0, a0, w_m0[t0], r_real, ph, x, False)).mean((1, 2))

    train = [t for (y, s, t) in season_starts(A, range(2003, 2018)) if s == "MAM"]
    E = np.stack([np.stack([np.asarray(implied(t + k)) for k in range(H)]) for t in train])
    rng = np.random.default_rng(SEED)
    paths = np.empty((B, H))
    for b in range(B):
        pick = rng.integers(0, len(train), H)
        paths[b] = np.asarray(roll(a0, w0, r_real, ph, jnp.asarray(np.stack([E[pick[k], k] for k in range(H)]))))
    lo, hi = np.percentile(paths, [5, 95], axis=0)

    pre = lambda v: np.concatenate([[float(np.asarray(a0).mean())], v])
    obs, expected, hist, lo, hi = map(pre, (obs, expected, hist, lo, hi))
    rain, resid = obs[-1] - cf[-1], obs[-1] - expected[-1]
    start = obs[-1] - rain - resid
    print(f"MAM2022 t0={t0}  obs {obs[-1]:+.3f}  expected {expected[-1]:+.3f}  cf {cf[-1]:+.3f}  history {hist[-1]:+.3f}")
    print(f"split: start {start:+.3f}  rain {rain:+.3f}  residual {resid:+.3f}  ({len(train)} training seasons, B={B})")
    print("band lo", lo.round(3), "\nband hi", hi.round(3))

    plt.rcParams.update({"font.size": 8})
    hh = np.arange(H + 1) * 16
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.5], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1], sharey=ax)
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax.axhline(0, color=LIGHT, lw=0.8)
    ax.fill_between(hh, lo, hi, color=BLUE, alpha=0.14, lw=0, label="90% range: same rain, ordinary-year residuals")
    ax.plot(hh, hist, "--", color=GREY, lw=1.4, label="history alone")
    ax.plot(hh, expected, "-", color=BLUE, lw=2, label="expected from the rain that fell")
    ax.plot(hh, obs, "o-", color=INK, lw=2, ms=4.5, mec="white", mew=0.8, label="observed", zorder=5)
    ax.set_xlim(-3, 100); ax.set_ylim(-0.2, 0.03); ax.set_xticks(hh)
    ax.set_xlabel("days from season start"); ax.set_ylabel("NDVI anomaly, box mean (0 = normal)")
    ax.set_title("Box A, Mar–May 2022 (long-rains season): rain 124 mm below normal", fontsize=8.5, loc="left")
    ax.grid(axis="y", color="#ececea", lw=0.6); ax.set_axisbelow(True)
    kw = dict(fontsize=7.5, ha="right")
    ax.text(95, hist[-1] + 0.007, f"history alone {hist[-1]:+.2f}", color=MUTED, **kw)
    ax.text(95, expected[-1] + 0.007, f"expected from rain {expected[-1]:+.2f}", color=INK, **kw)
    ax.text(95, obs[-1] - 0.016, f"observed {obs[-1]:+.2f}", color=INK, fontweight="bold", **kw)
    ax.legend(loc="lower left", fontsize=6.8, frameon=False, handlelength=1.8)
    segs = [(0.0, start, LIGHT, "starting deficit"), (start, start + rain, BLUE, "rain"),
            (start + rain, obs[-1], ORANGE, "not explained\nby rain")]
    for top, bot, col, lab in segs:
        bx.bar(0, bot - top, bottom=top, width=0.55, color=col, edgecolor="white", linewidth=1.2)
        bx.text(0.36, (top + bot) / 2, f"{lab}\n{abs(bot - top):.2f}", va="center", ha="left", fontsize=7,
                color=INK, linespacing=1.15)
    bx.set_xlim(-0.4, 1.9); bx.set_xticks([]); bx.tick_params(left=False, labelleft=False)
    bx.spines["left"].set_visible(False); bx.spines["bottom"].set_visible(False)
    fig.savefig("paper/figs/fig_attribution.pdf", bbox_inches="tight")
    fig.savefig("paper/figs/fig_attribution.png", dpi=220, bbox_inches="tight")
    print("wrote paper/figs/fig_attribution.pdf/.png")


if __name__ == "__main__":
    main()
