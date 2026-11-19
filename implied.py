"""The implied variable, made concrete with the fitted model.

Forecast mode: the seasonal total is given (a forecast gives totals); how it is spread
across the 16-day windows is the missing input. Same total, different spread, different
greenness, because the response is curved. Vega is d(greenness)/d(spread) at fixed total.

Attribution mode: given the observed greenness through the season, back out the
spread of rain that makes the model reproduce it, and compare with the rain that fell.

  python implied.py runs/<pipeline dir>  -> paper/figs/fig0_implied.png and numbers
"""
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

import model as M
from run import load, season_of, H

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/20260914_1708_pipeline_h100_fairfight"
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9891", "#0b0b0b", "#52514e"


def main():
    A = load("A")
    params = np.load(f"{RUN}/params.npy", allow_pickle=True).item()
    p = {k: jnp.asarray(v) for k, v in params["M2"].items()}
    D = {k: jnp.asarray(A[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    w = M.w_path(p, D)

    # season: the 2019 long rains (first held-out year), start = first composite with DOY >= 60
    year = A["year"]; doy = A["doy"]
    t0 = int(np.where((year == 2019) & (doy >= 60))[0][0]) - 1  # state at t0, rain windows t0+1..t0+H
    a0, w0 = D["a"][t0], w[t0]
    ph = D["phase"][t0 + 1:t0 + 1 + H]
    r_real = D["r"][t0 + 1:t0 + 1 + H]                       # [H,N,N] rain that fell
    obs = np.asarray(D["obs"][t0 + 1:t0 + 1 + H])
    a_obs = np.asarray(D["a"][t0 + 1:t0 + 1 + H])
    total = r_real.mean((1, 2)).sum()                        # box-mean seasonal total (100 mm units)

    def roll(path6):
        """path6: [H] box-uniform rain per window -> grid-mean expected anomaly [H]."""
        rp = jnp.broadcast_to(path6[:, None, None], r_real.shape)
        return M.rollout_mean(p, a0, w0, rp, ph, D["x"], True).mean((1, 2))

    # ---------------- forecast mode: same total, different spread
    even = jnp.full((H,), total / H)
    pattern = jnp.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]) * total   # two storms, same total
    g_even, g_burst = roll(even), roll(pattern)
    # Vega: d(grid-mean E[a_H]) / d s along path = even + s*(pattern - even)
    f = lambda s: roll(even + s * (pattern - even))[-1]
    vega = float(jax.grad(f)(0.0))
    s_grid = np.linspace(0, 1, 11)
    g_s = np.array([float(f(s)) for s in s_grid])

    # ---------------- attribution mode: infer the spread from observed greenness
    obs_mean = np.array([a_obs[h][obs[h]].mean() for h in range(H)])
    real_mean = np.asarray(r_real.mean((1, 2)))

    obs_j, a_obs_j = jnp.asarray(obs), jnp.asarray(a_obs)

    def loss(z):
        # fit the box-uniform 6-window path to every observed pixel-step, not to 6 box means
        path = total * jax.nn.softmax(z)
        rp = jnp.broadcast_to(path[:, None, None], r_real.shape)
        pred = M.rollout_mean(p, a0, w0, rp, ph, D["x"], True)
        return jnp.sum(((pred - a_obs_j) ** 2) * obs_j) / obs_j.sum()

    z = jnp.zeros(H); opt = optax.adam(0.05); st = opt.init(z)
    step = jax.jit(lambda z, st: (lambda l, g: (optax.apply_updates(z, opt.update(g, st, z)[0]), opt.update(g, st, z)[1], l))(*jax.value_and_grad(loss)(z)))
    for i in range(300):
        z, st, l = step(z, st)
    implied = np.asarray(total * jax.nn.softmax(z))
    g_fit = np.asarray(roll(jnp.asarray(implied)))
    g_real = np.asarray(roll(jnp.asarray(real_mean)))
    corr = float(np.corrcoef(implied, real_mean)[0, 1])
    print(f"season start {A['dates'][t0+1]}, total {total*100:.0f} mm; Vega {vega:.4f} NDVI per unit spread; "
          f"E[a_6] even {float(g_even[-1]):.4f} vs two-storm {float(g_burst[-1]):.4f}")
    print("implied path (mm):", (implied * 100).round(0), " CHIRPS:", (real_mean * 100).round(0), f" corr {corr:.2f}")
    print("implied std/mean:", float(implied.std() / implied.mean()), " CHIRPS:", float(real_mean.std() / real_mean.mean()))

    # ---------------- figure: one row, full text width
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7"})
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 1.75), constrained_layout=True)
    x = np.arange(1, H + 1); mm = 100
    # forecast mode: rain paths, then greenness
    ax[0].bar(x - 0.2, np.asarray(even) * mm, 0.38, color=BLUE, label=f"even: {total*mm:.0f} mm")
    ax[0].bar(x + 0.2, np.asarray(pattern) * mm, 0.38, color=ORANGE, label=f"two storms: {total*mm:.0f} mm")
    ax[0].set_ylabel("rain per 16 d (mm)"); ax[0].set_xlabel("16-day steps into the season")
    ax[0].legend(fontsize=6, frameon=False, loc="upper right")
    ax[0].set_title("Forecast: one total, two spreads", fontsize=7.5, loc="left")
    ax[1].plot(x, np.asarray(g_even), "o-", ms=3, color=BLUE, label="model: even rain")
    ax[1].plot(x, np.asarray(g_burst), "o-", ms=3, color=ORANGE, label="model: two storms")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("expected greenness anomaly"); ax[1].set_xlabel("16-day steps into the season")
    ax[1].legend(fontsize=6, frameon=False, loc="upper left")
    ax[1].set_title(f"Vega = {vega:+.3f} NDVI at 3 months", fontsize=7.5, loc="left")
    # attribution mode: greenness, then rain paths
    ax[2].plot(x, obs_mean, "ko", ms=3.5, label="observed (satellite)")
    ax[2].plot(x, g_real, "-", color=GREY, lw=1.5, label="model, rain that fell")
    ax[2].plot(x, g_fit, "--", color=ORANGE, lw=1.5, label="model, implied rain path")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("greenness anomaly"); ax[2].set_xlabel("16-day steps into the season")
    ax[2].legend(fontsize=6, frameon=False)
    ax[2].set_title("Attribution: infer the spread", fontsize=7.5, loc="left")
    ax[3].bar(x - 0.2, real_mean * mm, 0.38, color=GREY, label="rain that fell (CHIRPS)")
    ax[3].bar(x + 0.2, implied * mm, 0.38, color=ORANGE, label=f"implied by greenness (r = {corr:.2f})")
    ax[3].set_ylabel("rain per 16 d (mm)"); ax[3].set_xlabel("16-day steps into the season")
    ax[3].legend(fontsize=6, frameon=False)
    ax[3].set_title("Implied vs actual rain path", fontsize=7.5, loc="left")
    fig.suptitle(f"The missing input is the rain path within the season, given its total   (Box A, long rains {A['dates'][t0+1][:4]}, box means)", fontsize=8, x=0.01, ha="left")
    fig.savefig("paper/figs/fig0_implied.png", bbox_inches="tight", dpi=200)
    print("figure written")


if __name__ == "__main__":
    main()
