"""The implied residual, and plugging it back in.

Back out: eps_t = (a_{t+1} - a_t - mu(a_t, w_t, r_{t+1})) / sigma(.)  from observed greenness
          and observed rain, step by step (the implied-volatility move).
Plug in:  roll the same model along another rain path r' with eps_t held fixed
          (Pearl's abduction / action / prediction). r' = climatology gives "what the land
          would have done in a normal season, with everything that was not rain unchanged".

  .venv/bin/python residual.py [run dir]  -> runs/<ts>_residual/{metrics.json,fig data}, paper/figs/fig_residual.pdf
"""
import json
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import model as M
from run import load, season_of, run_dir, H

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/20260914_1708_pipeline_h100_fairfight"
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9a9891", "#0b0b0b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
TEST = range(2019, 2026)


def season_starts(d, years):
    """One start per rainy season: the state just before its first 16-day window."""
    out = []
    T = d["a"].shape[0]
    for t in range(T - H):
        s, y = season_of(d["doy"][t + 1]), int(d["year"][t + 1])
        if y in years and s in ("MAM", "OND") and season_of(d["doy"][t]) != s:
            out.append((y, s, t))
    return out


def main():
    A = load("A")
    params = np.load(f"{RUN}/params.npy", allow_pickle=True).item()
    p = {k: jnp.asarray(v) for k, v in params["M2"].items()}
    p0 = {k: jnp.asarray(v) for k, v in params["M0"].items()}
    D = {k: jnp.asarray(A[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    w = M.w_path(p, D)
    w_m0 = M.w_path(p0, D)
    x = D["x"]

    @jax.jit
    def implied(t):
        """eps_t and sigma_t from the observed transition t -> t+1."""
        mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], x, True)
        return (D["a"][t + 1] - mu) / sig, sig

    @jax.jit
    def roll(a0, w0, r_path, ph_path, eps_path):
        """Mean rollout along r_path with the innovation sequence eps_path re-injected."""
        def body(c, inp):
            a, wv = c
            r, ph, e = inp
            mu, sig, wv = M.step(p, a, wv, r, ph, x, True)
            a = mu + sig * e
            return (a, wv), a
        _, path = jax.lax.scan(body, (a0, w0), (r_path, ph_path, eps_path))
        return path

    rows, curves = [], {}
    for (y, s, t0) in season_starts(A, TEST):
        eps = jnp.stack([implied(t)[0] for t in range(t0, t0 + H)])          # [H,N,N]
        a0, w0 = D["a"][t0], w[t0]
        ph = D["phase"][t0 + 1:t0 + 1 + H]
        r_real, r_clim = D["r"][t0 + 1:t0 + 1 + H], D["rclim"][t0 + 1:t0 + 1 + H]
        zero = jnp.zeros_like(eps)
        obs = np.asarray(D["a"][t0 + 1:t0 + 1 + H])
        rebuilt = np.asarray(roll(a0, w0, r_real, ph, eps))                 # must equal obs
        expected = np.asarray(roll(a0, w0, r_real, ph, zero))               # F_h: rain only
        cf = np.asarray(roll(a0, w0, r_clim, ph, eps))                      # normal rain, same residual
        cf0 = np.asarray(roll(a0, w0, r_clim, ph, zero))                    # normal rain, no residual
        m0 = np.asarray(M.rollout_mean(p0, a0, w_m0[t0], r_real, ph, x, False))  # history only
        m = lambda z: [float(v) for v in z.mean((1, 2))]
        rain_dep = float((r_real - r_clim).mean((1, 2)).sum() * 100)       # mm over the season
        row = {"year": y, "season": s, "t0": int(t0), "rain_departure_mm": rain_dep,
               "recon_max_abs_err": float(np.abs(rebuilt - obs).max()),
               "a0": float(np.asarray(a0).mean()), "obs": m(obs), "expected": m(expected),
               "counterfactual": m(cf), "counterfactual_no_resid": m(cf0), "history_only": m(m0),
               "eps_gridmean_by_h": [float(v) for v in np.asarray(eps).mean((1, 2))],
               "eps_std_by_h": [float(v) for v in np.asarray(eps).std((1, 2))]}
        rows.append(row)
        curves[f"{s}{y}"] = {"obs": obs.mean((1, 2)), "expected": expected.mean((1, 2)),
                             "cf": cf.mean((1, 2)), "cf0": cf0.mean((1, 2)), "m0": m0.mean((1, 2)), "a0": float(np.asarray(a0).mean()),
                             "eps_map": np.asarray(eps).mean(0), "resid_map": (obs - expected)[-1]}
        print(f"{s} {y}: rain {rain_dep:+.0f} mm  a0 {row['a0']:+.3f}  h6 obs {row['obs'][-1]:+.3f} "
              f"expected {row['expected'][-1]:+.3f}  cf-normal {row['counterfactual'][-1]:+.3f} "
              f"cf-normal-noresid {row['counterfactual_no_resid'][-1]:+.3f}  recon err {row['recon_max_abs_err']:.1e}")

    # season-level attribution: observed = expected(rain) + residual;  counterfactual = normal rain + residual
    out = run_dir("residual")
    json.dump({"run": RUN, "rows": rows}, open(f"{out}/metrics.json", "w"), indent=1)
    np.savez(f"{out}/curves.npz", **{f"{k}_{q}": v for k, c in curves.items() for q, v in c.items()})

    # ------------------------------------------------------------------ figure 1: normal season, drought season
    def panel(ax, key, title, t0):
        c = curves[key]
        hh = np.arange(0, H + 1) * 16
        pre = lambda v: np.concatenate([[c["a0"]], v])
        ax.plot(hh, pre(c["m0"]), "--", color=GREY, lw=0.9, label="history only, no rain (the incumbent baseline)", zorder=1)
        ax.plot(hh, pre(c["obs"]), "o-", ms=3, color=INK, label="observed", zorder=3)
        ax.plot(hh, pre(c["expected"]), "-", color=BLUE, label="expected from the rain that fell (residual = 0)", zorder=2)
        ax.plot(hh, pre(c["cf"]), "-", color=ORANGE, label="counterfactual: normal rain, same residual", zorder=2)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlim(-3, 100); ax.set_ylim(-0.2, 0.12); ax.set_xticks([0, 32, 64, 96])
        ax.set_title(title, fontsize=8); ax.set_xlabel("days from season start")

    starts = {f"{s}{y}": t for (y, s, t) in season_starts(A, TEST)}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    panel(ax[0], "OND2020", "(a) Oct–Dec 2020, short-rains season: rain near normal", starts["OND2020"])
    panel(ax[1], "MAM2022", "(b) Mar–May 2022, long-rains season: drought", starts["MAM2022"])
    ax[0].set_ylabel("NDVI anomaly, box mean")
    ax[0].legend(fontsize=5.5, frameon=False, loc="lower left", handlelength=1.6)
    fig.tight_layout(w_pad=2.5)
    fig.savefig("paper/figs/fig_residual.pdf"); fig.savefig("paper/figs/fig_residual.png", dpi=200)

    # ------------------------------------------------------------------ figure 2: fire. burn on 2021-01-17, scored one composite on
    tb = int(np.where(A["dates"].astype("datetime64[D]") == np.datetime64("2021-01-17"))[0][0])
    tgt = tb + 1
    burn = A["burn"][tb].astype(bool)
    e = np.asarray(implied(tgt - 1)[0])                                   # residual, spread units
    z = A["a"][tgt] / A["cstd"][A["idx"][tgt]]                           # z-score, climatological s.d. units
    ys, xs = np.where(burn)
    m = 12
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, burn.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, burn.shape[1])
    fire = {"burn_date": str(A["dates"][tb]), "scored_date": str(A["dates"][tgt]), "n_burned": int(burn.sum()),
            "eps_burned": float(e[burn].mean()), "eps_unburned": float(e[~burn].mean()),
            "z_burned": float(z[burn].mean()), "z_unburned": float(z[~burn].mean()),
            "frac_burned_flagged_at_own_5pct": {"eps": float((e[burn] < np.percentile(e[~burn], 5)).mean()),
                                                "z": float((z[burn] < np.percentile(z[~burn], 5)).mean())},
            "crop_rows": [int(y0), int(y1)], "crop_cols": [int(x0), int(x1)]}
    print("fire:", fire)
    from mpl_toolkits.axes_grid1 import ImageGrid
    tc = np.load("data/A/fig_fire_rgb.npz")                                  # Sentinel-2 crop from fire_rgb.py
    fig = plt.figure(figsize=(7.0, 1.8))
    ax = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.12, cbar_mode="single", cbar_location="right",
                   cbar_size="3%", cbar_pad=0.08)                           # colour bar exactly as tall as the maps
    for k, (fld, title) in enumerate([(e, "(a) residual, vs.\ the rain that fell"), (z, "(b) z-score, vs.\ own history"),
                                      (None, f"(c) Sentinel-2, {str(tc['date'])}")]):
        if fld is None:
            ax[k].imshow(tc["rgb"], interpolation="lanczos", aspect="auto",
                         extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        else:
            im = ax[k].imshow(fld[y0:y1, x0:x1], cmap="RdBu", vmin=-3, vmax=3, origin="upper",
                              extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        ax[k].contour(np.arange(x0, x1), np.arange(y0, y1), burn[y0:y1, x0:x1].astype(float),
                      levels=[0.5], colors="w" if fld is None else "k", linewidths=0.6)
        ax[k].set_xlim(x0 - 0.5, x1 - 0.5); ax[k].set_ylim(y1 - 0.5, y0 - 0.5)
        ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].set_title(title.replace("\\ ", " "), fontsize=7)
        for sp in ax[k].spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, cax=ax.cbar_axes[0]); cb.set_label("standard deviations", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.savefig("paper/figs/fig_fire.pdf", bbox_inches="tight"); fig.savefig("paper/figs/fig_fire.png", dpi=200, bbox_inches="tight")
    json.dump({"run": RUN, "rows": rows, "fire": fire}, open(f"{out}/metrics.json", "w"), indent=1)
    print("wrote", out, "fig_residual.pdf, fig_fire.pdf")


if __name__ == "__main__":
    main()
