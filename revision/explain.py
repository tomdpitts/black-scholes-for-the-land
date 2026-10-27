"""Greeks explain of the weather term with M4 (5 fits and their mean), Box A held-out seasons 2019-25.

The weather term of the season split (split_seasons.py) is a full re-run: W = G(d) - G(0), where
G(x) is the season-end box mean of the zero-residual run from the state before the season under
weather (normal rain + x_r, zero ED anomaly + x_e) and d = (r - rclim, ED') is the weather that fell.
Taylor about normal weather, directional derivatives along d through the whole rollout:

  W  =  Delta_r + Delta_e                                    first order: dG[d_r] + dG[d_e]
      + Gamma_rr + Gamma_ee + Gamma_re                       second order: 1/2 d2G[d_r,d_r], 1/2 d2G[d_e,d_e], d2G[d_r,d_e]
      + Speed + unexplained_3                                third order: 1/6 d3G[d,d,d]; unexplained_3 = W - first - second - third

Delta_r sums the per-window rain deltas times each window's rain departure. For a quadratic G the
Shapley rain share is Delta_r + Gamma_rr + Gamma_re/2, so the explain also checks the Shapley split.
Per-pixel figures are 2 km block means (8x8 pixels), as the paper's maps.

  .venv/bin/python revision/explain.py <out dir>   -> <out dir>/explain.json, printed table
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual import season_starts  # noqa: E402
import m3  # noqa: E402
from m3_attribution import rerun  # noqa: E402
from run import H  # noqa: E402

R, CONFIG, SEEDS = "runs/20261021_0147_m3/gpu", "topo_pet", range(5)
TERMS = ("W", "Delta_r", "Delta_e", "Gamma_rr", "Gamma_ee", "Gamma_re", "unexplained", "shapley_r", "shapley_e", "Speed", "unexplained_3")


def blocks(z, b=8):
    n = z.shape[0] // b * b
    return z[:n, :n].reshape(n // b, b, n // b, b).mean((1, 3))


def explain_season(p, s, a0, w0, r, rc, ex, ph):
    """Season-end per-pixel maps [N,N] of every term in TERMS."""
    zero = jnp.zeros((H,) + a0.shape)
    G = lambda xr, xe: rerun(p, s, a0, w0, rc + xr, xe, ph, zero)[-1]
    dr, de = r - rc, ex
    zr, ze = jnp.zeros_like(dr), jnp.zeros_like(de)
    d1 = lambda u: jax.jvp(G, (zr, ze), u)[1]
    d2 = lambda u, v: jax.jvp(lambda xr, xe: jax.jvp(G, (xr, xe), u)[1], (zr, ze), v)[1]
    ur, ue = (dr, ze), (zr, de)
    g0, g_r, g_e, g_re = G(zr, ze), G(dr, ze), G(zr, de), G(dr, de)
    t = {"W": g_re - g0, "Delta_r": d1(ur), "Delta_e": d1(ue),
         "Gamma_rr": 0.5 * d2(ur, ur), "Gamma_ee": 0.5 * d2(ue, ue), "Gamma_re": d2(ur, ue),
         "shapley_r": 0.5 * ((g_re - g_e) + (g_r - g0)), "shapley_e": 0.5 * ((g_re - g_r) + (g_e - g0))}
    t["unexplained"] = t["W"] - sum(t[k] for k in TERMS[1:6])
    u = (dr, de)  # third order along the whole departure, the Greeks' speed
    d3 = jax.jvp(lambda xr, xe: jax.jvp(lambda yr, ye: jax.jvp(G, (yr, ye), u)[1], (xr, xe), u)[1], (zr, ze), u)[1]
    t["Speed"] = d3 / 6
    t["unexplained_3"] = t["unexplained"] - t["Speed"]
    return t


def main(out):
    os.makedirs(out, exist_ok=True)
    d = m3.load_box("A", True, "pet")
    D = m3.jaxify(d)
    s = D["s"]
    fits = []
    for k in SEEDS:
        p = {kk: jnp.asarray(v) for kk, v in np.load(f"{R}/params_{CONFIG}_s{k}.npy", allow_pickle=True).item().items()}
        fits.append((p, m3.w_path(p, D)))
    f = jax.jit(explain_season)
    rows = []
    for year, sn, t0 in season_starts(d, range(2019, 2026)):
        sl = slice(t0 + 1, t0 + 1 + H)
        r, rc, ex, ph = D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]
        maps = [f(p, s, D["a"][t0], w[t0], r, rc, ex, ph) for p, w in fits]
        mean = {k: np.mean([np.asarray(m[k]) for m in maps], 0) for k in TERMS}
        blk = {k: blocks(v) for k, v in mean.items()}
        row = {"season": f"{sn}{year}", "rain_departure_mm": float((r - rc).mean((1, 2)).sum() * 100),
               "ed_departure_mm": float(ex.mean((1, 2, 3)).sum() * 100),
               "box": {k: float(v.mean()) for k, v in mean.items()},
               "box_per_fit": {k: [float(np.asarray(m[k]).mean()) for m in maps] for k in TERMS},
               # share of the block-level weather term the Greeks reproduce: 1 - sum(unexpl^2)/sum(W^2)
               "blocks_r2_first": float(1 - ((blk["W"] - blk["Delta_r"] - blk["Delta_e"]) ** 2).sum() / (blk["W"] ** 2).sum()),
               "blocks_r2_second": float(1 - (blk["unexplained"] ** 2).sum() / (blk["W"] ** 2).sum()),
               "blocks_max_abs_unexplained": float(np.abs(blk["unexplained"]).max())}
        rows.append(row)
        b = row["box"]
        print(f"{row['season']:8s} rain {row['rain_departure_mm']:+5.0f} ED {row['ed_departure_mm']:+4.0f} | W {b['W']:+.3f} = "
              f"D_r {b['Delta_r']:+.3f} D_e {b['Delta_e']:+.3f} G_rr {b['Gamma_rr']:+.3f} G_ee {b['Gamma_ee']:+.3f} "
              f"G_re {b['Gamma_re']:+.3f} unexpl {b['unexplained']:+.4f} | Shapley r {b['shapley_r']:+.3f} e {b['shapley_e']:+.3f} "
              f"| blocks R2 1st {row['blocks_r2_first']:.3f} 2nd {row['blocks_r2_second']:.3f}", flush=True)
    json.dump(rows, open(f"{out}/explain.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
