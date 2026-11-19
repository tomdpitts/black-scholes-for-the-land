"""Cache the real maps drawn in the outputs of the rollout diagram (fig_rollout.py).

Box A, March-May 2022, M4 (mean of 5 fits), as fig_maps_split_m4.py: season-end observed = baseline + rain + evaporative
demand (Shapley) + residual, as Figure 1. Delta per 2.2 km block on unseen years, as fig_sens_c2.py,
and Gamma (its second derivative in the same rain window) on the same starts.
Averaged to 8x8-pixel blocks (~2.2 km).

  .venv/bin/python revision/rollout_maps.py   -> paper/figs/schematic/rollout_maps.npz
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import H  # noqa: E402
from residual import season_starts  # noqa: E402
from fig_maps_split import blocks  # noqa: E402
import m3  # noqa: E402
from m3_attribution import rerun  # noqa: E402
import suite as S  # noqa: E402
from fig_maps_split_m4 import R, CONFIG, SEEDS  # noqa: E402
from fig_sens_c2 import RUN, B, n  # noqa: E402

OUT = "paper/figs/schematic/rollout_maps.npz"


def split_2022():
    d = m3.load_box("A", True, "pet")
    D = m3.jaxify(d)
    s = D["s"]
    t = [t for (y, sn, t) in season_starts(d, [2022]) if sn == "MAM"][0]
    sl = slice(t + 1, t + 1 + H)
    a0, y = D["a"][t], D["a"][sl]
    r, rc, ex, ph = D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]
    e0, ex0 = jnp.zeros(y.shape), jnp.zeros_like(ex)
    out = []
    for k in SEEDS:
        p = {kk: jnp.asarray(v) for kk, v in np.load(f"{R}/params_{CONFIG}_s{k}.npy", allow_pickle=True).item().items()}
        w = m3.w_path(p, D)
        end = lambda rr, ee: np.asarray(rerun(p, s, a0, w[t], rr, ee, ph, e0)[-1])
        F, cf_r, cf_t, cf_rt = end(r, ex), end(rc, ex), end(r, ex0), end(rc, ex0)
        rain = 0.5 * ((F - cf_r) + (cf_t - cf_rt))
        demand = 0.5 * ((F - cf_t) + (cf_r - cf_rt))
        out.append((np.asarray(y[-1]), cf_rt, rain, demand, np.asarray(y[-1]) - F))
    ok = d["obs"][t + H].astype(bool)
    return [blocks(np.mean(z, 0), ok) for z in zip(*out)]


def delta_map(box="A"):
    P = np.load(f"{RUN}/plan.npz")
    st = P[f"{box}_sens_starts"].tolist()
    ng = P[f"{box}_sens_nogap"].astype(bool)
    return np.mean([S.deltas("M4", box, st, sd) for sd in S.SEEDS], 0)[ng].mean(0).reshape(n, B, n, B).mean((1, 3))


@jax.jit
def _gamma(p, s, a0, w0, rc, ex, ph):
    """d2E[a_{t+2}]/dr_{t+1}^2: Delta's jvp (suite._delta) differentiated again along the same window."""
    e1, zero = jnp.zeros_like(rc).at[0].set(1.0), jnp.zeros_like(rc)
    f = lambda r: rerun(p, s, a0, w0, r, jnp.zeros_like(ex), ph, zero)
    d = lambda r: jax.jvp(f, (r,), (e1,))[1]
    return jax.jvp(d, (rc,), (e1,))[1][S.G.H2 - 1]


def gamma_map(box="A"):
    """As delta_map, the same starts and horizon: per 2.2 km block, NDVI per (100 mm)^2."""
    P = np.load(f"{RUN}/plan.npz")
    st = P[f"{box}_sens_starts"].tolist()
    ng = P[f"{box}_sens_nogap"].astype(bool)
    D = S.data(box, "M4")[1]
    out = []
    for sd in S.SEEDS:
        p = S.params("M4", sd)
        w = m3.w_path(p, D)
        out.append(np.stack([np.asarray(_gamma(p, D["s"], D["a"][t], w[t], D["rclim"][t + 1:t + 1 + H],
                                               D["ex"][t + 1:t + 1 + H], D["phase"][t + 1:t + 1 + H])) for t in st]))
    return np.mean(out, 0)[ng].mean(0).reshape(n, B, n, B).mean((1, 3))


def main():
    obs, base, rain, demand, resid = split_2022()
    delta, gamma = delta_map(), gamma_map()
    for k, v in (("observed", obs), ("baseline", base), ("rain", rain), ("demand", demand), ("residual", resid),
                 ("delta", delta), ("gamma", gamma)):
        print(f"{k:9s} mean {np.nanmean(v):+.3f}  range {np.nanmin(v):+.3f} {np.nanmax(v):+.3f}")
    np.savez(OUT, observed=obs, baseline=base, rain=rain, demand=demand, residual=resid, delta=delta, gamma=gamma)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
