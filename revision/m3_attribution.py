"""Season attribution with M3: start deficit, rain share, temperature share, residual.

  obs = F + residual                    F: mean rollout along the forcings that fell, residual zero
  rain share, temperature share         Shapley over the two forcings, residual held fixed:
      rain = 1/2 [(obs - CF_r) + (CF_t - CF_rt)],  temp = 1/2 [(obs - CF_t) + (CF_r - CF_rt)]
      CF_x: re-run with forcing x set to normal (rain -> climatology, T' -> 0)
  start deficit = obs - residual - rain - temp
With rain as the only forcing this is the paper's split. Also the temperature Greek: the change in
season-end greenness per +1 C over the first window (jvp of the mean rollout), and the one-step
sensitivity g_mu. Box means over all pixels, Box A held-out seasons.

  .venv/bin/python revision/m3_attribution.py <run dir> <config> <seed>   -> <run dir>/attribution_<config>_s<seed>.json
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import H  # noqa: E402
from residual import season_starts  # noqa: E402
import m3  # noqa: E402


def rerun(p, s, a0, w0, r, ex, ph, eps):
    """Re-run a season along forcings (r, ex) with standardised residuals eps held fixed: [H,N,N]."""
    def body(c, i):
        a, w_ = c
        m, sg, w_, _ = m3.step(p, a, w_, r[i], ex[i], ph[i], s)
        a = m + sg * eps[i]
        return (a, w_), a
    return jax.lax.scan(body, (a0, w0), jnp.arange(H))[1]


def implied(p, s, a0, w0, y, r, ex, ph):
    """Invert a season: standardised residuals [H,N,N] and the one-step temperature sensitivity."""
    def body(c, i):
        a, w_ = c
        m, sg, w_, g = m3.step(p, a, w_, r[i], ex[i], ph[i], s)
        return (y[i], w_), ((y[i] - m) / sg, g[..., 0] if g.shape[-1] else jnp.zeros_like(m))
    return jax.lax.scan(body, (a0, w0), jnp.arange(H))[1]


def season_split(p, D, w, t0):
    """Split one season (state at t0, windows t0+1..t0+H) as in the module docstring. Box means, season end."""
    s = D["s"]
    roll = lambda *z: rerun(p, s, *z)
    sl = slice(t0 + 1, t0 + 1 + H)
    a0, w0, y = D["a"][t0], w[t0], D["a"][sl]
    r, rc, ex, ph = D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]
    ex0 = jnp.zeros_like(ex)
    eps, g_mu = implied(p, s, a0, w0, y, r, ex, ph)
    zero = jnp.zeros_like(eps)
    bm = lambda z: float(z[-1].mean())  # season-end box mean
    obs, F = bm(y), bm(roll(a0, w0, r, ex, ph, zero))
    cf_r = bm(roll(a0, w0, rc, ex, ph, eps))
    assert abs(bm(roll(a0, w0, r, ex, ph, eps)) - obs) < 1e-4  # re-injection reproduces the observation
    row = {"rain_departure_mm": float((r - rc).mean((1, 2)).sum() * 100), "obs": obs, "residual": obs - F}
    if ex.shape[-1]:
        cf_t, cf_rt = bm(roll(a0, w0, r, ex0, ph, eps)), bm(roll(a0, w0, rc, ex0, ph, eps))
        row["rain"] = 0.5 * ((obs - cf_r) + (cf_t - cf_rt))
        row["temp"] = 0.5 * ((obs - cf_t) + (cf_r - cf_rt))
        row["temp_anomaly_C"] = float(ex.mean())
        e1 = jnp.zeros_like(ex).at[0].set(1.0)
        _, dT = jax.jvp(lambda e: roll(a0, w0, r, e, ph, zero), (ex,), (e1,))
        row["delta_T_season_end_per_C"] = float(dT[-1].mean())
        row["g_mu_mean_per_C"] = float(g_mu.mean())
    else:
        row["rain"], row["temp"] = obs - cf_r, 0.0
    row["start_deficit"] = obs - row["residual"] - row["rain"] - row["temp"]
    return row


def attribution(R, config, seed):
    topo, temp = m3.CONFIGS[config][:2]
    d = m3.load_box("A", topo, temp)
    D = m3.jaxify(d)
    p = {k: jnp.asarray(v) for k, v in np.load(f"{R}/params_{config}_s{seed}.npy", allow_pickle=True).item().items()}
    w = m3.w_path(p, D)
    out = []
    for (y_, sn, t0) in season_starts(d, range(2019, 2026)):
        row = {"season": f"{sn}{y_}", **season_split(p, D, w, t0)}
        out.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    json.dump(out, open(f"{R}/attribution_{config}_s{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    attribution(sys.argv[1], sys.argv[2], int(sys.argv[3]))
