"""Season-end split of Box A seasons with M4 (5 fits and their mean), as fig_attribution_m4.py: box means over all
pixels; every run from the state before the season's first window, residual zero:
  baseline                   the run under normal weather
  rain, evaporative demand   Shapley over the two forcings (the four weather combinations)
  residual                   observed - the run under the weather that fell

  .venv/bin/python revision/split_seasons.py [YEAR:SEASON ...]   (default 2020:OND 2022:MAM)
"""
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import H  # noqa: E402
from residual import season_starts  # noqa: E402
import m3  # noqa: E402
from m3_attribution import rerun  # noqa: E402

R, CONFIG, SEEDS = "runs/20261021_0147_m3/gpu", "topo_pet", range(5)


def main():
    seasons = [(int(y), sn) for y, sn in (a.split(":") for a in (sys.argv[1:] or ["2020:OND", "2022:MAM"]))]
    d = m3.load_box("A", True, "pet")
    D = m3.jaxify(d)
    s = D["s"]
    bm = lambda z: float(np.asarray(z)[-1].mean())
    fits = []
    for k in SEEDS:
        p = {kk: jnp.asarray(v) for kk, v in np.load(f"{R}/params_{CONFIG}_s{k}.npy", allow_pickle=True).item().items()}
        fits.append((p, m3.w_path(p, D)))
    for year, sn in seasons:
        t0 = [t for (y, n, t) in season_starts(d, [year]) if n == sn][0]
        sl = slice(t0 + 1, t0 + 1 + H)
        a0, y = D["a"][t0], D["a"][sl]
        r, rc, ex, ph = D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]
        ex0, zero = jnp.zeros_like(ex), jnp.zeros_like(y)
        obs = bm(y)
        rows = []
        for p, w in fits:
            run = lambda rr, ee: bm(rerun(p, s, a0, w[t0], rr, ee, ph, zero))
            F, cf_r, cf_t, cf_rt = run(r, ex), run(rc, ex), run(r, ex0), run(rc, ex0)
            rows.append((cf_rt, 0.5 * ((F - cf_r) + (cf_t - cf_rt)), 0.5 * ((F - cf_t) + (cf_r - cf_rt)), obs - F))
        rows = np.array(rows)
        rain_mm = float((r - rc).mean((1, 2)).sum() * 100)
        pet_mm = float(ex.mean((1, 2, 3)).sum() * 100)
        print(f"{sn}{year}: rain {rain_mm:+.0f} mm, demand {pet_mm:+.0f} mm, observed {obs:+.3f}")
        for k, (b, ra, de, re) in zip(SEEDS, rows):
            print(f"  fit {k}: baseline {b:+.3f}  rain {ra:+.3f}  demand {de:+.3f}  weather {ra + de:+.3f}  residual {re:+.3f}")
        b, ra, de, re = rows.mean(0)
        print(f"  mean : baseline {b:+.3f}  rain {ra:+.3f}  demand {de:+.3f}  weather {ra + de:+.3f}  residual {re:+.3f}")


if __name__ == "__main__":
    main()
