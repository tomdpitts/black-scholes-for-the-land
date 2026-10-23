"""Box A, March-May 2022 with Chronos-2 (zero-shot, with weather): as fig_maps_split_m4.py, same layout.

Per pixel at season end, NDVI units, from runs/20261022_1639_suite/c2_drought.npz (revision/c2.py drought, full box):
  rain, evaporative demand   Shapley over the two forcings, percentiles held fixed (suite.py drought's formula)
  residual                   observed - median path under the weather that fell (percentile 50 at every step)
  baseline                   observed - rain - demand - residual
Residual blocks are not faded: Chronos-2 has no ordinary-season (2003-17) residual range yet.

  .venv/bin/python revision/fig_maps_split_c2.py   -> paper/figs/chronos2/fig_maps_split_2022_c2.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import H  # noqa: E402
from residual import season_starts  # noqa: E402
from fig_maps_split import blocks  # noqa: E402
import m3  # noqa: E402
from fig_maps_split_m4 import draw  # noqa: E402

C2 = "runs/20261022_1639_suite/c2_drought.npz"
OUT = "paper/figs/chronos2/fig_maps_split_2022_c2"


def main():
    d = m3.load_box("A", True, "pet")
    t0 = [t for (y, sn, t) in season_starts(d, [2022]) if sn == "MAM"][0]
    C = np.load(C2)
    assert int(C["t0"]) == t0 and len(C["px"]) == d["a"][0].size
    sq = lambda k: C[k].reshape(d["a"][0].shape)
    obs, F, cr, ct, crt = map(sq, ("obs", "F", "cf_r", "cf_t", "cf_rt"))
    assert np.allclose(obs, d["a"][t0 + H], atol=1e-6)
    rain = 0.5 * ((obs - cr) + (ct - crt))
    demand = 0.5 * ((obs - ct) + (cr - crt))
    resid = obs - F
    ok = d["obs"][t0 + H].astype(bool)
    maps = [blocks(m, ok) for m in (obs, obs - rain - demand - resid, rain, demand, resid)]
    elev = blocks(np.load("data/A/topo.npz")["elev"])
    means = [float(np.nanmean(m)) for m in maps]
    sl = slice(t0 + 1, t0 + 1 + H)
    rain_mm = float((d["r"][sl] - d["rclim"][sl]).mean((1, 2)).sum() * 100)
    pet_mm = float(d["ex"][sl].mean((1, 2, 3)).sum() * 100)
    corr = lambda m: float(np.corrcoef(elev.ravel(), m.ravel())[0, 1])
    print("box means obs/baseline/rain/demand/residual:", " ".join(f"{v:+.3f}" for v in means))
    print(f"corr with elevation (2 km blocks): rain {corr(maps[2]):+.2f}  demand {corr(maps[3]):+.2f}  residual {corr(maps[4]):+.2f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    draw(maps, None, elev, means, rain_mm, pet_mm, "Chronos-2, zero-shot", OUT)


if __name__ == "__main__":
    main()
