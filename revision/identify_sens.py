"""Sensitivity of the real-season counterfactuals (paper Fig. 1) to two residual contaminants
found by identify.py: rain-dependent residual bias, and marginal-quality MODIS pixels.

  .venv/bin/python revision/identify_sens.py   -> runs/20261006_1753_identify/sensitivity.json
"""
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identify import setup, make_fns, BINS  # noqa: E402
from run import H  # noqa: E402
from residual import season_starts  # noqa: E402

OUT = "runs/20261006_1753_identify"
m = json.load(open(f"{OUT}/metrics.json"))
bias = np.array(m["real"]["A"]["mean_eps_by_bin"])  # held-out-year bin means: in-sample, diagnostic only
d, p, D, w = setup("A")
_, invert, cf, _ = make_fns(p, D["x"])
md = np.load("data/A/modis.npz")
pos = {v: i for i, v in enumerate(md["dates"].astype("datetime64[D]"))}
rel = np.stack([md["rel"][pos[v]] for v in d["dates"].astype("datetime64[D]")])
rows = []
for (y, s, t0) in season_starts(d, range(2019, 2026)):
    sl = slice(t0 + 1, t0 + 1 + H)
    a0, w0, r, rc, ph, yobs = D["a"][t0], w[t0], D["r"][sl], D["rclim"][sl], D["phase"][sl], D["a"][sl]
    eh, _ = invert(a0, w0, yobs, r, ph)
    dep = np.asarray((r - rc) * 100)
    eh_db = eh - jnp.asarray(bias[np.digitize(dep, BINS[1:-1])], jnp.float32)
    c = np.asarray(cf(a0, w0, rc, ph, eh, False)[-1])
    c_db = np.asarray(cf(a0, w0, rc, ph, eh_db, False)[-1])
    clean = (rel[sl] <= 0).all(0)
    ob = np.asarray(yobs[-1])
    rows.append({"season": f"{s}{y}", "obs": float(ob.mean()), "cf": float(c.mean()), "cf_debiased": float(c_db.mean()),
                 "frac_px_clean": float(clean.mean()), "obs_clean": float(ob[clean].mean()), "cf_clean": float(c[clean].mean())})
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
json.dump(rows, open(f"{OUT}/sensitivity.json", "w"), indent=1)
