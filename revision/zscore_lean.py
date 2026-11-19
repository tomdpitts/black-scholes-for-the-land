"""Weather lean of the true z-score's leftover (the observed season-end anomaly itself), and of M0, M4 and RESTREND (post hoc; not in the suite's preregistered test 2a).

Axes: rain, ED, and the climatic water balance (rain minus ED departure), all in 100 mm.

suite.lean labelled observed - M0 forecast as "z-score". The z-score expects normal (0), so its leftover is the
observed anomaly. Same seasons, 2 km blocks, slope and season bootstrap as suite.lean. M0 is recomputed first and
must reproduce the suite's numbers (and block count) before the z-score row is trusted.

Run from the repo root: python revision/zscore_lean.py runs/<new dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite as S  # noqa: E402
import model as M  # noqa: E402
import m3  # noqa: E402

SUITE = "runs/20261022_1639_suite"

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
P = S.get_plan(SUITE)
d1, D1 = S.data("A", "M1")
p0 = S.params("M0")
w0 = M.w_path(p0, D1)
e5 = np.load("data/A/era5.npz")
R = S.r6(d1)
H = S.H
D4 = S.data("A", "M4")[1]
RT = S.restrend("A")  # per-pixel a + b * R6, fitted on 2003-17
fits = [(S.params("M4", s), m3.w_path(S.params("M4", s), D4)) for s in S.SEEDS]

cols = {k: [] for k in ("rain", "et0", "wb", "M0", "M4", "RESTREND", "z-score")}
table = []
for t0, name in zip(P["season_t0"].tolist(), P["season_name"]):
    sl = slice(t0 + 1, t0 + 1 + H)
    obs, ok = d1["a"][t0 + H], d1["obs"][t0] & d1["obs"][t0 + H]
    F0 = np.asarray(M.rollout_mean(p0, D1["a"][t0], w0[t0], D1["r"][sl], D1["phase"][sl], D1["x"], False)[-1])
    F4 = np.mean([np.asarray(S.forecast_m3(p, D4["s"], D4["a"][t0], w[t0], D4["r"][sl], D4["ex"][sl], D4["phase"][sl])) for p, w in fits], 0)
    v = {"rain": S.blocks(R[t0] * 100, ok), "et0": S.blocks((e5["pet"][sl] - e5["pet_clim"][sl]).sum(0) * 100, ok),
         "M0": S.blocks(obs - F0, ok), "M4": S.blocks(obs - F4, ok),
         "RESTREND": S.blocks(obs - (RT["alpha"] + RT["beta"] * R[t0]), ok), "z-score": S.blocks(obs, ok)}
    v["wb"] = v["rain"] - v["et0"]
    keep = np.isfinite(v["M4"])
    for k in cols:
        cols[k].append(v[k][keep])
    table.append({"season": str(name), "rain_departure_mm": float(v["rain"][keep].mean()),
                  "et0_departure_mm": float(v["et0"][keep].mean()), "leftover_M0": float(v["M0"][keep].mean()),
                  "leftover_z-score": float(v["z-score"][keep].mean()),
                  "leftover_RESTREND": float(v["RESTREND"][keep].mean()), "leftover_M4": float(v["M4"][keep].mean()), "n_blocks": int(keep.sum())})

rng = np.random.default_rng(0)  # same draws as suite.lean
draws = [rng.integers(0, len(table), len(table)) for _ in range(2000)]
res = {"seasons": table, "n_blocks": int(sum(len(x) for x in cols["rain"]))}
for m in ("M0", "M4", "RESTREND", "z-score"):
    for wx in ("rain", "et0", "wb"):
        f = lambda idx: S.slope(np.concatenate([cols[wx][i] for i in idx]) / 100, np.concatenate([cols[m][i] for i in idx]))
        bs = [f(ix) for ix in draws]
        res[f"{m}_{wx}"] = {"slope_per_100mm": f(range(len(table))),
                            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

ref = json.load(open(f"{SUITE}/lean.json"))
res["check_M0_reproduces_suite"] = {
    "n_blocks": [res["n_blocks"], ref["n_blocks"]],
    "rain": [res["M0_rain"]["slope_per_100mm"], ref["z-score_rain"]["slope_per_100mm"]],
    "et0": [res["M0_et0"]["slope_per_100mm"], ref["z-score_et0"]["slope_per_100mm"]],
    "M4_rain": [res["M4_rain"]["slope_per_100mm"], ref["M4_ens_rain"]["slope_per_100mm"]],
    "M4_et0": [res["M4_et0"]["slope_per_100mm"], ref["M4_ens_et0"]["slope_per_100mm"]],
}
json.dump(res, open(f"{out}/zscore_lean.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "seasons"}, indent=1))
print("wrote", f"{out}/zscore_lean.json")
