"""Size split of the Greeks test: does Delta rank rain exposure for small surprises but not large? See PREREG.md.

  python sizetest.py runs/<pipeline dir>/params.npy runs/<ts>_sizetest
"""
import json
import sys

import numpy as np
import jax.numpy as jnp

import greekstest as G
from run import load

NBOOT = 2000


def stats_by_year(grp, mask, y, x, a0, yr, yl):
    XtX = np.zeros((len(yl), 3, 3, 3)); Xty = np.zeros((len(yl), 3, 3))
    for s in range(len(yr)):
        yi = yl.index(int(yr[s]))
        for g in range(3):
            mm = mask[s] & (grp[s] == g)
            X = np.stack([np.ones(mm.sum()), x[s][mm], a0[s][mm]], 1)
            XtX[yi, g] += X.T @ X; Xty[yi, g] += X.T @ y[s][mm]
    return XtX, Xty


def slopes(XtX, Xty, wt):
    A = np.einsum("y,ygij->gij", wt, XtX); b = np.einsum("y,ygi->gi", wt, Xty)
    return np.array([np.linalg.solve(A[g], b[g])[1] for g in range(3)])


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def split_test(Gv, Draw, y, x, a0, ok, yr, rng):
    q = np.quantile(Gv[ok], [1 / 3, 2 / 3]); grp = np.digitize(Gv, q)
    med = float(np.median(np.abs(x[ok])))
    small = ok & (np.abs(x) <= med); large = ok & (np.abs(x) > med)
    yl = sorted(set(yr.tolist()))
    S = stats_by_year(grp, small, y, x, a0, yr, yl); L = stats_by_year(grp, large, y, x, a0, yr, yl)
    one = np.ones(len(yl))
    ps, pl = slopes(*S, one), slopes(*L, one)
    gs, gl, dd = [], [], []
    for _ in range(NBOOT):
        wt = np.bincount(rng.integers(0, len(yl), len(yl)), minlength=len(yl)).astype(float)
        a, b = slopes(*S, wt), slopes(*L, wt)
        gs.append(a[2] - a[0]); gl.append(b[2] - b[0]); dd.append((a[2] - a[0]) - (b[2] - b[0]))
    dmean = lambda m: [float(Draw[m & (grp == g)].mean()) for g in range(3)]
    return {"median_abs_rain_anomaly_100mm": med,
            "small": {"slopes_low_mid_high": ps.tolist(), "gap": float(ps[2] - ps[0]), "gap_ci95": ci(gs),
                      "model_delta_low_mid_high": dmean(small), "n": int(small.sum())},
            "large": {"slopes_low_mid_high": pl.tolist(), "gap": float(pl[2] - pl[0]), "gap_ci95": ci(gl),
                      "model_delta_low_mid_high": dmean(large), "n": int(large.sum())},
            "gap_small_minus_large": float((ps[2] - ps[0]) - (pl[2] - pl[0])), "diff_ci95": ci(dd),
            "S_diff_pass": bool(np.percentile(dd, 2.5) > 0), "S2_small_pass": bool(np.percentile(gs, 2.5) > 0)}


def main(params_path, out):
    P = {k: jnp.asarray(v) for k, v in np.load(params_path, allow_pickle=True).item()["M2"].items()}
    res = {}
    for box in ("A", "B"):
        rng = np.random.default_rng(0)
        d = load(box, "data")
        D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
        st = G.starts(d, G.TEST)
        y, x, a0, ok, yr = G.outcomes(d, st)
        De = G.deltas(P, D, st)
        res[box] = {"S1_raw": split_test(De, De, y, x, a0, ok, yr, rng),
                    "S3_within_pixel": split_test(De - De.mean(0, keepdims=True), De, y, x, a0, ok, yr, rng)}
        print(box, json.dumps(res[box], indent=1), flush=True)
    res["S1_primary_pass_both_boxes"] = bool(res["A"]["S1_raw"]["S_diff_pass"] and res["B"]["S1_raw"]["S_diff_pass"])
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
