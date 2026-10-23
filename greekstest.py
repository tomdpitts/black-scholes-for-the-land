"""Greeks utility test: does current-state Delta sort the real rain response? See PREREG.md.

  python greekstest.py runs/<pipeline dir>/params.npy runs/<ts>_greekstest
"""
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp

import model as M
from run import load, season_of, H

H2 = 2
TRAIN = set(range(2003, 2018))
TEST = set(range(2019, 2026))
NBOOT = 2000


def starts(d, years):
    T = d["a"].shape[0]
    return [t for t in range(T - H) if int(d["year"][t + 1]) in years
            and season_of(d["doy"][t + 1]) in ("MAM", "OND")]


def deltas(p, D, st):
    w = M.w_path(p, D)
    gk = jax.jit(lambda t: M.greeks(p, D["a"][t], w[t], jax.lax.dynamic_slice_in_dim(D["rclim"], t + 1, H, 0),
                                    jax.lax.dynamic_slice_in_dim(D["phase"], t + 1, H, 0), D["x"])["delta"][H2 - 1])
    return np.stack([np.asarray(gk(jnp.asarray(t))) for t in st]).astype(np.float64)


def outcomes(d, st):
    y = np.stack([d["a"][t + H2] - d["a"][t] for t in st]).astype(np.float64)
    x = np.stack([d["r"][t + 1] - d["rclim"][t + 1] for t in st]).astype(np.float64)
    a0 = np.stack([d["a"][t] for t in st]).astype(np.float64)
    ok = np.stack([d["obs"][t + H2] & d["obs"][t] for t in st])
    yr = np.array([int(d["year"][t + 1]) for t in st])
    return y, x, a0, ok, yr


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def sort_test(G, Draw, y, x, a0, ok, yr, rng, extra_mask=None, ctrl=None):
    """Terciles of G; per tercile slope of y on x controlling for a0 (and ctrl, if given), bootstrapped over years."""
    q = np.quantile(G[ok], [1 / 3, 2 / 3])
    grp = np.digitize(G, q)
    yl = sorted(set(yr.tolist()))
    k = 3 if ctrl is None else 4
    XtX = np.zeros((len(yl), 3, k, k)); Xty = np.zeros((len(yl), 3, k)); n = np.zeros((len(yl), 3))
    dsum = np.zeros(3); dn = np.zeros(3)
    for s in range(len(yr)):
        yi = yl.index(int(yr[s]))
        m = ok[s] if extra_mask is None else ok[s] & extra_mask[s]
        for g in range(3):
            mm = m & (grp[s] == g)
            X = np.stack([np.ones(mm.sum()), x[s][mm], a0[s][mm]] + ([] if ctrl is None else [ctrl[s][mm]]), 1)
            XtX[yi, g] += X.T @ X; Xty[yi, g] += X.T @ y[s][mm]; n[yi, g] += mm.sum()
            dsum[g] += Draw[s][mm].sum(); dn[g] += mm.sum()

    def slopes(wt):
        A = np.einsum("y,ygij->gij", wt, XtX); b = np.einsum("y,ygi->gi", wt, Xty)
        return np.array([np.linalg.solve(A[g], b[g])[1] for g in range(3)])

    pt = slopes(np.ones(len(yl)))
    bs = np.array([slopes(np.bincount(rng.integers(0, len(yl), len(yl)), minlength=len(yl)).astype(float))
                   for _ in range(NBOOT)])
    diff = bs[:, 2] - bs[:, 0]
    return {"slope_low_mid_high": pt.tolist(), "high_minus_low": float(pt[2] - pt[0]),
            "high_minus_low_ci95": ci(diff), "pass": bool(np.percentile(diff, 2.5) > 0),
            "monotone": bool(pt[0] < pt[1] < pt[2]), "model_delta_mean_low_mid_high": (dsum / dn).tolist(),
            "n_pixel_steps": int(n.sum())}


def hedge(cands_tr, cands_te, ytr, xtr, oktr, yte, xte, okte, yr, rng):
    yl = sorted(set(yr.tolist()))
    sse = {}; out = {}
    sse0 = np.array([(yte[yr == v][okte[yr == v]] ** 2).sum() for v in yl])
    cnt = np.array([okte[yr == v].sum() for v in yl])
    for k in cands_tr:
        ztr = cands_tr[k] * xtr; zte = cands_te[k] * xte
        A = np.stack([np.ones(oktr.sum()), ztr[oktr]], 1)
        coef, *_ = np.linalg.lstsq(A, ytr[oktr], rcond=None)
        res = coef[0] + coef[1] * zte - yte
        sse[k] = np.array([(res[yr == v][okte[yr == v]] ** 2).sum() for v in yl])
        out[k] = {"alpha": float(coef[0]), "beta": float(coef[1]),
                  "skill_vs_persistence": float(1 - np.sqrt(sse[k].sum() / sse0.sum()))}
    draws = [np.bincount(rng.integers(0, len(yl), len(yl)), minlength=len(yl)).astype(float) for _ in range(NBOOT)]
    sk = lambda k, wt: 1 - np.sqrt((wt @ sse[k]) / (wt @ sse0))
    for ref in ("constant", "frozen_M2_map"):
        dd = np.array([sk("M2_state", wt) - sk(ref, wt) for wt in draws])
        out[f"M2_state_minus_{ref}"] = {"point": out["M2_state"]["skill_vs_persistence"] - out[ref]["skill_vs_persistence"],
                                        "ci95": ci(dd), "pass": bool(np.percentile(dd, 2.5) > 0)}
    return out


def main(params_path, out):
    P = {k: {kk: jnp.asarray(v) for kk, v in d.items()}
         for k, d in np.load(params_path, allow_pickle=True).item().items() if k in ("M2", "M15")}
    res = {}
    for box in ("A", "B"):
        rng = np.random.default_rng(0)
        d = load(box, "data")
        D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
        st_te, st_tr = starts(d, TEST), starts(d, TRAIN)
        yte, xte, a0te, okte, yrte = outcomes(d, st_te)
        ytr, xtr, a0tr, oktr, _ = outcomes(d, st_tr)
        dl = {nm: (deltas(P[nm], D, st_tr), deltas(P[nm], D, st_te)) for nm in ("M2", "M15")}
        r = {"n_starts_test": len(st_te), "n_starts_train": len(st_tr),
             "corr_delta_rain_anomaly_M2": float(np.corrcoef(dl["M2"][1][okte], xte[okte])[0, 1])}
        for nm in ("M2", "M15"):
            Dtr, Dte = dl[nm]
            Dw = Dte - Dte.mean(0, keepdims=True)
            r[f"{nm}_G1_state_within_pixel"] = sort_test(Dw, Dte, yte, xte, a0te, okte, yrte, rng)
            r[f"{nm}_G2_sorting_raw"] = sort_test(Dte, Dte, yte, xte, a0te, okte, yrte, rng)
            r[f"{nm}_G1_by_sign"] = {sg: sort_test(Dw, Dte, yte, xte, a0te, okte, yrte, rng,
                                                   extra_mask=(xte < 0) if sg == "dry" else (xte >= 0))["slope_low_mid_high"]
                                     for sg in ("dry", "wet")}
            r[f"{nm}_corr_within_pixel_delta_rain_anomaly"] = float(np.corrcoef(Dw[okte], xte[okte])[0, 1])
        M2tr, M2te = dl["M2"]
        frozen = M2tr.mean(0)
        cands_tr = {"constant": np.full_like(M2tr, M2tr.mean()), "frozen_M2_map": np.broadcast_to(frozen, M2tr.shape),
                    "M2_state": M2tr, "M15_state": dl["M15"][0]}
        cands_te = {"constant": np.full_like(M2te, M2tr.mean()), "frozen_M2_map": np.broadcast_to(frozen, M2te.shape),
                    "M2_state": M2te, "M15_state": dl["M15"][1]}
        r["G3_hedge_error"] = hedge(cands_tr, cands_te, ytr, xtr, oktr, yte, xte, okte, yrte, rng)
        res[box] = r
        print(box, json.dumps(r, indent=1), flush=True)
        with open(f"{out}/metrics.json", "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
