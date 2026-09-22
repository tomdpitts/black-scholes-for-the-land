"""Is the Delta map a predictor, or only a by-product of a model that has skill?

Three tests, all with the fitted M2 and M1.5 from the fair-fight run:
  T1  transfer of the pattern: the Delta map on Box B (never seen) has the same relation to
      mean NDVI and annual rain as on Box A.
  T2  the map as a standalone predictor: a_{t+2} - a_t = alpha + beta * map(pixel) * rain anomaly(t+1),
      alpha, beta fitted on 2003-2017 long-rains starts, scored on 2019-2025, against the M1.5 map,
      a constant map, and the empirical per-pixel slope from 2003-2017.
  T3  out-of-sample seasons: rank correlation of the 2003-2017-derived map with the empirical
      per-pixel response slope of 2019-2025, against the same controls plus -mean NDVI.

  modal run --detach deltatest.py          -> /vol/runs/deltatest/{metrics.json,maps.npz,progress.log}
  modal volume get projectb-data runs/deltatest runs/<dir>
  python deltatest.py runs/<pipeline dir>  -> local CPU run, same outputs under runs/<ts>_deltatest
"""
import json
import sys

import modal

app = modal.App("projectb-deltatest")
vol = modal.Volume.from_name("projectb-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "scipy", "optax", "jax[cuda12]")
         .add_local_python_source("aoi", "model", "run"))

H2 = 2  # horizon of the map (16-day steps): Delta peaks here
TRAIN = list(range(2003, 2018))
TEST = list(range(2019, 2026))


def compute(data_dir, params_path, out, log=print, crop=None):
    import os
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import spearmanr
    import model as M
    from run import load, season_of, H

    os.makedirs(out, exist_ok=True)
    params = np.load(params_path, allow_pickle=True).item()
    P = {k: {kk: jnp.asarray(vv) for kk, vv in params[k].items()} for k in ("M2", "M15")}
    res, maps = {}, {}

    def J(d):
        return {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}

    def mam_starts(d, years):
        T = d["a"].shape[0]
        return [t for t in range(T - H) if d["year"][t + 1] in years and season_of(d["doy"][t + 1]) == "MAM"]

    def delta_map(name, D, d, years):
        p = P[name]
        w = M.w_path(p, D)
        gk = jax.jit(lambda t: M.greeks(p, D["a"][t], w[t], jax.lax.dynamic_slice_in_dim(D["r"], t + 1, H, 0),
                                        jax.lax.dynamic_slice_in_dim(D["phase"], t + 1, H, 0), D["x"])["delta"][H2 - 1])
        st = mam_starts(d, years)
        acc = 0.0
        for t in st:
            acc = acc + np.asarray(gk(jnp.asarray(t)))
        return (acc / len(st)).astype(np.float32), len(st)

    def empirical(d, years):
        """Per-pixel OLS slope of a_{t+2}-a_t on the rain anomaly of window t+1 over MAM starts."""
        st = mam_starts(d, years)
        y = np.stack([d["a"][t + H2] - d["a"][t] for t in st])
        ok = np.stack([d["obs"][t + H2] & d["obs"][t] for t in st])
        x = np.stack([d["r"][t + 1] - d["rclim"][t + 1] for t in st])
        y = np.where(ok, y, np.nan); x = np.where(ok, x, np.nan)
        xm, ym = np.nanmean(x, 0), np.nanmean(y, 0)
        cov = np.nanmean((x - xm) * (y - ym), 0); var = np.nanmean((x - xm) ** 2, 0)
        return (cov / np.maximum(var, 1e-6)).astype(np.float32), y, x, ok

    def stats(m, d):
        x = d["x"]
        return {"grid_mean": float(np.nanmean(m)), "frac_positive": float(np.nanmean(m > 0)),
                "corr_mean_ndvi": float(np.corrcoef(m.ravel(), x[..., 0].ravel())[0, 1]),
                "corr_annual_rain": float(np.corrcoef(m.ravel(), x[..., 2].ravel())[0, 1])}

    def fit_score(map_, ytr, xtr, oktr, yte, xte, okte):
        """alpha + beta * map * rain anomaly; pooled OLS on train pixel-steps, RMSE on test."""
        ztr = map_[None] * xtr; zte = map_[None] * xte
        a = np.stack([np.ones_like(ztr[oktr]), ztr[oktr]], 1)
        coef, *_ = np.linalg.lstsq(a, ytr[oktr], rcond=None)
        pred = coef[0] + coef[1] * zte
        rmse = float(np.sqrt(np.mean((pred[okte] - yte[okte]) ** 2)))
        rmse_alpha = float(np.sqrt(np.mean((coef[0] - yte[okte]) ** 2)))
        rmse_persist = float(np.sqrt(np.mean(yte[okte] ** 2)))
        return {"alpha": float(coef[0]), "beta": float(coef[1]), "rmse": rmse,
                "skill_vs_persistence": 1 - rmse / rmse_persist, "skill_vs_alpha_only": 1 - rmse / rmse_alpha}

    for box in ("A", "B"):
        d = load(box, data_dir, crop); D = J(d)
        r = {}
        # maps from the model on train-year and test-year starts
        for nm in ("M2", "M15"):
            for tag, yrs in (("train", TRAIN), ("test", TEST)):
                m, n = delta_map(nm, D, d, yrs)
                maps[f"{box}_{nm}_{tag}"] = m
                r[f"{nm}_{tag}"] = {**stats(m, d), "n_starts": n}
                log(f"{box} {nm} {tag}: {json.dumps(r[f'{nm}_{tag}'])}")
        r["corr_M2_vs_M15_test"] = float(np.corrcoef(maps[f"{box}_M2_test"].ravel(), maps[f"{box}_M15_test"].ravel())[0, 1])
        # empirical slopes
        emp_tr, ytr, xtr, oktr = empirical(d, TRAIN)
        emp_te, yte, xte, okte = empirical(d, TEST)
        maps[f"{box}_emp_train"], maps[f"{box}_emp_test"] = emp_tr, emp_te
        r["emp_train"] = stats(emp_tr, d); r["emp_test"] = stats(emp_te, d)
        # T2: standalone predictor, fitted on train, scored on test
        cands = {"M2_map": maps[f"{box}_M2_train"], "M15_map": maps[f"{box}_M15_train"],
                 "constant": np.full_like(emp_tr, float(maps[f"{box}_M2_train"].mean())),
                 "empirical_train": emp_tr,
                 "neg_mean_ndvi": (-d["x"][..., 0]).astype(np.float32)}
        r["T2_standalone"] = {k: fit_score(v, ytr, xtr, oktr, yte, xte, okte) for k, v in cands.items()}
        log(f"{box} T2: {json.dumps(r['T2_standalone'])}")
        # T3: rank correlation of train-derived maps with the test-period empirical slope
        r["T3_rank_with_test_slope"] = {k: {"spearman": float(spearmanr(v.ravel(), emp_te.ravel(), nan_policy="omit")[0]),
                                            "pearson": float(np.corrcoef(v.ravel(), np.nan_to_num(emp_te.ravel()))[0, 1])}
                                        for k, v in cands.items() if k != "constant"}
        r["T3_rank_with_test_slope"]["M2_map_test_starts"] = {"spearman": float(spearmanr(maps[f"{box}_M2_test"].ravel(), emp_te.ravel(), nan_policy="omit")[0])}
        log(f"{box} T3: {json.dumps(r['T3_rank_with_test_slope'])}")
        res[box] = r
        with open(f"{out}/metrics.json", "w") as f:
            json.dump(res, f, indent=1)
        np.savez_compressed(f"{out}/maps.npz", **maps)
    log("done")
    return res


@app.function(image=image, gpu="H100", memory=32768, timeout=3600 * 2, volumes={"/vol": vol})
def remote():
    out = "/vol/runs/deltatest"
    import os
    os.makedirs(out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(f"{out}/progress.log", "a") as fh:
            fh.write(msg + "\n")
        vol.commit()

    compute("/vol/data", "/vol/runs/fairfight/params.npy", out, log)
    vol.commit()


@app.local_entrypoint()
def main():
    remote.remote()


if __name__ == "__main__":
    from run import run_dir
    compute("data", f"{sys.argv[1]}/params.npy", run_dir("deltatest"), crop=int(sys.argv[2]) if len(sys.argv) > 2 else None)
