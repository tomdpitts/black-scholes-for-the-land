"""Pipeline: audit (numpy) and fit / score / products (JAX) for the workshop paper.

  python run.py audit                    # Box A substrate audit -> runs/<ts>_audit
  python run.py pipeline [--crop 32]     # fit on A, score on A-test and B, products
Everything writes metrics.json and arrays.npz into a run directory and prints the path.
"""
import json
import os
import sys
import time

import numpy as np

import aoi

H = 6      # rollout horizon in 16-day steps (~3 months)
S = 16     # samples per rollout
SEASONS = {"MAM": (60, 151), "OND": (274, 366)}


def season_of(doy):
    for k, (lo, hi) in SEASONS.items():
        if lo <= doy < hi:
            return k
    return "dry"


def load(box, data_dir="data", crop=None):
    d = dict(np.load(f"{data_dir}/{box}/built.npz"))
    if crop:
        for k in ("a", "obs", "r", "rclim", "clim", "cstd", "burn"):
            if k in d:
                d[k] = d[k][:, :crop, :crop]
        d["x"] = d["x"][:crop, :crop]
    d["doy"] = (d["dates"].astype("datetime64[D]") - d["dates"].astype("datetime64[Y]")).astype(int) + 1
    d["idx"] = (d["doy"] - 1) // 16
    return d


def run_dir(tag):
    ts = time.strftime("%Y%m%d_%H%M")
    path = f"runs/{ts}_{tag}"
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================ audit
def audit(d):
    a, obs, r, rclim, year = d["a"], d["obs"], d["r"], d["rclim"], d["year"]
    ndvi = d["clim"][d["idx"]] + a
    dev = np.isin(year, list(aoi.CLIM_YEARS))
    test = np.isin(year, list(aoi.TEST_YEARS))
    m = {}
    nd = np.where(obs, ndvi, np.nan)[dev]
    total = np.nanvar(nd)
    temporal = np.nanmean(np.nanvar(nd, axis=0))
    seasonal = np.nanmean(np.nanvar(d["clim"][d["idx"]][dev], axis=0))
    m["var_temporal_fraction"] = float(temporal / total)
    m["seasonal_fraction_of_temporal"] = float(seasonal / temporal)
    an = np.where(obs, a, np.nan)
    ac = {}
    for L in (1, 2, 4, 6, 12):
        x0, x1 = an[:-L][dev[:-L]], an[L:][dev[:-L]]
        ok = np.isfinite(x0) & np.isfinite(x1)
        ac[L] = float(np.corrcoef(x0[ok], x1[ok])[0, 1])
    m["anomaly_acf"] = ac
    da = an[1:] - an[:-1]
    ok = np.isfinite(da[:-1]) & np.isfinite(da[1:]) & dev[1:-1, None, None]
    m["change_lag1_corr"] = float(np.corrcoef(da[:-1][ok], da[1:][ok])[0, 1])
    # linear driver R^2 by lag: a_t on rain anomaly at t-l, pooled, fit dev, score test
    ra = r - rclim
    r2 = {}
    for L in range(0, 8):
        y, xx = an[L:], ra[: an.shape[0] - L]
        fit = dev[L:][:, None, None] & np.isfinite(y)
        te = test[L:][:, None, None] & np.isfinite(y)
        b = np.polyfit(xx[fit], y[fit], 1)
        pred = np.polyval(b, xx[te])
        r2[L] = float(1 - np.mean((y[te] - pred) ** 2) / np.var(y[te]))
    m["driver_r2_by_lag_oos"] = r2
    # concavity: seasonal mean anomaly vs seasonal rain-anomaly total, per pixel-year
    conc = {}
    for s in SEASONS:
        sel = np.array([season_of(x) == s for x in d["doy"]])
        ys, xs = [], []
        for yr in np.unique(year):
            t = sel & (year == yr)
            if t.sum() < 3:
                continue
            ys.append(np.nanmean(an[t], axis=0).ravel())
            xs.append(np.nansum(ra[t], axis=0).ravel())
        ys, xs = np.concatenate(ys), np.concatenate(xs)
        ok = np.isfinite(ys) & np.isfinite(xs)
        q = np.polyfit(xs[ok], ys[ok], 2)
        bins = np.quantile(xs[ok], np.linspace(0, 1, 11))
        which = np.clip(np.digitize(xs[ok], bins) - 1, 0, 9)
        conc[s] = {"quadratic_coef": float(q[0]), "linear_coef": float(q[1]),
                   "bin_x": [float(xs[ok][which == i].mean()) for i in range(10)],
                   "bin_y": [float(ys[ok][which == i].mean()) for i in range(10)]}
    m["concavity"] = conc
    wet = np.array([season_of(x) != "dry" for x in d["doy"]])
    m["observed_fraction"] = {"wet": float(obs[wet].mean()), "dry": float(obs[~wet].mean()),
                              "all": float(obs.mean())}
    return m


# ============================================================================ pipeline
def pipeline(data_dir="data", out=None, crop=None, steps=(800, 800, 2000), log=print):
    import jax
    import jax.numpy as jnp
    import model as M

    out = out or run_dir("pipeline")
    A, B = load("A", data_dir, crop), load("B", data_dir, crop)
    keys = ("a", "obs", "r", "rclim", "x", "phase")
    JA = {k: jnp.asarray(A[k]) for k in keys}
    JB = {k: jnp.asarray(B[k]) for k in keys}
    T = A["a"].shape[0]
    tgt_year = A["year"][1:]
    train_t = jnp.asarray(np.isin(tgt_year, list(aoi.TRAIN_YEARS)))
    val_t = jnp.asarray(np.isin(tgt_year, list(aoi.VAL_YEARS)))
    aoi.assert_allowed_for_training("A", np.unique(tgt_year[np.asarray(train_t) | np.asarray(val_t)]))

    # (use_driver, mlp, sigmoid). M15 is the hand-written S-curve: the fair fight for the Greeks.
    specs = {"M0": (False, False, False), "M1": (True, False, False), "M15": (True, False, True),
             "M2": (True, True, False), "M2nd": (False, True, False)}
    params, metrics = {}, {"fit": {}, "onestep": {}, "rollout": {}, "calibration": {},
                           "greeks": {}, "anomaly": {}}
    for name, (drv, mlp, sig) in specs.items():
        log(f"fit {name}")
        t0 = time.time()
        n = steps[2] if (mlp or sig) else steps[0]
        p, best = M.fit(JA, train_t, drv, mlp, steps=n, val_t=val_t, log=log, sigmoid=sig)
        params[name] = p
        metrics["fit"][name] = {"best_val_nll": best, "seconds": time.time() - t0,
                                "lambda_mean": float(M.lam(p, JA["x"]).mean())}
        if sig:
            metrics["fit"][name]["sigmoid"] = {k: float(v) for k, v in p.items() if k.startswith("sig_") and v.ndim == 0}
            metrics["fit"][name]["sigmoid"]["tau"] = float(jax.nn.softplus(p["sig_ltau"]))

    # ------------------------------------------------------------ one-step scores
    def onestep(p, D, drv, mask_t):
        nll = M.nll_sequence(p, D, drv)
        # recompute mu, sigma for CRPS / MSE / PIT
        w = M.w_path(p, D)
        x = D["x"]

        def body(_, t):
            mu, sig, _w = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], x, drv)
            return None, (mu, sig)

        _, (mu, sig) = jax.lax.scan(body, None, jnp.arange(D["a"].shape[0] - 1))
        y = D["a"][1:]
        m = (D["obs"][1:] & D["obs"][:-1]) & mask_t[:, None, None]
        crps = M.crps_gauss(mu, sig, y)
        z = (y - mu) / sig
        pit = 0.5 * (1 + jax.scipy.special.erf(z / jnp.sqrt(2)))
        return {"nll": nll, "crps": crps, "se": (y - mu) ** 2, "sig": sig, "pit": pit, "mask": m, "z": z}

    evalsets = {"A_test": (JA, jnp.asarray(np.isin(tgt_year, list(aoi.TEST_YEARS)))),
                "B_all": (JB, jnp.ones(B["a"].shape[0] - 1, bool))}
    os_out = {}
    for es, (D, mt) in evalsets.items():
        os_out[es] = {}
        for name, (drv, mlp, sig) in specs.items():
            o = onestep(params[name], D, drv, mt)
            os_out[es][name] = o
            m = o["mask"]
            metrics["onestep"].setdefault(es, {})[name] = {
                k: float(jnp.sum(o[k] * m) / m.sum()) for k in ("nll", "crps", "se")}
        base = metrics["onestep"][es]["M0"]["crps"]
        for name in specs:
            metrics["onestep"][es][name]["crps_skill_vs_M0"] = 1 - metrics["onestep"][es][name]["crps"] / base
        # calibration of M2: spread-skill by sigma decile, PIT histogram
        o = os_out[es]["M2"]
        m = np.asarray(o["mask"])
        sig, se, pit = np.asarray(o["sig"])[m], np.asarray(o["se"])[m], np.asarray(o["pit"])[m]
        q = np.quantile(sig, np.linspace(0, 1, 11))
        b = np.clip(np.digitize(sig, q) - 1, 0, 9)
        spread = [float(np.sqrt(np.mean(sig[b == i] ** 2))) for i in range(10)]
        rmse = [float(np.sqrt(np.mean(se[b == i]))) for i in range(10)]
        metrics["calibration"][es] = {"spread": spread, "rmse": rmse,
                                      "spread_skill_corr": float(np.corrcoef(spread, rmse)[0, 1]),
                                      "pit_hist": np.histogram(pit, bins=10, range=(0, 1))[0].tolist()}
    log(f"one-step: {json.dumps(metrics['onestep'])}")

    # ------------------------------------------------------------ rollouts
    def rollout_crps(p, D, starts, drv, clim_rain, key):
        w = M.w_path(p, D)
        x = D["x"]
        rain = D["rclim"] if clim_rain else D["r"]

        @jax.jit
        def chunk(ts, key):
            def one(t, k):
                rp = jax.lax.dynamic_slice_in_dim(rain, t + 1, H, 0)
                pp = jax.lax.dynamic_slice_in_dim(D["phase"], t + 1, H, 0)
                s = M.rollout_sample(p, k, D["a"][t], w[t], rp, pp, x, drv, S=S)
                y = jax.lax.dynamic_slice_in_dim(D["a"], t + 1, H, 0)
                ob = jax.lax.dynamic_slice_in_dim(D["obs"], t + 1, H, 0) & D["obs"][t][None]
                c = jax.vmap(lambda sh, yh: M.crps_samples(sh, yh), in_axes=(1, 0))(s, y)
                return jnp.sum(c * ob, (1, 2)), jnp.sum(ob, (1, 2))  # [H], [H]

            return jax.vmap(one)(ts, jax.random.split(key, ts.shape[0]))

        num, den = np.zeros(H), np.zeros(H)
        for i in range(0, len(starts), 8):
            ts = jnp.asarray(starts[i:i + 8])
            key, sub = jax.random.split(key)
            n_, d_ = chunk(ts, sub)
            num += np.asarray(n_).sum(0)
            den += np.asarray(d_).sum(0)
        return num / den

    key = jax.random.PRNGKey(1)
    roll_specs = {"M0": ("M0", False, False), "M1": ("M1", True, False), "M15": ("M15", True, False),
                  "M2": ("M2", True, False), "M1clim": ("M1", True, True), "M15clim": ("M15", True, True),
                  "M2clim": ("M2", True, True), "M2nd": ("M2nd", False, False)}
    for es, (D, mt) in evalsets.items():
        mt = np.asarray(mt)
        starts = np.array([t for t in range(D["a"].shape[0] - H) if mt[t]])
        metrics["rollout"][es] = {}
        for rn, (pn, drv, clim) in roll_specs.items():
            c = rollout_crps(params[pn], D, starts, drv, clim, key)
            metrics["rollout"][es][rn] = {"crps": c.tolist()}
        base = np.array(metrics["rollout"][es]["M0"]["crps"])
        for rn in roll_specs:
            c = np.array(metrics["rollout"][es][rn]["crps"])
            metrics["rollout"][es][rn]["skill_vs_M0"] = (1 - c / base).tolist()
        log(f"rollout {es}: " + json.dumps({k: [round(v, 3) for v in metrics['rollout'][es][k]['skill_vs_M0']] for k in roll_specs}))

    # ------------------------------------------------------------ Greeks (Box A test starts)
    p2 = params["M2"]
    wA = M.w_path(p2, JA)
    testA = np.asarray(evalsets["A_test"][1])
    starts = [t for t in range(T - H) if testA[t]]
    gk = jax.jit(lambda t: M.greeks(p2, JA["a"][t], wA[t],
                                    jax.lax.dynamic_slice_in_dim(JA["r"], t + 1, H, 0),
                                    jax.lax.dynamic_slice_in_dim(JA["phase"], t + 1, H, 0), JA["x"]))
    acc = {s: {k: 0 for k in ("delta", "gamma", "theta")} for s in ("MAM", "OND", "dry")}
    cnt = {s: 0 for s in acc}
    for t in starts:
        g = gk(jnp.asarray(t))
        s = season_of(A["doy"][t + 1])
        for k in acc[s]:
            acc[s][k] = acc[s][k] + np.asarray(g[k])
        cnt[s] += 1
    arrays = {}
    for s in acc:
        for k in acc[s]:
            arrays[f"{k}_{s}"] = (acc[s][k] / max(cnt[s], 1)).astype(np.float32)
            metrics["greeks"][f"{k}_{s}_gridmean_by_h"] = arrays[f"{k}_{s}"].mean((1, 2)).tolist()
    metrics["greeks"]["n_starts"] = cnt
    # linear-model delta for comparison (M1)
    p1 = params["M1"]
    w1 = M.w_path(p1, JA)
    t = starts[0]
    g1 = M.greeks(p1, JA["a"][t], w1[t], JA["r"][t + 1:t + 1 + H], JA["phase"][t + 1:t + 1 + H], JA["x"])
    metrics["greeks"]["M1_delta_gridmean_by_h"] = np.asarray(g1["delta"]).mean((1, 2)).tolist()
    # M1.5 Greeks averaged over the same MAM test starts as M2, for a like-for-like delta map
    p15 = params["M15"]
    w15 = M.w_path(p15, JA)
    gk15 = jax.jit(lambda t: M.greeks(p15, JA["a"][t], w15[t],
                                      jax.lax.dynamic_slice_in_dim(JA["r"], t + 1, H, 0),
                                      jax.lax.dynamic_slice_in_dim(JA["phase"], t + 1, H, 0), JA["x"]))
    acc15 = {k: 0 for k in ("delta", "gamma", "theta")}
    mam = [t for t in starts if season_of(A["doy"][t + 1]) == "MAM"]
    for t in mam:
        g = gk15(jnp.asarray(t))
        for k in acc15:
            acc15[k] = acc15[k] + np.asarray(g[k])
    for k in acc15:
        arrays[f"M15_{k}_MAM"] = (acc15[k] / max(len(mam), 1)).astype(np.float32)
        metrics["greeks"][f"M15_{k}_MAM_gridmean_by_h"] = arrays[f"M15_{k}_MAM"].mean((1, 2)).tolist()
    metrics["greeks"]["delta_MAM_h2_corr_M2_vs_M15"] = float(np.corrcoef(arrays["delta_MAM"][1].ravel(), arrays["M15_delta_MAM"][1].ravel())[0, 1])
    arrays["x"] = A["x"]
    arrays["clim_mean"] = np.nanmean(A["clim"], 0)

    # ------------------------------------------------------------ anomaly: fire detection
    if "burn" in A:
        burn = A["burn"][1:]  # aligned with targets
        yr = tgt_year
        test_m = np.isin(yr, list(aoi.TEST_YEARS))
        zc = np.asarray(A["a"][1:]) / np.maximum(A["cstd"][A["idx"][1:]], 1e-3)
        scores = {"zscore": np.abs(zc),
                  "M1": np.asarray(os_out["A_test"]["M1"]["nll"]),
                  "M15": np.asarray(os_out["A_test"]["M15"]["nll"]),
                  "M2": np.asarray(os_out["A_test"]["M2"]["nll"])}
        # one-sided (drop-only) variants, added after pre-registration and labelled as such:
        # -log P(a <= observed) under the model; z-score: -z (browner than usual)
        from scipy.stats import norm
        scores["zscore_1s"] = -zc
        for nm in ("M1", "M15", "M2"):
            scores[f"{nm}_1s"] = -norm.logcdf(np.asarray(os_out["A_test"][nm]["z"]))
        obs = np.asarray(os_out["A_test"]["M2"]["mask"])
        # event: burned in window t; score = max over t..t+2; negatives: no burn within +-3
        near = np.zeros_like(burn)
        for k in range(-3, 4):
            near |= np.roll(burn, k, axis=0)
        ev_t, ev_i, ev_j = np.where(burn & test_m[:, None, None])
        neg = (~near) & obs & test_m[:, None, None]
        hi = np.isin(yr, [2021, 2022]) & np.array([(season_of(x) == "OND" and y == 2021) or (season_of(x) == "MAM" and y == 2022)
                                                   for x, y in zip(A["doy"][1:], yr)])
        for name, sc in scores.items():
            sc = np.where(obs, sc, -np.inf)
            neg_sc = sc[neg]
            thr = np.quantile(neg_sc, 0.95)
            ev_sc = np.stack([sc[np.clip(ev_t + k, 0, len(sc) - 1), ev_i, ev_j] for k in range(3)], 0)
            hit = (ev_sc > thr).any(0)
            lag = np.argmax(ev_sc > thr, 0)[hit]
            far_hi = float(np.mean(sc[neg & hi[:, None, None]] > thr))
            # AUC: rank-based over events (max score) vs negatives
            ev_max = ev_sc.max(0)
            ranks = np.argsort(np.argsort(np.concatenate([ev_max, neg_sc])))
            auc = float((ranks[:len(ev_max)].sum() - len(ev_max) * (len(ev_max) - 1) / 2) / (len(ev_max) * len(neg_sc)))
            metrics["anomaly"][name] = {"n_events": int(len(ev_max)), "hit_rate_at_5pct_far": float(hit.mean()),
                                        "mean_lag_steps": float(lag.mean()) if hit.any() else None,
                                        "far_in_drought_seasons": far_hi, "auc": auc}
        log(f"anomaly: {json.dumps(metrics['anomaly'])}")

    np.savez_compressed(f"{out}/arrays.npz", **arrays)
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    np.save(f"{out}/params.npy", {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in params.items()}, allow_pickle=True)
    log(f"wrote {out}")
    return metrics, arrays, params


if __name__ == "__main__":
    stage = sys.argv[1]
    crop = int(sys.argv[sys.argv.index("--crop") + 1]) if "--crop" in sys.argv else None
    if stage == "audit":
        out = run_dir("audit")
        m = audit(load("A", crop=crop))
        with open(f"{out}/metrics.json", "w") as f:
            json.dump(m, f, indent=1)
        print(json.dumps(m, indent=1))
        print("wrote", out)
    elif stage == "pipeline":
        pipeline(crop=crop, steps=(200, 200, 300) if crop else (800, 800, 2000))
