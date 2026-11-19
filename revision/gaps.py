"""Re-score the fair-fight fits with every transition that spans a MODIS gap excluded
(reviewer 1: "steps spanning gaps are not yet excluded ... report sensitivity").

A gap step is t -> t+1 with more than 16 days between composites: 2021-11-01 (61 d),
2023-02-02, 2024-07-27, 2024-12-02 (~30 d), 2025-06-26 (144 d). One-step scores drop those
transitions; rollouts drop starts whose six windows contain one; the anomaly table drops gap
targets. No refit: no gap falls in the training or validation years.

  .venv/bin/python revision/gaps.py   -> runs/<ts>_gaps/metrics.json
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aoi  # noqa: E402
import model as M  # noqa: E402
from run import load, run_dir, season_of, H, S  # noqa: E402

REF = "runs/20260914_1708_pipeline_h100_fairfight"
SPECS = {"M0": False, "M1": True, "M15": True, "M2": True}


def gap_steps(d):
    dt = d["dates"].astype("datetime64[D]")
    return np.concatenate([np.diff(dt).astype(int) > 16, [False]])  # [T]: t -> t+1 spans a gap


def onestep(p, D, drv):
    w = M.w_path(p, D)

    def body(_, t):
        mu, sig, _w = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], D["x"], drv)
        return None, (mu, sig)
    _, (mu, sig) = jax.lax.scan(body, None, jnp.arange(D["a"].shape[0] - 1))
    y = D["a"][1:]
    return np.asarray(M.crps_gauss(mu, sig, y)), np.asarray((y - mu) / sig), np.asarray(M.nll_sequence(p, D, drv))


def rollout(p, D, starts, drv, key):
    w = M.w_path(p, D)

    @jax.jit
    def chunk(ts, key):
        def one(t, k):
            rp = jax.lax.dynamic_slice_in_dim(D["r"], t + 1, H, 0)
            pp = jax.lax.dynamic_slice_in_dim(D["phase"], t + 1, H, 0)
            s = M.rollout_sample(p, k, D["a"][t], w[t], rp, pp, D["x"], drv, S=S)
            y = jax.lax.dynamic_slice_in_dim(D["a"], t + 1, H, 0)
            ob = jax.lax.dynamic_slice_in_dim(D["obs"], t + 1, H, 0) & D["obs"][t][None]
            c = jax.vmap(lambda sh, yh: M.crps_samples(sh, yh), in_axes=(1, 0))(s, y)
            return jnp.sum(c * ob, (1, 2)), jnp.sum(ob, (1, 2))
        return jax.vmap(one)(ts, jax.random.split(key, ts.shape[0]))

    num, den = np.zeros(H), np.zeros(H)
    for i in range(0, len(starts), 8):
        key, sub = jax.random.split(key)
        n_, d_ = chunk(jnp.asarray(starts[i:i + 8]), sub)
        num += np.asarray(n_).sum(0)
        den += np.asarray(d_).sum(0)
    return num / den


def anomaly(A, os_A, test_m, keep_t):
    """Paper's fire protocol (run.py) with gap targets removed from events and negatives."""
    burn = A["burn"][1:]
    yr = A["year"][1:]
    zc = np.asarray(A["a"][1:]) / np.maximum(A["cstd"][A["idx"][1:]], 1e-3)
    scores = {"zscore": np.abs(zc), "zscore_1s": -zc}
    for nm in ("M1", "M15", "M2"):
        scores[nm] = os_A[nm]["nll"]
        scores[f"{nm}_1s"] = -norm.logcdf(os_A[nm]["z"])
    obs = (A["obs"][1:] & A["obs"][:-1]) & keep_t[:, None, None]
    near = np.zeros_like(burn)
    for k in range(-3, 4):
        near |= np.roll(burn, k, axis=0)
    ev_t, ev_i, ev_j = np.where(burn & (test_m & keep_t)[:, None, None])
    neg = (~near) & obs & test_m[:, None, None]
    hi = np.array([(season_of(x) == "OND" and y == 2021) or (season_of(x) == "MAM" and y == 2022)
                   for x, y in zip(A["doy"][1:], yr)])
    out = {}
    for name, sc in scores.items():
        sc = np.where(obs, sc, -np.inf)
        neg_sc = sc[neg]
        thr = np.quantile(neg_sc, 0.95)
        ev_sc = np.stack([sc[np.clip(ev_t + k, 0, len(sc) - 1), ev_i, ev_j] for k in range(3)], 0)
        hit = (ev_sc > thr).any(0)
        ev_max = ev_sc.max(0)
        ranks = np.argsort(np.argsort(np.concatenate([ev_max, neg_sc])))
        auc = float((ranks[:len(ev_max)].sum() - len(ev_max) * (len(ev_max) - 1) / 2) / (len(ev_max) * len(neg_sc)))
        out[name] = {"n_events": int(len(ev_max)), "hit_rate_at_5pct_far": float(hit.mean()),
                     "far_in_drought_seasons": float(np.mean(sc[neg & hi[:, None, None]] > thr)), "auc": auc}
    return out


def main():
    out_dir = run_dir("gaps")
    P = np.load(f"{REF}/params.npy", allow_pickle=True).item()
    params = {k: {kk: jnp.asarray(vv) for kk, vv in v.items()} for k, v in P.items()}
    A, B = load("A"), load("B")
    res = {"gap_steps": [str(A["dates"][t]) for t in np.where(gap_steps(A))[0]], "onestep": {}, "rollout": {}}
    os_A = {}
    for es, d, mt in (("A_test", A, np.isin(A["year"][1:], list(aoi.TEST_YEARS))),
                      ("B_all", B, np.ones(B["a"].shape[0] - 1, bool))):
        D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
        g = gap_steps(d)[:-1]
        base = np.asarray(d["obs"][1:] & d["obs"][:-1]) & mt[:, None, None]
        res["onestep"][es], res["rollout"][es] = {}, {}
        span = np.array([g[t:t + H].any() for t in range(len(g))])
        st_all = np.array([t for t in range(d["a"].shape[0] - H) if mt[t]])
        st_ok = np.array([t for t in st_all if not span[t]])
        res["rollout"][es]["n_starts"] = {"all": int(len(st_all)), "no_gap": int(len(st_ok))}
        for nm, drv in SPECS.items():
            cr, z, nll = onestep(params[nm], D, drv)
            if es == "A_test":
                os_A[nm] = {"z": z, "nll": nll}
            r = {}
            for tag, m in (("all", base), ("no_gap", base & ~g[:, None, None])):
                r[tag] = float(cr[m].mean())
            res["onestep"][es][nm] = r
            for tag, st in (("all", st_all), ("no_gap", st_ok)):
                res["rollout"][es].setdefault(nm, {})[tag] = rollout(params[nm], D, st, drv, jax.random.PRNGKey(1)).tolist()
            print(es, nm, res["onestep"][es][nm], {k: round(v[-1], 4) for k, v in res["rollout"][es][nm].items()}, flush=True)
        for tag in ("all", "no_gap"):
            b1 = res["onestep"][es]["M0"][tag]
            bh = np.array(res["rollout"][es]["M0"][tag])
            res["onestep"][es][f"skill_{tag}"] = {nm: 1 - res["onestep"][es][nm][tag] / b1 for nm in SPECS}
            res["rollout"][es][f"skill_{tag}"] = {nm: (1 - np.array(res["rollout"][es][nm][tag]) / bh).tolist() for nm in SPECS}
    mtA = np.isin(A["year"][1:], list(aoi.TEST_YEARS))
    gA = gap_steps(A)[:-1]
    res["anomaly"] = {"all": anomaly(A, os_A, mtA, np.ones_like(gA)), "no_gap": anomaly(A, os_A, mtA, ~gA)}
    json.dump(res, open(f"{out_dir}/metrics.json", "w"), indent=1)
    for es in ("A_test", "B_all"):
        for tag in ("all", "no_gap"):
            print(es, tag, "one-step skill", {k: round(v, 3) for k, v in res["onestep"][es][f"skill_{tag}"].items()},
                  "h6 skill", {k: round(v[-1], 3) for k, v in res["rollout"][es][f"skill_{tag}"].items()})
    for tag in ("all", "no_gap"):
        print("anomaly", tag, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res["anomaly"][tag].items()})
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
