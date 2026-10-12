"""Driver ablation, follow-ups on the saved fits (no refit).

(1) Non-stationarity check (post hoc): score Box B separately on the climatology-period years
    2003-2018 and the warm held-out years 2019-2025, for M0, M2 (rain), rain+temp, rain+all.
(2) D3 (pre-registered, reported): corr of the rain-only and rain+all implied residuals on
    held-out rainy-season pixel-steps; the 13 Box A season counterfactuals under each.

  .venv/bin/python revision/drivers_eval.py   -> runs/20261006_1802_drivers/eval.json
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from run import load, season_of, H, S  # noqa: E402
from residual import season_starts  # noqa: E402

R = "runs/20261006_1802_drivers"
REF = "runs/20260914_1708_pipeline_h100_fairfight"
EXTRA = {"rain_temp": ["tmean"], "rain_all": ["tmean", "pet", "vpd"]}
ORIG_F, ORIG_N = M.features, M.N_FEAT


def use(variant):
    """Point model.features at the variant's feature layout (None = the paper's)."""
    if variant is None:
        M.features, M.N_FEAT = ORIG_F, ORIG_N
        return

    def features(a, n, w, r, phase, x, use_driver):
        ph, ex = phase
        return jnp.concatenate([ORIG_F(a, n, w, r, ph, x, True), ex], -1)
    M.features, M.N_FEAT = features, ORIG_N + len(EXTRA[variant])


def data(box):
    d = load(box)
    e5 = np.load(f"data/{box}/era5.npz")
    D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    ex = {v: jnp.asarray(np.stack([e5[k] - e5[k + "_clim"] for k in EXTRA[v]], -1).astype(np.float32)) for v in EXTRA}
    return d, D, ex


def params():
    P = np.load(f"{REF}/params.npy", allow_pickle=True).item()
    out = {"M0": (None, False, P["M0"]), "M2": (None, True, P["M2"])}
    for v in EXTRA:
        out[v] = (v, True, np.load(f"{R}/params_{v}.npy", allow_pickle=True).item())
    return {k: (v, drv, {kk: jnp.asarray(vv) for kk, vv in p.items()}) for k, (v, drv, p) in out.items()}


def ph_of(D, ex, variant, sl):
    return D["phase"][sl] if variant is None else (D["phase"][sl], ex[variant][sl])


def score(name, variant, drv, p, D, ex, t_sets):
    """One-step CRPS and h=6 rollout CRPS for each set of target/start indices."""
    use(variant)
    w = M.w_path(p, D)
    T = D["a"].shape[0]

    def body(_, t):
        mu, sig, _w = M.step(p, D["a"][t], w[t], D["r"][t + 1], ph_of(D, ex, variant, t + 1), D["x"], drv)
        return None, (mu, sig)
    _, (mu, sig) = jax.lax.scan(body, None, jnp.arange(T - 1))
    cr = np.asarray(M.crps_gauss(mu, sig, D["a"][1:]))
    ok = np.asarray(D["obs"][1:] & D["obs"][:-1])

    @jax.jit
    def chunk(ts, key):
        def one(t, k):
            rp = jax.lax.dynamic_slice_in_dim(D["r"], t + 1, H, 0)
            pp = jax.lax.dynamic_slice_in_dim(D["phase"], t + 1, H, 0)
            if variant is not None:
                pp = (pp, jax.lax.dynamic_slice_in_dim(ex[variant], t + 1, H, 0))
            s = M.rollout_sample(p, k, D["a"][t], w[t], rp, pp, D["x"], drv, S=S)
            y = D["a"][t + H]
            ob = D["obs"][t + H] & D["obs"][t]
            c = M.crps_samples(s[:, -1], y)
            return jnp.sum(c * ob), jnp.sum(ob)
        return jax.vmap(one)(ts, jax.random.split(key, ts.shape[0]))

    res = {}
    for tag, ts in t_sets.items():
        m = ok & np.isin(np.arange(T - 1), ts)[:, None, None]
        st = np.array([t for t in ts if t < T - H])
        num = den = 0.0
        key = jax.random.PRNGKey(1)
        for i in range(0, len(st), 8):
            key, sub = jax.random.split(key)
            n_, d_ = chunk(jnp.asarray(st[i:i + 8]), sub)
            num += float(np.asarray(n_).sum())
            den += float(np.asarray(d_).sum())
        res[tag] = {"onestep_crps": float(cr[m].mean()), "h6_crps": num / den}
    print(name, json.dumps(res), flush=True)
    return res


def main():
    P = params()
    out = {"B_by_period": {}, "A_test": {}}
    dB, DB, exB = data("B")
    yrB = dB["year"][1:]
    sets = {"2003-2018": np.where(yrB <= 2018)[0], "2019-2025": np.where(yrB >= 2019)[0]}
    for k, (v, drv, p) in P.items():
        out["B_by_period"][k] = score(k, v, drv, p, DB, exB, sets)
    for tag in sets:
        b1 = out["B_by_period"]["M0"][tag]["onestep_crps"]
        bh = out["B_by_period"]["M0"][tag]["h6_crps"]
        out["B_by_period"][f"skill_{tag}"] = {k: {"onestep": 1 - out["B_by_period"][k][tag]["onestep_crps"] / b1,
                                                  "h6": 1 - out["B_by_period"][k][tag]["h6_crps"] / bh} for k in P}

    # D3: residuals and counterfactuals, rain-only M2 vs rain+all
    dA, DA, exA = data("A")
    resid = {}
    for box, d, D, ex, years in (("A", dA, DA, exA, range(2019, 2026)), ("B", dB, DB, exB, range(2003, 2026))):
        ts = [t for t in range(D["a"].shape[0] - 1)
              if int(d["year"][t + 1]) in years and season_of(d["doy"][t + 1]) in ("MAM", "OND")]
        E = {}
        for k in ("M2", "rain_all"):
            v, drv, p = P[k]
            use(v)
            w = M.w_path(p, D)
            es = []
            for t in ts:
                mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], ph_of(D, ex, v, t + 1), D["x"], True)
                ok = np.asarray(D["obs"][t] & D["obs"][t + 1])
                es.append(np.asarray((D["a"][t + 1] - mu) / sig)[ok])
            E[k] = np.concatenate(es)
        resid[box] = {"corr_resid_M2_vs_rain_all": float(np.corrcoef(E["M2"], E["rain_all"])[0, 1]),
                      "sd": {k: float(E[k].std()) for k in E}}
    out["D3_residual"] = resid

    cf = []
    for (y, s, t0) in season_starts(dA, range(2019, 2026)):
        row = {"season": f"{s}{y}", "obs": float(DA["a"][t0 + H].mean())}
        for k in ("M2", "rain_all"):
            v, drv, p = P[k]
            use(v)
            w = M.w_path(p, DA)
            sl = slice(t0 + 1, t0 + 1 + H)
            a, wv, ac, wc = DA["a"][t0], w[t0], DA["a"][t0], w[t0]
            for h in range(H):
                t = t0 + h
                ph = ph_of(DA, exA, v, t + 1)
                mu, sig, wn = M.step(p, a, wv, DA["r"][t + 1], ph, DA["x"], True)
                e = (DA["a"][t + 1] - mu) / sig
                a, wv = DA["a"][t + 1], wn
                muc, sigc, wc = M.step(p, ac, wc, DA["rclim"][t + 1], ph, DA["x"], True)
                ac = muc + sigc * e
            row[f"cf_{k}"] = float(ac.mean())
        cf.append(row)
        print(row, flush=True)
    out["D3_counterfactual_A"] = cf
    use(None)
    json.dump(out, open(f"{R}/eval.json", "w"), indent=1)
    print(json.dumps(out["B_by_period"]["skill_2003-2018"]), json.dumps(out["B_by_period"]["skill_2019-2025"]), json.dumps(resid))


if __name__ == "__main__":
    main()
