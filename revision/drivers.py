"""Second weather driver: does temperature (and evaporative demand) add to rain? Reviewer 2.
Also: does CHIRPS-vs-ERA5 rain disagreement leak into the residual? Reviewer 1.

Pre-registration: runs/<ts>_drivers/PREREG.md (written before any fit).

  .venv/bin/python revision/drivers.py build             -> data/<box>/era5.npz (16-day windows on the grid)
  .venv/bin/python revision/drivers.py rainerr           -> runs/20261006_1753_identify/rain_error.json
  .venv/bin/python revision/drivers.py fit <variant> <run dir>
        variant: temp (T only), rain_temp (rain + T), rain_all (rain + T + PET + VPD)
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aoi  # noqa: E402
from run import load, H, S  # noqa: E402

RAW = "data/era5/raw"
VARS = {"temperature_2m_mean": ("tmean", "mean"), "et0_fao_evapotranspiration": ("pet", "sum"),
        "vapour_pressure_deficit_max": ("vpd", "mean"), "precipitation_sum": ("era5_rain", "sum")}
SCALE = {"tmean": 1.0, "pet": 100.0, "vpd": 1.0, "era5_rain": 100.0}  # deg C; 100 mm; kPa; 100 mm
REF = "runs/20260914_1708_pipeline_h100_fairfight"


# ============================================================================ build
def build():
    from scipy.interpolate import RegularGridInterpolator
    files = sorted(glob.glob(f"{RAW}/*.json"))
    lats = sorted({float(os.path.basename(f).split("_")[0]) for f in files})
    lons = sorted({float(os.path.basename(f).split("_")[1]) for f in files})
    days = np.arange(np.datetime64("2002-01-01"), np.datetime64("2026-01-01"))
    cube = {v: np.full((len(days), len(lats), len(lons)), np.nan, np.float32) for v in VARS}
    for f in files:
        la, lo = (float(s) for s in os.path.basename(f).split("_")[:2])
        j = json.load(open(f))["daily"]
        t = np.array(j["time"], dtype="datetime64[D]")
        ii = (t - days[0]).astype(int)
        for v in VARS:
            cube[v][ii, lats.index(la), lons.index(lo)] = np.array([np.nan if x is None else x for x in j[v]], np.float32)
    for v in VARS:
        print(v, "missing fraction", float(np.isnan(cube[v]).mean()))
    for box in ("A", "B"):
        d = dict(np.load(f"data/{box}/built.npz"))
        dates = d["dates"].astype("datetime64[D]")
        w, s, e, n = aoi.BOXES[box]
        plat = n - (np.arange(aoi.N) + 0.5) * aoi.RES
        plon = w + (np.arange(aoi.N) + 0.5) * aoi.RES
        LA, LO = np.meshgrid(np.clip(plat, lats[0], lats[-1]), np.clip(plon, lons[0], lons[-1]), indexing="ij")
        pts = np.stack([LA.ravel(), LO.ravel()], 1)
        out = {}
        doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
        idx = (doy - 1) // 16
        year = dates.astype("datetime64[Y]").astype(int) + 1970
        for v, (name, how) in VARS.items():
            win = np.zeros((len(dates), aoi.N, aoi.N), np.float32)
            for k, s0 in enumerate(dates):
                sel = (days >= s0) & (days < s0 + 16)
                agg = np.nanmean(cube[v][sel], 0) if how == "mean" else np.nansum(cube[v][sel], 0)
                win[k] = RegularGridInterpolator((lats, lons), agg)(pts).reshape(aoi.N, aoi.N)
            win /= SCALE[name]
            clim = np.zeros_like(win)
            for q in range(23):
                m = np.isin(year, list(aoi.CLIM_YEARS)) & (idx == q)
                clim[idx == q] = win[m].mean(0)
            out[name] = win
            out[name + "_clim"] = clim
        np.savez_compressed(f"data/{box}/era5.npz", dates=d["dates"], **out)
        print(box, {k: float(np.mean(v)) for k, v in out.items()})


# ============================================================================ rain error vs residual
def rainerr():
    """Residual against CHIRPS-minus-ERA5 window rain (neither is truth; disagreement is a proxy
    for rain-product error). Held-out rainy-season pixel-steps, Box A 2019-25 and Box B."""
    import jax.numpy as jnp
    import model as M
    from run import season_of
    P = np.load(f"{REF}/params.npy", allow_pickle=True).item()
    p = {k: jnp.asarray(v) for k, v in P["M2"].items()}
    out = {}
    for box, years in (("A", range(2019, 2026)), ("B", range(2003, 2026))):
        d = load(box)
        e5 = np.load(f"data/{box}/era5.npz")
        D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
        w = M.w_path(p, D)
        E, DIF, DEP = [], [], []
        for t in range(d["a"].shape[0] - 1):
            if int(d["year"][t + 1]) not in years or season_of(d["doy"][t + 1]) not in ("MAM", "OND"):
                continue
            mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], D["x"], True)
            ok = np.asarray(D["obs"][t] & D["obs"][t + 1])
            E.append(np.asarray((D["a"][t + 1] - mu) / sig)[ok])
            DIF.append(((d["r"][t + 1] - e5["era5_rain"][t + 1]) * 100)[ok])
            DEP.append(((d["r"][t + 1] - d["rclim"][t + 1]) * 100)[ok])
        E, DIF, DEP = map(np.concatenate, (E, DIF, DEP))
        q = np.quantile(DIF, [0, .1, .25, .5, .75, .9, 1])
        b = np.clip(np.digitize(DIF, q[1:-1]), 0, 5)
        # partial: residual on disagreement controlling for CHIRPS departure (linear)
        X = np.stack([np.ones_like(DIF), DIF / 100, DEP / 100], 1)
        coef = np.linalg.lstsq(X, E, rcond=None)[0]
        out[box] = {"corr_resid_chirps_minus_era5": float(np.corrcoef(E, DIF)[0, 1]),
                    "corr_resid_abs_disagreement": float(np.corrcoef(E, np.abs(DIF))[0, 1]),
                    "disagreement_mm_quantiles": q.tolist(),
                    "mean_resid_by_disagreement_sextile": [float(E[b == k].mean()) for k in range(6)],
                    "sd_resid_by_disagreement_sextile": [float(E[b == k].std()) for k in range(6)],
                    "ols_resid_on_[1,disagree/100mm,chirps_dep/100mm]": coef.tolist(),
                    "corr_chirps_era5_window_rain": None}
        r1 = d["r"][1:].ravel() * 100
        r2 = e5["era5_rain"][1:].ravel() * 100
        out[box]["corr_chirps_era5_window_rain"] = float(np.corrcoef(r1, r2)[0, 1])
        print(box, json.dumps(out[box]), flush=True)
    json.dump(out, open("runs/20261006_1753_identify/rain_error.json", "w"), indent=1)


# ============================================================================ fit with extra drivers
def fit_variant(variant, out):
    """M2 with extra per-window driver anomalies appended to the feature vector. The fit,
    early stopping, and scoring follow run.py exactly; only the features change."""
    import jax
    import jax.numpy as jnp
    import model as M

    extra = {"temp": ["tmean"], "rain_temp": ["tmean"], "rain_all": ["tmean", "pet", "vpd"]}[variant]
    use_rain = variant != "temp"
    nx = len(extra)

    orig_features, orig_N = M.features, M.N_FEAT

    def load_x(box):
        d = load(box)
        e5 = np.load(f"data/{box}/era5.npz")
        assert (e5["dates"] == d["dates"]).all()
        d["ex"] = np.stack([e5[v] - e5[v + "_clim"] for v in extra], -1).astype(np.float32)  # [T,N,N,nx]
        return d

    A, B = load_x("A"), load_x("B")
    keys = ("a", "obs", "r", "rclim", "x", "phase", "ex")
    JA = {k: jnp.asarray(A[k]) for k in keys}
    JB = {k: jnp.asarray(B[k]) for k in keys}

    # Features: the M2 layout (a, n, phase, x, w, r) followed by the extra anomalies of window t+1.
    # The extra block is passed through `phase` as a pytree so step/rollout signatures are unchanged.
    def features(a, n, w, r, phase, x, use_driver):
        ph, ex = phase
        f = orig_features(a, n, w, r, ph, x, use_rain)
        return jnp.concatenate([f, ex], -1)

    M.features = features
    M.N_FEAT = orig_N + nx

    def wrap(D):
        """phase[t] -> (phase[t], ex[t]) so every existing indexing of D['phase'] carries the extras."""
        class P:
            def __init__(s, ph, ex):
                s.ph, s.ex = ph, ex

            def __getitem__(s, i):
                return (s.ph[i], s.ex[i])
        D2 = dict(D)
        D2["phase"] = P(D["phase"], D["ex"])
        return D2

    tgt = A["year"][1:]
    train_t = jnp.asarray(np.isin(tgt, list(aoi.TRAIN_YEARS)))
    val_t = jnp.asarray(np.isin(tgt, list(aoi.VAL_YEARS)))
    aoi.assert_allowed_for_training("A", np.unique(tgt[np.asarray(train_t) | np.asarray(val_t)]))
    t0 = time.time()
    logf = open(f"{out}/progress_{variant}.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"fit {variant}: extra {extra}, rain {use_rain}")
    p, best = M.fit(wrap(JA), train_t, True, True, steps=2000, val_t=val_t, log=log)
    log(f"fit done {time.time() - t0:.0f}s best val nll {best:.4f}")
    np.save(f"{out}/params_{variant}.npy", {k: np.asarray(v) for k, v in p.items()}, allow_pickle=True)

    ref = json.load(open(f"{REF}/metrics.json"))
    res = {"variant": variant, "extra": extra, "rain": use_rain, "best_val_nll": best}
    for es, D, mt in (("A_test", JA, np.isin(tgt, list(aoi.TEST_YEARS))), ("B_all", JB, np.ones(B["a"].shape[0] - 1, bool))):
        Dw = wrap(D)
        w = M.w_path(p, D)

        def body(_, t):
            mu, sig, _w = M.step(p, D["a"][t], w[t], D["r"][t + 1], (D["phase"][t + 1], D["ex"][t + 1]), D["x"], True)
            return None, (mu, sig)
        _, (mu, sig) = jax.lax.scan(body, None, jnp.arange(D["a"].shape[0] - 1))
        m = np.asarray(D["obs"][1:] & D["obs"][:-1]) & mt[:, None, None]
        cr = np.asarray(M.crps_gauss(mu, sig, D["a"][1:]))
        nll = np.asarray(M.nll_sequence(p, Dw, True))
        one = float(cr[m].mean())

        @jax.jit
        def chunk(ts, key):
            def one_(t, k):
                rp = jax.lax.dynamic_slice_in_dim(D["r"], t + 1, H, 0)
                pp = (jax.lax.dynamic_slice_in_dim(D["phase"], t + 1, H, 0), jax.lax.dynamic_slice_in_dim(D["ex"], t + 1, H, 0))
                s = M.rollout_sample(p, k, D["a"][t], w[t], rp, pp, D["x"], True, S=S)
                y = jax.lax.dynamic_slice_in_dim(D["a"], t + 1, H, 0)
                ob = jax.lax.dynamic_slice_in_dim(D["obs"], t + 1, H, 0) & D["obs"][t][None]
                c = jax.vmap(lambda sh, yh: M.crps_samples(sh, yh), in_axes=(1, 0))(s, y)
                return jnp.sum(c * ob, (1, 2)), jnp.sum(ob, (1, 2))
            return jax.vmap(one_)(ts, jax.random.split(key, ts.shape[0]))

        starts = np.array([t for t in range(D["a"].shape[0] - H) if mt[t]])
        num, den, key = np.zeros(H), np.zeros(H), jax.random.PRNGKey(1)
        for i in range(0, len(starts), 8):
            key, sub = jax.random.split(key)
            n_, d_ = chunk(jnp.asarray(starts[i:i + 8]), sub)
            num += np.asarray(n_).sum(0)
            den += np.asarray(d_).sum(0)
        roll = num / den
        m0_1, m0_h = ref["onestep"][es]["M0"]["crps"], np.array(ref["rollout"][es]["M0"]["crps"])
        res[es] = {"onestep_crps": one, "onestep_nll": float(nll[m].mean()), "onestep_skill_vs_M0": 1 - one / m0_1,
                   "rollout_crps": roll.tolist(), "rollout_skill_vs_M0": (1 - roll / m0_h).tolist()}
        log(f"{variant} {es}: one-step skill {res[es]['onestep_skill_vs_M0']:.3f} nll {res[es]['onestep_nll']:.3f}; "
            "rollout skill " + " ".join(f"{v:.3f}" for v in res[es]["rollout_skill_vs_M0"]))
    json.dump(res, open(f"{out}/metrics_{variant}.json", "w"), indent=1)
    log(f"wrote {out}/metrics_{variant}.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "rainerr":
        rainerr()
    elif cmd == "fit":
        fit_variant(sys.argv[2], sys.argv[3])
