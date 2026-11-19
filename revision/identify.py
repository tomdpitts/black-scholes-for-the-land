"""Is the residual counterfactual identified? Semi-synthetic tests plus real-data diagnostics.

Pre-registration: runs/20261006_1753_identify/PREREG.md (written before this ran).

  .venv/bin/python revision/identify.py [out dir]   -> <out>/metrics.json, <out>/RESULT.md (by hand)
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aoi  # noqa: E402
import model as M  # noqa: E402
from run import load, season_of, H  # noqa: E402
from residual import season_starts  # noqa: E402

RUN = "runs/20260914_1708_pipeline_h100_fairfight"
OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/20261006_1753_identify"
DELTA = 0.01
BINS = [-np.inf, -60, -30, 0, 30, 60, 120, np.inf]
BIN_NAMES = ["<-60", "-60..-30", "-30..0", "0..30", "30..60", "60..120", ">120"]


def setup(box):
    d = load(box)
    P = np.load(f"{RUN}/params.npy", allow_pickle=True).item()
    p = {k: jnp.asarray(v) for k, v in P["M2"].items()}
    D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    return d, p, D, M.w_path(p, D)


def make_fns(p, x):
    def step(a, w, r, ph):
        return M.step(p, a, w, r, ph, x, True)

    @jax.jit
    def truth(a0, w0, r_path, rc_path, ph_path, e_path, mask, k, delta):
        """True world: M2 + management m_t = -delta * clip(1 + k (rclim - r)/rclim, 0, 3) on mask."""
        def body(c, inp):
            a, w = c
            r, rc, ph, e = inp
            mean, sig, w = step(a, w, r, ph)
            m = -delta * jnp.clip(1 + k * (rc - r) / jnp.maximum(rc, 1e-3), 0, 3) * mask
            a = mean + sig * e + m
            return (a, w), a
        return jax.lax.scan(body, (a0, w0), (r_path, rc_path, ph_path, e_path))[1]

    @jax.jit
    def invert(a0, w0, y, r_path, ph_path):
        """Implied standardised and additive residuals from an observed path y [H,N,N]."""
        def body(c, inp):
            a, w = c
            yy, r, ph = inp
            mean, sig, w = step(a, w, r, ph)
            return (yy, w), ((yy - mean) / sig, yy - mean)
        return jax.lax.scan(body, (a0, w0), (y, r_path, ph_path))[1]

    @jax.jit
    def cf(a0, w0, r_path, ph_path, res, additive):
        def body(c, inp):
            a, w = c
            r, ph, e = inp
            mean, sig, w = step(a, w, r, ph)
            a = mean + jnp.where(additive, e, sig * e)
            return (a, w), a
        return jax.lax.scan(body, (a0, w0), (r_path, ph_path, res))[1]

    @jax.jit
    def implied(a, a1, w, r1, ph1):
        mean, sig, _ = step(a, w, r1, ph1)
        return (a1 - mean) / sig, a1 - mean, sig

    return truth, invert, cf, implied


def block_mask(N, size, rng):
    nb = N // size
    blocks = rng.random((nb, nb)) < 0.5
    return np.kron(blocks, np.ones((size, size))).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    d, p, D, w = setup("A")
    x, N = D["x"], D["a"].shape[1]
    truth, invert, cf, implied = make_fns(p, x)
    rng = np.random.default_rng(0)
    mask = jnp.asarray(block_mask(N, 8, rng))
    mb = np.asarray(mask).astype(bool)
    dt = d["dates"].astype("datetime64[D]")
    gap_step = np.concatenate([np.diff(dt).astype(int) > 16, [False]])  # step t -> t+1 spans a gap

    seasons = season_starts(d, range(2019, 2026))
    out = {"run": RUN, "n_managed_px": int(mb.sum()), "seasons": [], "synthetic": {}}
    scen = {"S0": (0.0, 0.0), "S1": (0.0, DELTA), "S2a_incursion": (1.0, DELTA), "S2b_destock": (-1.0, DELTA)}
    rows = {s: [] for s in list(scen) + ["S3_s0.25", "S3_s0.5"]}

    for (y, s, t0) in seasons:
        sl = slice(t0 + 1, t0 + 1 + H)
        a0, w0 = D["a"][t0], w[t0]
        r, rc, ph = D["r"][sl], D["rclim"][sl], D["phase"][sl]
        e_real = jnp.stack([implied(D["a"][t], D["a"][t + 1], w[t], D["r"][t + 1], D["phase"][t + 1])[0]
                            for t in range(t0, t0 + H)])
        rain_dep = float((r - rc).mean((1, 2)).sum() * 100)
        drought = rain_dep < -60
        gaps = int(gap_step[t0:t0 + H].sum())
        out["seasons"].append({"year": y, "season": s, "rain_departure_mm": rain_dep, "gap_steps": gaps})

        for name, (k, delta) in scen.items():
            yobs = truth(a0, w0, r, rc, ph, e_real, mask, k, delta)
            cf_true = truth(a0, w0, rc, rc, ph, e_real, mask, k, delta)[-1]
            cf_true0 = truth(a0, w0, rc, rc, ph, e_real, mask, k, 0.0)[-1]
            eh, ad = invert(a0, w0, yobs, r, ph)
            row = {"year": y, "season": s, "rain_departure_mm": rain_dep, "drought": drought}
            eff = float((cf_true - cf_true0)[mb].mean())
            row["true_mgmt_effect_normal_rain"] = eff
            for en, add, res in (("E1_standardised", False, eh), ("E2_additive", True, ad)):
                err = np.asarray(cf(a0, w0, rc, ph, res, add)[-1] - cf_true)
                row[en] = {"boxmean_err_managed": float(err[mb].mean()), "boxmean_err_unmanaged": float(err[~mb].mean()),
                           "px_rmse": float(np.sqrt((err ** 2).mean())),
                           "rel_err": float(err[mb].mean() / eff) if abs(eff) > 1e-6 else None}
            rows[name].append(row)

        # S3: rain-product error, no management; the world runs on r_true, the estimator sees r
        for sd in (0.25, 0.5):
            errs = {"E1_standardised": [], "E2_additive": []}
            for seed in range(5):
                g = np.random.default_rng(1000 + seed + 17 * t0)
                eta = g.normal(0, sd, (H, N // 20, N // 20))
                eta = np.kron(eta, np.ones((1, 20, 20)))
                r_true = r * jnp.asarray(np.exp(eta - sd ** 2 / 2), jnp.float32)
                yobs = truth(a0, w0, r_true, rc, ph, e_real, mask, 0.0, 0.0)
                cf_true = truth(a0, w0, rc, rc, ph, e_real, mask, 0.0, 0.0)[-1]
                eh, ad = invert(a0, w0, yobs, r, ph)
                for en, add, res in (("E1_standardised", False, eh), ("E2_additive", True, ad)):
                    errs[en].append(np.asarray(cf(a0, w0, rc, ph, res, add)[-1] - cf_true))
            row = {"year": y, "season": s, "rain_departure_mm": rain_dep, "drought": drought}
            for en, ee in errs.items():
                ee = np.stack(ee)
                row[en] = {"boxmean_err": float(ee.mean((1, 2)).mean()), "boxmean_err_seed_sd": float(ee.mean((1, 2)).std()),
                           "px_rmse": float(np.sqrt((ee ** 2).mean()))}
            rows[f"S3_s{sd}"].append(row)
        print(f"{s}{y} rain {rain_dep:+.0f} mm gaps {gaps}", flush=True)

    # summaries by season class
    for name, rr in rows.items():
        summ = {}
        for cls, sel in (("drought", [r_["drought"] for r_ in rr]), ("other", [not r_["drought"] for r_ in rr])):
            sub = [r_ for r_, k in zip(rr, sel) if k]
            summ[cls] = {"n": len(sub)}
            for en in ("E1_standardised", "E2_additive"):
                key = "boxmean_err_managed" if name[:2] != "S3" else "boxmean_err"
                be = [r_[en][key] for r_ in sub]
                summ[cls][en] = {"mean_boxmean_err": float(np.mean(be)), "max_abs_boxmean_err": float(np.max(np.abs(be))),
                                 "mean_px_rmse": float(np.mean([r_[en]["px_rmse"] for r_ in sub]))}
                if name[:2] != "S3" and name != "S0":
                    rel = [r_[en]["rel_err"] for r_ in sub]
                    summ[cls][en]["mean_rel_err"] = float(np.mean(rel))
                    summ[cls][en]["max_abs_rel_err"] = float(np.max(np.abs(rel)))
            if name[:2] != "S3":
                summ[cls]["mean_true_effect"] = float(np.mean([r_["true_mgmt_effect_normal_rain"] for r_ in sub]))
        out["synthetic"][name] = {"summary": summ, "rows": rr}

    # ------------------------------------------------------------ real-data diagnostics
    out["real"] = {}
    rel_flag = None
    try:
        m = np.load("data/A/modis.npz")
        md = m["dates"].astype("datetime64[D]")
        pos = {v: i for i, v in enumerate(md)}
        rel_flag = np.stack([m["rel"][pos[v]] for v in dt])  # [T,N,N] aligned with built.npz
    except Exception as ex:  # noqa: BLE001
        print("reliability flags unavailable:", ex)

    for box in ("A", "B"):
        db, pb, Db, wb = (d, p, D, w) if box == "A" else setup("B")
        _, _, _, impl_b = make_fns(pb, Db["x"])
        years = range(2019, 2026) if box == "A" else range(2003, 2026)
        ts = [t for t in range(Db["a"].shape[0] - 1)
              if int(db["year"][t + 1]) in years and season_of(db["doy"][t + 1]) in ("MAM", "OND")]
        E, DEP, YR, GAP, REL = [], [], [], [], []
        for t in ts:
            e, _, _ = impl_b(Db["a"][t], Db["a"][t + 1], wb[t], Db["r"][t + 1], Db["phase"][t + 1])
            ok = np.asarray(Db["obs"][t] & Db["obs"][t + 1])
            E.append(np.asarray(e)[ok])
            DEP.append(np.asarray((Db["r"][t + 1] - Db["rclim"][t + 1]) * 100)[ok])
            YR.append(np.full(ok.sum(), int(db["year"][t + 1])))
            GAP.append(np.full(ok.sum(), bool(np.diff(db["dates"].astype("datetime64[D]")[t:t + 2]).astype(int)[0] > 16)))
            if box == "A" and rel_flag is not None:
                REL.append(rel_flag[t + 1][ok])
        E, DEP, YR, GAP = map(np.concatenate, (E, DEP, YR, GAP))
        which = np.digitize(DEP, BINS[1:-1])
        uy = np.unique(YR)
        boot = np.random.default_rng(7)

        def bin_means(sel):
            return [float(E[sel & (which == b)].mean()) if (sel & (which == b)).any() else np.nan for b in range(len(BIN_NAMES))]

        allsel = np.ones_like(E, bool)
        draws = []
        idx_by_year = {yy: np.where(YR == yy)[0] for yy in uy}
        for _ in range(2000):
            ys = boot.choice(uy, len(uy))
            ii = np.concatenate([idx_by_year[yy] for yy in ys])
            e_, w_ = E[ii], which[ii]
            draws.append([e_[w_ == b].mean() if (w_ == b).any() else np.nan for b in range(len(BIN_NAMES))])
        draws = np.array(draws)
        res = {"n_px_steps": int(len(E)), "corr_eps_rain": float(np.corrcoef(E, DEP)[0, 1]),
               "bins": BIN_NAMES, "mean_eps_by_bin": bin_means(allsel),
               "ci95_by_bin": [[float(np.nanpercentile(draws[:, b], 2.5)), float(np.nanpercentile(draws[:, b], 97.5))]
                               for b in range(len(BIN_NAMES))],
               "n_by_bin": [int((which == b).sum()) for b in range(len(BIN_NAMES))],
               "mean_eps_by_bin_no_gap_steps": bin_means(~GAP), "n_gap_px_steps": int(GAP.sum())}
        if REL:
            R = np.concatenate(REL)
            res["eps_by_reliability"] = {str(v): {"n": int((R == v).sum()), "mean": float(E[R == v].mean()),
                                                  "sd": float(E[R == v].std())} for v in (0, 1)}
        out["real"][box] = res
        print(box, json.dumps({k: v for k, v in res.items() if k != "bins"}), flush=True)

    # E1 vs E2 on the real seasons: assumption-sensitivity band for the paper's numbers
    real_cf = []
    for (y, s, t0) in seasons:
        sl = slice(t0 + 1, t0 + 1 + H)
        a0, w0 = D["a"][t0], w[t0]
        r, rc, ph, yobs = D["r"][sl], D["rclim"][sl], D["phase"][sl], D["a"][sl]
        eh, ad = invert(a0, w0, yobs, r, ph)
        c1 = float(cf(a0, w0, rc, ph, eh, False)[-1].mean())
        c2 = float(cf(a0, w0, rc, ph, ad, True)[-1].mean())
        real_cf.append({"year": y, "season": s, "obs_h6": float(yobs[-1].mean()), "cf_E1": c1, "cf_E2": c2,
                        "rain_departure_mm": float((r - rc).mean((1, 2)).sum() * 100),
                        "gap_steps": int(gap_step[t0:t0 + H].sum())})
        print(f"real {s}{y}: obs {real_cf[-1]['obs_h6']:+.3f} cf E1 {c1:+.3f} E2 {c2:+.3f}", flush=True)
    out["real"]["cf_E1_vs_E2"] = real_cf

    json.dump(out, open(f"{OUT}/metrics.json", "w"), indent=1)
    for name in out["synthetic"]:
        print(name, json.dumps(out["synthetic"][name]["summary"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
