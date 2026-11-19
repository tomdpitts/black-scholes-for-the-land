"""Does removing rain get closer to the management effect than baselines do? AgriAI item 1b.

Pre-registration: runs/20261019_1928_attribution/PREREG.md (written before this script).
Two synthetic worlds (fitted M2; refitted B1 GBM), the identification test's management
scenarios, and four estimators of the land under normal rain: z-score (Z), M1 (L), M1.5 (S), M2.

  .venv/bin/python revision/attribution.py [out dir]   -> <out>/metrics.json, <out>/RESULT.md (by hand)
"""
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
from run import H  # noqa: E402
from residual import season_starts  # noqa: E402
from identify import RUN, DELTA, setup, make_fns, block_mask  # noqa: E402
from baselines import fit_gbm, gbm_features, nbhd, LAGS_A  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/20261019_1928_attribution"
SCEN = {"S0": (0.0, 0.0), "S1": (0.0, DELTA), "S2a_incursion": (1.0, DELTA), "S2b_destock": (-1.0, DELTA)}
INVERTED = {"L": "M1", "S": "M15", "M2": "M2"}
EST = ("Z", "L", "S", "M2", "M2_additive")


def management(r, rc, k, delta, mask):
    """m_t = -delta * clip(1 + k (rclim - r)/rclim, 0, 3) on managed pixels (identify.py's truth)."""
    return -delta * np.clip(1 + k * (rc - r) / np.maximum(rc, 1e-3), 0, 3) * mask


def gbm_world(predict, D, t0, r_season, e, mask, k, delta):
    """GBM truth over the season: observed lags before t0, synthetic lags after; CHIRPS rain before
    the season, r_season [H,N,N] within it. Returns the path [H,N,N]."""
    N = D["a"].shape[1]
    r = np.array(D["r"])
    r[t0 + 1:t0 + 1 + H] = r_season
    Dm = {"phase": D["phase"], "r": r, "rclim": D["rclim"]}
    lags = [D["a"][t0 - j] for j in range(LAGS_A)]
    path = []
    for h in range(H):
        tn = t0 + 1 + h
        mu, sg = predict(gbm_features(lags, nbhd(lags[0]), Dm, tn, D["x"]))
        new = (lags[0] + mu.reshape(N, N) + sg.reshape(N, N) * e[h]
               + management(r[tn], D["rclim"][tn], k, delta, mask))
        lags = [new] + lags[:-1]
        path.append(new)
    return np.stack(path).astype(np.float32)


def gbm_innovations(predict, D, t0):
    """Real GBM-implied innovations of the season [H,N,N]."""
    N = D["a"].shape[1]
    e = []
    for t in range(t0, t0 + H):
        lags = [D["a"][t - j] for j in range(LAGS_A)]
        mu, sg = predict(gbm_features(lags, nbhd(D["a"][t]), D, t + 1, D["x"]))
        e.append((D["a"][t + 1] - D["a"][t] - mu.reshape(N, N)) / sg.reshape(N, N))
    return np.stack(e)


def main():
    os.makedirs(OUT, exist_ok=True)
    d, p2, D, w2 = setup("A")
    x, N = D["x"], D["a"].shape[1]
    P = np.load(f"{RUN}/params.npy", allow_pickle=True).item()
    fns, ws = {}, {}
    for est, key in INVERTED.items():
        p = {k: jnp.asarray(v) for k, v in P[key].items()}
        fns[est], ws[est] = make_fns(p, x), M.w_path(p, D)
    truth_m2, _, _, implied_m2 = fns["M2"]

    mask = block_mask(N, 8, np.random.default_rng(0))
    mb = mask.astype(bool)
    Dn = {k: np.asarray(v) for k, v in D.items()}
    assert np.isfinite(Dn["a"]).all()
    predict, fit_info = fit_gbm(d)

    def estimates(t0, yobs, r, rc, ph):
        """Each estimator's end-of-season greenness under climatological rain."""
        a0 = D["a"][t0]
        out = {"Z": np.asarray(yobs[-1])}
        for est in INVERTED:
            _, invert, cf, _ = fns[est]
            eh, ad = invert(a0, ws[est][t0], yobs, r, ph)
            out[est] = np.asarray(cf(a0, ws[est][t0], rc, ph, eh, False)[-1])
            if est == "M2":
                out["M2_additive"] = np.asarray(cf(a0, ws[est][t0], rc, ph, ad, True)[-1])
        return out

    def score(cf_hat, cf_true, cf_true0):
        eff = float((cf_true - cf_true0)[mb].mean())
        row = {"true_mgmt_effect_normal_rain": eff}
        for est, c in cf_hat.items():
            err = c - cf_true
            em, eu = float(err[mb].mean()), float(err[~mb].mean())
            row[est] = {"err_managed": em, "err_unmanaged": eu, "err_all": float(err.mean()),
                        "rel_err": em / eff if abs(eff) > 1e-6 else None,
                        "invented_share": eu / abs(eff) if abs(eff) > 1e-6 else None}
        return row

    seasons = season_starts(d, range(2019, 2026))
    rows = {wn: {s: [] for s in SCEN} for wn in ("W1_M2", "W2_GBM")}
    for (y, s, t0) in seasons:
        sl = slice(t0 + 1, t0 + 1 + H)
        a0, w0 = D["a"][t0], w2[t0]
        r, rc, ph = D["r"][sl], D["rclim"][sl], D["phase"][sl]
        rain_dep = float((r - rc).mean((1, 2)).sum() * 100)
        meta = {"year": y, "season": s, "rain_departure_mm": rain_dep, "drought": rain_dep < -60}
        e_m2 = jnp.stack([implied_m2(D["a"][t], D["a"][t + 1], w2[t], D["r"][t + 1], D["phase"][t + 1])[0]
                          for t in range(t0, t0 + H)])
        e_gbm = gbm_innovations(predict, Dn, t0)
        jm = jnp.asarray(mask)
        for name, (k, delta) in SCEN.items():
            # W1: M2 world, exactly as the identification test
            yobs = truth_m2(a0, w0, r, rc, ph, e_m2, jm, k, delta)
            cf_true = np.asarray(truth_m2(a0, w0, rc, rc, ph, e_m2, jm, k, delta)[-1])
            cf_true0 = np.asarray(truth_m2(a0, w0, rc, rc, ph, e_m2, jm, k, 0.0)[-1])
            rows["W1_M2"][name].append({**meta, **score(estimates(t0, yobs, r, rc, ph), cf_true, cf_true0)})
            # W2: GBM world
            yobs = gbm_world(predict, Dn, t0, Dn["r"][sl], e_gbm, mask, k, delta)
            cf_true = gbm_world(predict, Dn, t0, Dn["rclim"][sl], e_gbm, mask, k, delta)[-1]
            cf_true0 = gbm_world(predict, Dn, t0, Dn["rclim"][sl], e_gbm, mask, k, 0.0)[-1]
            rows["W2_GBM"][name].append({**meta, **score(estimates(t0, jnp.asarray(yobs), r, rc, ph),
                                                         cf_true, cf_true0)})
        r1 = {wn: rows[wn]["S1"][-1] for wn in rows}
        print(f"{s}{y} rain {rain_dep:+.0f} mm | S1 rel err "
              + " | ".join(f"{wn}: " + " ".join(f"{e} {r1[wn][e]['rel_err']:+.2f}" for e in EST) for wn in rows),
              flush=True)

    out = {"run": RUN, "gbm_fit": fit_info, "n_managed_px": int(mb.sum()), "worlds": {}}
    for wn, by_scen in rows.items():
        out["worlds"][wn] = {}
        for name, rr in by_scen.items():
            summ = {}
            for cls in ("drought", "other"):
                sub = [r_ for r_ in rr if r_["drought"] == (cls == "drought")]
                summ[cls] = {"n": len(sub)}
                if name != "S0":
                    summ[cls]["mean_true_effect"] = float(np.mean([r_["true_mgmt_effect_normal_rain"] for r_ in sub]))
                for est in EST:
                    v = {"mean_err_unmanaged": float(np.mean([r_[est]["err_unmanaged"] for r_ in sub])),
                         "mean_abs_err_unmanaged": float(np.mean([abs(r_[est]["err_unmanaged"]) for r_ in sub]))}
                    if name == "S0":
                        v["mean_abs_err_all"] = float(np.mean([abs(r_[est]["err_all"]) for r_ in sub]))
                    else:
                        rel = [r_[est]["rel_err"] for r_ in sub]
                        v.update({"mean_rel_err": float(np.mean(rel)), "mean_abs_rel_err": float(np.mean(np.abs(rel))),
                                  "max_abs_rel_err": float(np.max(np.abs(rel))),
                                  "mean_abs_invented_share": float(np.mean([abs(r_[est]["invented_share"]) for r_ in sub]))})
                    summ[cls][est] = v
            out["worlds"][wn][name] = {"summary": summ, "rows": rr}

    crit = {}
    for wn in rows:
        dr = out["worlds"][wn]["S1"]["summary"]["drought"]
        m2 = dr["M2"]["mean_abs_rel_err"]
        crit[wn] = {"P1_M2_below_Z": bool(m2 < dr["Z"]["mean_abs_rel_err"]),
                    "P2_M2_below_L_and_S": bool(m2 < dr["L"]["mean_abs_rel_err"] and m2 < dr["S"]["mean_abs_rel_err"]),
                    "mean_abs_rel_err": {est: dr[est]["mean_abs_rel_err"] for est in EST}}
    out["criteria"] = crit
    json.dump(out, open(f"{OUT}/metrics.json", "w"), indent=1)
    for wn in rows:
        for name in SCEN:
            print(wn, name, json.dumps(out["worlds"][wn][name]["summary"]))
    print(json.dumps(crit, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
