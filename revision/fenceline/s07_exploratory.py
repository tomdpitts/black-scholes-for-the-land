"""Step 07: post-hoc exploratory analyses declared in DEVIATIONS.md (E1-E3). Written after the
pre-registered results; does not alter them. Writes RUN/outputs/exploratory/* once."""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C

C.start("07_exploratory", __file__)
C.verify_freeze("07_exploratory")
sys.path.insert(0, C.ROOT)
import numpy as np, jax, jax.numpy as jnp
from scipy import stats
from scipy.spatial import cKDTree
import model as M
from run import load

dz = np.load(C.out("design", "design.npz"), allow_pickle=True)
meta = json.load(open(C.out("design", "design_meta.json")))
oz = np.load(C.out("outputs", "outcomes.npz"), allow_pickle=True)
res = json.load(open(C.out("outputs", "results.json")))
P = res["placebo_years_MAM"]
t_idx, m_idx = dz["t_idx"], dz["m_idx"]
out = {}


def contrast(R, valid, ti, mi):
    Rf, Vf = R.ravel(), valid.ravel()
    mv = Vf[mi]; n = mv.sum(1); keep = Vf[ti] & (n > 0)
    cm = np.where(mv, Rf[mi], 0).sum(1)[keep] / n[keep]
    return float((Rf[ti][keep] - cm).mean()), float(Rf[ti][keep].mean()), float(cm.mean())


# ------------------------------------------------------------------ E1 corrected S4
treated = dz["treated_id"] >= 0; pool = dz["control_pool"]; cov = dz["cov"]
elig = treated | pool
Z = ((cov - cov[elig].mean(0)) / cov[elig].std(0)).reshape(-1, cov.shape[-1])
ii, jj = np.mgrid[:C.N, :C.N]; pi = np.flatnonzero(pool.ravel())
rng = np.random.default_rng(20172)
k17 = int(np.where(oz["MAM_years"] == 2017)[0][0])
fakes, draws = [], 0
while len(fakes) < 200 and draws < 5000:
    draws += 1
    th = rng.uniform(0, 2 * np.pi)
    s = (jj * np.cos(th) + ii * np.sin(th)).ravel()
    c = rng.uniform(np.percentile(s[pi], 30), np.percentile(s[pi], 70))
    ft, fp = pi[s[pi] < c - 3], pi[s[pi] > c + 3]
    if len(ft) < 100 or len(fp) < 25:
        continue
    d, idx = cKDTree(Z[fp]).query(Z[ft], k=5)
    keep = d[:, 0] <= meta["caliper"]
    if keep.sum() < 100:
        continue
    fakes.append(contrast(oz["MAM_R"][k17], oz["MAM_valid"][k17], ft[keep], fp[idx[keep]])[0])
fakes = np.array(fakes); Dt = res["P1"]["D_2017"]
out["E1"] = {"n_splits": int(fakes.size), "draws": draws, "D_true": Dt, "fake_median": float(np.median(fakes)),
             "fake_p5": float(np.percentile(fakes, 5)), "fake_p95": float(np.percentile(fakes, 95)),
             "p_one_sided": (1 + int((fakes <= Dt).sum())) / (1 + fakes.size)}
C.log("07_exploratory", "E1 corrected S4 (theta on [0, 2pi))", **out["E1"])

# ------------------------------------------------------------------ E3 trend in D, and which arm moves
years = [int(y) for y in oz["MAM_years"]]
rows = []
for k, y in enumerate(years):
    D, tr, ct = contrast(oz["MAM_R"][k], oz["MAM_valid"][k], t_idx, m_idx)
    rows.append({"year": y, "D": D, "treated_mean_R": tr, "control_mean_R": ct, "gap": bool(oz["MAM_gap"][k])})
use = [r for r in rows if not r["gap"]]
early = [r["D"] for r in use if r["year"] <= 2012]; late = [r["D"] for r in use if r["year"] >= 2013]
wt = stats.ttest_ind(late, early, equal_var=False)
lr = stats.linregress([r["year"] for r in use], [r["D"] for r in use])
lr_t = stats.linregress([r["year"] for r in use], [r["treated_mean_R"] for r in use])
lr_c = stats.linregress([r["year"] for r in use], [r["control_mean_R"] for r in use])
out["E3"] = {"mean_D_2003_12": float(np.mean(early)), "mean_D_2013_25": float(np.mean(late)),
             "welch_t": float(wt.statistic), "welch_p_two_sided": float(wt.pvalue),
             "slope_D_per_year": lr.slope, "slope_p": lr.pvalue,
             "slope_treated_R_per_year": lr_t.slope, "slope_treated_p": lr_t.pvalue,
             "slope_control_R_per_year": lr_c.slope, "slope_control_p": lr_c.pvalue,
             "by_year": rows, "note": "excludes gap-flagged 2023; includes invasion-coded years"}
C.log("07_exploratory", "E3 trend in D and arm decomposition", **{k: v for k, v in out["E3"].items() if k != "by_year"})

# ------------------------------------------------------------------ E2 composite-level event study
d = load("B")
params = np.load(os.path.join(C.MODEL_RUN, "params.npy"), allow_pickle=True).item()
p = {k: jnp.asarray(v) for k, v in params["M2"].items()}
D = {k: jnp.asarray(d[k]) for k in ("a", "r", "rclim", "x", "phase")}
w = M.w_path(p, D)
dates = d["dates"].astype("datetime64[D]")


@jax.jit
def one_step(t):
    mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], D["x"], True)
    return D["a"][t + 1] - mu


def comp_contrast(t):
    e = np.asarray(one_step(t)).ravel()
    v = (d["obs"][t + 1] & ~d["burn"][t + 1]).ravel()
    mv = v[m_idx]; n = mv.sum(1); keep = v[t_idx] & (n > 0)
    return float((e[t_idx][keep] - np.where(mv, e[m_idx], 0).sum(1)[keep] / n[keep]).mean())


doy = d["doy"]; yr = d["year"]
targets = [t for t in range(len(dates) - 1) if np.datetime64("2016-09-01") <= dates[t + 1] <= np.datetime64("2017-12-31")]
ev = []
for t in targets:
    ref = []
    for y in P:
        cand = [u for u in range(len(dates) - 1) if int(yr[u + 1]) == y + (int(yr[t + 1]) - 2017) and int(doy[u + 1]) == int(doy[t + 1])]
        if cand:
            ref.append(comp_contrast(cand[0]))
    ev.append({"date": str(dates[t + 1]), "contrast": comp_contrast(t),
               "placebo_median": float(np.median(ref)), "placebo_p5": float(np.percentile(ref, 5)),
               "placebo_p95": float(np.percentile(ref, 95)), "n_ref": len(ref)})
out["E2"] = ev
C.log("07_exploratory", "E2 composite event study", n_composites=len(ev),
      below_p5=[e["date"] for e in ev if e["contrast"] < e["placebo_p5"]],
      above_p95=[e["date"] for e in ev if e["contrast"] > e["placebo_p95"]])

fj = C.out("outputs", "exploratory", "exploratory.json")
C.write_new(fj, json.dumps(out, indent=1, default=float))
C.log("07_exploratory", "saved", file=os.path.relpath(fj, C.ROOT), sha256=C.sha256(fj))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
x = np.array([np.datetime64(e["date"]) for e in ev])
ax[0].fill_between(x, [e["placebo_p5"] for e in ev], [e["placebo_p95"] for e in ev], color="#2d6db5", alpha=0.18, label="placebo years, 5-95%")
ax[0].plot(x, [e["placebo_median"] for e in ev], color="#2d6db5", lw=0.8)
ax[0].plot(x, [e["contrast"] for e in ev], "o-", ms=2.5, color="#d9622b", label="2016-17")
for a, b in (("2017-01-29", "2017-04-22"),):
    ax[0].axvspan(np.datetime64(a), np.datetime64(b), color="#a8761f", alpha=0.12, label="main invasion period")
ax[0].axhline(0, color="k", lw=0.4); ax[0].legend(frameon=False, fontsize=6)
ax[0].set_ylabel("one-step residual, invaded minus control (NDVI)"); ax[0].set_title("E2 (exploratory): by composite", fontsize=7)
ax[0].tick_params(axis="x", labelrotation=45)
yy = [r["year"] for r in use]
ax[1].plot(yy, [r["treated_mean_R"] for r in use], "o-", ms=2.5, color="#d9622b", label="invaded ranches (matched)")
ax[1].plot(yy, [r["control_mean_R"] for r in use], "o-", ms=2.5, color="#2d6db5", label="matched community controls")
ax[1].axvline(2012.5, color="k", lw=0.4, ls=":"); ax[1].axhline(0, color="k", lw=0.4)
ax[1].set_ylabel("season-end residual, MAM (NDVI)"); ax[1].legend(frameon=False, fontsize=6)
ax[1].set_title("E3 (exploratory): which arm moves", fontsize=7)
fig.tight_layout()
fp = C.out("outputs", "exploratory", "fig_exploratory.png")
fig.savefig(fp, dpi=200)
C.log("07_exploratory", "figure", file=os.path.relpath(fp, C.ROOT), sha256=C.sha256(fp))
