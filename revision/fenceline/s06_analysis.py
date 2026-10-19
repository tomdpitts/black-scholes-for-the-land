"""Step 06: the pre-registered statistics (P1, S1-S8), exactly as PREREG.md specifies.
Reads design (step 03), exposure years (frozen) and outcomes (step 05). Writes RUN/outputs/results.json
and figures once."""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C

C.start("06_analysis", __file__)
C.verify_freeze("06_analysis")
import numpy as np
from scipy import stats

dz = np.load(C.out("design", "design.npz"), allow_pickle=True)
meta = json.load(open(C.out("design", "design_meta.json")))
oz = np.load(C.out("outputs", "outcomes.npz"), allow_pickle=True)
yrs_rows = list(csv.DictReader(open(C.out("inputs", "exposure", "years.csv"))))
invaded_years = {int(r["year"]) for r in yrs_rows if r["invasion_treated_janmay"].strip() == "1"}
clean_years = {int(r["year"]) for r in yrs_rows if r["invasion_treated_janmay"].strip() == "0"}   # Amendment 1
props = meta["treated_properties"]
dose = {r["property"]: r["dose_2017"] for r in meta["property_rows"]}
tid = dz["treated_id"].ravel()
t_idx, m_idx = dz["t_idx"], dz["m_idx"]


def contrast(R, valid, ti, mi):
    """D = mean_i [R_i - mean_{valid matches} R_j]; per-pixel diffs returned for grouping."""
    Rf, Vf = R.ravel(), valid.ravel()
    ok_t = Vf[ti]
    mv = Vf[mi]                                     # [nt,k]
    n = mv.sum(1)
    keep = ok_t & (n > 0)
    cm = np.where(mv, Rf[mi], 0).sum(1)[keep] / n[keep]
    diff = Rf[ti][keep] - cm
    return (float(diff.mean()) if diff.size else np.nan), diff, ti[keep]


def series(kind, ti=t_idx, mi=m_idx, field="R"):
    Y = oz[f"{kind}_years"]
    out = {}
    for k, y in enumerate(Y):
        D, diff, kept = contrast(oz[f"{kind}_{field}"][k], oz[f"{kind}_valid"][k], ti, mi)
        out[int(y)] = {"D": D, "n": int(diff.size), "diff": diff, "kept": kept}
    return out


def placebo_set(kind):
    Y, G = oz[f"{kind}_years"], oz[f"{kind}_gap"]
    return [int(y) for y, g in zip(Y, G) if y != 2017 and not g and int(y) not in invaded_years]


def rank_p(s, P, target=2017):
    vals = np.array([s[y]["D"] for y in P])
    Dt = s[target]["D"]
    rank = 1 + int((vals < Dt).sum())
    return {"D_2017": Dt, "rank_among_all": rank, "n_placebo": len(P),
            "p_one_sided": (1 + int((vals <= Dt).sum())) / (1 + len(P)),
            "placebo_median": float(np.median(vals)), "placebo_min": float(vals.min()), "placebo_max": float(vals.max())}


results = {"placebo_years_MAM": placebo_set("MAM"), "invasion_coded_years": sorted(invaded_years)}
# ------------------------------------------------------------------ P1
sM = series("MAM")
gapM = dict(zip(oz["MAM_years"].astype(int), oz["MAM_gap"]))
assert not gapM[2017], "MAM 2017 is gap-flagged"
P = results["placebo_years_MAM"]
p1 = rank_p(sM, P)
p1["verdict"] = "pass" if p1["rank_among_all"] == 1 else ("supportive" if p1["rank_among_all"] == 2 else "fail")
p1["n_treated_pixels_2017"] = sM[2017]["n"]
results["P1"] = p1
results["D_by_year_MAM"] = {y: {"D": v["D"], "n": v["n"], "gap": bool(gapM[y]), "invasion_coded": y in invaded_years}
                            for y, v in sM.items()}
C.log("06_analysis", "P1 placebo-in-time", **{k: v for k, v in p1.items()})

# ------------------------------------------------------------------ S1 drought-adjusted
dep = dict(zip(oz["MAM_years"].astype(int), oz["MAM_rain_dep_mm"].astype(float)))
xv = np.array([dep[y] for y in P]); yv = np.array([sM[y]["D"] for y in P])
lr = stats.linregress(xv, yv)
n = len(P); yhat = lr.intercept + lr.slope * dep[2017]
s = np.sqrt(np.sum((yv - (lr.intercept + lr.slope * xv)) ** 2) / (n - 2))
se = s * np.sqrt(1 + 1 / n + (dep[2017] - xv.mean()) ** 2 / np.sum((xv - xv.mean()) ** 2))
t = (sM[2017]["D"] - yhat) / se
driest = sorted(P, key=lambda y: dep[y])[:5]
s1 = {"slope_D_per_mm": lr.slope, "intercept": lr.intercept, "r": lr.rvalue, "rain_dep_2017_mm": dep[2017],
      "D_pred_2017": yhat, "t": t, "df": n - 2, "p_one_sided": float(stats.t.cdf(t, n - 2)),
      "five_driest_placebo": {y: {"rain_dep_mm": dep[y], "D": sM[y]["D"]} for y in driest},
      "rank_among_five_driest_plus_2017": 1 + int(sum(sM[y]["D"] < sM[2017]["D"] for y in driest)),
      "rain_dep_rank_2017_among_all": 1 + int(sum(dep[y] < dep[2017] for y in P))}
s1["verdict"] = "pass" if s1["p_one_sided"] < 0.05 else "fail"
results["S1"] = s1
C.log("06_analysis", "S1 drought-adjusted", **{k: v for k, v in s1.items() if k != "five_driest_placebo"})

# ------------------------------------------------------------------ S2 per property, S3 dose
s2 = {}
for n_, prop in enumerate(props):
    per = {}
    for y, v in sM.items():
        sel = tid[v["kept"]] == n_
        per[y] = {"D": float(v["diff"][sel].mean()) if sel.any() else np.nan, "n": int(sel.sum())}
    vals = np.array([per[y]["D"] for y in P if np.isfinite(per[y]["D"])])
    s2[prop] = {"D_2017": per[2017]["D"], "n_2017": per[2017]["n"], "dose": dose.get(prop),
                "rank": 1 + int((vals < per[2017]["D"]).sum()), "n_placebo": int(vals.size)}
results["S2"] = s2
C.log("06_analysis", "S2 per property", **{k: f"D={v['D_2017']:+.4f} rank {v['rank']}/{v['n_placebo'] + 1} n={v['n_2017']} dose={v['dose']}" for k, v in s2.items()})
s3 = {}
for lvl in ("heavy", "light"):
    ids = [n_ for n_, pr in enumerate(props) if dose.get(pr) == lvl]
    if ids:
        sel = np.isin(tid[sM[2017]["kept"]], ids)
        per = {}
        for y, v in sM.items():
            sy = np.isin(tid[v["kept"]], ids)
            per[y] = float(v["diff"][sy].mean()) if sy.any() else np.nan
        vals = np.array([per[y] for y in P if np.isfinite(per[y])])
        s3[lvl] = {"properties": [props[i] for i in ids], "D_2017": per[2017], "n": int(sel.sum()),
                   "rank": 1 + int((vals < per[2017]).sum()), "n_placebo": int(vals.size),
                   "p_one_sided": (1 + int((vals <= per[2017]).sum())) / (1 + vals.size)}
results["S3"] = s3
C.log("06_analysis", "S3 dose", **{k: f"D={v['D_2017']:+.4f} n={v['n']} rank {v['rank']}/{v['n_placebo'] + 1} p={v['p_one_sided']:.3f}" for k, v in s3.items()})

# ------------------------------------------------------------------ S4 placebo-in-space
k17 = int(np.where(oz["MAM_years"] == 2017)[0][0])
sp = np.load(C.out("design", "s4_line_splits.npz"), allow_pickle=True)      # Amendment 2: line splits
fakes = [contrast(oz["MAM_R"][k17], oz["MAM_valid"][k17], ft, fm)[0]
         for ft, fm in zip(sp["t"], sp["m"]) if len(ft)]
fakes = np.array([f for f in fakes if np.isfinite(f)])
s4 = {"design": "random line splits of control pool (Amendment 2)", "n_splits": int(fakes.size), "D_true": sM[2017]["D"],
      "p_one_sided": (1 + int((fakes <= sM[2017]["D"]).sum())) / (1 + fakes.size),
      "fake_median": float(np.median(fakes)), "fake_p5": float(np.percentile(fakes, 5))}
results["S4"] = s4
C.log("06_analysis", "S4 placebo-in-space", **s4)

# ------------------------------------------------------------------ S5 Jan-Feb window
sJ = series("JF")
PJ = placebo_set("JF")
gapJ = dict(zip(oz["JF_years"].astype(int), oz["JF_gap"]))
results["S5"] = {**rank_p(sJ, PJ), "gap_2017": bool(gapJ[2017]), "placebo_years": PJ}
C.log("06_analysis", "S5 Jan-Feb", **{k: v for k, v in results["S5"].items() if k != "placebo_years"})

# ------------------------------------------------------------------ S6 community-only controls
if len(dz["t6"]):
    s6s = series("MAM", dz["t6"], dz["m6"])
    results["S6"] = rank_p(s6s, P)
else:
    results["S6"] = {"skipped": "fewer than 300 eligible community-polygon pixels"}
C.log("06_analysis", "S6 community-only controls", **results["S6"])

# ------------------------------------------------------------------ S7 one-step residual sum
s7s = series("MAM", field="S7")
results["S7"] = rank_p(s7s, P)
C.log("06_analysis", "S7 one-step sum (method check)", **results["S7"])

# ------------------------------------------------------------------ S10 clean placebo years only (Amendment 1)
P0 = [y for y in P if y in clean_years]
s10 = rank_p(sM, P0)
x0 = np.array([dep[y] for y in P0]); y0 = np.array([sM[y]["D"] for y in P0])
lr0 = stats.linregress(x0, y0); n0 = len(P0)
s0 = np.sqrt(np.sum((y0 - (lr0.intercept + lr0.slope * x0)) ** 2) / (n0 - 2))
se0 = s0 * np.sqrt(1 + 1 / n0 + (dep[2017] - x0.mean()) ** 2 / np.sum((x0 - x0.mean()) ** 2))
t0_ = (sM[2017]["D"] - (lr0.intercept + lr0.slope * dep[2017])) / se0
s10.update({"placebo_years": P0, "S1_t": t0_, "S1_df": n0 - 2, "S1_p_one_sided": float(stats.t.cdf(t0_, n0 - 2))})
results["S10"] = s10
C.log("06_analysis", "S10 clean placebo years", **{k: v for k, v in s10.items()})

# ------------------------------------------------------------------ S8 other invasion-coded seasons
results["S8"] = {y: {"D": sM[y]["D"], "rank_vs_placebo": 1 + int(sum(sM[q]["D"] < sM[y]["D"] for q in P)),
                     "gap": bool(gapM[y])} for y in sorted(invaded_years) if y != 2017 and y in sM}
C.log("06_analysis", "S8 other invasion-coded seasons", **{str(k): v for k, v in results["S8"].items()})

# ------------------------------------------------------------------ save
rj = C.out("outputs", "results.json")
C.write_new(rj, json.dumps(results, indent=1, default=float))
C.log("06_analysis", "results saved", file=os.path.relpath(rj, C.ROOT), sha256=C.sha256(rj))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
Y = sorted(sM)
col = ["#d9622b" if y == 2017 else ("#a8761f" if y in invaded_years else ("#bbbbbb" if gapM[y] else "#2d6db5")) for y in Y]
ax[0].bar(Y, [sM[y]["D"] for y in Y], color=col)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel("D: invaded minus matched control\n(season-end residual, NDVI)")
ax[0].set_title("MAM season, by year (orange 2017; ochre other invasion years; grey gap)", fontsize=7)
ax[1].scatter(xv, yv, s=14, color="#2d6db5", label="placebo years")
xx = np.linspace(min(xv.min(), dep[2017]) - 10, max(xv.max(), dep[2017]) + 10, 50)
ax[1].plot(xx, lr.intercept + lr.slope * xx, color="#2d6db5", lw=0.8)
ax[1].scatter([dep[2017]], [sM[2017]["D"]], s=30, color="#d9622b", label="2017", zorder=3)
for y in P:
    ax[1].annotate(str(y)[2:], (dep[y], sM[y]["D"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
ax[1].set_xlabel("Box B MAM rain departure (mm)"); ax[1].legend(frameon=False, fontsize=6)
ax[1].set_title("S1: is 2017 low for its rain?", fontsize=7)
fig.tight_layout()
fp = C.out("outputs", "fig_D_by_year.png")
fig.savefig(fp, dpi=200)
C.log("06_analysis", "figure", file=os.path.relpath(fp, C.ROOT), sha256=C.sha256(fp))
