"""Step 03: the pre-registered tests Q1-Q9 (PREREG.md). Uses the first run's frozen matched design, read-only."""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fs_common as C

C.start("03_analysis", __file__)
C.verify_freeze("03_analysis")
C.verify_prev()
import numpy as np
from scipy import stats

dz = np.load(os.path.join(C.PREV, "design", "design.npz"), allow_pickle=True)
meta = json.load(open(os.path.join(C.PREV, "design", "design_meta.json")))
prev = json.load(open(os.path.join(C.PREV, "outputs", "results.json")))
yrows = list(csv.DictReader(open(os.path.join(C.PREV, "inputs", "exposure", "years.csv"))))
clean = {int(r["year"]) for r in yrows if r["invasion_treated_janmay"].strip() == "0"}
oz = np.load(C.out("outputs", "outcomes.npz"))
YEARS = [int(y) for y in oz["years"]]
t_idx, m_idx = dz["t_idx"], dz["m_idx"]
tid = dz["treated_id"].ravel()
props = meta["treated_properties"]
dose = {r["property"]: r["dose_2017"] for r in meta["property_rows"]}


def contrast(Y, valid, ti, mi):
    Yf, Vf = Y.ravel(), valid.ravel()
    mv = Vf[mi]; n = mv.sum(1); keep = Vf[ti] & (n > 0)
    cm = np.where(mv, np.nan_to_num(Yf[mi]), 0).sum(1)[keep] / n[keep]
    diff = Yf[ti][keep] - cm
    return (float(diff.mean()) if diff.size else np.nan), diff, ti[keep]


def series(key, ti=t_idx, mi=m_idx):
    Yall = oz[key]
    out = {}
    for k, y in enumerate(YEARS):
        valid = np.isfinite(Yall[k]) & ~oz["burn"][k]
        D, diff, kept = contrast(Yall[k], valid, ti, mi)
        out[y] = {"D": D, "n": int(diff.size), "diff": diff, "kept": kept}
    return out


def data_ok(sensor):
    med = dict(zip(YEARS, oz[f"{sensor}_ndti_main_median_valid_composites"]))
    return {y: bool(med[y] >= 3) for y in YEARS}


def rank_low(s, P, t=2017):
    v = np.array([s[y]["D"] for y in P if np.isfinite(s[y]["D"])])
    return {"D_2017": s[t]["D"], "n_2017": s[t]["n"], "rank_low": 1 + int((v < s[t]["D"]).sum()), "n_placebo": int(v.size),
            "p_low": (1 + int((v <= s[t]["D"]).sum())) / (1 + v.size), "placebo_median": float(np.median(v)),
            "placebo_min": float(v.min()), "placebo_max": float(v.max())}


def rank_high(s, P, t=2017):
    v = np.array([s[y]["D"] for y in P if np.isfinite(s[y]["D"])])
    return {"D_2017": s[t]["D"], "rank_high": 1 + int((v > s[t]["D"]).sum()), "n_placebo": int(v.size),
            "p_high": (1 + int((v >= s[t]["D"]).sum())) / (1 + v.size), "placebo_median": float(np.median(v))}


verdict = lambda r: "pass" if r == 1 else ("supportive" if r == 2 else "fail")
R = {}
okT, okA = data_ok("terra"), data_ok("aqua")
if not okT[2017]:
    C.log("03_analysis", "VOID: 2017 fails the data rule (Terra)"); R["void"] = True
P0 = prev["placebo_years_MAM"]
P = [y for y in P0 if okT[y]]
R["placebo_years"] = P; R["placebo_excluded_for_data"] = [y for y in P0 if not okT[y]]
C.log("03_analysis", "placebo set", P=P, excluded_for_data=R["placebo_excluded_for_data"])

sN = series("terra_ndti_main_Y")
R["Q1"] = {**rank_low(sN, P)}; R["Q1"]["verdict"] = verdict(R["Q1"]["rank_low"])
C.log("03_analysis", "Q1 NDTI Terra (primary)", **R["Q1"])

dep = dict(zip(YEARS, oz["rain_dep_octapr_mm"].astype(float)))
xv = np.array([dep[y] for y in P]); yv = np.array([sN[y]["D"] for y in P])
lr = stats.linregress(xv, yv); n = len(P)
s_ = np.sqrt(np.sum((yv - (lr.intercept + lr.slope * xv)) ** 2) / (n - 2))
se = s_ * np.sqrt(1 + 1 / n + (dep[2017] - xv.mean()) ** 2 / np.sum((xv - xv.mean()) ** 2))
t = (sN[2017]["D"] - (lr.intercept + lr.slope * dep[2017])) / se
R["Q2"] = {"rain_dep_2017_mm": dep[2017], "slope": lr.slope, "r": lr.rvalue, "t": t, "df": n - 2,
           "p_one_sided": float(stats.t.cdf(t, n - 2)), "verdict": "pass" if stats.t.cdf(t, n - 2) < 0.05 else "fail"}
C.log("03_analysis", "Q2 rain-adjusted", **R["Q2"])

sB = series("terra_b07_main_Y")
R["Q3"] = rank_high(sB, P); R["Q3"]["verdict"] = verdict(R["Q3"]["rank_high"])
C.log("03_analysis", "Q3 SWIR2 b07 (bare soil)", **R["Q3"])

sV = series("terra_ndvi_main_Y")
R["Q4"] = rank_low(sV, P)
C.log("03_analysis", "Q4 NDVI 500 m (descriptive)", **R["Q4"])

heavy = [n_ for n_, p in enumerate(props) if dose.get(p) == "heavy"]
selh = np.isin(tid[t_idx], heavy)
sH = series("terra_ndti_main_Y", t_idx[selh], m_idx[selh])
R["Q5"] = rank_low(sH, P); R["Q5"]["verdict"] = verdict(R["Q5"]["rank_low"])
C.log("03_analysis", "Q5 heavy-dose NDTI", **R["Q5"])

PA = [y for y in P0 if okA[y]]
sA = series("aqua_ndti_main_Y")
if okA[2017] and len(PA) >= 3:
    R["Q6"] = rank_low(sA, PA); R["Q6"]["verdict"] = verdict(R["Q6"]["rank_low"])
else:                                                    # DEVIATIONS D1: pre-registered void rule
    R["Q6"] = {"verdict": "void", "reason": "Aqua fails the data rule (median valid window composites < 3)",
               "aqua_median_valid_composites": dict(zip(YEARS, oz["aqua_ndti_main_median_valid_composites"].tolist()))}
R["Q6"]["placebo_years"] = PA
C.log("03_analysis", "Q6 Aqua NDTI replicate", **R["Q6"])

q7 = {}
for n_, prop in enumerate(props):
    sel = tid[t_idx] == n_
    if sel.sum() == 0:
        continue
    sp = series("terra_ndti_main_Y", t_idx[sel], m_idx[sel])
    q7[prop] = rank_low(sp, P)
R["Q7"] = q7
C.log("03_analysis", "Q7 per property NDTI", **{k: f"D={v['D_2017']:+.4f} rank {v['rank_low']}/{v['n_placebo'] + 1} n={v['n_2017']}" for k, v in q7.items()})

PC = [y for y in P if y in clean]
R["Q8"] = rank_low(sN, PC); R["Q8"]["verdict"] = verdict(R["Q8"]["rank_low"]); R["Q8"]["placebo_years"] = PC
C.log("03_analysis", "Q8 clean placebo years", **R["Q8"])

sP = series("terra_ndti_post_Y")
R["Q9"] = rank_low(sP, P)
C.log("03_analysis", "Q9 post-invasion NDTI (descriptive)", **R["Q9"])

R["D_by_year"] = {y: {"ndti": sN[y]["D"], "b07": sB[y]["D"], "ndvi": sV[y]["D"], "ndti_aqua": sA[y]["D"],
                      "rain_dep_octapr_mm": dep[y], "in_placebo": y in P} for y in YEARS}
rj = C.out("outputs", "results.json")
C.write_new(rj, json.dumps(R, indent=1, default=float))
C.log("03_analysis", "results saved", file=os.path.relpath(rj, C.ROOT), sha256=C.sha256(rj))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.6))
for a, (key, s, title) in zip(ax, [("ndti", sN, "Q1 NDTI (dry litter), Terra: predicted low"),
                                   ("b07", sB, "Q3 SWIR2 b07 (bare soil): predicted high"),
                                   ("ndvi", sV, "Q4 NDVI 500 m: no prediction")]):
    col = ["#d9622b" if y == 2017 else ("#2d6db5" if y in P else "#bbbbbb") for y in YEARS]
    a.bar(YEARS, [s[y]["D"] if np.isfinite(s[y]["D"]) else 0 for y in YEARS], color=col)
    a.axhline(0, color="k", lw=0.5); a.set_title(title, fontsize=7)
ax[0].set_ylabel("invaded minus matched control\n(window anomaly, 26 Feb-22 Apr)")
fig.tight_layout()
fp = C.out("outputs", "fig_swir_by_year.png"); fig.savefig(fp, dpi=200)
C.log("03_analysis", "figure", file=os.path.relpath(fp, C.ROOT), sha256=C.sha256(fp))
