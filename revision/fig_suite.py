"""One figure per test of the suite (revision/TESTS.md), from the saved metrics of a suite run directory.
Style of revision/fig_onestep_skill.py. Chronos-2 rows read "GPU pending" until its cells exist.

  .venv/bin/python revision/fig_suite.py runs/20261022_1639_suite
      -> <run>/fig_suite_{skill,attribution,2022,sens}.png and the same in paper/figs/candidates/
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
from matplotlib import font_manager
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_onestep_skill import INK, INK2, GRID, SURF  # noqa: E402

for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "OpenSans-*.ttf")):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Open Sans", "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9891"
SEEDS = range(5)
CAND = "paper/figs/candidates"


def frame(ax, xlabel, title):
    ax.set_facecolor(SURF)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(colors=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold")


def save(fig, R, name):
    for d in (R, CAND):
        fig.savefig(f"{d}/{name}.png", facecolor=SURF)
    plt.close(fig)
    print("wrote", f"{R}/{name}.png", f"{CAND}/{name}.png")


def paired_bars(ax, labels, vals, lo=None, hi=None, series=("Box A, 2019–25", "Box B, 2019–25"), cols=(BLUE, ORANGE), fmt="{:.2f}", at=None, gap=0.08):
    """Horizontal pairs of bars (one per box), optional seed-range whiskers, 'GPU pending' for missing rows."""
    y = np.arange(len(labels))[::-1].astype(float)
    h = 0.34
    for j, (name, col) in enumerate(zip(series, cols)):
        yy = y + (h / 2 + 0.02) * (1 if j == 0 else -1)
        v = np.array([np.nan if r is None else r[j] for r in vals])
        ax.barh(yy, np.nan_to_num(v), height=h, color=col, label=name, zorder=2)
        for k, (vi, yi) in enumerate(zip(v, yy)):
            if lo is not None and lo[k] is not None:
                ax.plot([lo[k][j], hi[k][j]], [yi, yi], color=INK, lw=0.8, zorder=3)
            if np.isfinite(vi):
                x = max(vi, hi[k][j] if hi is not None and hi[k] is not None else vi, at[k][j] if at is not None else vi)
                ax.text(x + (0.004 if at is None else gap), yi, fmt.format(vi), va="center", fontsize=7, color=INK, fontweight="bold" if at is not None else None)
    for k, r in enumerate(vals):
        if r is None:
            ax.text(0.004, y[k], "GPU pending (Go/No-go)", va="center", fontsize=7, color=INK2, style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.axvline(0, color=INK2, lw=1.0, zorder=3)
    return y


def fig_skill(R):
    sk = json.load(open(f"{R}/skill.json"))
    sk1 = lambda es, n: 1 - sk[es]["all"][n]["rollout_crps"][-1] / sk[es]["all"]["M0"]["rollout_crps"][-1]
    trees = (json.load(open("runs/20261006_1759_baselines/metrics_gbm.json"))["A_test"]["rollout_skill_vs_M0"][-1],
             1 - json.load(open("runs/20261021_0147_m3/metrics_gbm_B_test.json"))["B_test"]["rollout_crps"][-1]
             / sk["B_test"]["all"]["M0"]["rollout_crps"][-1])
    rows, lo, hi = [], [], []
    for lab, k in [("M1, linear rain", "M1"), ("RESTREND (per-pixel regression)", "RESTREND"), ("Gradient-boosted trees", None),
                   ("M2, learned", "M2"), ("M3 (+ topography)", "M3"),
                   ("M4 ensemble (+ evaporative demand; ours)", "M4_ens"), ("Chronos-2, with weather (4,000 px)", "c2w"), ("Chronos-2, history only (4,000 px)", "c2h")]:
        if k is None:
            v, l_, h_ = trees, None, None
        elif k.startswith("c2"):  # 4,000-px sample (the pre-registered comparison)
            c2 = f"{R}/c2_skill.npz"
            if os.path.exists(c2):
                C = np.load(c2)
                m = "weather" if k == "c2w" else "history"
                v = tuple(1 - C[f"{b}_{m}_num"][:, -1].sum() / C[f"{b}_{m}_den"][:, -1].sum() / sk[f"{b}_test"]["px"]["M0"]["rollout_crps"][-1] for b in "AB")
            else:
                v = None
            l_, h_ = None, None
        elif k in ("M2", "M3", "M4"):
            s = [[sk1(es, f"{k}_s{i}") for i in SEEDS] for es in ("A_test", "B_test")]
            v, l_, h_ = tuple(np.mean(x) for x in s), tuple(min(x) for x in s), tuple(max(x) for x in s)
        else:
            v, l_, h_ = (sk1("A_test", k), sk1("B_test", k)), None, None
        rows.append((lab, v)); lo.append(l_); hi.append(h_)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(SURF)
    paired_bars(ax, [r[0] for r in rows], [r[1] for r in rows], lo, hi, fmt="{:.3f}")
    ax.get_yticklabels()[5].set_fontweight("bold")
    ax.set_xlim(0, 0.34)
    frame(ax, "3-month skill vs history alone (0 = no better); M2, M3: mean and range of 5 fits",
          "Test 1. Skill three months ahead on unseen years")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    save(fig, R, "fig_suite_skill")


def fig_attribution(R):
    """Five M4 worlds (PREREG addendum 2): bars = mean over worlds, dots = the five worlds."""
    from suite import EST5, world_avg
    rows = json.load(open(f"{R}/attrib_m4x5.json"))["rows"]
    worlds = sorted({r["world_seed"] for r in rows})
    est = [(lab.replace("**", ""), e) for lab, e in EST5 if not lab.startswith("M4 single")]
    est = [({"z-score": "z-score (current practice)", "M2 (5 seeds)": "M2", "M3 (5 seeds)": "M3", "M4 ensemble, 5 seeds": "M4 ensemble"}.get(l, l), e)
           for l, e in est]
    per = lambda e, dro: [100 * world_avg([r for r in rows if r["world_seed"] == ws], e if not callable(e) else (lambda _, ws=ws: e(ws)), dro, "size")[0]
                          for ws in worlds]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    fig.patch.set_facecolor(SURF)
    y = paired_bars(ax, [e[0] for e in est], [(100 * world_avg(rows, e, True, "size")[0], 100 * world_avg(rows, e, False, "size")[0]) for _, e in est],
                    series=("Drought seasons (n = 4)", "Other seasons (n = 9)"), cols=(ORANGE, BLUE), fmt="{:.0f}%",
                    at=[(max(per(e, True)), max(per(e, False))) for _, e in est], gap=6)
    for k, (_, e) in enumerate(est):
        for j, dro in enumerate((True, False)):
            v = per(e, dro)
            ax.scatter(v, np.full(len(v), y[k] + (0.19 if j == 0 else -0.19)), s=6, color=INK, alpha=0.5, zorder=4, lw=0)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0, 420)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    frame(ax, "Error, % of the true hidden damage (0% = perfect; bar: mean of 5 worlds; dots: each world)",
          "Test 2c. Finding hidden land damage in simulated worlds")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    save(fig, R, "fig_suite_attribution")


def fig_lean(R):
    """Test 2a: each season's box-mean leftover against its rain departure; lines: pooled block-level slopes."""
    ln = json.load(open(f"{R}/lean.json"))
    t = ln["seasons"]
    x = np.array([r["rain_departure_mm"] for r in t])
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    fig.patch.set_facecolor(SURF)
    for key, lab, col in (("z-score", "z-score (current practice)", GREY), ("M4_ens", "M4 ensemble", BLUE)):
        y = np.array([r[f"leftover_{key}"] for r in t])
        b = ln[f"{key}_rain"]
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, y.mean() + b["slope_per_100mm"] * (xx - x.mean()) / 100, color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(x, y, s=34, color=col, edgecolor=SURF, lw=1.5, zorder=3,
                   label=f"{lab}: {b['slope_per_100mm']:+.3f} per 100 mm [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]")
    ax.axhline(0, color=INK2, lw=1.0, zorder=1)
    ax.axvline(0, color=GRID, lw=0.6, zorder=0)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    frame(ax, "Season rain departure, Box A (mm; dots: the 13 held-out seasons, 2019–25)",
          "Test 2a. Does what the model leaves unexplained lean with the weather?")
    ax.set_ylabel("Leftover: observed − forecast, NDVI", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    save(fig, R, "fig_suite_lean")


def fig_2022(R):
    dr = json.load(open(f"{R}/drought.json"))
    parts = [("start_deficit", "Starting deficit", AQUA), ("rain", "Rain", BLUE), ("temp", "Evaporative demand", ORANGE),
             ("residual", "Not explained (the land)", GREY)]
    rows = [("z-score", "z-score"), ("RESTREND", "RESTREND"), ("M1", "M1"), ("M2", "M2"), ("M3", "M3"), ("M4", "M4 ensemble"), ("Chronos-2", "Chronos-2")]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    fig.patch.set_facecolor(SURF)
    y = np.arange(len(rows))[::-1].astype(float)
    for k, (key, lab) in enumerate(rows):
        if key not in dr["rows"]:
            ax.text(-0.004, y[k], "GPU pending (Go/No-go)", va="center", ha="right", fontsize=7, color=INK2, style="italic")
            continue
        rr = dr["rows"][key]
        neg = pos = 0.0
        for p, _, col in parts:
            v = float(np.mean([r[p] or 0.0 for r in rr]))
            left = neg if v < 0 else pos
            ax.barh(y[k], v, left=left, height=0.5, color=col, edgecolor=SURF, lw=1.0, zorder=2)
            neg, pos = (neg + v, pos) if v < 0 else (neg, pos + v)
        wt = [r["rain"] + r["temp"] for r in rr]
        ax.text(neg - 0.003, y[k], f"weather {np.mean(wt):+.3f}" + (f" [{min(wt):+.3f}, {max(wt):+.3f}]" if len(wt) > 1 else ""),
                va="center", ha="right", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], color=INK)
    ax.axvline(0, color=INK2, lw=1.0, zorder=3)
    ax.set_xlim(-0.30, 0.01)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for _, lab, col in parts:
        ax.barh([np.nan], [0], color=col, label=lab)
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=4, bbox_to_anchor=(0, -0.42))
    frame(ax, f"Box-mean NDVI anomaly at season end (observed {dr['obs']:+.3f}); M2, M3, M4 ensemble: mean [range] of 5 fits",
          "Test 3. Splitting the March–May 2022 drought, Box A")
    save(fig, R, "fig_suite_2022")


def fig_sens(R):
    se = json.load(open(f"{R}/sens.json"))
    rows = [("M2", "M2"), ("M3", "M3"), ("M4_ens", "M4 ensemble"), ("RESTREND", "RESTREND"), ("M1", "M1"), ("Chronos-2", "Chronos-2 (4,000 px)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    fig.patch.set_facecolor(SURF)
    y = np.arange(len(rows))[::-1].astype(float)
    for j, (box, col) in enumerate((("A", BLUE), ("B", ORANGE))):
        for k, (key, lab) in enumerate(rows):
            yy = y[k] + (0.15 if j == 0 else -0.15)
            if key in ("M2", "M3"):
                xs = [se[box]["tests"][f"{key}_s{s}"]["nogap"]["G1"]["primary"] for s in SEEDS]
                ax.scatter([x["high_minus_low"] for x in xs], np.full(5, yy), s=14, color=col, zorder=3, lw=0)
                continue
            if key == "Chronos-2" and "px" in se[box]:
                x = se[box]["px"]["tests"]["Chronos-2"]["nogap"]["G1"]["primary"]
                ax.plot(x["high_minus_low_ci95"], [yy, yy], color=col, lw=2, solid_capstyle="round", zorder=2, alpha=0.6)
                ax.scatter([x["high_minus_low"]], [yy], s=30, color=col, edgecolor=SURF, lw=1.5, zorder=3)
                continue
            if key in ("Chronos-2", "M1", "RESTREND"):
                if j == 0:
                    ax.text(-0.075, y[k], {"Chronos-2": "GPU pending (Go/No-go)", "M1": "n/a: same value everywhere",
                                           "RESTREND": "n/a: one fixed slope per pixel"}[key], va="center", fontsize=7, color=INK2, style="italic")
                continue
            x = se[box]["tests"][key]["nogap"]["G1"]["primary"]
            ax.plot(x["high_minus_low_ci95"], [yy, yy], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([x["high_minus_low"]], [yy], s=30, color=col, edgecolor=SURF, lw=1.5, zorder=3)
        ax.scatter([np.nan], [np.nan], color=col, label=f"Box {box}, 2019–25")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], color=INK)
    ax.axvline(0, color=INK2, lw=1.0, zorder=1)
    ax.set_xlim(-0.08, 0.13)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    frame(ax, "Extra greenness per 100 mm of rain, most- vs least-sensitive third (line: 95% interval; dots: M2, M3's 5 fits)",
          "Test 4. Do pixels the model calls more rain-sensitive really respond more?")
    save(fig, R, "fig_suite_sens")


if __name__ == "__main__":
    R = sys.argv[1]
    fig_skill(R)
    fig_attribution(R)
    fig_lean(R)
    fig_2022(R)
    fig_sens(R)
