"""Figures and tables for the paper from a run directory.

  python figs.py runs/<pipeline_dir> runs/<audit_dir>
Writes paper/figs/*.pdf and paper/tables.tex.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"M0": "#888888", "M1": "#1f77b4", "M15": "#2ca02c", "M2": "#d62728", "M2clim": "#ff9896", "M1clim": "#aec7e8", "M2nd": "#9467bd"}
LABEL = {"M0": "M0 history only", "M1": "M1 linear, realised rain", "M15": "M1.5 S-curve, realised rain", "M2": "M2 learned, realised rain",
         "M2clim": "M2, climatological rain", "M1clim": "M1, climatological rain", "M2nd": "M2 without driver"}


def main(run, audit):
    m = json.load(open(f"{run}/metrics.json"))
    au = json.load(open(f"{audit}/metrics.json"))
    ar = np.load(f"{run}/arrays.npz")
    import os
    os.makedirs("paper/figs", exist_ok=True)

    # ---- Fig 1: delta map (MAM, h=2) + concavity curve + theta decay
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.7), gridspec_kw={"width_ratios": [1.25, 1, 1]}, constrained_layout=True)
    d = ar["delta_MAM"][1]
    im = ax[0].imshow(d, cmap="viridis", vmin=0, vmax=np.nanpercentile(d, 99))
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title(r"$\Delta$: $\partial\,\mathbb{E}[a_{t+2}]/\partial r_{t+1}$, long rains", fontsize=8)
    cb = plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02); cb.set_label("NDVI per 100 mm", fontsize=7)
    for s, c in (("MAM", "#2ca02c"), ("OND", "#ff7f0e")):
        cc = au["concavity"][s]
        ax[1].plot(np.array(cc["bin_x"]) * 100, cc["bin_y"], "o-", ms=3, color=c,
                   label=f"{s}: quad. coef {cc['quadratic_coef']:.3f}")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_xlabel("seasonal rain anomaly (mm)"); ax[1].set_ylabel("seasonal NDVI anomaly")
    ax[1].set_title("observed seasonal response (data, not model)", fontsize=8); ax[1].legend(fontsize=6, frameon=False)
    h = np.arange(1, 7)
    for s, c in (("MAM", "#2ca02c"), ("OND", "#ff7f0e"), ("dry", "#8c564b")):
        ax[2].plot(h, m["greeks"][f"theta_{s}_gridmean_by_h"], "o-", ms=3, color=c, label=s)
    ax[2].set_xlabel("horizon (16-day steps)"); ax[2].set_ylabel(r"$\partial\,\mathbb{E}[a_{t+h}]/\partial a_t$")
    ax[2].set_title(r"$\Theta$: anomaly persistence", fontsize=8); ax[2].legend(fontsize=6, frameon=False)
    ax[2].set_ylim(0, 1)
    fig.savefig("paper/figs/fig1_greeks.pdf"); plt.close(fig)

    # ---- Fig 2: error recovering the observed land response, given the rain that fell (absolute CRPS)
    def response_chart(figsize, fs, out, title=None, footnote=None, right_axis=True, short=False, xmax=10.6):
        Cc = {"M0": "#52514e", "M2": "#2a78d6", "M15": "#eb6834", "M2clim": "#9a9891"}
        Ll = {"M0": "History only (no rain)", "M2": "Our model, actual rain", "M15": "Best formula (S-curve), actual rain",
              "M2clim": "Control: our model, rain withheld"}
        if short:
            Ll = {"M0": "M0 history only", "M2": "M2 learned", "M15": "M1.5 S-curve", "M2clim": "M2, rain withheld"}
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)
        for ax, es, ttl in zip(axes, ("A_test", "B_all"), ("Unseen years 2019–25, training landscape", "Unseen landscape, all years")):
            ys = {k: np.array(m["rollout"][es][k]["crps"]) for k in Cc}
            for k in ("M2clim", "M0", "M15", "M2"):
                ax.plot(h, ys[k], color=Cc[k], lw=1.6, marker="o", ms=4, mfc="white", mew=1.4, ls="--" if k == "M2clim" else "-")
            order = sorted(Cc, key=lambda k: ys[k][-1]); pos = {}; last = None
            gap = 0.0034 * fs / 9
            for k in order:
                y = ys[k][-1]
                if last is not None and y < last + gap: y = last + gap
                pos[k] = y; last = y
            for k in Cc:
                ax.annotate(Ll[k], xy=(6, ys[k][-1]), xytext=(6.25, pos[k]), va="center", ha="left", fontsize=fs,
                            color="#0b0b0b" if k != "M0" else "#52514e", arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.6, shrinkA=0, shrinkB=3))
            yF, yL = ys["M15"][-1], ys["M2"][-1]
            ax.annotate("", xy=(5.5, yL), xytext=(5.5, yF), arrowprops=dict(arrowstyle="<->", color="#0b0b0b", lw=1.0))
            ax.annotate(f"{100*(1 - yL/yF):.0f}% less error\nthan the formula", xy=(5.5, (yL + yF) / 2), xytext=(4.5, yL - 0.012),
                        ha="center", va="top", fontsize=fs + 1, fontweight="bold", color="#0b0b0b",
                        arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.6, shrinkB=4))
            ax.set_title(ttl, fontsize=fs + 1, loc="left")
            ax.set_xticks(h); ax.set_xticklabels([f"{int(x*16)} d" if x < 6 else "3 mo" for x in h], fontsize=fs)
            ax.tick_params(axis="y", labelsize=fs)
            ax.set_xlim(0.7, xmax); ax.grid(axis="y", color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
            ax.set_xlabel("Steps ahead along the known rain path", fontsize=fs)
        axes[0].set_ylabel("Error recovering observed greenness\n(CRPS, NDVI units; lower is better)", fontsize=fs)
        axes[0].set_ylim(0.02, 0.075)
        if right_axis:
            sec = axes[1].secondary_yaxis("right", functions=(lambda y: y / 0.2 * 100, lambda p: p * 0.2 / 100))
            sec.set_ylabel("Error as % of a typical seasonal swing (0.2 NDVI)", color="#52514e", fontsize=fs)
            sec.tick_params(colors="#52514e", labelsize=fs); sec.spines["right"].set_color("#c3c2b7")
            sec.set_yticks([10, 15, 20, 25, 30, 35]); sec.set_yticklabels([f"{v}%" for v in [10, 15, 20, 25, 30, 35]])
        if title:
            fig.suptitle(title, fontsize=fs + 4, x=0.01, ha="left")
        if footnote:
            fig.text(0.01, -0.02, footnote, fontsize=fs - 1, color="#52514e", ha="left")
        fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb"); plt.close(fig)

    response_chart((7.0, 2.35), 7, "paper/figs/fig2_response.pdf", right_axis=True, short=True, xmax=8.6)
    response_chart((12.5, 4.8), 9, "paper/figs/headline_error_vs_horizon.png",
                   title="Given the rain that fell, how well does each model recover what the land did?",
                   footnote="MODIS NDVI 250 m, Laikipia, Kenya. Every model is fed the rain that actually fell over the following steps; none uses a rain forecast. "
                            "CRPS = mean gap between the model's distribution and the observed greenness anomaly, over identical held-out pixel-dates.")
    response_chart((12.5, 4.8), 9, "paper/figs/headline_error_vs_horizon.pdf",
                   title="Given the rain that fell, how well does each model recover what the land did?")

    # ---- Fig 3: calibration (spread-skill + PIT)
    fig, ax = plt.subplots(1, 2, figsize=(4.0, 1.9), constrained_layout=True)
    for es, c in (("A_test", C["M2"]), ("B_all", "#e377c2")):
        cal = m["calibration"][es]
        ax[0].plot(cal["spread"], cal["rmse"], "o-", ms=3, color=c, label=f"{es} r={cal['spread_skill_corr']:.2f}")
    lim = max(max(m["calibration"][e]["rmse"]) for e in ("A_test", "B_all"))
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.5)
    ax[0].set_xlabel("predicted spread"); ax[0].set_ylabel("RMSE"); ax[0].legend(fontsize=6, frameon=False)
    ax[1].bar(np.arange(10) / 10 + 0.05, np.array(m["calibration"]["A_test"]["pit_hist"]) / sum(m["calibration"]["A_test"]["pit_hist"]), width=0.1, color=C["M2"])
    ax[1].axhline(0.1, color="k", lw=0.5); ax[1].set_xlabel("PIT"); ax[1].set_ylabel("frequency")
    fig.savefig("paper/figs/fig3_calibration.pdf"); plt.close(fig)

    # ---- tables
    o = m["onestep"]
    an = m.get("anomaly", {})
    with open("paper/tables.tex", "w") as f:
        f.write("\\begin{tabular}{lcccc}\\toprule\n & \\multicolumn{2}{c}{Box A, 2019--25} & \\multicolumn{2}{c}{Box B, all years}\\\\\n")
        f.write("Model & NLL & CRPS skill & NLL & CRPS skill\\\\\\midrule\n")
        for k in ("M0", "M1", "M15", "M2nd", "M2"):
            f.write(f"{LABEL[k]} & {o['A_test'][k]['nll']:.3f} & {o['A_test'][k]['crps_skill_vs_M0']:.3f} & "
                    f"{o['B_all'][k]['nll']:.3f} & {o['B_all'][k]['crps_skill_vs_M0']:.3f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
        f.write("%%ANOMALY\n")
        if an:
            f.write("\\begin{tabular}{llcccc}\\toprule\n & Score & hit rate @5\\% FAR & lag (steps) & FAR, drought seasons & AUC\\\\\\midrule\n")
            rows = (("two-sided, pre-reg.", "zscore", "z-score"), ("", "M1", "M1"), ("", "M15", "M1.5"), ("", "M2", "M2"),
                    ("one-sided, post hoc", "zscore_1s", "z-score"), ("", "M1_1s", "M1"), ("", "M15_1s", "M1.5"), ("", "M2_1s", "M2"))
            for grp, k, lab in rows:
                if k == "zscore_1s":
                    f.write("\\midrule\n")
                a = an[k]
                lag = f"{a['mean_lag_steps']:.2f}" if a["mean_lag_steps"] is not None else "--"
                f.write(f"{grp} & {lab} & {a['hit_rate_at_5pct_far']:.2f} & {lag} & {a['far_in_drought_seasons']:.3f} & {a['auc']:.3f}\\\\\n")
            f.write("\\bottomrule\\end{tabular}\n")
    print("figures and tables written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
