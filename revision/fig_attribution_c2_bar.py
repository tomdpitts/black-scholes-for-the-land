"""The 2022 drought split with Chronos-2: the bar of fig_attribution_m4.py alone, numbers only, same scale and colours.

Box means of the season-end maps in c2_drought0.npz (revision/c2.py drought0, full box; median paths):
  baseline                   the run under normal weather
  rain, evaporative demand   Shapley over the two forcings (the four weather combinations)
  residual                   observed - the run under the weather that fell

  .venv/bin/python revision/fig_attribution_c2_bar.py   -> paper/figs/candidates/fig_2022_attribution_c2_bar.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig_attribution_m4 import BLUE, DARK_BLUE, ORANGE, INK, LIGHT

C2 = "runs/20261026_1312_c2_split0/c2_drought0.npz"
OUT = "paper/figs/candidates/fig_2022_attribution_c2_bar"


def main():
    C = np.load(C2)
    obs, F, cr, ct, crt = (float(np.nanmean(C[k])) for k in ("obs", "F0", "cf_r0", "cf_t0", "cf_rt0"))
    rain = 0.5 * ((F - cr) + (ct - crt))
    demand = 0.5 * ((F - ct) + (cr - crt))
    print(f"baseline {crt:+.3f}  rain {rain:+.3f}  demand {demand:+.3f}  residual {obs - F:+.3f}  observed {obs:+.3f}")

    plt.rcParams.update({"font.size": 8})
    fig, bx = plt.subplots(figsize=(1.0, 3.6))
    top = 0.0
    for i, (v, col) in enumerate(((crt, LIGHT), (rain, DARK_BLUE), (demand, BLUE), (obs - F, ORANGE))):
        bx.bar(0, v, bottom=top, width=0.55, color=col, edgecolor="white", linewidth=1.2)
        # baseline is a sliver (0.003): one more decimal, and its number sits above the bar, clear of rain's
        y, va, s = (max(top, top + v) + 0.002, "bottom", f"{abs(v):.3f}") if i == 0 else (top + v / 2, "center", f"{abs(v):.2f}")
        bx.text(0.36, y, s, va=va, ha="left", fontsize=7, color=INK)
        top += v
    bx.set_ylim(-0.2, 0.03); bx.set_xlim(-0.4, 0.9)
    bx.axis("off")
    fig.savefig(f"{OUT}.png", dpi=220, bbox_inches="tight")
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
