"""Method schematic, one column: the transition density and its three uses, in the notation of eq. (3).

Colours follow fig_attribution_m4.py (weather blue, residual orange); Delta takes the green of the sensitivity maps.

  .venv/bin/python revision/fig_schematic.py   -> paper/figs/schematic/fig_schematic.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "Open Sans", "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
BLUE, ORANGE, GREEN, INK, MUTED, SURF = "#3d8fd6", "#e0703a", "#4a9a5c", "#2b2620", "#6b645a", "#ffffff"
OUT = "paper/figs/schematic/fig_schematic"
W, H = 3.33, 1.85  # inches; AAAI column width


def arrow(ax, x0, y0, x1, y1, col=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=3,head_width=1.8", color=col, lw=0.8,
                                 shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    t = lambda x, y, s, **k: ax.text(x, y, s, ha=k.pop("ha", "center"), va=k.pop("va", "center"), color=k.pop("color", INK), **k)

    # inputs
    t(0.85, 1.74, r"state  $a_t,\ w_t,\ x$", fontsize=7)
    t(W - 0.85, 1.74, r"weather  $r,\ e$", fontsize=7)
    arrow(ax, 0.85, 1.65, 1.25, 1.49)
    arrow(ax, W - 0.85, 1.65, W - 1.25, 1.49)

    # the transition density
    cw, ch, cy = 1.9, 0.38, 1.11
    ax.add_patch(FancyBboxPatch((W / 2 - cw / 2, cy), cw, ch, boxstyle="round,pad=0,rounding_size=0.03", fc=SURF, ec=INK, lw=0.9))
    t(W / 2, cy + 0.24, r"$p_\theta(a_{t+1} \mid \mathrm{state},\ \mathrm{weather})$", fontsize=7.5)
    t(W / 2, cy + 0.09, "M4 or Chronos-2", fontsize=5.8, color=MUTED)

    # three uses: verb on the arrow, output, its formula, its user
    xs = (0.5, W / 2, W - 0.5)
    uses = [(ORANGE, "invert", "residual", r"$a_{t+h} - F_h(r,e)$", "carbon credit"),
            (BLUE, "roll forward", "weather's share", r"$F_h(r,e) - F_h(\bar r,0)$", "index insurance"),
            (GREEN, "differentiate", r"$\Delta$", r"$\partial F_h / \partial r$", "exposure to rain")]
    for x, (col, verb, out, f, user) in zip(xs, uses):
        x0 = W / 2 + (x - W / 2) * 0.45
        arrow(ax, x0, cy, x, 0.66, col=col)
        side = (x > W / 2) - (x < W / 2)  # -1 left, 0 centre, +1 right: label sits outside its arrow
        t((x0 + x) / 2 + (0.12 * side if side else 0.05), 0.89, verb, fontsize=6.2, color=MUTED, style="italic",
          ha="right" if side < 0 else "left")
        t(x, 0.54, out, fontsize=7.4, fontweight="bold", color=col)
        t(x, 0.35, f, fontsize=6.6)
        t(x, 0.14, user, fontsize=6.2, color=MUTED)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, facecolor=SURF)
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
