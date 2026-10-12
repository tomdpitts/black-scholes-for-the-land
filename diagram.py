"""Schematic: the land model as an option-pricing model, with the one missing input.

  python diagram.py   -> paper/figs/fig0_diagram.pdf and .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"


def box(ax, x, y, w, h, title, sub=None, analogue=None, fc="white", ec=GREY, ls="-", lw=1.2, tfs=9, sfs=7.5, step=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    cy = y + h / 2
    lines = [(title, tfs, INK, "bold")]
    if sub:
        lines.append((sub, sfs, INK, "normal"))
    if analogue:
        lines.append((analogue, sfs, MUTED, "italic"))
    step = step or (0.26 if h > 0.9 else 0.22) * (tfs / 8.5)
    # each entry may wrap; lay out by cumulative line count
    counts = [t.count("\n") + 1 for t, *_ in lines]
    total = sum(counts)
    ytop = cy + step * (total - 1) / 2
    cur = 0
    for (txt, fs, col, wt), c in zip(lines, counts):
        yy = ytop - step * (cur + (c - 1) / 2)
        ax.text(x + w / 2, yy, txt, ha="center", va="center", fontsize=fs, color=col, linespacing=1.0,
                fontweight=wt if wt != "italic" else "normal", style="italic" if wt == "italic" else "normal")
        cur += c


def arrow(ax, p, q, color=INK, ls="-", lw=1.3, rad=0.0, label=None, lpos=None, lfs=7.5, lcolor=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text(*lpos, label, ha="center", va="center", fontsize=lfs, color=lcolor or color,
                bbox=dict(fc=SURF, ec="none", pad=1.5))


def draw(out, fscale=1.0):
    fig, ax = plt.subplots(figsize=(14, 5.6))
    fig.patch.set_facecolor(SURF)
    ax.set_xlim(0, 14); ax.set_ylim(-0.35, 5.1); ax.axis("off")
    T, S = 8.5 * fscale, 7 * fscale
    F7, F8 = 7 * fscale, 8 * fscale

    # observable inputs (left column)
    xi, wi = 2.9, 2.9
    box(ax, xi, 3.95, wi, 0.72, "Today's greenness", "$a_t$, and its 1 km neighbourhood", "≈ spot price", tfs=T, sfs=S)
    box(ax, xi, 3.0, wi, 0.72, "Rain that fell", "$r_{t+1}$, and the bucket memory $w_t$", "≈ path of the underlying", tfs=T, sfs=S)
    box(ax, xi, 2.05, wi, 0.72, "Pixel character", "mean greenness, seasonality, mean rain", "≈ terms of the contract", tfs=T, sfs=S)

    # the missing input
    box(ax, xi, 0.35, wi, 0.85, "Within-season rain variability", "how the seasonal total was distributed",
        "≈ volatility: the one unobserved input", fc="#fff3ec", ec=ORANGE, ls="--", lw=1.6, tfs=T, sfs=S)
    box(ax, 0.2, 0.35, 2.3, 0.85, "Scenario or seasonal\nrain forecast", None, "supplied, as σ is supplied", fc="white", ec=ORANGE, lw=1.2, tfs=T, sfs=S)
    arrow(ax, (2.5, 0.775), (xi, 0.775), color=ORANGE, label="plug in", lpos=(2.7, 1.12), lfs=F7)

    # the model
    xm, wm = 6.6, 2.9
    box(ax, xm, 1.45, wm, 3.2, "Learned transition density", "$p_\\theta(a_{t+1}\\mid a_t, w_t, r_{t+1})$\nrolled out $h$ steps",
        "≈ the pricing model", fc="#eaf2fc", ec=BLUE, lw=1.6, tfs=9.5 * fscale, sfs=F8)
    ax.text(xm + wm / 2, 1.62, "drift and spread learned from\n25,600 pixels × 24 years", ha="center", va="bottom", fontsize=F7, color=MUTED)
    for y in (4.31, 3.36, 2.41):
        arrow(ax, (xi + wi, y), (xm, y))
    arrow(ax, (xi + wi / 2, 1.2), (xm, 1.9), color=ORANGE, rad=-0.15)

    # outputs (right column)
    xo, wo = 10.3, 3.4
    box(ax, xo, 3.95, wo, 0.72, "Expected greenness and spread", "$\\mathbb{E}[a_{t+h}]$ with a calibrated interval", "≈ price and its distribution", ec=AQUA, tfs=T, sfs=S)
    box(ax, xo, 3.0, wo, 0.72, "Greeks", "$\\Delta=\\partial/\\partial r$,  $\\Gamma=\\partial^2/\\partial r^2$,  $\\Theta=\\partial/\\partial a_t$,  Vega", "exact derivatives of the rollout", ec=AQUA, tfs=T, sfs=S)
    box(ax, xo, 2.05, wo, 0.72, "Residual", "observed − expected, given the rain", "≈ mispricing → attribution, anomaly", ec=AQUA, tfs=T, sfs=S)
    for y in (4.31, 3.36, 2.41):
        arrow(ax, (xm + wm, y), (xo, y))

    # observed outcome (the 'market prices')
    box(ax, xo, 0.35, wo, 0.85, "Observed greenness", "$a_{t+h}$ across the landscape, from the satellite",
        "≈ option prices quoted by the market", fc="#f1f1ef", ec=MUTED, tfs=T, sfs=S)
    arrow(ax, (xo + wo / 2, 1.2), (xo + wo / 2, 2.05), color=MUTED)

    # inference arrow: observed -> missing variable
    arrow(ax, (xo, 0.6), (xi + wi, 0.6), color=ORANGE, ls="--", lw=1.4,
          label="infer: the rain distribution that makes the model reproduce the observed land  (≈ implied volatility)",
          lpos=((xo + xi + wi) / 2, 0.28), lfs=F7)

    # mode labels
    ax.text(0.2, 1.45, "Forecast mode: supply the missing input", fontsize=F8, color=ORANGE, fontweight="bold", ha="left")
    ax.text(xo + wo, -0.15, "Attribution mode: infer it from what the land did", fontsize=F8, color=ORANGE, fontweight="bold", ha="right")

    fig.savefig(out, bbox_inches="tight", facecolor=SURF, dpi=220)
    plt.close(fig)


def draw_paper(out, f=1.45):
    """Compact variant with short labels for a 5.5-inch column."""
    fig, ax = plt.subplots(figsize=(14, 6.2))
    fig.patch.set_facecolor(SURF)
    ax.set_xlim(0, 14); ax.set_ylim(-0.8, 5.6); ax.axis("off")
    T, S, F7, F8 = 8.5 * f, 7 * f, 7 * f, 8 * f
    xi, wi, hb = 2.9, 3.0, 0.95
    ys = (4.45, 3.3, 2.15)
    box(ax, xi, ys[0], wi, hb, "Today's greenness", "$a_t$ + 1 km neighbourhood", "≈ spot price", tfs=T, sfs=S)
    box(ax, xi, ys[1], wi, hb, "Rain that fell", "$r_{t+1}$ + bucket memory $w_t$", "≈ path of the underlying", tfs=T, sfs=S)
    box(ax, xi, ys[2], wi, hb, "Pixel character", "mean greenness, seasonality, rain", "≈ terms of the contract", tfs=T, sfs=S)
    box(ax, xi, -0.05, wi, 1.55, "Within-season\nrain variability", "how the total was distributed",
        "≈ volatility, the unobserved input", fc="#fff3ec", ec=ORANGE, ls="--", lw=1.6, tfs=T, sfs=S)
    box(ax, 0.15, -0.05, 2.35, 1.55, "Scenario or\nseasonal forecast", None, "supplied, as σ is", fc="white", ec=ORANGE, lw=1.2, tfs=T, sfs=S)
    arrow(ax, (2.5, 0.75), (xi, 0.75), color=ORANGE)
    xm, wm = 6.7, 2.9
    box(ax, xm, 1.6, wm, 3.8, "Learned\ntransition density", "$p_\\theta(a_{t+1}\\mid a_t, w_t, r_{t+1})$\nrolled out $h$ steps",
        "≈ the pricing model", fc="#eaf2fc", ec=BLUE, lw=1.6, tfs=9.5 * f, sfs=F8)
    ax.text(xm + wm / 2, 1.72, "learned from 25,600 pixels\n× 24 years", ha="center", va="bottom", fontsize=F7, color=MUTED, linespacing=1.0)
    for y in ys:
        arrow(ax, (xi + wi, y + hb / 2), (xm, y + hb / 2))
    arrow(ax, (xi + wi / 2, 1.5), (xm, 2.1), color=ORANGE, rad=-0.15)
    xo, wo = 10.4, 3.45
    box(ax, xo, ys[0], wo, hb, "Expected greenness", "$\\mathbb{E}[a_{t+h}]$ + calibrated interval", "≈ price and its distribution", ec=AQUA, tfs=T, sfs=S)
    box(ax, xo, ys[1], wo, hb, "Greeks", "$\\Delta,\\ \\Gamma,\\ \\Theta$, Vega", "exact derivatives of the rollout", ec=AQUA, tfs=T, sfs=S)
    box(ax, xo, ys[2], wo, hb, "Residual", "observed − expected, given rain", "≈ mispricing: attribution, anomaly", ec=AQUA, tfs=T, sfs=S)
    for y in ys:
        arrow(ax, (xm + wm, y + hb / 2), (xo, y + hb / 2))
    box(ax, xo, -0.05, wo, 1.55, "Observed greenness", "$a_{t+h}$ from the satellite", "≈ quoted option prices", fc="#f1f1ef", ec=MUTED, tfs=T, sfs=S)
    arrow(ax, (xo + wo / 2, 1.5), (xo + wo / 2, ys[2]), color=MUTED)
    arrow(ax, (xo, 0.3), (xi + wi, 0.3), color=ORANGE, ls="--", lw=1.4,
          label="infer: the rain distribution that reproduces the observed land (≈ implied volatility)",
          lpos=((xo + xi + wi) / 2, -0.15), lfs=F7)
    ax.text(0.15, 1.72, "Forecast mode: supply it", fontsize=F8, color=ORANGE, fontweight="bold", ha="left")
    ax.text(xo + wo, -0.65, "Attribution mode: infer it from what the land did", fontsize=F8, color=ORANGE, fontweight="bold", ha="right")
    fig.savefig(out, bbox_inches="tight", facecolor=SURF, dpi=220)
    plt.close(fig)


def draw_strip(out, f=1.0):
    """Figure 1 strip: one transition, inverted and re-used. Drawn at 6.2 in for a 5.5 in text width (~7 pt type)."""
    fig, ax = plt.subplots(figsize=(6.2, 1.55))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor(SURF)
    ax.set_xlim(-0.06, 7.06); ax.set_ylim(0, 1.75); ax.axis("off")
    T, S = 9 * f, 8 * f
    y, h, w, g = 0.03, 1.38, 1.55, 0.267
    xs = [i * (w + g) for i in range(4)]
    st = 0.19
    box(ax, xs[0], y, w, h, "1  State", "$a_t$: greenness\n$w_t$: rain memory\n$x$: pixel traits",
        "≈ spot price", tfs=T, sfs=S, step=st)
    box(ax, xs[1], y, w, h, "2  Transition", "$a_{t+1}=a_t+\\mu+\\sigma\\varepsilon_t$\n$w_{t+1}=\\lambda w_t+r_{t+1}$\none $\\theta$, all pixels",
        "≈ pricing formula", fc="#eaf2fc", ec=BLUE, lw=1.3, tfs=T, sfs=S, step=st)
    box(ax, xs[2], y, w, h, "3  Invert", "$\\hat\\varepsilon_t=(a_{t+1}-a_t-\\mu)/\\sigma$\none unknown, so\nunique given $\\theta$",
        "≈ implied volatility", fc="#fff3ec", ec=ORANGE, lw=1.3, tfs=T, sfs=S, step=st)
    box(ax, xs[3], y, w, h, "4  Re-use", "rain $r'$: counterfactual\n$\\partial_r,\\ \\partial_a$: $\\Delta,\\ \\Theta$\n$-\\log p_\\theta$: score",
        "≈ Greeks", ec=AQUA, lw=1.2, tfs=T, sfs=S, step=st)
    for i in range(3):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))
    # what is observed enters from above
    for k, lab in ((1, "rain $r_{t+1}$ (CHIRPS)"), (2, "observed $a_{t+1}$ (MODIS)")):
        ax.text(xs[k] + w / 2, 1.66, lab, ha="center", va="center", fontsize=S, color=MUTED)
        arrow(ax, (xs[k] + w / 2, 1.58), (xs[k] + w / 2, y + h), color=MUTED)
    fig.savefig(out, bbox_inches="tight", facecolor=SURF, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    draw_strip("paper/figs/fig1_strip.pdf")
    draw_strip("paper/figs/fig1_strip.png")
    draw_paper("paper/figs/fig0_diagram.pdf")
    draw("paper/figs/fig0_diagram.png")
    print("diagram written")
