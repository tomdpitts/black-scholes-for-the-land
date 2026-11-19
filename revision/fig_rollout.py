"""Rollout diagram after Ha & Schmidhuber (2018) Fig. 4: static traits once, three steps, two outputs. Mostly visual.

Static traits are the six real Box A rasters, drawn as a layer cake. Each step takes that fortnight's real rain and
evaporative demand (Box A, March-April 2022) into the shared network, alongside the carried state (greenness with
its 5x5 neighbourhood, and water memory). Its mean and spread set the predicted greenness, on which the observed
greenness lands; the gap is the implied residual. Over the season this feeds attribution and sensitivities.

  .venv/bin/python revision/fig_rollout.py   -> paper/figs/schematic/fig_rollout.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

plt.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
                     "lines.solid_capstyle": "butt"})  # lines end exactly where they start: no overshoot
BLUE, DARK_BLUE, ORANGE, SAGE, OCHRE = "#61adf2", "#2e78c0", "#ee814d", "#5cbf9f", "#f7bc50"
DARK_SAGE = "#2f7a5f"
VIOLET = "#9080d7"  # linear skip
INK, MUTED, LIGHT = "#2b2620", "#6b645a", "#cfc8bc"
BLUE_TXT = "#3d8fd6"
OUT = "paper/figs/schematic/fig_rollout"
W, H = 7.0, 3.93  # inches; AAAI full width
YPAD = 0.10  # extra height below y = 0 for the output maps and their colour bars
STEPS = (463, 464, 465)  # 2022-04-07, -04-23, -05-09: 84, 34, 11 mm, all below normal, demand above: drying
FS = 7.2
BOX = dict(boxstyle="round,pad=0,rounding_size=0.035", lw=0.8, ec=INK)  # every box: one border, gentle corners


def ramp(*cols):
    return LinearSegmentedColormap.from_list("r", cols)


def tint(c, a):
    return tuple(1 - a * (1 - v) for v in to_rgb(c))


def load():
    d = np.load("data/A/built.npz")
    t = np.load("data/A/topo.npz")
    e = np.load("data/A/era5.npz")
    traits = [  # top slice first
        ("mean NDVI", d["x"][..., 0], ramp("#ffffff", SAGE, DARK_SAGE)),
        ("seasonal amplitude", d["x"][..., 1], ramp("#ffffff", "#9ab04d", "#5a6b25")),
        ("annual rain", d["x"][..., 2], ramp("#ffffff", BLUE, DARK_BLUE)),
        ("elevation", t["elev"], ramp("#ffffff", OCHRE, "#a8742a")),
        ("slope", t["slope"], ramp("#ffffff", "#9c958a", "#4a453e")),
        ("topographic position", t["tpi"], "tpi"),
    ]
    return traits, [d["r"][i] for i in STEPS], [e["pet"][i] for i in STEPS]


def main():
    traits, rain, pet = load()
    fig = plt.figure(figsize=(W, H + YPAD))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(-YPAD, H); ax.axis("off")

    def t(x, y, s, **k):
        return ax.text(x, y, s, ha=k.pop("ha", "center"), va=k.pop("va", "center"), color=k.pop("color", INK),
                       fontsize=k.pop("fontsize", FS), **k)

    def arrow(x0, y0, x1, y1, col=INK, lw=0.9, z=3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=3.4,head_width=2.0",
                                     color=col, lw=lw, shrinkA=0, shrinkB=0, zorder=z, capstyle="butt"))

    def slab(img, x0, y0, w, dx, dh, cmap, norm=None, z=2, interp="bilinear"):
        """img on the parallelogram (x0,y0) + u(w,0) + v(dx,dh): a 2-D slice lying flat."""
        im = ax.imshow(np.ma.masked_invalid(img[::-1]), extent=(0, 1, 0, 1), cmap=cmap, norm=norm, interpolation=interp,
                       origin="lower", zorder=z)
        im.set_transform(Affine2D(np.array([[w, dx, x0], [0, dh, y0], [0, 0, 1]])) + ax.transData)
        ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w + dx, y0 + dh), (x0 + dx, y0 + dh)],
                             closed=True, fc="none", ec=INK, lw=0.45, zorder=z + 0.1))

    def box(x0, y0, w, h, fc="white", z=1):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, fc=fc, zorder=z, **BOX))

    def group(x0, y0, x1, y1, label, at=None):
        """Thin frame with its label cut into the border: top right by default, or at=(x, y) right-aligned."""
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.035",
                                    fc="none", ec="#9c958a", lw=0.6, zorder=8))
        lx_, ly_ = at if at else (x1 - 0.12, y1)
        t(lx_, ly_, label, ha="right", fontsize=7.8, fontweight="bold", zorder=9,
          bbox=dict(fc="white", ec="none", pad=1.5))

    top = 3.78
    xs = (3.05, 4.65, 6.25)  # step centres
    bw, bh, by = 0.80, 0.74, 1.98  # network boxes
    bus_y = by + bh + 0.20
    frame_r = 2.38  # right edge shared by the static-traits and dynamic-state frames

    # ---------------------------------------------------------------- static traits: the layer cake, once
    sw, sdx, sdh, gap = 0.52, 0.25, 0.20, 0.118
    cx0 = 0.50
    ys = [bus_y + 0.15 - 0.22 + (5 - i) * gap + 0.05 * ((i < 2) + (i < 3)) for i in range(6)]
    # each slice shows only the band below the slice above it
    band = [(ys[i], ys[i] + sdh if i == 0 else ys[i - 1]) for i in range(6)]
    fb = ys[-1] - 0.07
    # label on the bottom border, just before the line that leaves the frame
    group(0.06, fb, frame_r, ys[0] + sdh + 0.07, "Static traits", at=(frame_r - 0.12, fb))
    # brackets: which product each trait comes from, spanning exactly the visible edges of its slices
    for lab, g0, g1 in (("MODIS", 0, 1), ("CHIRPS", 2, 2), ("DEM", 3, 5)):
        # anchored on the slices' bottom-left corners, one tick per slice
        corners = [ys[i] for i in range(g0, g1 + 1)]
        y_hi, y_lo = max(corners) + 0.02 * (g0 == g1), min(corners) - 0.02 * (g0 == g1)
        bxl = cx0 - 0.05
        ax.plot([bxl, bxl], [y_lo, y_hi], color=MUTED, lw=0.7)
        for yc in corners:
            ax.plot([bxl, cx0 - 0.012], [yc, yc], color=MUTED, lw=0.7)
        t(bxl - 0.045, (y_hi + y_lo) / 2, lab, ha="right", fontsize=6.2, color=MUTED)
    for i in reversed(range(6)):  # bottom slice first so upper slices sit on top
        name, img, cmap = traits[i]
        if cmap == "tpi":
            v = np.nanpercentile(np.abs(img), 98)
            cmap, norm = ramp("#a8742a", "#ffffff", SAGE), TwoSlopeNorm(0, -v, v)
        else:
            norm = plt.Normalize(*np.nanpercentile(img, (2, 98)))
        slab(img, cx0, ys[i], sw, sdx, sdh, cmap, norm, z=2 + (6 - i) * 0.2)
        ly = ys[i] + sdh  # anchored on the slice's top-right corner, which clears the slice above
        ex0 = cx0 + sw + sdx
        lx = cx0 + sw + sdx + 0.08
        ax.plot([ex0 + 0.01, lx - 0.03], [ly, ly], color=LIGHT, lw=0.5, zorder=4)
        t(lx, ly, name, ha="left", fontsize=6.6)

    ax.plot([frame_r, xs[-1]], [bus_y, bus_y], color=INK, lw=0.8, zorder=1)  # leaves the frame's right edge
    for cx in xs:
        arrow(cx, bus_y, cx, by + bh, lw=0.8)

    # ---------------------------------------------------------------- state: greenness with its 5x5 neighbours, water memory
    sy = by + bh / 2
    c, mw, sgap = 0.055, 0.07, 0.07  # grid cell, gauge width, space between grid and gauge
    sw_state = 5 * c + sgap + mw

    def state(x0, level, fade, seed, label=False):
        """Greenness of a pixel and its 5x5 neighbourhood, and the water-memory gauge, at one step."""
        rng = np.random.default_rng(7)  # one spatial pattern at every step: the same place, drying
        gy = sy - 2.5 * c
        green = ramp("#ffffff", SAGE, DARK_SAGE)
        for i in range(5):
            for j in range(5):
                w_ = (i + j) / 8 if seed == "tr" else ((4 - i) + (4 - j)) / 8  # which corner is greener
                v = 1.0 if i == 2 and j == 2 else 0.12 + 0.62 * w_ ** 1.4 + 0.12 * rng.random()
                fc = green(v * fade)  # greener towards the top-right corner; everything pales as it dries
                ax.add_patch(Rectangle((x0 + i * c, gy + j * c), c, c, fc=fc, ec="white", lw=0.5, zorder=3))
        mx = x0 + 5 * c + sgap
        ax.add_patch(Rectangle((mx, gy), mw, 5 * c * level, fc=BLUE, ec="none", zorder=3))
        ax.add_patch(FancyBboxPatch((mx, gy), mw, 5 * c, boxstyle="round,pad=0,rounding_size=0.012", fc="none",
                                    ec=INK, lw=0.6, zorder=4))
        if label:
            t(x0 - 0.06, sy, "5×5 pixel\ngreenness", fontsize=6.6, color=MUTED, ha="right",
              linespacing=1.0)
            t(mx + mw + 0.06, sy, "water\nmemory", fontsize=6.6, color=MUTED, ha="left", linespacing=1.0)

    # the state each step hands to the next: in the drought, greenness pales and the water runs down
    states = ((0.55, 1.00, "tr"), (0.40, 0.80, "tr"), (0.27, 0.62, "bl"))
    dl, dr = 0.62, frame_r  # dynamic-state frame
    x_first = (dl + dr) / 2 - sw_state / 2
    state(x_first, *states[0], label=True)
    group(dl, sy - 2.5 * c - 0.12, dr, sy + 2.5 * c + 0.14, "Dynamic state")
    arrow(dr, sy, xs[0] - bw / 2, sy)
    for k in range(2):
        l, r_ = xs[k] + bw / 2, xs[k + 1] - bw / 2
        x0 = (l + r_ - 0.12) / 2 - sw_state / 2  # centred in the gap before the skip branch
        ax.plot([l, x0 - 0.05], [sy, sy], color=INK, lw=0.9, zorder=3)
        state(x0, *states[k + 1])
        arrow(x0 + sw_state + 0.05, sy, r_, sy)
    arrow(xs[-1] + bw / 2, sy, W - 0.06, sy)

    # ---------------------------------------------------------------- three steps
    tw, tdx, tdh, ty = 0.38, 0.18, 0.15, 3.36  # driver tiles
    wb = bus_y + 0.08  # bottom of the weather-drivers frame, just above the traits line
    group(xs[0] - 0.58, wb, xs[-1] + 0.68, ty + tdh + 0.08, "Weather drivers")
    rnorm = plt.Normalize(0, max(np.nanpercentile(r, 98) for r in rain))
    pnorm = plt.Normalize(min(np.nanmin(p) for p in pet) * 0.9, max(np.nanmax(p) for p in pet) * 1.05)
    yb, pk, sg = 1.30, 0.30, 0.105  # bells
    for k, cx in enumerate(xs):
        t(cx, top, ("$t$", "$t+1$", "$t+2$")[k], fontsize=9.5)
        rx, ex = cx - 0.54, cx + 0.04
        slab(rain[k], rx, ty, tw, tdx, tdh, ramp("#ffffff", "#4892d9", "#1f5f9e"), rnorm)
        slab(pet[k], ex, ty, tw, tdx, tdh, ramp("#ffffff", "#cfe6fb", BLUE), pnorm)
        rc, ec = rx + (tw + tdx) / 2, ex + (tw + tdx) / 2
        t(rc, ty - 0.08, "rain", color=DARK_BLUE, va="top", fontsize=6.8)
        t(ec, ty - 0.08, "evaporative\ndemand", color=BLUE_TXT, va="top", linespacing=1.0, fontsize=6.8)
        for x_, y0_, col in ((rc, wb, DARK_BLUE), (ec, wb, BLUE_TXT)):
            ax.plot([x_, x_], [bus_y - 0.035, bus_y + 0.035], color="white", lw=3.5, zorder=2)
            arrow(x_, y0_, x_, by + bh, col=col)

        # the network: one set of weights, shared by every step and pixel
        box(cx - bw / 2, by, bw, bh, z=5)
        # inside: a linear part plus an MLP correction, summed
        py_ = by + 0.36
        lx_ = cx - 0.20  # linear: a straight line on small axes
        ax.plot([lx_ - 0.10, lx_ - 0.10, lx_ + 0.10], [py_ + 0.10, py_ - 0.09, py_ - 0.09], color=MUTED, lw=0.6,
                zorder=6)
        ax.plot([lx_ - 0.08, lx_ + 0.09], [py_ - 0.06, py_ + 0.08], color=VIOLET, lw=1.5, zorder=7,
                solid_capstyle="round")
        t(cx, py_, "+", fontsize=10, zorder=7)
        mx_ = cx + 0.21  # MLP: 3 inputs, 4 hidden units, 2 outputs
        rows = [(py_ + 0.10, 3), (py_, 4), (py_ - 0.10, 2)]
        nodes = [[(mx_ + (i - (n - 1) / 2) * 0.072, y) for i in range(n)] for y, n in rows]
        for a_, b_ in zip(nodes, nodes[1:]):
            for p in a_:
                for q in b_:
                    ax.plot([p[0], q[0]], [p[1], q[1]], color=LIGHT, lw=0.4, zorder=6)
        for li, row in enumerate(nodes):
            for p in row:
                ax.add_patch(Circle(p, 0.015, fc=INK if li == 2 else "white", ec=INK, lw=0.6, zorder=7))
        t(lx_, by + 0.12, "linear", fontsize=6.2, color=MUTED, zorder=7)
        t(mx_, by + 0.12, "MLP", fontsize=6.2, color=MUTED, zorder=7)
        t(cx, by + bh - 0.11, "M4", fontsize=8, fontweight="bold", zorder=7)

        # mean and spread leave the two output nodes and set the predicted greenness
        u = np.linspace(-0.46, 0.46, 200)
        g = yb + pk * np.exp(-0.5 * (u / sg) ** 2)
        ax.fill_between(cx + u, yb, g, color=SAGE, alpha=0.20, lw=0, zorder=2)
        ax.plot(cx + u, g, color=SAGE, lw=1.2, zorder=3)
        ax.plot([cx - 0.48, cx + 0.48], [yb, yb], color=INK, lw=0.6, zorder=3)
        hy = yb + pk * np.exp(-0.5)
        ax.add_patch(FancyArrowPatch((cx, hy), (cx + sg, hy), arrowstyle="|-|,widthA=1.6,widthB=1.6",
                                     color=INK, lw=0.6, shrinkA=0, shrinkB=0, zorder=4))
        # mu straight down through the sum node, which adds the incoming greenness; sigma straight down to the spread
        px, py, pr = cx, by - 0.20, 0.035
        arrow(cx, by, cx, py + pr)
        arrow(cx, py - pr, cx, yb + pk + 0.01)
        ax.add_patch(Circle((px, py), pr, fc="white", ec=INK, lw=0.8, zorder=7))
        ax.plot([px - pr * 0.6, px + pr * 0.6], [py, py], color=INK, lw=0.8, zorder=8)
        ax.plot([px, px], [py - pr * 0.6, py + pr * 0.6], color=INK, lw=0.8, zorder=8)
        jx = cx - bw / 2 - 0.12  # greenness branches off the incoming state
        ax.plot([jx, jx], [sy, py], color=INK, lw=0.7, zorder=3)
        arrow(jx, py, px - pr, py, lw=0.7)
        arrow(cx + sg, by, cx + sg, hy + 0.01)
        t(cx - 0.05, (by + py) / 2 + 0.01, r"$\mu$", fontsize=9, ha="right")
        t(cx + sg + 0.05, (by + hy) / 2, r"$\sigma$", fontsize=9, ha="left")
        # observed greenness on the axis; the gap to the mean, on the axis, is the implied residual
        ob = -sg * (1.5, 2.0, 2.5)[k]  # the drought deepens: 1.5, 2 and 2.5 sigma below the mean
        ax.plot([cx + ob, cx], [yb, yb], color=ORANGE, lw=2.4, solid_capstyle="butt", zorder=4)
        ax.plot([cx, cx], [yb, yb + pk], color=SAGE, lw=0.7, ls=(0, (2, 1.5)), zorder=3)
        ax.plot(cx + ob, yb, "s", ms=4.8, color=DARK_SAGE, mec="white", mew=0.8, zorder=6)  # a pixel, as the grid centre
        t(cx - 0.055, yb + 0.06, r"$\hat\varepsilon$", fontsize=7.5, color=ORANGE, zorder=6)  # inside the bell, over its gap, near the mean where the bell is tall
        # inverting: the observed greenness, not the model's own mean, is carried into the next step's state
        if k < 2:
            gxs = (xs[k] + bw / 2 + xs[k + 1] - bw / 2 - 0.12) / 2 - sw_state / 2 + 2.5 * c  # next 5x5 icon, centre
            yl = yb - 0.08
            ax.plot([cx + ob, cx + ob, gxs, gxs], [yb - 0.03, yl, yl, sy - 2.5 * c - 0.06], color=DARK_SAGE,
                    lw=0.8, ls=(0, (2.5, 1.8)), zorder=3, solid_joinstyle="miter")
            arrow(gxs, sy - 2.5 * c - 0.07, gxs, sy - 2.5 * c - 0.005, col=DARK_SAGE, lw=0.8)

    # ---------------------------------------------------------------- key
    kx, ky, kw, kh = 0.06, 1.12, 1.62, 0.60
    ax.add_patch(FancyBboxPatch((kx, ky), kw, kh, boxstyle="round,pad=0,rounding_size=0.035", fc="none",
                                ec="#9c958a", lw=0.6))
    for j, (draw, lab) in enumerate((("bell", "predicted greenness"), ("dot", "observed greenness"),
                                     ("gap", r"implied residual $\hat\varepsilon$"))):
        y = ky + kh - 0.15 - j * 0.155
        x0 = kx + 0.12
        if draw == "bell":
            u = np.linspace(-0.09, 0.09, 50)
            g = y - 0.05 + 0.11 * np.exp(-0.5 * (u / 0.03) ** 2)
            ax.fill_between(x0 + 0.09 + u, y - 0.05, g, color=SAGE, alpha=0.2, lw=0, zorder=2)
            ax.plot(x0 + 0.09 + u, g, color=SAGE, lw=1.1, zorder=3)
        elif draw == "dot":
            ax.plot(x0 + 0.09, y, "s", ms=4.8, color=DARK_SAGE, mec="white", mew=0.8, zorder=3)
        else:
            ax.plot([x0, x0 + 0.18], [y, y], color=ORANGE, lw=2.4, solid_capstyle="butt", zorder=3)
        t(x0 + 0.28, y, lab, ha="left", fontsize=6.8)

    # ---------------------------------------------------------------- the season's rollout -> two outputs
    # in the visual language of the paper: Figure 1's split (BrBG, +-0.3) and Figure 5's Delta (YlGn, 0-0.14)
    M = np.load("paper/figs/schematic/rollout_maps.npz")  # revision/rollout_maps.py: Box A, MAM 2022, M4
    ry = 1.14
    x_l, x_r = xs[0] - 0.52, xs[-1] + 0.52
    oy, otop = -0.07, 1.00
    a_l, a_r, s_l, s_r = 0.08, 5.16, 5.32, 6.92
    a_in, s_in = 3.62, s_l + 0.32  # both land left of their cut-out labels
    ax.plot([x_l, x_l, x_r, x_r], [ry + 0.07, ry, ry, ry + 0.07], color=MUTED, lw=0.7)
    for x in (a_in, s_in):
        arrow(x, ry, x, otop, col=MUTED)
    group(a_l, oy, a_r, otop, "Attribution")
    group(s_l, oy, s_r, otop, "Sensitivities")

    m_, op_ = 0.62, 0.24  # map side, operator gap
    mb = 0.10  # map bottom
    split = [("Observed", "observed"), ("Baseline", "baseline"), ("Rain", "rain"), ("Evap. demand", "demand"),
             ("Residual", "residual")]
    x0m = (a_l + a_r) / 2 - (5 * m_ + 4 * op_) / 2
    ext_of = lambda x: (x, x + m_, mb, mb + m_)
    xpos = [x0m + i * (m_ + op_) for i in range(5)]

    def square(img, x, cmap, vmin, vmax):
        ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                  extent=ext_of(x), origin="upper", zorder=3)
        ax.add_patch(Rectangle((x, mb), m_, m_, fc="none", ec="#c3c2b7", lw=0.6, zorder=4))

    for i, ((title, key), x) in enumerate(zip(split, xpos)):
        square(M[key], x, "BrBG", -0.3, 0.3)
        # titles carry the attribution colours: weather in blue, implied residual in orange
        t(x + m_ / 2, mb + m_ + 0.08, title, fontsize=7, fontweight="bold",
          color={"rain": BLUE_TXT, "demand": BLUE_TXT, "residual": ORANGE}.get(key, INK))
        if i < 4:
            t(x + m_ + op_ / 2, mb + m_ / 2, "=" if i == 0 else "+", fontsize=12, color=MUTED)
    # Delta as Figure 5 (YlGn, 0-0.14); Gamma, its curvature, diverging about zero: violet saturating, green not
    gap_ = 0.14
    xd = (s_l + s_r) / 2 - m_ - gap_ / 2
    square(M["delta"], xd, "YlGn", 0, 0.14)
    # Gamma as it is, on Delta's numbers (+-0.14): NDVI per (100 mm)^2 beside NDVI per 100 mm
    gv = 0.14
    yg = plt.get_cmap("YlGn")  # Gamma's positive side borrows Delta's greens: first and second order of one response
    # positive half is exactly YlGn as Delta uses it (value v gets YlGn(v / 0.14)); negative half runs to violet
    pos = [yg(x) for x in np.linspace(0, 1, 9)]
    neg = [tuple((1 - f) * np.array(to_rgb(yg(0.0))) + f * np.array(to_rgb("#4a3a98"))) for f in np.linspace(1, 0, 9)]
    gcm = ramp(*neg[:-1], *pos)
    square(M["gamma"], xd + m_ + gap_, gcm, -gv, gv)

    # slim colour bars beneath each panel
    def cbar(x0, x1, cmap, lo, hi, ticks, label):
        by_, bh_ = mb - 0.10, 0.035
        cax = fig.add_axes([x0 / W, (by_ + YPAD) / (H + YPAD), (x1 - x0) / W, bh_ / (H + YPAD)])
        cb = ColorbarBase(cax, cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap, norm=Normalize(lo, hi),
                          orientation="horizontal", ticks=ticks)
        cb.outline.set_linewidth(0.4)
        cb.set_ticks([])  # no numbers: the bars show direction and footing, the figures in the paper give values
        t(x0 - 0.05, by_ + bh_ / 2, label, fontsize=6.2, ha="right", color=MUTED, linespacing=1.0)  # unit, left end

    cbar(xpos[1], xpos[4] + m_, "BrBG", -0.3, 0.3, [-0.3, 0, 0.3], "NDVI")
    cbar(xd + 0.72, xd + 2 * m_ + gap_ - 0.06, gcm, -0.14, 0.14, [-0.14, 0, 0.14],
         "per 100 mm rain")  # exact for both: Delta is NDVI per 100 mm, Gamma is Delta per 100 mm
    for x_, g_ in ((xd, r"Delta $\Delta$"), (xd + m_ + gap_, r"Gamma $\Gamma$")):
        t(x_ + m_ / 2, mb + m_ + 0.08, g_, fontsize=7, fontweight="bold")  # as the attribution titles

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, facecolor="white")
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
