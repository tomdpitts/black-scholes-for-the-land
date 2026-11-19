"""Slide versions of the Figure (map) box panels: Sentinel-2 true colour of Box A and Box B, with lon/lat ticks.

  python mapfig_slide.py  -> paper/figs/slide_box_A.png / slide_box_B.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aoi
from mapfig import true_colour

NPIX = 900  # ~49 m pixels over 0.40 deg; S2 overview 1 is 40 m

for b, name, c in (("A", "Box A: Lewa–Borana–Il Ngwesi–Mukogodo", "#2a78d6"),
                   ("B", "Box B: Mpala–Ol Jogi–Loisaba", "#eb6834")):
    rgb, dates = true_colour(b, NPIX=NPIX, ovr=1)
    w, s, e, n = aoi.BOXES[b]
    fig, ax = plt.subplots(figsize=(5.5, 5.8), constrained_layout=True)
    ax.imshow(rgb, extent=(w, e, s, n), interpolation="bilinear")
    for sp in ax.spines.values():
        sp.set_color(c); sp.set_linewidth(2.5)
    xs = [w, (w + e) / 2, e]; ys = [s, (s + n) / 2, n]
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.set_xticklabels([f"{x:.2f}°E" for x in xs], fontsize=11)
    ax.set_yticklabels([f"{y:.2f}°N" for y in ys], fontsize=11)
    ax.set_title(name, fontsize=13, loc="left", color=c)
    fig.savefig(f"paper/figs/slide_box_{b}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(b, "scenes:", dates)
