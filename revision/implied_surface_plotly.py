"""Implied residual surface rendered with Plotly (same cell means as implied_surface.py, read from surface.json).

Faceted: the surface through the cell means, nothing added; grey lines mark the cells.
Colour: BrBG, as Figure 1 (brown = browner than expected, teal = greener). Flatness, not zero, is the test
(the level is the land's signal plus drift).

  .venv/bin/python revision/implied_surface_plotly.py runs/<dir>
      -> paper/figs/implied_surface/plotly_faceted_col.png   (stacked, one paper column; the paper's figure)
"""
import json
import sys

import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps
from plotly.subplots import make_subplots

FIGS = "paper/figs/implied_surface"
METHODS = ("z-score", "RESTREND", "M4")
TITLE = {"z-score": "z-score", "RESTREND": "RESTREND", "M4": "<b>M4 (ours)</b>"}
INK, MUTED, LIGHT, GRID = "#2b2620", "#6b645a", "#cfc8bc", "#ebe5da"
FONT = "Open Sans, Helvetica Neue, Arial, sans-serif"
VMAX = 0.2
ZR = (-0.21, 0.16)
ZERO = np.array([0.86, 0.84, 0.80])  # warm grey for zero, so a flat surface stays visible on white


def _brbg(v):
    """BrBG (Figure 1) with its near-white centre pulled toward a warm grey; ends unchanged."""
    c = np.array(colormaps["BrBG"](v)[:3])
    w = max(0.0, 1 - abs(v - 0.5) / 0.06)  # 1 at zero, 0 beyond |v - 0.5| = 0.06 (+-0.024 NDVI)
    return "rgb({},{},{})".format(*(int(255 * x) for x in (1 - w) * c + w * ZERO))


BRBG = [[v, _brbg(v)] for v in sorted(set(np.round(np.r_[np.linspace(0, 1, 21), 0.5 + np.linspace(-0.06, 0.06, 7)], 4)))]


XLAB = "Water balance vs normal (sd)"


def figure(J, layout="row"):
    """layout "row": three panels side by side (7 in, talk / arXiv). "col": stacked, for one paper column (3.3 in)."""
    E = np.array(J["edges_sd"])
    xc = (E[:-1] + E[1:]) / 2
    days = 16.0 * np.arange(1, len(J["n_years"]) + 1)
    col = layout == "col"
    fs = dict(tick=10, axis=11, title=13, sub=11) if col else dict(tick=12, axis=13, title=15, sub=12)
    if col:
        fig = make_subplots(rows=3, cols=1, specs=[[{"type": "surface"}]] * 3, vertical_spacing=0.0)
    else:
        fig = make_subplots(rows=1, cols=3, specs=[[{"type": "surface"}] * 3], horizontal_spacing=0.0)
    for k, m in enumerate(METHODS):
        z = np.array(J[m]["cell_mean"], float)
        cb = (dict(orientation="h", x=0.5, xanchor="center", y=-0.075, yanchor="top", len=0.7, thickness=10)
              if col else dict(len=0.6, thickness=12, x=1.0, y=0.45))
        fig.add_trace(go.Surface(
            x=xc, y=days, z=z, colorscale=BRBG, cmin=-VMAX, cmax=VMAX, showscale=(k == 2),
            colorbar=dict(cb, title=dict(text="Residual (NDVI)" if col else "Residual<br>(NDVI)",
                                         side="bottom" if col else "top", font=dict(size=fs["axis"], color=INK)),
                          tickvals=[-0.2, -0.1, 0, 0.1, 0.2], tickfont=dict(size=fs["tick"], color=MUTED), outlinewidth=0),
            lighting=dict(ambient=0.85, diffuse=0.3, specular=0.05, roughness=1.0, fresnel=0.0),
            lightposition=dict(x=-1e4, y=-2e4, z=3e4),
            contours=dict(x=dict(show=True, start=xc[0], end=xc[-1], size=1.0, color="#a39c90", width=2),
                          y=dict(show=True, start=days[0], end=days[-1], size=16.0, color="#a39c90", width=2)),
            hovertemplate="water balance %{x:.1f} sd<br>horizon %{y:.0f} d<br>residual %{z:.3f}<extra></extra>"),
            *((k + 1, 1) if col else (1, k + 1)))
        ax = dict(showbackground=True, backgroundcolor="white", gridcolor=GRID, zerolinecolor=LIGHT, linecolor=LIGHT,
                  tickfont=dict(size=fs["tick"], color=MUTED), title_font=dict(size=fs["axis"], color=INK))
        last = k == len(METHODS) - 1
        fig.update_layout(**{f"scene{k + 1 if k else ''}": dict(
            xaxis=dict(ax, title="" if col else XLAB, range=[E[0], E[-1]], tickvals=[-3, 0, 3, 6]),
            yaxis=dict(ax, title="" if col else "Horizon (days)", range=[days[0], days[-1]], tickvals=[16, 48, 96],
                       ticktext=["16 d", "48 d", "96 d"] if col else None),
            zaxis=dict(ax, title="" if col else "Residual (NDVI)", range=list(ZR), tickvals=[-0.2, -0.1, 0, 0.1]),
            aspectmode="manual", aspectratio=dict(x=1.5, y=0.9, z=0.6) if col else dict(x=1.3, y=1.0, z=0.75),
            camera=dict(eye=dict(x=0.72, y=-1.72, z=0.8) if col else dict(x=0.95, y=-2.25, z=1.15),
                        center=dict(x=0, y=0, z=-0.15)),
            annotations=[dict(x=x0, y=days[0], z=ZR[0], text=t, showarrow=False, xanchor=a, yshift=-34 if col else -40,
                              font=dict(size=fs["axis"], color=MUTED))
                         for x0, t, a in ((E[0], "\u2190 drier", "right"), (E[-1], "wetter \u2192", "left"))]
                        if not col else [])})
        dom = fig.layout[f"scene{k + 1 if k else ''}"].domain
        txt = (f"{TITLE[m]}<br><span style='font-size:{fs['sub']}px;color:{MUTED}'>"
               f"Spread {J[m]['spread_over_valid_cells']:.3f} NDVI</span>")
        if col:
            fig.add_annotation(text=txt, x=0.0, xanchor="left", y=dom.y[1] - 0.01, yanchor="top", xref="paper",
                               yref="paper", showarrow=False, align="left", font=dict(size=fs["title"], color=INK))
        else:
            fig.add_annotation(text=txt, x=sum(dom.x) / 2, xanchor="center", y=0.98, yanchor="top", xref="paper",
                               yref="paper", showarrow=False, font=dict(size=fs["title"], color=INK))
    fig.update_layout(font=dict(family=FONT, color=INK), paper_bgcolor="white",
                      width=500 if col else 1500, height=640 if col else 520,
                      margin=dict(l=0, r=0, t=0, b=95) if col else dict(l=0, r=10, t=40, b=0))
    if col:
        fig.add_annotation(text=f"<span style='color:{MUTED}'>\u2190 drier</span>      {XLAB}      "
                                f"<span style='color:{MUTED}'>wetter \u2192</span>", x=0.5, y=-0.005, xref="paper",
                           yref="paper", yanchor="top", showarrow=False, font=dict(size=fs["axis"], color=INK))
    return fig


def main(out):
    J = json.load(open(f"{out}/surface.json"))
    name, layout = "plotly_faceted_col", "col"
    fig = figure(J, layout)
    fig.write_image(f"{FIGS}/{name}.png", scale=5)
    print("wrote", f"{FIGS}/{name}.png")


if __name__ == "__main__":
    main(sys.argv[1])
