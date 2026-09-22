"""Location figure: Kenya inset with the two boxes, and Sentinel-2 true colour of each box.

  python mapfig.py  -> paper/figs/fig_map.pdf / .png
"""
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import aoi

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
NPIX = 240


def kenya_outline():
    with urllib.request.urlopen(NE, timeout=60) as r:
        gj = json.load(r)
    for f in gj["features"]:
        if f["properties"].get("ADMIN") == "Kenya" or f["properties"].get("NAME") == "Kenya":
            g = f["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            return [np.array(p[0]) for p in polys]
    raise RuntimeError("Kenya not found")


def true_colour(box):
    w, s, e, n = aoi.BOXES[box]
    res = (e - w) / NPIX
    tr = from_origin(w, n, res, res)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=[w, s, e, n],
                            datetime="2023-01-15/2023-03-10", query={"eo:cloud_cover": {"lt": 15}}).items())
    items.sort(key=lambda it: it.properties["eo:cloud_cover"])
    # the box straddles tiles: take the two clearest scenes per MGRS tile so every part is covered
    per_tile = {}
    for it in items:
        per_tile.setdefault(it.properties.get("s2:mgrs_tile"), []).append(it)
    items = [it for tile in per_tile.values() for it in tile[:2]]
    acc = np.full((3, NPIX, NPIX), np.nan, np.float32)
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES", GDAL_CACHEMAX=512)
    env.__enter__()
    for it in items:
        for i, b in enumerate(("B04", "B03", "B02")):
            with rasterio.open(it.assets[b].href, overview_level=3) as src:
                with WarpedVRT(src, crs="EPSG:4326", transform=tr, width=NPIX, height=NPIX,
                               resampling=Resampling.average, nodata=0) as vrt:
                    arr = vrt.read(1).astype(np.float32)
            arr[arr == 0] = np.nan
            arr = (arr - 1000) / 10000  # processing baseline >= 04.00 offset
            fill = np.isnan(acc[i]) & ~np.isnan(arr)
            acc[i][fill] = arr[fill]
    env.__exit__(None, None, None)
    rgb = np.clip(acc / 0.28, 0, 1) ** 0.8
    rgb = np.moveaxis(np.nan_to_num(rgb, nan=1.0), 0, -1)
    return rgb, [(it.properties.get('s2:mgrs_tile'), it.datetime.date().isoformat()) for it in items]


def main():
    fig = plt.figure(figsize=(7.0, 2.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    for p in kenya_outline():
        ax0.fill(p[:, 0], p[:, 1], fc="#f1f1ef", ec="#52514e", lw=0.8)
    for b, c in (("A", "#2a78d6"), ("B", "#eb6834")):
        w, s, e, n = aoi.BOXES[b]
        ax0.add_patch(Rectangle((w, s), e - w, n - s, fc=c, ec=c, lw=1.2, alpha=0.9))
    ax0.plot(36.82, -1.29, "k.", ms=4); ax0.text(36.95, -1.45, "Nairobi", fontsize=6.5, va="top")
    ax0.plot(37.31, -0.15, "k^", ms=4); ax0.text(37.5, -0.55, "Mt Kenya", fontsize=6.5)
    ax0.text(37.6, 0.85, "A", color="#2a78d6", fontsize=8, fontweight="bold")
    ax0.text(35.9, 0.85, "B", color="#eb6834", fontsize=8, fontweight="bold")
    ax0.set_xlim(33.5, 42.2); ax0.set_ylim(-5, 5.3); ax0.set_aspect("equal")
    ax0.set_xticks([34, 36, 38, 40, 42]); ax0.set_yticks([-4, -2, 0, 2, 4])
    ax0.set_xticklabels([f"{x}°E" for x in (34, 36, 38, 40, 42)], fontsize=6)
    ax0.set_yticklabels([f"{abs(y)}°{'S' if y < 0 else 'N'}" if y else "0°" for y in (-4, -2, 0, 2, 4)], fontsize=6)
    ax0.set_title("Kenya", fontsize=8, loc="left")
    for sp in ax0.spines.values():
        sp.set_color("#c3c2b7")
    for k, (b, name, c) in enumerate((("A", "Box A: Lewa–Borana–Il Ngwesi–Mukogodo", "#2a78d6"),
                                      ("B", "Box B: Mpala–Ol Jogi–Loisaba", "#eb6834"))):
        ax = fig.add_subplot(gs[k + 1])
        rgb, dates = true_colour(b)
        w, s, e, n = aoi.BOXES[b]
        ax.imshow(rgb, extent=(w, e, s, n), interpolation="bilinear")
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(1.5)
        ax.set_xticks([w, (w + e) / 2, e]); ax.set_yticks([s, (s + n) / 2, n])
        ax.set_xticklabels([f"{x:.2f}°E" for x in (w, (w + e) / 2, e)], fontsize=6)
        ax.set_yticklabels([f"{y:.2f}°N" for y in (s, (s + n) / 2, n)], fontsize=6)
        ax.set_title(name, fontsize=7.5, loc="left", color=c)
        print(b, "scenes:", dates)
    fig.savefig("paper/figs/fig_map.pdf", bbox_inches="tight")
    fig.savefig("paper/figs/fig_map.png", bbox_inches="tight", dpi=200)
    print("map written")


if __name__ == "__main__":
    main()
