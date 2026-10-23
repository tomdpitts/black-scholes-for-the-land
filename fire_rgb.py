"""Sentinel-2 true colour of the exact Figure 2 crop (Box A rows 136-160, cols 50-89), just after the 2021-01-17 burn.

  .venv/bin/python fire_rgb.py  -> paper/figs/fig_fire_rgb.png, fig_fire_rgb_outline.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import aoi

R0, R1, C0, C1 = 136, 160, 50, 89                      # crop used by residual.py for Figure 2
W0, S0, E0, N0 = aoi.BOXES["A"]
W, E = W0 + C0 * aoi.RES, W0 + C1 * aoi.RES
N, S = N0 - R0 * aoi.RES, N0 - R1 * aoi.RES
RES = 0.00009                                          # ~10 m
NX, NY = int(round((E - W) / RES)), int(round((N - S) / RES))
TR = from_origin(W, N, RES, RES)
WINDOW = sys.argv[1] if len(sys.argv) > 1 else "2021-01-18/2021-03-05"
PICK = sys.argv[2] if len(sys.argv) > 2 else "first"
STEM = sys.argv[3] if len(sys.argv) > 3 else "fig_fire_rgb"


def read(href, res_scale=1, resampling=Resampling.bilinear):
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs="EPSG:4326", transform=TR, width=NX, height=NY, resampling=resampling, nodata=0) as v:
            return v.read(1).astype(np.float32)


def main():
    print(f"bbox W {W:.4f} S {S:.4f} E {E:.4f} N {N:.4f}  -> {NX} x {NY} px")
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=[W, S, E, N],
                            datetime=WINDOW, query={"eo:cloud_cover": {"lt": 40}}).items())
    print(len(items), "candidate scenes")
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                       GDAL_HTTP_MULTIRANGE="YES", GDAL_CACHEMAX=512)
    env.__enter__()
    scored = []
    for it in sorted(items, key=lambda i: i.datetime):
        scl = read(it.assets["SCL"].href, resampling=Resampling.nearest)
        valid = scl > 0
        cloud = np.isin(scl, [3, 8, 9, 10]) & valid     # shadow, cloud medium/high, cirrus
        cov, cf = valid.mean(), cloud.sum() / max(valid.sum(), 1)
        print(f"  {it.datetime.date()} tile {it.properties.get('s2:mgrs_tile')} coverage {cov:.2f} cloud-in-crop {cf:.3f}")
        if cov > 0.99 and cf < 0.02:
            scored.append(it)
    if not scored:
        raise SystemExit("no clear scene fully covering the crop")
    it = scored[0] if PICK == "first" else scored[-1]     # earliest after the burn, or latest before it
    base = float(it.properties.get("s2:processing_baseline", "0"))
    off = 1000 if base >= 4.0 else 0
    rgb = np.stack([(read(it.assets[b].href) - off) / 10000 for b in ("B04", "B03", "B02")], -1)
    env.__exit__(None, None, None)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)
    img = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.9
    date = it.datetime.date().isoformat()
    print("chosen", it.id, date, "baseline", base)
    np.savez_compressed(f"data/A/{STEM}.npz", rgb=(img[::2, ::2] * 255).astype(np.uint8),
                        bbox=np.array([W, S, E, N]), date=np.array(date), scene=np.array(it.id))

    d = np.load("data/A/built.npz")
    tb = int(np.where(d["dates"].astype("datetime64[D]") == np.datetime64("2021-01-17"))[0][0])
    burn = d["burn"][tb][R0:R1, C0:C1].astype(float)
    xs = W + (np.arange(C1 - C0) + 0.5) * aoi.RES
    ys = N - (np.arange(R1 - R0) + 0.5) * aoi.RES

    for outline, name in ((False, STEM), (True, STEM + "_outline")):
        fig, ax = plt.subplots(figsize=(3.2, 3.2 * NY / NX))
        ax.imshow(img, extent=(W, E, S, N), interpolation="lanczos")
        if outline:
            ax.contour(xs, ys, burn, levels=[0.5], colors="w", linewidths=0.8)
        ax.set_xlim(W, E); ax.set_ylim(S, N); ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(f"paper/figs/{name}.png", dpi=400)
        plt.close(fig)
    print("wrote paper/figs/fig_fire_rgb.png and fig_fire_rgb_outline.png, scene", date)


if __name__ == "__main__":
    main()
