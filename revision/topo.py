"""Site traits from the Copernicus DEM GLO-30 (public AWS bucket): elevation, slope and
topographic position for every grid pixel of both boxes.

GLO-30 pixels are 1 arcsecond; the target grid is 9 arcseconds (aoi.RES = 0.0025 degrees), so
slope is computed at 30 m and every trait is an exact 9x9 block mean: no resampling.

  .venv/bin/python revision/topo.py     -> data/<box>/topo.npz (elev km, slope m/m, tpi 100 m)
"""
import os
import sys

import numpy as np
import rasterio
from rasterio.merge import merge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aoi  # noqa: E402

TILE = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/"
        "Copernicus_DSM_COG_10_{t}_DEM.tif")
TILES = ["N00_00_E036_00", "N00_00_E037_00"]
K = 9          # 1-arcsecond pixels per grid pixel
TPI_PX = 9     # topographic position: elevation minus the mean of a 9x9-pixel (~2.5 km) neighbourhood


def build(box):
    w, s, e, n = aoi.BOXES[box]
    m = TPI_PX // 2 * aoi.RES  # margin so the neighbourhood mean is defined at the box edge
    srcs = [rasterio.open(TILE.format(t=t)) for t in TILES]
    z, tr = merge(srcs, bounds=(w - m, s - m, e + m, n + m), res=aoi.RES / K)
    z = z[0].astype(np.float64)
    lat = tr.f + (np.arange(z.shape[0]) + 0.5) * tr.e
    dy = abs(tr.e) * 111_320.0
    dx = tr.a * 111_320.0 * np.cos(np.deg2rad(lat))[:, None]
    gy, gx = np.gradient(z)
    slope = np.hypot(gx / dx, gy / dy)

    def blocks(a):
        H, W = (a.shape[0] // K) * K, (a.shape[1] // K) * K
        return a[:H, :W].reshape(H // K, K, W // K, K).mean((1, 3))

    zb, sb = blocks(z), blocks(slope)
    from scipy.ndimage import uniform_filter
    tpi = zb - uniform_filter(zb, TPI_PX, mode="nearest")
    c = TPI_PX // 2
    cut = (slice(c, c + aoi.N), slice(c, c + aoi.N))
    out = {"elev": zb[cut] / 1000.0, "slope": sb[cut], "tpi": tpi[cut] / 100.0}
    assert all(v.shape == (aoi.N, aoi.N) and np.isfinite(v).all() for v in out.values())
    np.savez_compressed(f"data/{box}/topo.npz", **{k: v.astype(np.float32) for k, v in out.items()})
    print(box, {k: (round(float(v.min()), 3), round(float(v.mean()), 3), round(float(v.max()), 3)) for k, v in out.items()})


if __name__ == "__main__":
    for b in ("A", "B"):
        build(b)
