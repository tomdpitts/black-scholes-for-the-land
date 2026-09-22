"""Modal ingest app: MODIS MOD13Q1 NDVI, CHIRPS daily rain, MCD64A1 burned area,
Copernicus DEM, all warped onto the shared target grid in aoi.py and written to
local data/<box>/*.npz by the local entrypoint.

Run:
  modal run ingest.py --source modis --box A
  modal run ingest.py --source chirps           # both boxes at once
  modal run ingest.py --source fire  --box A
  modal run ingest.py --source dem   --box A
  modal run ingest.py --source check --box A    # prints coverage of what is on disk
"""
import io
import gzip
import datetime as dt

import modal

import aoi

app = modal.App("projectb-ingest")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "rasterio", "pystac-client", "planetary-computer", "requests")
    .add_local_python_source("aoi")
)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
CHIRPS_DAILY = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05"


# ----------------------------------------------------------------------------- helpers
def _warp_to_grid(href, box, resampling="nearest", band=1):
    """Read one raster (any CRS) onto the target grid for `box`. Returns float32 array
    with NaN for nodata."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    rs = {"nearest": Resampling.nearest, "average": Resampling.average,
          "bilinear": Resampling.bilinear}[resampling]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.hdf,.h5"):
        with rasterio.open(href) as src:
            nodata = src.nodata
            with WarpedVRT(src, crs="EPSG:4326", transform=aoi.grid_transform(box),
                           width=aoi.N, height=aoi.N, resampling=rs,
                           nodata=nodata, src_nodata=nodata) as vrt:
                arr = vrt.read(band).astype(np.float32)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def _stac_items(collection, box, start, end):
    import planetary_computer as pc
    import pystac_client

    cat = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    search = cat.search(collections=[collection], bbox=list(aoi.BOXES[box]),
                        datetime=f"{start}/{end}", limit=500)
    return list(search.items())


# ----------------------------------------------------------------------------- MODIS
@app.function(image=image, timeout=1800, retries=2)
def modis_year(box: str, year: int):
    """MOD13Q1 v061 NDVI and pixel reliability for one box-year.
    Returns (dates[T] as iso strings, ndvi[T,N,N] float32 scaled, rel[T,N,N] int8)."""
    import numpy as np

    items = _stac_items("modis-13Q1-061", box, f"{year}-01-01", f"{year}-12-31")
    by_date = {}
    for it in items:
        d = (it.properties.get("start_datetime") or it.properties["datetime"])[:10]
        by_date.setdefault(d, []).append(it)
    dates, ndvis, rels = [], [], []
    for d in sorted(by_date):
        acc_n, acc_r = [], []
        for it in by_date[d]:
            a = it.assets
            acc_n.append(_warp_to_grid(a["250m_16_days_NDVI"].href, box))
            acc_r.append(_warp_to_grid(a["250m_16_days_pixel_reliability"].href, box))
        n = np.nanmean(np.stack(acc_n), axis=0)
        r = np.nanmin(np.stack(acc_r), axis=0)  # keep best reliability where tiles overlap
        n = n * 1e-4
        n[(n < -0.3) | (n > 1.0)] = np.nan  # fill = -3000 -> -0.3
        r = np.where(np.isnan(r), -1, r).astype(np.int8)
        dates.append(d)
        ndvis.append(n.astype(np.float32))
        rels.append(r)
    if not dates:
        return [], None, None
    return dates, np.stack(ndvis), np.stack(rels)


# ----------------------------------------------------------------------------- CHIRPS
@app.function(image=image, timeout=1800, retries=3)
def chirps_month(year: int, month: int):
    """Daily CHIRPS v2 Africa p05, cropped to both boxes.
    Returns (dates, {box: [T, h, w] float32}, {box: (transform, shape)})."""
    import calendar

    import numpy as np
    import rasterio
    import requests
    from rasterio.io import MemoryFile
    from rasterio.windows import from_bounds

    out = {b: [] for b in aoi.BOXES}
    meta = {}
    dates = []
    s = requests.Session()
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        url = f"{CHIRPS_DAILY}/{year}/chirps-v2.0.{year}.{month:02d}.{day:02d}.tif.gz"
        r = s.get(url, timeout=120)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        raw = gzip.decompress(r.content)
        with MemoryFile(raw) as mf, mf.open() as src:
            for b, (w, so, e, n) in aoi.BOXES.items():
                win = from_bounds(w - 0.05, so - 0.05, e + 0.05, n + 0.05, src.transform)
                win = win.round_offsets().round_lengths()
                arr = src.read(1, window=win).astype(np.float32)
                arr[arr == src.nodata] = np.nan
                arr[arr < 0] = np.nan  # -9999 fill
                out[b].append(arr)
                if b not in meta:
                    meta[b] = (tuple(src.window_transform(win))[:6], arr.shape)
        dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates, {b: (np.stack(v) if v else None) for b, v in out.items()}, meta


# ----------------------------------------------------------------------------- FIRE
@app.function(image=image, timeout=1800, retries=2)
def fire_year(box: str, year: int):
    """MCD64A1 v061 Burn_Date (day of year, 0 = unburned) per month on the grid.
    Returns (months[T] iso strings, burn_doy[T,N,N] int16)."""
    import numpy as np

    items = _stac_items("modis-64A1-061", box, f"{year}-01-01", f"{year}-12-31")
    by_date = {}
    for it in items:
        d = (it.properties.get("start_datetime") or it.properties["datetime"])[:7]
        by_date.setdefault(d, []).append(it)
    months, arrs = [], []
    for d in sorted(by_date):
        acc = [_warp_to_grid(it.assets["Burn_Date"].href, box) for it in by_date[d]]
        a = np.nanmax(np.stack(acc), axis=0)
        a = np.where(np.isnan(a), -2, a)
        months.append(d)
        arrs.append(a.astype(np.int16))
    if not months:
        return [], None
    return months, np.stack(arrs)


# ----------------------------------------------------------------------------- DEM
@app.function(image=image, timeout=1800, retries=2)
def dem(box: str):
    import numpy as np

    items = _stac_items("cop-dem-glo-30", box, "1900-01-01", "2100-01-01")
    acc = [_warp_to_grid(it.assets["data"].href, box, resampling="average") for it in items]
    return np.nanmean(np.stack(acc), axis=0).astype(np.float32)


# ----------------------------------------------------------------------------- driver
@app.local_entrypoint()
def main(source: str, box: str = "A", start: int = 2002, end: int = 2025):
    import os
    import numpy as np

    os.makedirs(f"data/{box}", exist_ok=True)
    if source == "modis":
        years = list(range(start, end + 1))
        res = list(modis_year.map([box] * len(years), years))
        dates = sum([r[0] for r in res], [])
        ndvi = np.concatenate([r[1] for r in res if r[1] is not None])
        rel = np.concatenate([r[2] for r in res if r[2] is not None])
        # composites straddling a year boundary are returned by both year queries
        dates, keep = np.unique(np.array(dates), return_index=True)
        dates, ndvi, rel = list(dates), ndvi[keep], rel[keep]
        np.savez_compressed(f"data/{box}/modis.npz", dates=np.array(dates), ndvi=ndvi, rel=rel)
        good = (rel >= 0) & (rel <= 1) & ~np.isnan(ndvi)
        print(f"MODIS box {box}: {ndvi.shape}, {dates[0]}..{dates[-1]}, good-pixel fraction {good.mean():.3f}")
    elif source == "chirps":
        ym = [(y, m) for y in range(start, end + 1) for m in range(1, 13)]
        res = list(chirps_month.map([y for y, _ in ym], [m for _, m in ym]))
        dates = sum([r[0] for r in res], [])
        for b in aoi.BOXES:
            os.makedirs(f"data/{b}", exist_ok=True)
            rain = np.concatenate([r[1][b] for r in res if r[1][b] is not None])
            tr, shape = next(r[2][b] for r in res if b in r[2])
            np.savez_compressed(f"data/{b}/chirps_daily.npz", dates=np.array(dates), rain=rain,
                                transform=np.array(tr), shape=np.array(shape))
            print(f"CHIRPS box {b}: {rain.shape}, {dates[0]}..{dates[-1]}, nan fraction {np.isnan(rain).mean():.4f}")
    elif source == "fire":
        years = list(range(max(start, 2003), end + 1))
        res = list(fire_year.map([box] * len(years), years))
        months = sum([r[0] for r in res], [])
        burn = np.concatenate([r[1] for r in res if r[1] is not None])
        months, keep = np.unique(np.array(months), return_index=True)
        months, burn = list(months), burn[keep]
        np.savez_compressed(f"data/{box}/fire.npz", months=np.array(months), burn_doy=burn)
        print(f"FIRE box {box}: {burn.shape}, {months[0]}..{months[-1]}, burned-pixel-months {(burn > 0).sum()}")
    elif source == "dem":
        z = dem.remote(box)
        np.savez_compressed(f"data/{box}/dem.npz", elev=z)
        print(f"DEM box {box}: {z.shape}, {np.nanmin(z):.0f}..{np.nanmax(z):.0f} m")
    elif source == "check":
        for f in sorted(os.listdir(f"data/{box}")):
            d = np.load(f"data/{box}/{f}")
            print(f, {k: d[k].shape for k in d.files})
    else:
        raise SystemExit(f"unknown source {source}")
