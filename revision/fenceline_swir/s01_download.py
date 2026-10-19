"""Step 01: download MOD09A1 (Terra) and MYD09A1 (Aqua) v061 bands b01, b02, b06, b07 and state flags for Box B,
2003-2025, warped by nearest neighbour onto the Box B 0.0025 deg grid (same route as ingest.py for MOD13Q1).
Raw integers are saved unmodified. Logged: counts and box-wide clear-sky coverage only (no indices, no groups)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import fs_common as C
import numpy as np, rasterio, pystac_client, planetary_computer as pc
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from concurrent.futures import ThreadPoolExecutor, as_completed

C.start("01_download", __file__)
TR = from_origin(C.BOX_B[0], C.BOX_B[3], C.RES, C.RES)
BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b06", "sur_refl_b07", "sur_refl_state_500m"]
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
items = []
for y in range(2003, 2026):
    items += list(cat.search(collections=["modis-09A1-061"], bbox=list(C.BOX_B), datetime=f"{y}-01-01/{y}-12-31", limit=500).items())
C.log("01_download", "STAC search done", n_items=len(items), collection="modis-09A1-061")


def warp(href):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF"):
        with rasterio.open(href) as src:
            nd = src.nodata
            with WarpedVRT(src, crs="EPSG:4326", transform=TR, width=C.N, height=C.N,
                           resampling=Resampling.nearest, nodata=nd, src_nodata=nd) as vrt:
                return vrt.read(1), nd


def fetch(item):
    for attempt in range(4):
        try:
            s = pc.sign(item)
            out = {}
            for b in BANDS:
                arr, nd = warp(s.assets[b].href)
                out[b] = (arr, nd)
            return item.id, (item.properties.get("start_datetime") or item.properties["datetime"])[:10], out
        except Exception as ex:
            err = repr(ex)[:200]; time.sleep(2 + 3 * attempt)
    return item.id, None, err


t0 = time.time()
res, failed = {}, []
with ThreadPoolExecutor(12) as ex:
    futs = [ex.submit(fetch, it) for it in items]
    for k, f in enumerate(as_completed(futs)):
        iid, d, out = f.result()
        if d is None:
            failed.append((iid, out)); continue
        res[iid] = (d, out)
        if (k + 1) % 250 == 0:
            C.log("01_download", "progress", done=k + 1, of=len(items), seconds=round(time.time() - t0))
C.log("01_download", "fetch finished", ok=len(res), failed=len(failed), seconds=round(time.time() - t0),
      failed_ids=[f[0] for f in failed][:50])

for sensor, prefix in (("terra", "MOD09A1"), ("aqua", "MYD09A1")):
    ids = sorted([i for i in res if i.startswith(prefix)], key=lambda i: res[i][0])
    by_date = {}
    for i in ids:
        by_date.setdefault(res[i][0], []).append(i)
    dates = sorted(by_date)
    stacks = {b: [] for b in BANDS}
    for d in dates:
        for b in BANDS:
            arrs = [res[i][1][b] for i in by_date[d]]
            if len(arrs) == 1:
                stacks[b].append(arrs[0][0])
            else:                                     # tile overlap: first non-nodata value
                a = arrs[0][0].copy(); nd = arrs[0][1]
                for arr, _ in arrs[1:]:
                    m = (a == nd); a[m] = arr[m]
                stacks[b].append(a)
    nodata = {b: res[ids[0]][1][b][1] for b in BANDS}
    p = C.out("inputs", "raw", f"{prefix.lower()}_{sensor}.npz")
    assert not os.path.exists(p)
    np.savez_compressed(p, dates=np.array(dates), item_ids=np.array([";".join(by_date[d]) for d in dates]),
                        nodata=json.dumps({b: (None if v is None else float(v)) for b, v in nodata.items()}),
                        **{b: np.stack(v) for b, v in stacks.items()})
    st = np.stack(stacks["sur_refl_state_500m"]).astype(np.int64)
    clear = (((st & 0b11) == 0) & (((st >> 2) & 1) == 0) & (((st >> 10) & 1) == 0))
    yrs = np.array([int(d[:4]) for d in dates])
    C.log("01_download", f"{sensor} saved", file=os.path.relpath(p, C.ROOT), sha256=C.sha256(p), n_dates=len(dates),
          first=dates[0], last=dates[-1], dates_per_year={int(y): int((yrs == y).sum()) for y in np.unique(yrs)},
          boxwide_clear_fraction_by_year={int(y): round(float(clear[yrs == y].mean()), 3) for y in np.unique(yrs)})
