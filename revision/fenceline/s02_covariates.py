"""Step 02: static covariates for matching and masking, fetched into RUN/inputs/covariates.
 - ESA WorldCover 2021 v200 (10 m) class fractions per Box-B pixel. 1/12000 deg pixels nest
   exactly 30x30 into the 0.0025 deg grid (offsets are integers), so fractions are exact counts.
 - SoilGrids 2.0 clay content (g/kg), 5-15 cm, mean; averaged onto the Box-B grid.
No outcome (NDVI anomaly or residual) is read in this step."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C
import numpy as np, rasterio, requests
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

C.start("02_covariates", __file__)
w, s, e, n = C.BOX_B
# ---------------------------------------------------------------- WorldCover
url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N00E036_Map.tif"
t0 = time.time()
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
    with rasterio.open("/vsicurl/" + url) as src:
        tr = src.transform
        assert abs(tr.a - 1 / 12000) < 1e-12 and abs(tr.c - 36.0) < 1e-9 and abs(tr.f - 3.0) < 1e-9, tr
        row0, col0 = round((3.0 - n) * 12000), round((w - 36.0) * 12000)
        k = 30
        wc = src.read(1, window=Window(col0, row0, C.N * k, C.N * k))
assert wc.shape == (C.N * 30, C.N * 30), wc.shape
classes = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow",
           80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
blocks = wc.reshape(C.N, 30, C.N, 30)
frac = {v: (blocks == c).mean(axis=(1, 3)).astype(np.float32) for c, v in classes.items()}
p = C.out("inputs", "covariates", "worldcover2021_fractions.npz")
np.savez_compressed(p, **frac)
C.log("02_covariates", "WorldCover 2021 class fractions on Box-B grid", url=url, window_rowcol=(row0, col0),
      seconds=round(time.time() - t0, 1), file=os.path.relpath(p, C.ROOT), sha256=C.sha256(p),
      box_mean_fractions={v: round(float(f.mean()), 4) for v, f in frac.items()})
# ---------------------------------------------------------------- SoilGrids clay
wcs = ("https://maps.isric.org/mapserv?map=/map/clay.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
       "&COVERAGEID=clay_5-15cm_mean&FORMAT=image/tiff"
       f"&SUBSET=long({w - 0.02},{e + 0.02})&SUBSET=lat({s - 0.02},{n + 0.02})"
       "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326")
t0 = time.time()
r = requests.get(wcs, timeout=300); r.raise_for_status()
rawp = C.out("inputs", "covariates", "soilgrids_clay_5-15cm_raw.tif")
C.write_new(rawp, r.content, mode="wb")
with rasterio.open(rawp) as src:
    dst = np.full((C.N, C.N), np.nan, np.float32)
    reproject(src.read(1).astype(np.float32), dst, src_transform=src.transform, src_crs=src.crs,
              src_nodata=src.nodata, dst_transform=from_origin(w, n, C.RES, C.RES), dst_crs="EPSG:4326",
              dst_nodata=np.nan, resampling=Resampling.average)
p2 = C.out("inputs", "covariates", "clay_5-15cm_gkg.npy")
np.save(p2, dst)
C.log("02_covariates", "SoilGrids clay 5-15 cm averaged onto Box-B grid", url=wcs,
      seconds=round(time.time() - t0, 1), raw=os.path.relpath(rawp, C.ROOT), raw_sha256=C.sha256(rawp),
      file=os.path.relpath(p2, C.ROOT), sha256=C.sha256(p2),
      nan_pixels=int(np.isnan(dst).sum()), clay_p5_p50_p95=[round(float(v), 1) for v in np.nanpercentile(dst, [5, 50, 95])])
