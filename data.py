"""Build the model-ready arrays for one box from data/<box>/*.npz.

Output data/<box>/built.npz with, on the MODIS 16-day calendar (T steps) and the
160 x 160 grid:
  a     [T,N,N]  NDVI anomaly vs the 2003-2018 pixel climatology, short gaps interpolated
  obs   [T,N,N]  bool, pixel observed (reliability 0/1) at that step
  r     [T,N,N]  rain in the composite window, units of 100 mm
  rclim [T,N,N]  climatological rain for that step of year, units of 100 mm
  x     [N,N,3]  static features on fixed scales: mean NDVI, seasonal amplitude,
                 climatological annual rain
  phase [T,2]    sin, cos of the day of year at composite start
  year  [T]      calendar year of composite start
  clim  [23,N,N] NDVI climatology; cstd [23,N,N] its std (for the z-score baseline)
  burn  [T,N,N]  bool, pixel burned during the composite window (MCD64A1), if present
"""
import sys
import numpy as np
from scipy.ndimage import map_coordinates

import aoi

STEP_DAYS = 16


def _doy_index(dates):
    d = dates.astype("datetime64[D]")
    doy = (d - d.astype("datetime64[Y]")).astype(int) + 1
    return (doy - 1) // STEP_DAYS, doy


def _rain_on_grid(box, mod_dates, daily, ddates, transform, shape):
    """Sum daily CHIRPS over each 16-day window, then bilinear onto the grid."""
    ddates = ddates.astype("datetime64[D]")
    start = mod_dates.astype("datetime64[D]")
    T = len(start)
    h, w = shape
    sums = np.zeros((T, h, w), np.float32)
    for t, s in enumerate(start):
        sel = (ddates >= s) & (ddates < s + STEP_DAYS)
        sums[t] = np.nansum(daily[sel], axis=0) if sel.any() else np.nan
    # grid pixel centres in CHIRPS fractional index coordinates
    a, b, c, d, e, f = transform  # (xres, 0, west, 0, -yres, north)
    W, S, E, Nn = aoi.BOXES[box]
    lon = W + (np.arange(aoi.N) + 0.5) * aoi.RES
    lat = Nn - (np.arange(aoi.N) + 0.5) * aoi.RES
    col = (lon - c) / a - 0.5
    row = (lat - f) / e - 0.5
    rr, cc = np.meshgrid(row, col, indexing="ij")
    out = np.empty((T, aoi.N, aoi.N), np.float32)
    for t in range(T):
        out[t] = map_coordinates(sums[t], [rr, cc], order=1, mode="nearest")
    return out


def _interp_gaps(a, obs):
    """Linear interpolation along time of unobserved values, per pixel."""
    T = a.shape[0]
    af = a.reshape(T, -1).copy()
    ob = obs.reshape(T, -1)
    t = np.arange(T)
    for p in range(af.shape[1]):
        m = ob[:, p]
        if m.sum() >= 2:
            af[~m, p] = np.interp(t[~m], t[m], af[m, p])
        else:
            af[:, p] = 0.0
    return af.reshape(a.shape)


def build(box):
    m = np.load(f"data/{box}/modis.npz")
    c = np.load(f"data/{box}/chirps_daily.npz")
    z = np.load(f"data/{box}/dem.npz")["elev"]
    dates = m["dates"].astype("datetime64[D]")
    ndvi, rel = m["ndvi"].astype(np.float32), m["rel"]
    # the catalogue interleaves Terra (DOY 1, 17, ...) and Aqua (DOY 9, 25, ...); keep Terra
    _, doy_all = _doy_index(dates)
    terra = (doy_all - 1) % STEP_DAYS == 0
    dates, ndvi, rel = dates[terra], ndvi[terra], rel[terra]
    obs = (rel >= 0) & (rel <= 1) & np.isfinite(ndvi)
    ndvi = np.where(obs, ndvi, np.nan)

    idx, doy = _doy_index(dates)
    year = dates.astype("datetime64[Y]").astype(int) + 1970
    clim_mask = np.isin(year, list(aoi.CLIM_YEARS))

    clim = np.full((23, aoi.N, aoi.N), np.nan, np.float32)
    cstd = np.full_like(clim, np.nan)
    for k in range(23):
        sel = clim_mask & (idx == k)
        with np.errstate(all="ignore"):
            clim[k] = np.nanmean(ndvi[sel], axis=0)
            cstd[k] = np.nanstd(ndvi[sel], axis=0)
    a = ndvi - clim[idx]
    obs &= np.isfinite(a)
    a = _interp_gaps(np.where(obs, a, 0.0), obs).astype(np.float32)

    # keep only composites whose 16-day window is fully covered by CHIRPS days
    dd = c["dates"].astype("datetime64[D]")
    full = np.array([((dd >= s) & (dd < s + STEP_DAYS)).sum() == STEP_DAYS for s in dates])
    dates, ndvi, rel, obs, a, idx, doy, year = (v[full] for v in (dates, ndvi, rel, obs, a, idx, doy, year))
    clim_mask = clim_mask[full]
    r = _rain_on_grid(box, dates, c["rain"], c["dates"], c["transform"], tuple(c["shape"])) / 100.0
    rclim = np.zeros_like(r)
    for k in range(23):
        sel = clim_mask & (idx == k)
        rclim[idx == k] = np.nanmean(r[sel], axis=0)

    phase = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1).astype(np.float32)

    # three climatological statistics of the pixel, on fixed scales so both boxes share them
    x = np.stack([
        np.nanmean(clim, axis=0) / 0.3,
        (np.nanmax(clim, axis=0) - np.nanmin(clim, axis=0)) / 0.2,
        np.nansum(rclim[:23], axis=0) / 10.0,  # climatological annual rain, in metres
    ], axis=-1).astype(np.float32)

    out = dict(a=a, obs=obs, r=r.astype(np.float32), rclim=rclim.astype(np.float32), x=x,
               phase=phase, year=year.astype(np.int16), clim=clim, cstd=cstd,
               dates=dates.astype(str))
    try:
        f = np.load(f"data/{box}/fire.npz")
        months = f["months"].astype("datetime64[M]")
        bd = f["burn_doy"]
        burn = np.zeros(a.shape, bool)
        for t, s in enumerate(dates):
            for k in range(len(months)):
                if months[k] <= s.astype("datetime64[M]") <= (s + STEP_DAYS).astype("datetime64[M]"):
                    ydoy0 = (months[k].astype("datetime64[D]") - months[k].astype("datetime64[Y]").astype("datetime64[D]")).astype(int)
                    burn_day = months[k].astype("datetime64[Y]").astype("datetime64[D]") + (bd[k] - 1)
                    burn[t] |= (bd[k] > 0) & (burn_day >= s) & (burn_day < s + STEP_DAYS)
        out["burn"] = burn
    except FileNotFoundError:
        pass
    np.savez_compressed(f"data/{box}/built.npz", **out)
    print(f"box {box}: T={len(dates)} {dates[0]}..{dates[-1]} obs={obs.mean():.3f} "
          f"|a| mean={np.nanmean(np.abs(a[obs])):.3f} rain/step mean={np.nanmean(r)*100:.1f} mm "
          f"burn steps={out.get('burn', np.zeros(1)).sum()}")


if __name__ == "__main__":
    for b in sys.argv[1:] or ["A", "B"]:
        build(b)
