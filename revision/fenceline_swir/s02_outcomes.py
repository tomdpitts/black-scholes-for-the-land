"""Step 02: indices, climatology, window statistics. Runs only after FREEZE.json (verified) and with the previous
run's design verified unchanged. Writes RUN/outputs/outcomes.npz once. No statistics across groups here."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fs_common as C

C.start("02_outcomes", __file__)
C.verify_freeze("02_outcomes")
C.verify_prev()
import numpy as np

B = np.load(C.DATA_B)
bdates = B["dates"].astype("datetime64[D]")
YEARS = list(range(2003, 2026))
WIN = {"main": (57, 105), "post": (113, 177)}


def load(sensor_file):
    z = np.load(C.out("inputs", "raw", sensor_file), allow_pickle=True)
    nd = json.loads(str(z["nodata"]))
    dates = z["dates"].astype("datetime64[D]")
    ref = {}
    for b in ("sur_refl_b01", "sur_refl_b02", "sur_refl_b06", "sur_refl_b07"):
        a = z[b].astype(np.float32)
        if nd[b] is not None:
            a[z[b] == nd[b]] = np.nan
        a *= 1e-4
        a[(a <= 0) | (a > 1)] = np.nan
        ref[b] = a
    st = z["sur_refl_state_500m"].astype(np.int64)
    clear = ((st & 0b11) == 0) & (((st >> 2) & 1) == 0) & (((st >> 10) & 1) == 0)
    for b in ref:
        ref[b][~clear] = np.nan
    return dates, ref


def window_stats(dates, idx):
    yrs = dates.astype("datetime64[Y]").astype(int) + 1970
    doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    slots = np.unique(doy)
    clim_mask = (yrs >= 2003) & (yrs <= 2018)
    anom = np.full_like(idx, np.nan)
    for s in slots:
        sel = doy == s
        cm = sel & clim_mask
        n = np.isfinite(idx[cm]).sum(0)
        c = np.nanmean(idx[cm], axis=0) if cm.any() else np.full(idx.shape[1:], np.nan)
        c = np.where(n >= 5, c, np.nan)
        anom[sel] = idx[sel] - c
    out = {}
    for w, (lo, hi) in WIN.items():
        Y, cnt, med = [], [], []
        for y in YEARS:
            sel = (yrs == y) & (doy >= lo) & (doy <= hi)
            k = np.isfinite(anom[sel]).sum(0)
            with np.errstate(all="ignore"):
                m = np.nanmean(anom[sel], axis=0) if sel.any() else np.full(idx.shape[1:], np.nan)
            Y.append(np.where(k >= 3, m, np.nan)); cnt.append(k); med.append(float(np.median(k)))
        out[w] = (np.stack(Y).astype(np.float32), np.array(med))
    return out


burn = np.zeros((len(YEARS), C.N, C.N), bool)
for k, y in enumerate(YEARS):
    sel = (bdates >= np.datetime64(f"{y}-01-01")) & (bdates <= np.datetime64(f"{y}-04-30"))
    burn[k] = B["burn"][sel].any(0)
dep = []
for y in YEARS:
    sel = (bdates >= np.datetime64(f"{y - 1}-10-01")) & (bdates <= np.datetime64(f"{y}-04-22"))
    dep.append(float(((B["r"][sel] - B["rclim"][sel]).sum(0) * 100.0).mean()))

save = {"years": np.array(YEARS), "burn": burn, "rain_dep_octapr_mm": np.array(dep)}
for sensor, f in (("terra", "mod09a1_terra.npz"), ("aqua", "myd09a1_aqua.npz")):
    dates, ref = load(f)
    with np.errstate(all="ignore"):
        idx = {"ndti": (ref["sur_refl_b06"] - ref["sur_refl_b07"]) / (ref["sur_refl_b06"] + ref["sur_refl_b07"]),
               "b07": ref["sur_refl_b07"],
               "ndvi": (ref["sur_refl_b02"] - ref["sur_refl_b01"]) / (ref["sur_refl_b02"] + ref["sur_refl_b01"])}
    for name, arr in idx.items():
        ws = window_stats(dates, arr)
        for w, (Y, med) in ws.items():
            save[f"{sensor}_{name}_{w}_Y"] = Y
            save[f"{sensor}_{name}_{w}_median_valid_composites"] = med
    C.log("02_outcomes", f"{sensor} indices computed", n_composites=len(dates),
          median_valid_window_composites_main=dict(zip(YEARS, save[f"{sensor}_ndti_main_median_valid_composites"].tolist())))
fo = C.out("outputs", "outcomes.npz")
assert not os.path.exists(fo)
np.savez_compressed(fo, **save)
C.log("02_outcomes", "outcomes saved", file=os.path.relpath(fo, C.ROOT), sha256=C.sha256(fo))
