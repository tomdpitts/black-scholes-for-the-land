"""Daily ERA5 / ERA5-Land weather for both boxes from the Open-Meteo archive API.

Second driver for the temperature ablation (reviewer 2) and an independent rain product
for the CHIRPS-error diagnostic (reviewer 1). Points on a 0.2 degree lattice covering both
boxes; the archive's `era5_seamless` model gives ERA5-Land temperature (0.1 degree) and ERA5
PET, VPD and precipitation (0.25 degree), so a finer lattice adds no information.

Resumable and paced for the free-tier rate limit: one file per (point, 4-year chunk).

  .venv/bin/python revision/fetch_era5.py      -> data/era5/raw/*.json
"""
import json
import os
import time
import urllib.error
import urllib.request

OUT = "data/era5/raw"
LATS = [0.15, 0.35, 0.55]
LONS = [36.8, 37.0, 37.2, 37.4]
VARS = "temperature_2m_mean,et0_fao_evapotranspiration,vapour_pressure_deficit_max,precipitation_sum"
CHUNKS = [(y, min(y + 3, 2025)) for y in range(2002, 2026, 4)]
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
       "&start_date={y0}-01-01&end_date={y1}-12-31&daily={vars}&models=era5_seamless&timezone=Africa%2FNairobi")


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(la, lo, c) for la in LATS for lo in LONS for c in CHUNKS]
    for i, (la, lo, (y0, y1)) in enumerate(jobs):
        f = f"{OUT}/{la:.2f}_{lo:.2f}_{y0}_{y1}.json"
        if os.path.exists(f):
            continue
        wait = 30
        while True:
            try:
                with urllib.request.urlopen(URL.format(lat=la, lon=lo, y0=y0, y1=y1, vars=VARS), timeout=120) as r:
                    body = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                msg = e.read()[:200]
                print(f"  {e.code} {msg}; sleeping {wait}s", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 1800)
            except Exception as e:  # network blips
                print(f"  {e}; sleeping {wait}s", flush=True)
                time.sleep(wait)
        json.dump(body, open(f, "w"))
        print(f"{i + 1}/{len(jobs)} {f}", flush=True)
        time.sleep(15)
    print("done")


if __name__ == "__main__":
    main()
