# Second weather driver: pre-registered before any fit

Written 2026-10-06 before running revision/drivers.py fit. Prompted by reviewer 2 ("rainfall-only,
temperature-only, and combined inputs ... how omitted weather effects might influence the
inferred residual") and reviewer 1 (CHIRPS error). Seen beforehand: no ERA5-based result.

Data: ERA5-Land 2 m mean temperature, ERA5 FAO-56 reference evapotranspiration (PET) and daily
max vapour-pressure deficit (VPD), via the Open-Meteo archive (era5_seamless), 0.2 degree lattice,
bilinear to the grid, aggregated to each composite's 16-day window, as anomalies against the
2003-2018 window climatology. Same window as the rain.

Variants (M2 architecture, fit, early stopping and scoring exactly as run.py; only the
features change; extra anomalies are those of window t+1, as for rain):
- M2 rain (the paper's fit, fair-fight run)
- M2 temp: temperature anomaly, rain and rain memory removed
- M2 rain+temp
- M2 rain+temp+PET+VPD

Questions and criteria (Box A 2019-25 and Box B all years; h = 6 rollout CRPS skill vs M0):
D1 Is rain the right first driver? RAIN DOMINANT if M2 rain beats M2 temp by > 0.05 on both sets.
D2 Does temperature add? TEMP ADDS if rain+temp beats rain by > 0.01 on both sets (refit noise
   is ~0.005 per the paper).
D3 Does the residual change? Reported, no pass/fail: corr of the rain-only and rain+temp+PET+VPD
   implied residuals on held-out rainy-season pixel-steps; the 2022 counterfactual under each.

Also (reviewer 1, CHIRPS error; reported, no pass/fail): residual against CHIRPS minus ERA5
window rain, held-out rainy-season pixel-steps, controlling linearly for CHIRPS departure.
