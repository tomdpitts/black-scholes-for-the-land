# Greeks explain of the weather term (M4, mean of 5 fits, Box A, 13 held-out seasons 2019-25)

Script: revision/explain.py (explain.json); scan.py (summary table and the 2022 scan below).
Weather term W = full re-run under the weather that fell minus the re-run under normal weather
(the paper's split, zero residual, season end). Taylor about normal weather along the realised
departures d = (rain - clim, ED'), directional derivatives through the whole 6-window rollout:
W = Delta_r + Delta_e + Gamma_rr + Gamma_ee + Gamma_re + unexplained.

| Season | rain mm | ED mm | W | Delta (1st) | Delta+Gamma (2nd) | unexpl. 1st / abs(W) | unexpl. 2nd / abs(W) |
|---|---|---|---|---|---|---|---|
| MAM2019 | -56 | +36 | -0.047 | -0.048 | -0.064 | +0.01 | +0.35 |
| OND2019 | +434 | -48 | +0.135 | +0.277 | +0.135 | -1.05 | +0.00 |
| MAM2020 | +255 | -31 | +0.124 | +0.179 | +0.119 | -0.45 | +0.03 |
| OND2020 | -38 | +22 | -0.057 | -0.057 | -0.070 | +0.00 | +0.23 |
| MAM2021 | +9 | +13 | +0.011 | +0.016 | +0.009 | -0.44 | +0.21 |
| OND2021 | -172 | +47 | -0.064 | -0.082 | -0.091 | +0.29 | +0.44 |
| MAM2022 | -124 | +48 | -0.105 | -0.110 | -0.117 | +0.05 | +0.11 |
| OND2022 | -75 | +31 | -0.066 | -0.060 | -0.071 | -0.08 | +0.09 |
| MAM2023 | +180 | -16 | +0.098 | +0.122 | +0.100 | -0.24 | -0.02 |
| OND2023 | +215 | -21 | +0.057 | +0.097 | +0.055 | -0.71 | +0.04 |
| MAM2024 | +133 | -13 | +0.079 | +0.099 | +0.077 | -0.26 | +0.02 |
| OND2024 | -77 | +32 | -0.056 | -0.056 | -0.061 | -0.01 | +0.07 |
| MAM2025 | +83 | -11 | +0.022 | +0.028 | +0.025 | -0.26 | -0.10 |

Gamma_rr < 0 in every season (concave in rain at normal); Gamma_ee ~ 0; Gamma_re < 0 and small.
Wet seasons: Delta alone overshoots (by 105% in OND2019); Delta + Gamma closes it to within 4%.
Dry seasons: Delta alone is within 1-8% except OND2021 (29%); adding Gamma makes it worse (9-44%).
MAM2022: unexplained after 2nd order +0.012 (per fit 0.008-0.018); Delta alone -0.110 vs W -0.105.

Why: the response along the 2022 direction is S-shaped, not quadratic. Season-end box mean with
weather t * d (t=1 what fell, t=-1 the mirror-image wet season; rain clamped at 0 beyond t=1):
t -2 +0.153, -1.5 +0.125, -1 +0.090, -0.5 +0.045, 0 -0.007, +0.5 -0.063, +1 -0.112, +1.5 -0.142, +2 -0.158.
Slope per half step is largest near normal (0.052, 0.056) and falls off both ways (0.045 wet, 0.049 dry):
concave on the wet side, flattening on the dry side, inflection close to normal. A quadratic fitted at
normal carries the wet-side curvature into droughts, where the true curve flattens.

Per 2 km block, share of the weather-term map reproduced (1 - SSE/SS): first order 0.84-0.97 in dry
seasons, second order 0.78-0.99; OND2019 (+434 mm) only 0.33 even at second order.

## Third order (Speed = 1/6 d3G[d,d,d] along the whole departure)
| Season | W | unexpl. 2nd | Speed | unexpl. 3rd | 2nd / abs(W) | 3rd / abs(W) |
|---|---|---|---|---|---|---|
| MAM2019 | -0.047 | +0.017 | +0.009 | +0.007 | +0.35 | +0.15 |
| OND2019 | +0.135 | +0.000 | -0.200 | +0.200 | +0.00 | +1.48 |
| MAM2020 | +0.124 | +0.004 | -0.030 | +0.034 | +0.03 | +0.28 |
| OND2020 | -0.057 | +0.013 | +0.009 | +0.004 | +0.23 | +0.07 |
| MAM2021 | +0.011 | +0.002 | -0.001 | +0.003 | +0.21 | +0.29 |
| OND2021 | -0.064 | +0.028 | +0.018 | +0.009 | +0.44 | +0.15 |
| MAM2022 | -0.105 | +0.012 | +0.010 | +0.002 | +0.11 | +0.02 |
| OND2022 | -0.066 | +0.006 | +0.004 | +0.002 | +0.09 | +0.03 |
| MAM2023 | +0.098 | -0.002 | -0.011 | +0.009 | -0.02 | +0.09 |
| OND2023 | +0.057 | +0.002 | -0.026 | +0.028 | +0.04 | +0.49 |
| MAM2024 | +0.079 | +0.001 | -0.004 | +0.005 | +0.02 | +0.06 |
| OND2024 | -0.056 | +0.004 | +0.003 | +0.001 | +0.07 | +0.02 |
| MAM2025 | +0.022 | -0.002 | -0.003 | +0.001 | -0.10 | +0.06 |

Droughts (all six): Speed > 0, same sign as the 2nd-order leftover; it removes 55-85% of it, leaving 2-15% of W.
Wet seasons: Speed < 0 and overshoots; past ~+200 mm the series diverges (OND2019: Delta +0.277,
Gamma -0.142, Speed -0.200, each correction larger than the error it corrects). The wet side is
outside the expansion's radius of convergence; full re-run is the only reliable number there.

## Gap seasons (MODIS composite gaps, FINDINGS.md section 8)
OND2021's six windows are 16 Oct, 1 Nov, then 1 Jan-18 Feb 2022: the 61-day gap after 2021-11-01 is
stepped as one window, and windows 3-6 carry January-February rain and normals. Its explain (29% at
first order) is not a test of the Greeks. OND2024 (30-day gap after 2024-12-02) and MAM2023 (32-day gap
before its first window) also span a gap. Gap-free droughts (MAM2019, OND2020, MAM2022, OND2022): first
order within 0-8% of the full re-run.

## Correction: block R2 above is uncentred (1 - SSE / sum W^2)
It mostly credits the box-wide drop. Centred on the box mean (does first order get which blocks lost more?),
gap-free droughts: MAM2019 -0.33 (corr 0.38), OND2020 0.51 (0.78), MAM2022 0.59 (0.80), OND2022 0.76 (0.88).
Not claimed in the paper.
