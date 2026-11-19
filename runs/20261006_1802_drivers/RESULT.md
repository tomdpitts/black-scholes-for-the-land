# Second weather driver: result against PREREG.md

Scripts: revision/fetch_era5.py, revision/drivers.py (build, rainerr, fit), revision/drivers_eval.py.
Fits: params_{rain_temp,temp,rain_all}.npy (2000 steps each, CPU, ~1.6-1.9 h). Metrics:
metrics_*.json, eval.json. M2 rain = the paper's fair-fight fit.

CRPS skill vs M0:

| Model | A 1-step | A 3-month | B 1-step | B 3-month | val NLL 2018 |
|---|---|---|---|---|---|
| M2 rain (paper) | **0.085** | **0.226** | 0.104 | 0.170 | -1.465 |
| temp only | -0.064 | -0.430 | 0.062 | 0.047 | -1.399 |
| rain + temp | 0.049 | -0.024 | 0.095 | 0.146 | -1.414 |
| rain + temp + PET + VPD | 0.070 | 0.075 | **0.124** | **0.220** | -1.451 |

D1 RAIN DOMINANT (rain beats temp by > 0.05 at 3 months, both sets): **PASS** (A +0.66, B +0.12).
D2 TEMP ADDS (rain+temp beats rain by > 0.01, both sets): **FAIL**; it subtracts (A -0.25, B -0.02).

## Why (post hoc)
Temperature anomalies are non-stationary: against the 2003-2018 climatology, the held-out years
2019-2025 are +0.59 C (A) and +0.48 C (B) warmer, about one window-to-window s.d.; VPD +0.11 /
+0.08 kPa. Temperature is no smoother in space than CHIRPS rain (within/across-window sd ratio
0.25 vs 0.30).

Box B split by period, 3-month skill vs M0:

| | 2003-2018 | 2019-2025 |
|---|---|---|
| M2 rain | 0.163 | **0.188** |
| rain + temp | 0.241 | -0.061 |
| rain + temp + PET + VPD | **0.287** | 0.074 |

Extra weather drivers help in the climatology-period years and hurt in the warm years. The
in-period gain is confounded: Box B's 2003-2018 are the calendar years Box A was fitted on, and
a four-variable regional weather vector can identify the composite, letting the network carry
Box A's training shocks across. The two clean out-of-time tests (A and B, 2019-2025) both say
the extra drivers hurt. M2's rain-only skill does not inflate in-period (0.163 vs 0.188).

## D3: does the residual change?
corr(rain-only residual, rain+all residual), held-out rainy-season pixel-steps: 0.91 (A), 0.91 (B);
residual sd 1.09 -> 1.18 (A). Season counterfactuals under normal rain (other weather as
observed), Box A box mean at season end:

| Season | obs | rain-only | rain+all |
|---|---|---|---|
| MAM2022 | -0.168 | -0.099 | -0.158 |
| MAM2020 | +0.135 | -0.004 | +0.064 |
| MAM2024 | +0.171 | +0.071 | +0.124 |
| OND2021 | -0.054 | +0.028 | -0.020 |

Reading: with temperature and evaporative demand in the model, "normal rain" keeps the season's
heat, so part of what the rain-only model calls residual becomes modelled weather that stays in
the counterfactual. What the residual means depends on which drivers are modelled, and a
counterfactual must say which drivers it sets to normal. Here the richer model fails out of
time, so its counterfactuals are not trustworthy; the rain-only model is the one that validates.
