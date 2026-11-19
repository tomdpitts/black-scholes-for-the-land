# Gradient-boosted trees with M4's inputs: pre-registered before any result was computed

Written 2026-10-27 before running `revision/baselines.py gbm_full`. Prompted by the paper's use of gradient-boosted
trees as the canonical machine-learning baseline: the trees in the suite (B1, runs/20261006_1759_baselines) saw rain
but not evaporative demand or topography, so M4 had more information. This run gives the trees the same information
sources as the M4 ensemble.

Seen beforehand: the suite's results (runs/20261022_1639_suite/RESULT.md), including B1's 3-month skill (Box A
2019-25 0.222, Box B 2019-25 0.197) and the M4 ensemble's (0.273, 0.274).

## Model
B1 unchanged (sklearn HistGradientBoostingRegressor, same hyperparameters, target a_{t+1} - a_t, variance from a
second GBM on log squared residual rescaled on 2018, 3 M training rows subsampled), with two additions to the features:
- evaporative demand: FAO-56 ET0 anomaly (era5.npz pet - pet_clim, 100 mm per window) for windows t+1 ... t-4
  (six windows, as the rain);
- topography: elevation, slope and topographic position (topo.npz), raw, after the three existing traits.
All other features as B1: a_t, a_{t-1}, a_{t-2}, 5x5 neighbourhood mean, seasonal phase (2), mean NDVI, seasonal
amplitude, climatological annual rain, rain r_{t+1} ... r_{t-4}, climatological rain r_clim_{t+1}.
Not given: M4's water-balance memory w_t, which depends on M4's fitted parameters.

## Data, splits, scores
Fit Box A 2003-2017, early-stop and calibrate on 2018. Score on Box A 2019-25 (A_test) and Box B 2019-25 (B_test):
one-step Gaussian CRPS and 1..6-step sampled-rollout CRPS (S = 16) along the observed weather, as B1. Skill is
1 - CRPS / CRPS_M0 with M0 from the suite (skill.json, same starts and masks).

## Question and criterion
Q: does the M4 ensemble beat trees given the same information? M4 BETTER if its 3-month (h = 6) CRPS is lower than
the trees' on both A_test and B_test; TREES BETTER if higher on both; MIXED otherwise. The result replaces B1 as the
paper's trees row whatever it is.
