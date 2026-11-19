# Stronger rain-conditioned baselines: pre-registered before any result was computed

Written 2026-10-06 before running revision/baselines.py. Prompted by reviewer 2 ("the
baselines ... seem not very strong; add at least one competitive rain-conditioned temporal
baseline"). Seen beforehand: the paper's M0/M1/M1.5/M2 numbers only.

Same data, splits, masks and scores as run.py: fit Box A 2003-2017, early-stop/calibrate on
2018, score one-step CRPS (Gaussian) and 1..6-step sampled-rollout CRPS (S=16, realised rain)
on Box A 2019-25 and Box B all years; skill vs M0 from the fair-fight run, recomputed on the
same starts.

## B1 GBM: gradient-boosted trees with distributed lags (non-Markovian, standard tabular baseline)
sklearn HistGradientBoostingRegressor. Target a_{t+1} - a_t. Features: a_t, a_{t-1}, a_{t-2},
5x5 neighbourhood mean, seasonal phase (2), static x (3), rain r_{t+1} ... r_{t-4} (six
windows), climatological rain r_clim_{t+1}. Variance: second GBM on log squared residual,
rescaled so that the mean predicted variance matches 2018. 3 M training rows subsampled.

## B2 GRU-M2: M2 plus a learned per-pixel memory (non-Markovian twin of M2)
Same features and linear base as M2, plus a 32-unit GRU over the pixel's history whose
zero-initialised head adds to mu and log sigma: M2's capacity and more. Trained on a fixed
random 25 % of Box A pixels (CPU budget), neighbourhood features from the full field.

## Questions and criteria
Q1 (reviewer 2): is M2 competitive? M2 COMPETITIVE if its three-month (h=6) rollout CRPS is
within 2 % of, or better than, the best of B1/B2 on both evaluation sets.
Q2 (Markov sufficiency, reviewer 1's "model simplicity"): MARKOV SUFFICIENT if B2's h=6 skill
exceeds M2's by less than 0.02 on both sets.

## Addendum 2026-10-06, after the 25 %-pixel GRU result and before the full-pixel run
B2 on 25 % of pixels early-stopped at step 750 with validation NLL -1.420 (M2: -1.465), training
NLL still falling: overfitting, possibly caused by the subsample. B2 is re-run on all pixels
(GRU_FRAC=1, metrics_gru_full.json); the same Q1/Q2 criteria apply, and the full-pixel run
replaces the subsample run as B2 whatever its result.
