# Stronger baselines: result against PREREG.md

Scripts: revision/baselines.py (B1, B2), revision/gbm_diag*.py (post hoc diagnostics).
Metrics: metrics_gbm.json, metrics_gru.json (25 % pixels), metrics_gru_full.json (all pixels;
replaces the subsample run per the addendum), diag*.json.

CRPS skill vs M0 (fair-fight M0, same starts and masks):

| Model | A 1-step | A 3-month | B 1-step | B 3-month |
|---|---|---|---|---|
| M2 (paper) | **0.085** | **0.226** | **0.104** | 0.170 |
| B1 GBM, 3 greenness lags + 6 rain lags | 0.046 | 0.222 | 0.090 | **0.260** |
| B2 GRU-M2, all pixels | 0.052 | 0.205 | 0.097 | 0.206 |
| (B2 on 25 % pixels, superseded) | 0.039 | 0.184 | 0.084 | 0.174 |

Q1 M2 COMPETITIVE (h=6 CRPS within 2 % of the best baseline, both sets): A yes (best);
B **no**: GBM's three-month CRPS is 12 % lower. **FAIL.**
Q2 MARKOV SUFFICIENT (B2 h=6 lead < 0.02, both sets): A yes (B2 worse by 0.021);
B **no**: B2 leads by 0.036. **FAIL.**

## Diagnostics (post hoc): why the baselines win at three months on Box B
Mean rollouts (no sampling), error at h = 6 months, all held-out starts:

| Model | A bias | A MAE | B bias | B MAE |
|---|---|---|---|---|
| M2 | -0.020 | 0.0724 | -0.011 | 0.0630 |
| GBM, full lags | -0.014 | 0.0754 | -0.005 | 0.0583 |
| GBM, no greenness lags (Markov in a) | -0.012 | 0.0754 | -0.009 | 0.0589 |
| GBM, M2's exact inputs (a, n, w bucket, r_{t+1}) | +0.003 | 0.0728 | +0.000 | 0.0586 |

Reading: removing greenness memory or replacing six rain lags by M2's one-parameter bucket
leaves GBM's Box B advantage intact (0.0586-0.0589 vs M2's 0.0630). The edge is not memory and
not the rain representation: with M2's own state, trees roll out without drift, while M2's mean
drifts browner by 0.01-0.02 NDVI over three months. The gap is in M2's fit (one-step Gaussian
likelihood, early-stopped on likelihood; cf. LESSONS 9), not in the Markov state. The GRU's
Box B gain is consistent with the same story but is not separately diagnosed.

M2 remains the best one-step density on both sets, which is what the residual (Eq. 2) and the
anomaly score are. The counterfactual rolls M2 forward, so the drift can matter there; its
effect on the 2022 numbers is not measured here and is a stated limitation. Trees are
piecewise constant in rain and give no derivative, so GBM cannot supply Delta.
