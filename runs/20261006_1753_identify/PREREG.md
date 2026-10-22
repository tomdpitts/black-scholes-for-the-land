# Identification of the residual counterfactual: pre-registered before any result was computed

Written 2026-10-06 before running revision/identify.py. Prompted by reviewers 1 and 3: the
counterfactual of Eq. (3) holds the implied residual fixed while the rain changes, an
assumption the paper states only implicitly and never tests. Seen beforehand: the paper's
real-data residual-vs-rain bins (appendix) and one spot check that M2's sigma rises steeply
with rain (box mean 0.018 / 0.043 / 0.062 at -100 / 0 / +100 mm, one 2015 step).

Model: saved fair-fight M2 (runs/20260914_1708_pipeline_h100_fairfight/params.npy); no refit.
Seasons: the 14 held-out Box A rainy seasons of residual.py (2019-2025), six 16-day windows each,
start state observed. Counterfactual rain r' = climatology, as in the paper.

## Semi-synthetic design
The fitted M2 is taken as the true world, so the model is correctly specified by construction
and any error is due to the identification assumption alone. Within each season the truth is

    a_{t+1} = a_t + mu(a_t, n_t, w_t, r_{t+1}) + sigma(.) e_t + m_t(r),

with e_t the real implied innovations of that season (realistic noise), and m_t a management
term applied on "managed" paddocks: a fixed random half of 8x8-pixel (2 km) blocks. The true
counterfactual re-runs the same equation under r' with the same e_t and the management policy
evaluated at r'. The estimator only sees the synthetic observations and the CHIRPS rain.

Estimators: (E1) the paper's: invert to standardised e-hat, roll under r' with e-hat fixed;
(E2) additive: hold the residual in NDVI units (a_{t+1} - mu) fixed instead of e-hat.

Scenarios (management size delta = 0.01 NDVI per window, -0.06 over a season, the scale of
the 2022 residual):
- S0 none (sanity: both estimators exact, error < 1e-5).
- S1 rain-independent: m_t = -delta on managed blocks.
- S2a drought incursion: m_t = -delta (1 + k (rclim - r)/rclim), k = 1, clipped to [0, 3 delta]
  (grazing pressure rises when rain fails, as in the 2017 and 2022 Laikipia incursions).
- S2b destocking: same with k = -1 (pressure falls when rain fails).
- S3 rain-product error: truth is driven by r_true = r * exp(eta - s^2/2), eta Gaussian,
  constant over each 20x20-pixel (0.05 degree, one CHIRPS cell) block and independent per
  window; s in {0.25, 0.5}. No management. Estimator sees r.

## Outcomes (season end, h = 6)
Counterfactual error CF_hat - CF_true, box mean over managed pixels (S1-S2) or all pixels (S3);
pixel RMSE; and as a share of the true management effect under normal rain (S1-S2).
Reported separately for drought seasons (rain departure < -60 mm) and the rest.

## Criteria (per scenario and estimator)
IDENTIFIED if |box-mean error| < 10 % of the true management effect in every season class.
For S3, IDENTIFIED if |box-mean error| < 0.01 NDVI (a tenth of the 2022 counterfactual gap).

## Real-data diagnostics (reported, no pass/fail)
- Mean standardised residual by rain-departure bin, Box A test and Box B, with 95 % bootstrap
  intervals over years (cluster bootstrap, 2000 draws).
- Residual by MODIS reliability flag (0 good, 1 marginal) at the target step.
- 2022 and every other season: counterfactual under E1 vs E2 on real data; their gap is an
  assumption-sensitivity band for the paper's numbers.
- Steps spanning MODIS gaps (> 16 days between composites) flagged; seasons containing one
  reported with and without them.
