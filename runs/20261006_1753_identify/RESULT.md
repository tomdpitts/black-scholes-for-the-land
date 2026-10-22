# Identification of the residual counterfactual: result

Pre-registration: PREREG.md (this directory). Scripts: revision/identify.py, revision/identify_sens.py.
Numbers: metrics.json, sensitivity.json. 13 held-out Box A seasons (MAM2019-MAM2025; the paper's
"14" counts OND2025, which falls in the 144-day MODIS gap from 2025-06-26 and has no start).
Drought class (rain departure < -60 mm): OND2021, MAM2022, OND2022, OND2024.

## Semi-synthetic (M2 is the true world; management -0.01 NDVI per window on half the 2 km blocks)
True management effect at season end under normal rain: -0.027 NDVI (mean reversion eats ~half
of the -0.06 injected).

| Scenario | Estimator | Error / true effect, drought (mean, max) | other seasons (mean, max) | Verdict |
|---|---|---|---|---|
| S0 none | E1 paper | exact (1e-9) | exact | sanity pass |
| S0 none | E2 additive | box err max 0.022 NDVI | max 0.017 | n/a |
| S1 rain-independent | E1 paper | +18 %, 30 % | +1 %, 7 % | **FAIL** (drought only) |
| S1 rain-independent | E2 additive | 8 %, 102 % | 9 %, 82 % | FAIL |
| S2a drought incursion | E1 paper | +47 %, 81 % | 4 %, 39 % | **FAIL** |
| S2b destocking | E1 paper | -7 %, 24 % | 11 %, 72 % | **FAIL** |

| Scenario | Estimator | box-mean error max (NDVI) | pixel RMSE | Verdict |
|---|---|---|---|---|
| S3 rain error s=0.25 | E1 | 0.005 | 0.03-0.04 | identified (box) |
| S3 rain error s=0.5 | E1 | 0.013 | 0.05-0.11 | FAIL narrowly (box) |

Reading:
1. Even when management ignores rain and the model is exactly right, the paper's estimator
   overstates the management effect by ~18 % (MAM2022: 28 %) in drought seasons. Cause: the
   residual is held in spread units and M2's spread rises steeply with rain (sigma 0.018 -> 0.043
   -> 0.062 at -100/0/+100 mm), so re-running a drought under normal rain scales the residual
   up by sigma'/sigma. In near-normal seasons it is within 1 %. Holding the residual in NDVI
   units (E2) mis-scales the noise instead and is worse; the two do not bracket the truth
   (5 of 13 seasons), so no interval is claimed.
2. If management responds to rain the counterfactual is not identified from NDVI and rain
   alone: drought incursions inflate the drought-season management effect by ~half (MAM2022:
   81 %). This must be stated as an assumption, not fixed.
3. Rain-product error averages out at box scale (<= 0.005 NDVI at 25 % multiplicative error,
   0.013 at 50 %) but not at pixel scale (RMSE 0.03-0.11). Project-level attribution is
   robust to it; pixel-level attribution is not.

## Real-data diagnostics
Mean standardised residual by window rain departure, rainy-season pixel-steps, 95 % year-cluster CI:

| Bin (mm) | <-60 | -60..-30 | -30..0 | 0..30 | 30..60 | 60..120 | >120 |
|---|---|---|---|---|---|---|---|
| A 2019-25 | **-0.24 [-.37,-.04]** | -0.03 | -0.03 | **+0.22 [.00,.49]** | +0.05 | +0.10 | -0.01 |
| B all years | -0.32 [-.69,+.12] | +0.03 | **+0.13 [.03,.22]** | **+0.26 [.08,.43]** | **+0.34 [.10,.60]** | +0.17 | +0.15 |

corr(residual, rain departure): 0.047 (A), 0.060 (B). Excluding gap-spanning steps changes no bin
by more than 0.04. The residual is not rain-free: the model under-responds to deep drought on A
and under-predicts greening in moderate-wet windows on the unseen landscape B.

MODIS reliability: flag 1 (marginal, 16 % of A test pixel-steps) residual mean +0.13, sd 1.31,
against -0.01, sd 1.04 for flag 0. Satellite quality leaks into the residual.

## Sensitivity of the paper's numbers (MAM2022, counterfactual box mean at season end)
paper -0.099; residual de-biased by rain bin -0.096; clean-quality pixels only (40 %) -0.094;
E2 additive -0.087. The "0.10 below under normal rain" headline survives every variant to
within 0.012; the synthetic S1 result says about a quarter of the management share in that
season may be spread-scaling inflation.

## Rain-product error on real data (revision/drivers.py rainerr -> rain_error.json)
Proxy for CHIRPS error: CHIRPS minus ERA5 window rain (neither is truth). The two products agree
poorly at 16-day pixel scale: corr 0.52 (A), 0.56 (B). Held-out rainy-season pixel-steps:

| | Box A 2019-25 | Box B all years |
|---|---|---|
| corr(residual, CHIRPS - ERA5) | -0.00 | -0.07 |
| residual per +100 mm CHIRPS excess, controlling CHIRPS departure | -0.07 sigma | -0.22 sigma |
| residual sd, middle vs outer disagreement sextiles | 0.97-0.99 vs 1.07-1.18 | 0.99-1.07 vs 1.12-1.23 |

Reading: where the two rain products disagree most, the residual is 10-20 % noisier, and on
Box B it leans negative where CHIRPS reports more rain than ERA5 (the land greened less than
CHIRPS rain implies). Rain-product error is in the residual; its mean effect is small
(|corr| <= 0.07), consistent with the semi-synthetic S3 result that it averages out at box scale
but not at pixel scale.
