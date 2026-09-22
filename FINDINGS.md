# FINDINGS

Every number cites its run directory under `runs/`. Criteria in §1 were written before
any model was trained (see git history). Incumbent numbers are filled after `baselines.py`
and before `train.py`.

## 0. Setup (fixed 2026-09-14)

- Substrate: MODIS MOD13Q1 v061 NDVI, 250 m, 16-day, 2003–2025, warped to a 0.0025°
  grid (160 × 160 per box), pixel reliability ≤ 1 kept.
- Driver: CHIRPS v2 daily 0.05°, summed over each MODIS composite window, bilinear to
  the grid. Anomalies relative to the 2003–2018 pixel climatology for both.
- Boxes: A (dev) 37.15–37.55°E, 0.05–0.45°N; B (held out) 36.70–37.10°E, 0.25–0.65°N.
- Splits: train 2003–2017, validate 2018, test 2019–2025 (Box A); Box B all years test.
- Models: M0 climatology + AR(1) shrink; M1 linear distributed-lag on rain anomalies
  lags 0–6 + AR(1) + neighbourhood; M2 GRU + 3-Gaussian mixture, zero-initialised on M1;
  M2-clim = M2 rolled out with climatological rain.

## 1. Pre-registered criteria

| Product | Metric | Incumbent | Incumbent value | Kill if | Result |
|---|---|---|---|---|---|
| Calibration gate | spread–skill corr, PIT (Box A 2019–25) | — | — | reported before any product | corr 0.993 (A), 0.984 (B); PIT top decile 12 % |
| P1 sensitivity | one-step CRPS skill vs M0, Box A test & Box B | M1 | A: 0.013 B: 0.052 | M2 < M1 + 0.02 on either | **PASS** M2 A: 0.090 B: 0.101 |
| P1 gamma | concavity of binned seasonal response | — | — | reported, not gated | concave (MAM −0.0096, OND −0.0039); local Γ positive beyond h=1 |
| P2 impact: value of response | CRPS skill M2 − M1, horizons 1–3 (realised rain) | M1 | A: 0.014/0.032/0.054 B: 0.053/0.071/0.080 | ≤ 0 | **PASS** A: +0.072/+0.107/+0.127 B: +0.050/+0.064/+0.071 |
| P2 impact: value of driver | CRPS skill M2 − M2-clim, horizons 1–3 | M2-clim | A: 0.061/0.037/0.019 B: 0.063/0.004/−0.040 | ≤ 0 | **PASS** A: +0.025/+0.102/+0.162 B: +0.040/+0.131/+0.191 |
| P3 anomaly curve 1 | hit rate within 2 composites at 5 % FAR, MCD64A1 burned pixels 2019–25 (n=208) | z-score, M1 likelihood | z: 0.063 M1: 0.192 | M2 not better than both | pass: M2 0.197 (AUC 0.727 vs 0.558, 0.713) |
| P3 anomaly curve 2 | FAR in OND 2021 and MAM 2022 at the curve-1 threshold | z-score, M1 likelihood | z: 0.017 M1: 0.060 | M2 not better than both | **KILL**: M2 0.022 beats M1, not z-score |

Skill = 1 − CRPS(model)/CRPS(M0), pooled over test pixels and steps, observed steps only.

## 2. Substrate audit (Box A; statistics on 2003–2018, out-of-sample R² on 2019–25)

Run: `runs/20260914_1422_audit`. 537 Terra composites 2002-01-01..2025-12-03, 25,600 pixels.

| Quantity | Value |
|---|---|
| Fraction of NDVI variance that is temporal | 0.584 |
| Of temporal, fraction that is the pixel seasonal cycle | 0.450 |
| Anomaly autocorrelation at 1 / 2 / 4 / 6 / 12 steps (16 d each) | 0.732 / 0.556 / 0.299 / 0.171 / 0.075 |
| Lag-1 correlation of anomaly change | −0.174 |
| Single-lag linear R² of anomaly on rain anomaly, OOS, lags 0..7 | 0.007, 0.053, 0.044, 0.033, 0.030, 0.023, 0.011, 0.001 |
| Seasonal response curvature: quadratic / linear coefficient | MAM −0.0096 / 0.0597; OND −0.0039 / 0.0346 (concave in both) |
| Fraction of composites observed, wet / dry / all | 0.976 / 0.982 / 0.979 |

**Gate outcome.** The pre-registered gate (single-lag linear R² ≥ 0.10 at some lag) was
not met: the best lag explains 5.3 % of anomaly variance out of sample. Per the plan
this is reported as a headline fact, not a stop: the anomaly is dominated by its own
persistence (0.73 at one step), so the driver's role is in the transition, not the
level, and the transition models below are scored on exactly that. Any driver value
claimed later must be read against this number.

## 3. Results

Run: `runs/20260914_1500_pipeline_h100` (H100, spawned call fc-01M2F4SKD9CRM8MZJS0RQT465B;
`progress.log` in the directory is the remote log). Audit: `runs/20260914_1422_audit`.

One-step CRPS skill vs M0: A test M1 0.013, M2nd 0.015, M2 0.090; B all M1 0.052, M2nd 0.025, M2 0.101.
Rollout skill by horizon 1..6 (A test): M1 0.014 0.032 0.054 0.074 0.092 0.107; M2 0.086 0.139 0.181 0.211 0.228 0.241;
M2clim 0.061 0.037 0.019 0.004 −0.018 −0.029; M2nd 0.013 0.002 −0.003 −0.010 −0.018 −0.023.
(B all): M1 0.053 0.071 0.080 0.088 0.094 0.100; M2 0.103 0.135 0.151 0.159 0.163 0.166; M2clim 0.063 0.004 −0.040 −0.074 −0.095 −0.105.
Greeks (grid mean, A test starts; MAM n=41, OND 26, dry 74): Δ_MAM by h 0.027 0.071 0.068 0.055 0.043 0.033 (M1: 0.009 0.034 0.031 0.025 0.020 0.016);
Γ_MAM −0.029 0.019 0.017 0.013 0.010 0.007; Θ_MAM 0.68 0.47 0.33 0.24 0.17 0.13. Δ_MAM(h=2) vs mean NDVI r = −0.84, vs annual rain r = −0.66; positive at 100 % of pixels.
Fit: λ mean M1 0.157, M2 0.382; M2 196 s on H100. First A10G attempt OOM'd at 20.6 GB (M2 full-batch backprop).
Anomaly (fire, A test, n=208): hit@5%FAR z 0.063 / M1 0.192 / M2 0.197; lag 0.54 / 0.23 / 0.27; FAR drought seasons 0.017 / 0.060 / 0.022; AUC 0.558 / 0.713 / 0.727.


## 4. Fair fight: hand-written S-curve (M1.5) versus learned residual (M2)

Run: `runs/20260914_1708_pipeline_h100_fairfight` (H100, call fc-01M2FB8M2GV4XT7NZB6PM6E9MQ; 698 s).
Added after the paper draft; M2 numbers here are a re-fit (GPU nondeterminism moves one-step skill by ~0.005).

M1.5 = M1 plus s(x)·A·[σ((w+r−θ)/τ) − σ(−θ/τ)], per-pixel scale s(x)=exp(c·x); 6 extra parameters; 2000 steps like M2.
Fitted: A = −0.023, θ = 0.73 (73 mm effective water), τ = 0.245. The curve barely engaged.
A starts at 0 and is unconstrained; the fit chose the negative sign (not a code sign error: g is increasing in u and s(x)>0).
With A<0 the term trims the linear rain response near u = 73 mm; the linear w, r coefficients steepen vs M1 (0.039, 0.023 vs 0.027, 0.009).
M1.5 Δ_MAM h=2 map (fair-fight arrays.npz): p5/p50/p95 = 0.034 / 0.038 / 0.042, i.e. within ~10 % of its mean (M2: 0.039 / 0.081 / 0.088).
corr(M1.5 map, s(x)) = −0.06: its spatial pattern comes from g'(u), where a pixel's water sits relative to θ, not from the per-pixel scale.
M1's two-step Δ is identical at every pixel (linear drift, uniform perturbation), so all M1.5 spatial variation is the S-curve term.

| Metric | M1 | M1.5 | M2 |
|---|---|---|---|
| Val NLL (2018) | −1.298 | −1.298 | −1.465 |
| One-step CRPS skill, A test | 0.013 | 0.009 | 0.085 |
| One-step CRPS skill, B all | 0.052 | 0.052 | 0.104 |
| Rollout skill h=1..6, A test | .014 .032 .054 .074 .092 .107 | .008 .023 .042 .065 .087 .107 | .081 .133 .171 .198 .214 .226 |
| Rollout skill h=1..6, B all | .053 .070 .080 .087 .093 .099 | .052 .068 .077 .085 .092 .098 | .106 .141 .158 .165 .168 .170 |
| Δ_MAM grid mean by h | .009 .034 .031 .025 .020 .016 | .012 .038 .036 .029 .023 .018 | .029 .075 .073 .060 .047 .036 |
| Γ_MAM grid mean by h | 0 | .004 .002 .001 .001 .001 .001 | −.039 .013 .013 .011 .008 .006 |
| Θ_MAM grid mean by h | — | .79 .62 .49 .38 .30 .24 | .70 .50 .36 .26 .19 .15 |
| corr(Δ_MAM h=2, mean NDVI) | — | +0.84 | −0.86 |

corr(Δ map M2, Δ map M1.5) at MAM h=2: −0.80. The two models disagree on *where* the land is
rain-sensitive; M2's pattern (dry, sparse pixels most sensitive) is the one that predicts the
held-out response (skill 0.085 vs 0.009). **Verdict:** this hand-written curve does not close the
gap; the gain is in interactions (pixel × state × season × rain) a 6-parameter curve does not
express. One parametrisation tried; not a proof that no parametric form could.

### One-sided (drop-only) anomaly scores, added post hoc, not pre-registered
Score = −log P(a ≤ observed) under the model; z-score = −a/σ_clim. Same 208 fire events, same protocol.

| Score | hit@5 % FAR | lag | FAR drought seasons | AUC |
|---|---|---|---|---|
| z-score two-sided (pre-reg) | 0.062 | 0.54 | 0.017 | 0.558 |
| M2 two-sided (pre-reg) | 0.192 | 0.28 | 0.018 | 0.735 |
| z-score one-sided | 0.149 | 0.45 | **0.244** | 0.755 |
| M1 one-sided | 0.274 | 0.25 | 0.148 | 0.711 |
| M1.5 one-sided | 0.274 | 0.25 | 0.146 | 0.705 |
| M2 one-sided | **0.279** | 0.24 | **0.051** | **0.773** |

On the one-sided score M2 beats both baselines on both curves. The z-score's 24 % drought-season
false-alarm rate is the driver-blind failure the brief predicted: it calls the drought itself an
anomaly. This is a post-hoc variant and is labelled as such wherever it is reported.

## 5. Is the Delta map a predictor on its own? (added after the paper draft)

Run: `runs/20260914_2137_deltatest_h100` (H100, `deltatest.py`, fair-fight params). Map = MAM-start mean of
∂E[a_{t+2}]/∂r_{t+1}; "train" = 2003–17 starts, "test" = 2019–25 starts. Empirical slope = per-pixel OLS of
a_{t+2}−a_t on the window-(t+1) rain anomaly over MAM starts.

| | Box A | Box B |
|---|---|---|
| corr(M2 map, mean NDVI), train / test starts | −0.67 / −0.86 | **+0.83 / −0.51** |
| corr(M1.5 map, mean NDVI), test starts | +0.84 | +0.71 |
| T2 skill vs persistence of α+β·map·rain-anomaly, fitted 2003–17, scored 2019–25: M2 / M1.5 / constant / empirical-train / −meanNDVI | 0.013 / 0.009 / 0.010 / 0.014 / 0.004 | 0.042 / 0.042 / 0.043 / 0.038 / 0.040 |
| T3 Spearman of train-derived map with 2019–25 empirical slope: M2 / M1.5 / empirical-train / −meanNDVI | 0.38 / −0.44 / 0.51 / 0.58 | −0.17 / −0.07 / 0.31 / 0.04 |
| T3 with the M2 map from the test starts themselves (in-period) | 0.70 | 0.08 |

**Verdict:** the frozen Δ map is not a standalone predictor: no better than a constant coefficient (T2),
ranked below trivial controls (T3), and its spatial pattern is period-dependent (sign flip on Box B).
The "driest pixels respond most" pattern is a feature of the drought-heavy held-out years on Box A.
The sensitivity is state-dependent; it is the model evaluated at the current state that carries the
information, not a map. Paper text changed accordingly (abstract, Sensitivities, limitations).

## 6. Does current-state Delta rank real rain exposure? (pre-registered, added 2026-09-15)

Run: `runs/20260915_1228_greekstest` (CPU, 12 s, `greekstest.py`, fair-fight params). Criteria in PREREG.md
were written before running; full table in RESULT.md. Delta = dE[a_{t+2}]/dr_{t+1} from the state at t along
climatological rain; held-out 2019–25 MAM+OND starts; per-tercile slope of a_{t+2}−a_t on rain anomaly,
controlling for a_t; bootstrap over 7 years.

| | Box A | Box B |
|---|---|---|
| G1 within-pixel (primary), high − low real slope | +.026 [−.026, +.070] **FAIL** | +.064 [+.014, +.116] pass |
| G2 raw sorting, high − low real slope | +.047 [+.005, +.084] **PASS** | +.062 [+.009, +.110] pass |
| G2 real slopes low/mid/high vs model Delta | .031/.066/.078 vs .036/.088/.124 | .074/.087/.136 vs .065/.098/.123 |
| M1.5 G2 high − low | −.052 [−.122, −.001] (backwards) | +.008 [−.117, +.085] |
| G3 skill: constant / frozen map / M2 state | .025 / .032 / .034 (diffs n.s.) | .081 / .080 / .087 (diffs n.s.) |

**Verdict:** pre-registered primary fails on Box A. M2's Delta ranks which patches really respond most to rain on
both boxes (top tercile responds 2.5× bottom on A, 1.8× on B), including the unseen landscape where its magnitudes
match the real slopes; state-dependence within a pixel is significant only on Box B. The hand-built M1.5 Delta ranks
exposure backwards on A. Error-score gains are real in sign but within noise.

## 7. Does Delta fail on large rain surprises, like Black–Scholes Delta? (pre-registered, 2026-09-15)

Run: `runs/20260915_1251_sizetest` (`sizetest.py`, same setup as section 6, pixel-steps split at the median
|rain anomaly|: 30 mm on A, 21 mm on B). PREREG.md and RESULT.md in the run folder.

| | Box A | Box B |
|---|---|---|
| Ranking gap, small surprises | +.051 [−.060, +.242] | +.154 [+.006, +.287] |
| Ranking gap, large surprises | +.044 [+.009, +.086] | +.070 [+.010, +.112] |
| Small minus large (primary) | +.008 [−.102, +.176] **FAIL** | +.084 [−.060, +.221] **FAIL** |

**Verdict:** hypothesis not supported. Delta's ranking holds on large surprises on both boxes; small surprises are
too noisy to measure. Do not claim the Greeks "share Black–Scholes Delta's failure on large moves". Box A's 2021
failure remains unexplained.

## 8. Does the residual carry forward as a latent regime? (pre-registered, 2026-09-15)

Run: `runs/20260915_0250_latent` (H100, `latent.py`, fair-fight M2 frozen). PREREG.md and RESULT.md in the run folder.
Latent z = (z0, z1) per pixel-year in the drift, μ + z0 + z1·(w + r); z_y by MAP on year y; year y forecast with z_{y−1}.
Prior scale 0.003 (grid edge; every scale worsened 2018 NLL vs z = 0). Skill vs M2, bootstrap over years.

| | Box A (2019–25) | Box B (2004–25) |
|---|---|---|
| lag vs M2, h6 (L1 primary on A) | +.0060 [−.0025, +.0156] **FAIL** | +.0017 [−.0072, +.0109] **FAIL** |
| lag vs M2, one-step | +.0033 [+.0025, +.0045] **PASS** | +.0014 [−.0003, +.0033] |
| lag vs pixel-shuffled, h6 | +.0058 [+.0048, +.0068] **PASS** | +.0034 [+.0013, +.0056] |
| same-year z (in-sample ceiling), h6 | +.027 | +.031 |
| corr(z0_y, z0_{y−1}) across pixels | .33 | .28 |
| regime effect on E[a_{t+6}], p5/p95 | about ±.01 NDVI | |

**Verdict:** a pixel-specific residual pattern persists a year (r ≈ 0.3, beats shuffled on both boxes) but buys
0.3–0.6 % of M2's error, not significant at three months on A (2019 reverses it) and not on B. Not usable for
forecasting at this cadence; the carbon counterfactual cannot be built from it. Do not claim the residual is a
forecastable management regime.

## 8. Data gap and a correction (2026-09-15)

OND 2021 has only 2 of ~6 composites in built.npz (a 61-day gap after 2021-11-01; OND 2024 has 4). Cause not yet
checked (MODIS availability or the full-window filter in data.py). The Greeks tests' "2021" is therefore mostly
MAM 2021, when rain was near normal (A +3%, B −12%), not the drought; the test-period drought is 2022
(MAM A −40%, OND A −21%). The earlier per-year reading "Delta failed in the 2021 drought" is wrong.
P3's drought seasons (OND 2021, MAM 2022) rest on only two OND 2021 composites; the paper does not say so.

Training (2003–17) contains droughts as severe as the test drought: Box A MAM 2017 −50%, MAM 2009 −44%,
OND 2005 −40%, MAM 2011 −32%, with mean greenness anomalies down to −0.117 (OND 2010) against −0.124 in MAM 2022.

## 9. The residual, backed out and plugged in (2026-09-15)

Run: `runs/20260915_1456_residual` (CPU, `residual.py`, fair-fight M2 frozen). Per-step implied innovation
eps_t = (a_{t+1} − a_t − mu)/sigma from observed greenness and observed rain; counterfactual = roll M2 from the same
state along climatological rain with eps_t re-injected (abduction / action / prediction). One start per held-out
rainy season (Box A, 2019–25, 13 seasons; OND 2025 not in record), H = 6.

- Re-injecting eps along the rain that fell reproduces the observed anomaly to max abs error 1e-6 in every season.
- Box-mean anomaly at h=6 (observed / expected under realised rain, eps=0 / counterfactual: clim rain + eps / clim rain, eps=0):
  MAM 2022 (rain −124 mm): −.168 / −.097 / −.099 / −.023. MAM 2023 (+180 mm): +.043 / +.089 / −.078 / −.028.
  Full list in `metrics.json` and the paper appendix.
- Over the 13 seasons, mean |residual| (obs − expected) = 0.041 NDVI; mean |rain term| (expected − clim-rain-no-residual) = 0.069.
- OND 2023 has the largest residual (+0.12): consistent with the PIT's under-predicted upward surprises; part of the residual is model error.
- Residual vs rain, rainy-season pixel-steps: corr 0.03 (train) / 0.05 (test). Mean eps by rain-departure bin
  (<−60, −60..−30, −30..0, 0..30, 30..60, 60..120, >120 mm): test −0.24/−0.03/−0.03/+0.22/+0.05/+0.10/−0.01;
  train −0.04/+0.07/+0.12/+0.16/+0.13/+0.09/+0.20. The driest held-out windows carry a systematic −0.24 σ residual:
  part of the 2022 residual is the model under-responding to the drought.
- Fire: burn 2021-01-17 (50 px), scored on 2021-02-02: mean eps burned −0.75 vs −0.04 unburned; z-score −0.17 vs −0.13;
  36 % of burned px below eps's own 5 % threshold vs 14 % for z. The 2022-02-18 burn (49 px) registers in neither score.
  Paper Figure 2 (`paper/figs/fig_fire.pdf`).

**Verdict:** the residual is exact by construction, of the same order as the rain term, and plugs into a counterfactual.
What it is made of (grazing/clearing/fire vs CHIRPS error vs model error vs noise) is not separated here; §8 shows it is
not a forecastable regime at three months. Paper: Approach, "The residual, backed out and plugged in", Figure 1, Appendix.
