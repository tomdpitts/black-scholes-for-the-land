# AgriAI test suite: four tests, one cast, Chronos-2 in every test. Pre-registered before any result

Written 2026-10-22 before any suite number exists. Plan: revision/TESTS.md (agreed 2026-10-22); Chronos-2 specifics:
revision/FM_HANDOFF.md. This file fixes what those leave open. Seen beforehand: every number in
runs/20261021_0147_m3/RESULT.md and its gpu/ files; the Box A season rain departures (inputs, listed below); the
Chronos-2 API check (synthetic series only, facts below). No suite metric has been computed on any data; no Chronos-2
call has touched real data.

## Fixed state
- git HEAD e839c8cfeac5380eacd0f0430899e1b8bd0973ca, plus uncommitted, behaviour-preserving extensions made for the
  suite before this file (defaults checked identical to HEAD on 2016-17 starts and MAM2009: differences 0.0):
  revision/m3.py `score` and `score_ensemble` take an optional pixel mask `px` (TESTS.md test 1);
  greekstest.py `sort_test` takes an optional extra control `ctrl` (test 4 secondary);
  revision/greeks_nogap.py runs its old main only under `__main__`, so `starts_nogap` can be imported;
  revision/m3_attribution.py: the per-season split is now the function `season_split`, `attribution` calls it.
- sha256 of the plan, code, data and fits: PREREG_inputs.sha256 beside this file (TESTS.md, FM_HANDOFF.md, m3.py,
  model.py, the three edited files, data/{A,B}/{built,topo,era5}.npz, the fair-fight params.npy, the 15 gpu params).
- Environments: .venv (Python 3.10.20, JAX 0.6.2 CPU, numpy 2.2.6); .venv-chronos (Python 3.10.20,
  chronos-forecasting 2.3.2, torch 2.14.1, transformers 5.20.2). Laptop: Apple M4 Max, 16 cores, 64 GB.
- New scripts (after this file): revision/suite.py (JAX: every non-Chronos cell, the scoring of Chronos-2 outputs,
  timing, tables), revision/c2.py (Chronos-2 engine, .venv-chronos), revision/c2_modal.py (GPU wrapper, not run
  before the Go), revision/fig_suite.py.

## Chronos-2 API check (synthetic series only)
amazon/chronos-2, 119,477,664 parameters, float32. 21 quantile levels 0.01, 0.05, 0.10, ..., 0.95, 0.99. Context
limit 8192 steps, input/output patch 16 (1-6 step forecasts are one forward pass). Covariates: per task a target row
plus one row per covariate, linked by a group id; known-future covariate values go in `future_covariates` (NaN for
the target row). Calling the network directly with that layout reproduces `pipeline.predict` to 2e-7. Without
cross-learning a series' forecast does not depend on the rest of the batch (4e-8). NaN in the context is accepted.
Gradients reach the future covariates (autodiff vs finite differences on a synthetic rain response: corr 0.996).
CPU, L = 537 context steps: about 3.4 ms per row-forward (a task with two covariates is three rows).

## Chronos-2 protocol
- Series: the same gap-filled NDVI anomaly every other model sees (built.npz `a`), from the first window (2002-01-01)
  to the forecast origin, no cap (at most 537 steps). Scores use the same `obs` masks as `m3.score`.
- With weather: two covariates, rain anomaly r - rclim and FAO-56 ET0 anomaly pet - pet_clim (both 100 mm per
  window), past values over the context and known-future values over the forecast. Without weather: target only.
- Each pixel is its own task: no cross-learning, no grouping across pixels.
- Normal weather = both covariates 0 over the season windows (rain at climatology, zero demand anomaly); the
  history before the season is real.
- Predictive distribution from the 21 quantiles: values sorted (removes any crossing); quantile function linear in
  z = Phi^-1(tau) between the knots and extended linearly in z beyond 0.01 and 0.99 using the outer segments
  (Gaussian tails; exact for a Gaussian forecast). CRPS = 2 x the mean pinball loss over 1,000 levels
  tau_j = (j - 0.5)/1000. Percentile (PIT) of a value: the inverse of the same function, z clipped to [-8, 8].
- 1-step and 3-month skill: one forecast of 6 steps per start (direct, prediction length 6), h = 1 and h = 6
  marginals, against M0's `rollout_crps[0]` and `[5]` on the same pixels, starts and masks.
- Step-by-step runs (test 2 world, test 3): prediction length 1 per step; each step's value = the given percentile of
  that step's predictive distribution, appended to the context before the next step. Inversion: the 6 one-step
  forecasts given the real history (independent, batched); percentile of the observed value in each.
- Test 3 split exactly as m3_attribution.season_split: F = the median path (percentile 0.5) along the weather that
  fell; counterfactuals rain-normal, demand-normal, both, with the season's percentiles held fixed; Shapley over
  the two forcings; start deficit = the rest. Total weather share = rain + demand (NDVI; negative = browning).
- Test 4 Delta: from the real history to t, a 2-step forecast along normal weather; Delta = d(median at t+2) /
  d(rain anomaly at t+1), by autodiff through the network (the API check shows gradients reach the inputs);
  finite differences (+0.01) are timed and compared on the debug data only. On the 4,000-pixel samples.
- GPU: float32 weights, TF32 matmuls (torch.set_float32_matmul_precision("high")), batch size does not change outputs.

## Samples and season classes
- 4,000-pixel sample per box: numpy default_rng(0).choice(25600, 4000, replace=False) on the flattened 160 x 160
  grid, the same indices for A and B; saved as samples.npz.
- Drought start (test 1): box-mean rain departure over windows t+1..t+6 below -60 mm (full box). Drought season
  (test 2): the same over the season's six windows. Box A 2019-25 seasons (start index, departure mm):
  MAM2019 394 -56, OND2019 408 +434, MAM2020 417 +255, OND2020 431 -38, MAM2021 440 +9, OND2021 454 -172,
  MAM2022 460 -124, OND2022 474 -75, MAM2023 482 +180, OND2023 496 +215, MAM2024 505 +133, OND2024 518 -77,
  MAM2025 526 +83. Drought: OND2021, MAM2022, OND2022, OND2024 (n = 4); other n = 9.

## Test 1 details
- New full-grid cells: M1 (fair-fight, model.py, raw traits) 1-step and 3-month; RESTREND 3-month (Gaussian CRPS,
  starts t < T-6 with year[t+1] in 2019-25, pixels obs[t] & obs[t+6]). A_test and B_test.
- Drought starts only: M0, M1, M2-M4 (5 seeds each), M4 ensemble, RESTREND, full grid; plus M0, M4 (5) and the
  M4 ensemble on the 4,000-px samples. Number of starts reported. Skill always against M0 on the same starts/pixels.
- 4,000-px: M0, M4 (5 seeds) and the M4 ensemble re-scored by the extended `m3.score` (`px`), all starts and drought.
- Before the first 2019-25 number: reproduce gpu/metrics_topo_pet_s0.json A_test (1-step and 3-month CRPS) within
  1e-3 with the extended score. If it fails, stop and report.

## Test 2 details
- Mask: identify.block_mask(160, 8, default_rng(0)) (8 px = 2.2 km blocks); DELTA = 0.01; management formula
  attribution.management; scenarios S0, S1 (headline), S2a (k = +1), S2b (k = -1).
- M4 world: M4 seed 0 on the full grid. Unexplained part = m3_attribution.implied on the real season. World = rerun
  plus the management term after each step (suite.world). Truth = world under normal weather (rc, zero demand)
  with and without management. Estimators' counterfactual = their own inversion (z-score: none; RESTREND:
  y_end - beta R6; M1: identify.make_fns standardised; M2-M4: implied + rerun) re-run under normal rain (and zero
  demand for M4). Ensembles average the members' counterfactual maps (5 seeds; seeds 1-4).
- Metrics per season, managed pixels unless noted: rel = mean(cf_hat - cf_true) / mean(cf_true - cf_true0); signed
  mean, mean size and worst (largest size, signed) over the season class; invented = mean over unmanaged pixels of
  (cf_hat - cf_true) / |true effect|, mean size; S0 = mean over all pixels of (cf_hat - cf_true), NDVI, signed mean
  and mean size; pixel RMSE over all pixels and box error over managed pixels, NDVI. Single-fit rows: mean over the
  seeds and the seed range of each class metric. S2a/S2b are reported as the stated limit.
- Difficulty check: world(season end, rain = max(rclim - 1/6, 0) per window, zero demand anomaly) minus world(normal
  weather), same unexplained part, no management, box mean; mean over the 13 seasons and by class; the achieved
  mean rain departure is reported beside it.
- Chronos-2 world (GPU): a centred square tile of side L (multiple of 16, so blocks align; the block mask cropped).
  Inside the tile the world is Chronos-2 with weather (inverted real season, management added to each step's value
  before it is fed back); outside it every scenario keeps the real season, so all JAX estimators run on the full grid
  as in the M4 world; metrics use the tile only. Chronos-2's own estimator: invert the world's greenness, re-run
  under normal weather. The M4 world's metrics are also reported on the same tile (descriptive).
- **Sample-size rule** (fixed now, the number goes in a dated addendum after the CPU cost test): L = the largest of
  160, 144, ..., 32 whose central-estimate H100 cost for the whole Chronos-2 world cell (13 seasons, 4 scenarios,
  the own estimator, the difficulty check) is at most $5. If L = 32 costs more than $5, the 4 drought seasons only.
- Claim and outcome framings: TESTS.md, verbatim. Wins counted season by season (|rel| smaller) beside the means.

## Test 3 details
M1: model.py, rain only (identify.make_fns, standardised residual). M2, M3, M4: m3_attribution.season_split per
seed (M4 recomputed and checked against gpu/attribution_topo_pet_s*.json, tolerance 1e-4). RESTREND: weather =
box mean of beta x R6, rest not explained, no deficit term. z-score: all to the land. Rows report mean and seed range.

## Test 4 details
greekstest `starts`, `outcomes`, `sort_test`, `ci`; greeks_nogap `starts_nogap` (primary). M1: greekstest.deltas.
M2-M4: jvp of the mean rollout (zero residual) along rclim (and zero ET0 anomaly for M4) for +1 (100 mm) in window
1, read at step 2. Ensemble: mean Delta of the 5 fits. G1 = Delta minus its per-pixel mean over the starts; G2 =
raw. RESTREND: beta (Box B: from the trait regression), G2 only. Secondary: also controlling for the ET0 anomaly at
t+1 (`ctrl`). Every sort_test call gets a fresh default_rng(0); 2000 bootstrap draws over years. Single-fit rows:
mean high-minus-low over seeds, range, number of seeds passing. Chronos-2: same tests on the 4,000-px samples (the
M4 ensemble re-run on the same pixels beside it), reported, not deciding.

## Compute
One dedicated timing pass after the local run, one job at a time. Per model: test 1 scoring per set; test 2 world
simulation + attribution per season and total; test 3 full-box 2022 first call and warm (median of 5); test 4
full-box Delta map, first and warm, autodiff, and finite differences for M4. Chronos-2: CPU ms per row-forward
measured on 2003-17 contexts; task counts per cell from the protocol above; H100 time = CPU time x (CPU throughput
/ GPU throughput) with CPU throughput from the measured time and a FLOP count, GPU 120 TFLOP/s effective (central,
TF32) and 33 TFLOP/s (pessimistic, plain FP32); price $3.95 per H100-hour (Modal list price; to be checked before
the Go).

## Outcomes
TESTS.md "Claims and outcomes" and FM_HANDOFF.md outcomes A-F, verbatim. A cell that fails is recorded and the run
carries on. Tests 1 and 3 claims wait for the Chronos-2 GPU cells; tests 2 and 4 are decided by the local run.

## Addendum 1, 2026-10-22 (local time, after the local run and the CPU cost test; before any Chronos-2 call on 2019-25 and before the tile re-run of the M4 world)
Seen: every local-run number (tests 1-4, M4 world full box), the timing pass, c2_cost.json (CPU measured on 2003-17
and synthetic contexts), and the debug outputs (2015-17, in-sample).
- **Test 2 world size (the pre-registered rule applied):** L = 128 (16,384 px, offset 16). Central-estimate H100 cost
  of the world cell $4.23 (L = 144 would cost $5.35, over the $5 bar); pessimistic $15.36. All 13 seasons.
- **M1 in test 4:** M1 is linear, so its Delta is one number (0.034 NDVI per 100 mm) at every pixel and start; G1 and
  G2 have nothing to sort and are recorded as undefined, not as fails. (Found in debugging; a property of the model.)
- **Test 3 compute column:** the time reported for M2-M4 is the split's five maps as one compiled call (as the 16 ms on
  record); the uncompiled season_split time is reported beside it.
- The M4 world is re-run with the tile (`suite.py attrib --tile 128`) for the descriptive tile metrics; its full-box
  numbers must equal the first run's.

## Addendum 2, 2026-10-22 (local time, after the test 2 seed-0 results were seen; before any run below). User decision.
Seen: all local-run results, including test 2 in the M4 seed-0 world (verdict "seed 0 passes, seeds 1-4 do not";
the ensemble of seeds 1-4 lost to RESTREND on the drought mean, 0.93 vs 0.86). Reason for the change: the cast's
model is the 5-seed ensemble (tests 1, 3, 4), which cannot be scored cleanly in a world built from one of its members.
The seed-0 results stay on record (attrib_m4.json) and are reported beside the new ones.
- **Test 2 M4 worlds:** five worlds, one per M4 seed 0-4, each built exactly as before (that seed's standardised
  residuals of the real season + the management term). Every estimator is scored in every world; each season-class
  metric is computed per world, then averaged over the five worlds (range over worlds reported). Wins are counted
  per world and averaged.
- **Headline estimator:** the M4 5-seed ensemble (mean of the five fits' counterfactuals, the test 3 definition). In
  each world one of its members built the world: the truth is one of the ensemble's own fits, each equally likely.
  Also reported: the single fit in its own world, and single fits in the other four worlds (averaged).
- **Claim (replaces TESTS.md's test 2 claim):** the M4 5-seed ensemble has a smaller S1 drought mean error size than
  the z-score and than RESTREND, averaged over the five worlds. Framings: beats both: given controlled drivers, the
  inversion recovers management better than current practice and the standard research method; beats z-score, not
  RESTREND: including rain is what matters at box scale; beats neither: box-scale attribution is limited by how well
  the rain response is known. The Chronos-2 world (no member of the cast built it) remains the cross-check.
- All 13 seasons pooled are reported beside drought / other (descriptive; the claim stays on droughts).
- The descriptive tile metrics use the same five worlds. No other change: same seasons, mask, scenarios, metrics, code.

## Addendum 3, 2026-10-22 (local time), before any run below. User decision: add test 2a.
Seen beforehand: all results above; and, from runs/20261021_0147_m3/gpu/attribution_topo_pet_s*.json, each single M4
fit's box-mean "residual" (observed minus the forecast along the weather that fell) and rain departure for the 13
seasons (box level, single seeds). The block-level, ensemble statistic below has not been computed.
- **Test 2a, weather lean. Question:** does what the model leaves unexplained lean with the weather?
- **Leftover** of a season, per pixel: observed season-end NDVI anomaly minus the method's season-end forecast from the
  observed start along the weather that fell, residual zero. M4 ensemble: mean of the five fits' forecasts
  (m3_attribution.rerun with zero residual; the "F" of season_split). Yardstick, the z-score: the same with M0's
  forecast from the start (history only; model.rollout_mean, as residual.py), which leaves all the weather in.
- **Unit:** 2 km blocks (8 x 8 px, 400 per box), Box A, the 13 held-out seasons 2019-25. Block value = mean over pixels
  observed at the season's start and end; blocks with under half their pixels observed are dropped.
- **Weather:** the block's rain departure summed over the six windows (mm). ET0 departure: reported, not deciding.
- **Statistic:** pooled OLS slope of leftover (NDVI) on rain departure (per 100 mm) over seasons x blocks; 95% CI by
  bootstrap over seasons (2000 draws, default_rng(0)). Also the M4 slope as a share of the z-score's, and the 13
  season box means as a table.
- **Claim:** the M4 ensemble removes the weather lean: its slope's 95% CI includes 0.
- **Outcomes:** CI includes 0: weather removed; test 3's weather share stands on real data. Slope > 0 (leftover browner
  in dry seasons): M4 underplays the weather, or management worsened in droughts (incursions); cannot say which.
  Slope < 0 (leftover greener in dry seasons): M4 overplays the weather, or management eased in droughts
  (destocking); cannot say which. Either way the share of the z-score's lean is reported ("removes x% of the weather
  bias current practice carries").
- **Assumption, stated in the paper:** real management does not follow the weather. Test 2b (injection) is not run.
- Code: revision/suite.py `lean`, reusing split_maps / rerun / implied, model.rollout_mean, r6, the plan's seasons.
