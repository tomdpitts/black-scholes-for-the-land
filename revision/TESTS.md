# Test suite for the AgriAI paper (agreed 2026-10-22)

The paper's evidence is four tests, run on one cast of models. This file is the plan. Each run's pre-registration
(runs/<ts>_suite/PREREG.md) fixes the details below before any result exists. The Chronos-2 engine details are in
revision/FM_HANDOFF.md.

## The chain
1. **Skill**: is the model accurate on unseen data?
2. **Attribution**: does it correctly separate the weather's effect from everything else, and better than
   current practice?
3. **2022 drought**: what does it say about a real season?
4. **Sensitivities**: do pixels the model calls more rain-sensitive really respond more?

The model is accurate (1) → its attribution is correct and beats current practice (2) → here it is on a real
drought (3) → its sensitivities are real (4). Every link is also tested against a foundation model (Chronos-2).
Every comparison reports compute alongside accuracy.

"Weather" means rain and evaporative demand (FAO-56 reference ET anomaly). "Attributions" is the word for the split.

## The cast
| Model | What it is | Source |
|---|---|---|
| M0 / z-score | history only. Inverting M0 is exactly the z-score, which is current practice: index insurance (IBLI) pays on it, and the VM0032/SNAP carbon baseline treats a given year's rain the same way | fair-fight params, runs/20260914_1708_pipeline_h100_fairfight/params.npy `M0` |
| RESTREND | the standard research method: per-pixel linear regression of greenness on rain; the residual is the land (spec below) | new, revision/suite.py |
| M1 | linear rain model, 16-day, shared weights (model.py) | fair-fight params `M1` |
| M2 | M1 + 64-unit network, standardised traits (m3.py config `base`) | runs/20261021_0147_m3/gpu/params_base_s{0..4}.npy |
| M3 | M2 + topography traits (`topo`) | params_topo_s{0..4}.npy |
| M4 | M3 + evaporative demand draining the water input (`topo_pet`) | params_topo_pet_s{0..4}.npy |
| M4 ensemble | equal-weight mixture of the 5 M4 seeds | the same 5 fits |
| Chronos-2 | foundation model, zero-shot, with and without weather covariates | amazon/chronos-2, see FM_HANDOFF.md |

M2, M3 and M4 are the refits from one code base, so each step adds one ingredient. M2 is the `base` refit in every
test; the paper's fair-fight M2 is not in the cast (older harnesses, revision/identify.py and attribution.py, load
it: do not reuse them for M2). Single-fit rows are means over the 5 seeds, with the seed range; the ensemble row is
the mixture. M1.5 (S-curve) is not in the cast. The trees (GBM) appear in test 1 only, as an existing number.

No model is refitted for the suite: every cell uses the fits above. The only fitting is RESTREND's regressions
(seconds on CPU). Do not call attribution.py's `fit_gbm`.

**RESTREND spec.** Per pixel, ordinary least squares on Box A training starts (any 16-day start whose target year is
2003-2017, both ends observed): a_{t+6} = alpha + beta R6, where R6 is the rain departure summed over windows t+1..t+6
(100 mm). Spread = the pixel's training residual sd (Gaussian). Its attribution: rain share = beta x the season's R6,
and the rest is the land. Its sensitivity: beta, a fixed map. Box B is never used for fitting, so on Box B,
RESTREND's alpha, beta and spread come from a Box A linear regression of each on the three standardised traits.
Box A is the primary set for RESTREND.

## Test 1: Skill
- Sets: Box A 2019-25 (A_test) and Box B 2019-25 (B_test). Box B all-years is inflated by shared regional weather
  and is not reported.
- Metric: CRPS skill vs M0, 1-step and 3-month (6-step sampled rollout along the weather that fell), same code,
  starts and masks as `m3.score`. RESTREND: 3-month only, Gaussian CRPS on the same starts and masks; shown for
  completeness (it has no starting state, so it is not a forecaster).
- Indexing: mask_t[t] = year[t+1] in test years; valid pairs are obs[t] & obs[t+1+h], h = 0..5, so 3-month =
  obs[t] & obs[t+6]. The 4,000-pixel scores mask the full-grid score (model.py uses neighbourhood features, so the
  grid cannot be subset): extend `m3.score` with an optional pixel mask, default unchanged.
- Status: M0, M2, M3, M4 and the ensemble are done (runs/20261021_0147_m3/gpu). New: RESTREND, and M1 re-scored on
  A_test and B_test, both horizons. Chronos-2: on a fixed 4,000-pixel sample per box (seed 0), with M0, M4 and the
  ensemble re-scored on the same pixels (CPU, in the local run).
- Also: drought starts only, as the referee's secondary measure: a start is a drought start if its box-mean rain
  departure over windows t+1..t+6 is below -60 mm. Report the number of starts.

## Test 2: Attribution
Inject management of a known size into a fake world, give each estimator only the fake greenness and the weather
that fell, and score how much management it recovers.

- **Worlds**: the M4 world (M4 seed 0) and the Chronos-2 world (Chronos-2 with weather; built on GPU, see
  FM_HANDOFF.md). A world is the truth for that run. The z-score cannot be a world: rain would do nothing there.
- **Seasons**: the 13 held-out Box A rainy seasons 2019-25 (`residual.season_starts`), six windows each, from the
  observed state. Each season's real unexplained part comes from inverting the world model on the real data.
  Real background management therefore sits in both the truth and the estimate and cancels.
- **Management**: the identification test's mask (random half of 2 km blocks, seed 0) and size (-0.01 NDVI per
  window). Scenarios: S0 none; **S1 weather-independent (headline)**; S2a incursion and S2b destocking (management
  follows rain; reported as the stated limit).
- **Truth**: the world re-run under normal weather (climatological rain, zero demand anomaly) with the same
  unexplained part, with management minus without.
- **Estimators**: z-score, RESTREND, M1, M2, M3, M4, M4 ensemble (mean of the 5 fits' counterfactuals), plus
  Chronos-2 in its own world. Rain-only estimators replay normal rain only, so demand's effect stays in their
  "land" share; that is part of what is being compared.
  - In the M4 world, "M4" is the world's own fit (seed 0): the exact own-world cell. Seeds 1-4 are reported
    separately: how much does seed-to-seed model error cost? The 5-fit ensemble contains the world's own fit, so
    the ensemble of seeds 1-4 is reported beside it as the fair out-of-world row.
  - In the Chronos-2 world: z-score, RESTREND, M1, M2, M3, M4, the ensemble and Chronos-2. Chronos-2 is not scored
    in the M4 world.
- **Metrics** (S1 unless noted; drought seasons, rain departure < -60 mm, and other seasons separately: 4 and 9 in
  Box A 2019-25, with MAM 2019 at -56 mm just outside; report n beside every number). For every
  error metric, zero is perfect and further from zero is worse; the sign says the direction (+ = management blamed
  for drought damage).
  - error in the management estimate as a share of the true effect: signed mean, mean size, worst season
  - management invented on untouched land, as a share of the true effect
  - management found when none was injected (S0, NDVI)
  - pixel-level error (RMSE, NDVI) alongside the box-level error
  - difficulty check (not an error): each world's box-mean season-end greenness change for a season 100 mm drier
    than normal, spread evenly over the six windows, from the observed starts. Shows whether the two worlds are like for like.
- Code: reuse `block_mask`, `DELTA` and the management formula (identify.py, attribution.py) and their scoring.
  `identify.make_fns` runs on model.py, so it serves M1 only. For the M4 world and the M2-M4 estimators, use
  `implied` and `rerun` from revision/m3_attribution.py; the world is `rerun` plus the management term (one new
  function).

## Test 3: 2022 drought
- Box A, March-May 2022. Split of the observed -0.168: starting deficit, weather (rain + demand), not explained.
- Per model: z-score (all to the land), RESTREND, M1, M2, M3, M4 (5-fit mean and seed range), M4 ensemble,
  Chronos-2.
- **Headline: the total weather share.** The rain/demand division is descriptive only (they correlate about -0.6).
- Models without a state (z-score, RESTREND) have no starting-deficit term: their split is weather + not explained.
- Chronos-2 agreement test and referee: FM_HANDOFF.md (pre-registered range 0.066-0.104 NDVI).
- Existing: M4 per seed (runs/20261021_0147_m3/gpu/attribution_topo_pet_s*.json). The ensemble's split equals the
  5-fit mean (the shares are linear in the counterfactuals), so one number serves both rows. M2 and M3 are computed
  from their 5 refits; the paper's M2 numbers and the CPU seed-0 M3 (runs/20261021_0147_m3/attribution_topo_s0.json)
  are not used.

## Test 4: Sensitivities
- The pre-registered Delta ranking design (greekstest.py, runs/20260915_1228_greekstest/PREREG.md): Delta =
  dE[a_{t+2}]/dr_{t+1} from the state at t, rolled out along normal weather (for M4: climatological rain and zero
  ET0 anomaly); outcome a_{t+2} - a_t against the rain anomaly, controlling for a_t; terciles; bootstrap over years
  (2000 draws). Secondary: the same, also controlling for the ET0 anomaly (rain and demand move together).
- Reuse greekstest.py's `starts`, `outcomes`, `sort_test`, `ci` and greeks_nogap's `starts_nogap`; its `deltas` and
  `main` are model.py-only (M1). M2-M4 Delta: jvp through the m3 rollout.
- G1 within-pixel (primary) and G2 raw sorting, Box A and Box B 2019-25. **Primary set: starts whose two-step
  outcome does not span a MODIS gap** (the known data-validity correction, runs/20261006_1801_gaps). Report all starts too.
- Models: M1, M2, M3, M4, M4 ensemble (mean Delta of 5 fits), RESTREND (G2 only: a fixed map has no within-pixel
  variation), Chronos-2 (reported, not deciding).
- Delta by autodiff (jvp) for M1-M4. Chronos-2: autodiff if possible, otherwise finite differences; say which.

## Compute (every model, same laptop, same task)
Timed in one dedicated pass, one job at a time, nothing else running; record chip, cores, JAX and torch versions.
| Measure | Notes |
|---|---|
| parameters | M4 1,002; Chronos-2 ~120M |
| training | M4 ~80 min CPU / ~2.5 min H100 per fit, x5 seeds; Chronos-2 none (cite pretraining) |
| test 1 scoring time | per set |
| test 2 simulation + attribution time | per season and total |
| test 3 full-box 2022 attribution | laptop CPU, first call and warm (M4: 16 ms warm, 0.2 s first) |
| test 4 full-box Delta map | same; autodiff vs finite differences |
| hardware and cloud cost | CPU or GPU; $ for Modal runs |

## Cut from this paper
Calibration (CRPS already penalises a badly stated spread), fire / drought-vs-disturbance (follow-up paper),
shock half-life (Theta) and rain widening the spread (one line pointing to future work at most), the rain-vs-demand
separation check. Remove their M2 numbers from the paper text. The 2017 fence-line test is not cited.

## Status
This table is the suite's deliverable: each run's RESULT.md opens with it, a number in every cell, detail tables
below. Legend: ✅ reuse the existing number (path relative to runs/20261021_0147_m3/) · 🆕 compute in the local run
· GPU = Chronos-2 cell, left for the Go/No-go (the local run gives its cost estimate) · — does not apply.

| Model | 1 Skill | 2 Attribution (M4 world) | 3 2022 drought | 4 Sensitivities | Compute |
|---|---|---|---|---|---|
| M0 / z-score | ✅ the reference (gpu/metrics_M0.json) | 🆕 | ✅ all 0.168 blamed on the land | — (no rain response) | 🆕 |
| RESTREND | 🆕 3-month only | 🆕 | 🆕 | 🆕 G2 only (fixed per-pixel slope) | 🆕 |
| M1 | 🆕 re-score, both horizons | 🆕 | 🆕 | 🆕 | 🆕 |
| M2 (`base` refit) | ✅ gpu/metrics_base_s* | 🆕 | 🆕 | 🆕 | 🆕 |
| M3 | ✅ gpu/metrics_topo_s* | 🆕 | 🆕 (5 seeds) | 🆕 | 🆕 |
| M4 | ✅ gpu/metrics_topo_pet_s* | 🆕 own world (seed 0), seeds 1-4 separately | ✅ gpu/attribution_topo_pet_s* | 🆕 | 🆕 |
| M4 ensemble | ✅ gpu/metrics_ens_topo_pet.json | 🆕 5 seeds, and seeds 1-4 | ✅ same as the M4 5-fit mean | 🆕 | 🆕 |
| Trees (GBM) | ✅ RESULT.md skill table (test 1 only) | — | — | — | — |
| Chronos-2 | GPU | not scored (see its own world) | GPU | GPU (reported, not deciding) | 🆕 CPU estimate |

Headline number per column:
| Column | Number in each cell |
|---|---|
| 1 Skill | 3-month CRPS skill vs M0, A_test / B_test |
| 2 Attribution | S1 error in the management estimate as a share of the true effect, signed, drought / other seasons |
| 3 2022 drought | total weather share (rain + demand), NDVI |
| 4 Sensitivities | G1 high-minus-low tercile slope, gap-free starts, Box A / B, with 95% CI and pass (G2 for RESTREND) |
| Compute | laptop CPU warm time for the full-box 2022 attribution, and the parameter count |

Test 2 in the Chronos-2 world is a second table, all GPU: z-score, RESTREND, M1, M2, M3, M4, the M4 ensemble, and
Chronos-2 (its own-world cell), same metrics.

## Claims and outcomes (confirmatory; everything else is descriptive)
One claim per test, ordinal only (no size thresholds). With 4 drought seasons there are no significance tests: report
the mean and how many seasons each comparison wins. Each outcome has its framing written here, so no result is
re-framed after the fact.

| Test | Claim | Outcomes |
|---|---|---|
| 1 Skill | 3-month skill vs M0 on the 4,000-px samples, A_test and B_test: M4 ensemble vs Chronos-2 with weather (the referee) and without (outcome F). The M4 ensemble decides; the 5-seed mean is reported | FM_HANDOFF.md outcomes A-F |
| 2 Attribution | M4 world: M4 (seed 0) and the seeds 1-4 ensemble each have a smaller S1 drought mean error size than the z-score and RESTREND. The Chronos-2 world is the cross-check | Both beat both: given controlled drivers, the inversion recovers management better than current practice and the standard research method. Seed 0 passes, seeds 1-4 do not: the inversion works; model uncertainty, not the method, limits it. Beats z-score, not RESTREND: including rain is what matters at box scale. Beats neither: box-scale attribution is limited by how well the rain response is known; the paper leans on tests 3-4 and compute |
| 3 2022 | Chronos-2 total weather share within M4's seed range, 0.066-0.104 NDVI | FM_HANDOFF.md outcomes A-F |
| 4 Sensitivities | M4 ensemble G1 passes (greekstest criterion: high-minus-low 95% CI above 0), gap-free starts, Box A | Box B only, or G2 only: rankings hold across pixels, not within a pixel over time. Neither: the sensitivities are descriptive, not validated at 16-day scale |

Test 2 checks the inversion given controlled drivers; test 1 checks the model against reality. The paper describes
test 2 that way.

## Run protocol
- **Order**: Chronos-2 API check → pre-register the whole suite → build and debug → local run (laptop CPU) →
  Chronos-2 CPU cost test → show the user tables and figures → the user decides Go/No-go on Modal for Chronos-2.
- **Held-out data**: develop and debug on 2003-17 only. The first run on 2019-25 is the run. Before it, reproduce
  one existing number (M4 seed 0, A_test, gpu/metrics_topo_pet_s0.json) within 1e-3 with the extended `m3.score`.
- **Chronos-2**: a separate venv (.venv-chronos), so the JAX .venv is untouched. The API check (package, version,
  prediction call, quantile levels, covariate format, context limit, whether gradients reach the inputs, CPU time)
  and the cost test use synthetic series or 2003-17 only: no Chronos-2 number on 2019-25 before the Go. Every
  Chronos-2 cell is coded and tested end to end in the local run, so the GPU run only executes.
- **The PREREG fixes**: git HEAD; sha256 of TESTS.md, FM_HANDOFF.md, m3.py, model.py, data/{A,B}/{built,topo,era5}.npz
  and the params files; for Chronos-2, CRPS from quantiles (the formula), percentile interpolation and the tail rule
  outside the outer quantiles, context length, covariate encoding, gap handling, whether pixels are forecast
  independently or as a group (cross-learning changes the outputs), the 4,000-px samples (seed 0), and the rule
  that sets the test 2 world's sample size from the cost test (the number goes in a dated addendum).
- **No refits** beyond RESTREND's regressions (see the cast).
- A cell that fails or misses its claim is recorded and reported; the run carries on.
