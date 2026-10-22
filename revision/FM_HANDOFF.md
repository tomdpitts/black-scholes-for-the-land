# Handoff: foundation-model (Chronos-2) attribution experiment

You are taking over one experiment in /Users/tompitts/dphil/ProjectB. Read this whole file before acting.

## The paper and what matters
AgriAI workshop paper (AAAI-27, deadline 20 Nov 2026, source paper/agriai/main.tex). Claim: invert a learned
model of satellite greenness (NDVI anomaly) conditioned on weather to split a season's greenness change into
weather attributions and a residual, and to get sensitivities (Greeks) by differentiation. The contribution is the
**reverse attribution and sensitivities**, not the neural net. Use the word "attributions" for the split.

## Models already built (do not refit)
- revision/m3.py: M3 = rain + topography traits (config `topo`); **M4 = M3 + evaporative demand** (config
  `topo_pet`): net water input = rain - k(traits) x ET0 anomaly. ~1,000 parameters. Configs, `load_box`,
  `score`, `score_ensemble`, `evalsets` (A_test = Box A 2019-25, B_all, B_test = Box B 2019-25) live there.
- Fits: runs/20261021_0147_m3/gpu/params_<config>_s{0..4}.npy (5 seeds each of base, topo, topo_pet);
  metrics_*.json; metrics_ens_*.json (5-seed ensembles); metrics_M0.json (history-only reference, M0).
- Attribution code: revision/m3_attribution.py (`rerun`, `implied`; Shapley over two forcings with the residual
  held fixed). 2022 figure: revision/fig_attribution_m4.py. Map version: revision/fig_maps_split_m4.py.
- Data: data/<box>/built.npz (a, obs, r and rclim in 100 mm per 16-day window, x, phase, year, dates),
  data/<box>/topo.npz, data/<box>/era5.npz (pet, pet_clim = FAO-56 ET0, 100 mm per window).
  `m3.load_box(box, topo=True, temp="pet")` returns everything M4 uses. Box B is never used for fitting.

## Reference numbers (M4)
- 2022 Mar-May, Box A, 5-seed mean, NDVI: observed -0.168 = starting deficit -0.028 + rain -0.049
  + evaporative demand -0.035 + not explained -0.055.
- **Total weather attribution** (rain + demand) per seed: 0.090, 0.066, 0.104, 0.077, 0.084 (mean 0.084).
- 3-month skill vs M0, clean sets: M4 ensemble 0.273 (A 2019-25), 0.274 (B 2019-25); 5-seed means 0.260, 0.250.
  Trees: 0.222 (A), 0.197 (B 2019-25).
- One-step skill chart: revision/fig_onestep_skill.py -> paper/figs/fig_onestep_skill.png (Box A 2019-25 and
  Box B 2019-25 columns).
- Compute: M4 full-box (25,600 pixels) 2022 attribution + rain Greek map on laptop CPU = 16 ms after JIT.

## The experiment (agreed with the user)
Model: **Chronos-2 only** (amazon/chronos-2, ~120M params, open weights, zero-shot, past and known-future
covariates). Verify the current package and API before building. Two modes:
1. **With weather**: past and known-future rain and ET0 anomalies as covariates (attribution given the weather
   that fell, not forecasting). This is the alternative attribution engine.
2. **Without weather**: history only. An extra row on the skill chart (M0 stays the reference for all skill).
TimesFM is a possible later second model, decided after Chronos-2 results. Inference only, no fine-tuning.

Steps:
1. **Pre-register first** (runs/<YYYYMMDD_HHMM>_fm/PREREG.md, then `shasum -a 256` into PREREG.sha256 with a UTC
   timestamp) before any Chronos-2 output exists. Any later change is a dated addendum written before the run it
   affects, appended and re-hashed. Follow runs/20261021_0147_m3/PREREG.md as the template.
2. **Skill**: rolling 1-step and 6-step (3-month) forecasts, CRPS skill vs M0 using the same code and starts as
   `m3.score`, on A_test and B_test, on a fixed random sample of 4,000 pixels per box (seed 0). Re-score M4 on
   the same pixels. Add Chronos-2 rows (with / without weather) to the one-step chart.
3. **2022 attribution with Chronos-2 as engine** (full Box A):
   - Invert: at each 16-day step take the observation's percentile in Chronos-2's one-step predictive
     distribution (probability integral transform; interpolate the quantiles).
   - Counterfactuals: re-run the season one step at a time under normal rain, normal ET0, and both, taking the
     same percentile of each new predictive distribution and feeding results back as history.
   - Shapley split exactly as m3_attribution.py; box totals and maps.
4. **Greeks**: rain Delta by finite differences (+100 mm in window 1, response 2 steps ahead); autodiff if practical.
5. **Diagnostics for both engines (reported, not deciding)**: exact reconstruction when percentiles are
   re-injected; Delta-ranking test; residual trend against rain and ET0 departure, binned.
6. **Compute**: time the same full-box 2022 attribution on the laptop CPU for Chronos-2 vs M4 (16 ms).

## Pre-registered test and referee
- **Agreement**: Chronos-2's 2022 total weather attribution (rain + ET0) lies within M4's seed range
  **0.066-0.104 NDVI**. Report the difference as a number too. The rain/ET0 division and the maps are
  descriptive only (rain and ET0 correlate about -0.6, so any engine divides them unstably).
- **Referee if they disagree**: 3-month CRPS skill on clean 2019-25 data (A_test and B_test, same pixels).
  Secondary: the same restricted to drought seasons (rain departure < -60 mm). Diagnostics explain, do not decide.
- **Outcomes**:
  A M4 better skill, agree: best case; independent engine confirms ~0.08 weather attribution.
  B M4 better skill, disagree: keep M4's attributions; structure matters.
  C Chronos-2 better skill, agree: method validated; M4 same answer at ~1/100,000 compute, on a laptop, with terrain.
  D Chronos-2 better skill, disagree: adopt Chronos-2's weather attribution; pivot to "general method, foundation
    model engine, M4 as lightweight terrain-aware version".
  E Chronos-2 forecasts barely change with weather: weather attribution near zero; zero-shot models cannot
    attribute (consistent with fev-bench's covariate finding).
  F Chronos-2 without weather >= M4 at 3 months: weather adds little over a strong history model (serious; unlikely).

## How to work
- Local CPU first: build and test on a small pixel subset with .venv/bin/python (JAX on CPU). Use the CPU test to
  estimate full-run cost. **The user is cost-sensitive: ask before any Modal spend** (expected ~$3-6 for one
  Chronos-2 run). Modal pattern: revision/m3_modal.py, volume `projectb-data`.
- Do not modify model.py (hashed by earlier runs). New scripts go in revision/, outputs in the new run dir,
  figures in paper/figs/ (candidates in paper/figs/candidates/). Do not commit unless asked.
- Elegance over breadth: few clean scripts, no ad-hoc clutter; match the style of revision/m3.py.
- The user wants short, plain answers, tables for numbers, no jargon without a one-line explanation.
  Self-critique results before reporting; report failures as failures.
