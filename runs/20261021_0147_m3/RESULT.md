# M3 / M4: site traits and evaporative demand. Result against PREREG.md (and addenda 1-5)

Scripts: revision/topo.py (DEM traits), revision/m3.py (model, fit, scoring, ensembles), revision/m3_modal.py
(GPU fits), revision/m3_attribution.py, revision/fig_attribution_m4.py, revision/fig_maps_split_m4.py,
revision/fig_onestep_skill.py. Final comparisons: 5 seeds per configuration fitted on Modal H100 (gpu/);
GPU and CPU fits agree to about 0.005. Skill = CRPS reduction vs M0 (metrics_M0.json, same code and starts).

## Models
- **M3** (`topo`): M2 plus three site traits from Copernicus DEM GLO-30 (elevation, slope, topographic position),
  all six traits standardised on Box A. Traits shape the drift network and set the rain memory lam(s).
- **M4** (`topo_pet`): M3 with evaporative demand (FAO-56 reference ET, ERA5 via Open-Meteo, anomaly vs
  2003-18) draining the water input: r_net = r - k(traits) PET'. k starts at 0 (= M3).
- **Ensemble**: equal-weight mixture of the 5 seeds (score_ensemble; the stacked-vmap version was wrong under
  jit and was replaced before any reported ensemble number).

## Skill, 5 seeds (mean, and sd at 3 months) and 5-seed ensemble
| Model | A 2019-25 1-step | A 3-month | B all 1-step | B all 3-month | **B 2019-25 1-step** | **B 2019-25 3-month** |
|---|---|---|---|---|---|---|
| base (M2 refit) | .092 | .241 ± .005 | .090 | .135 ± .027 | .075 | .177 ± .026 |
| base, ensemble | .094 | .249 | .100 | .158 | .088 | .197 |
| M3 | .099 | .254 ± .005 | .099 | .168 ± .025 | .083 | .204 ± .027 |
| M3, ensemble | .103 | .264 | .110 | .192 | .096 | .224 |
| M4 | .103 | .260 ± .009 | .130 | .251 ± .018 | .112 | .250 ± .020 |
| **M4, ensemble** | **.111** | **.273** | **.145** | **.280** | **.131** | **.274** |
| Trees (B1 GBM) | .046 | .222 | .090 | .260 | .003 | .197 |
| M2 (paper) | .085 | .226 | .104 | .170 | .084 | .188 |

Box B all-years scores are inflated for models reading regional weather (2003-17 are Box A's training years):
trees fall from .260 to .197 on Box B 2019-25. Box B 2019-25 is the clean test.

## Verdicts
- **C0** (base reproduces paper M2 within 0.01): fails in the favourable direction (A 3-month .239-.245 vs .226),
  attributed to standardised traits. Deviation, reported.
- **T1 topography helps**: passes. 2 seeds: +0.013 A, +0.020 B all. 5 seeds: +0.013 A, +0.033 B all
  (+0.027 B 2019-25).
- **T2 temperature adds** (three designs, all fail; seed-0 3-month A / B all / B 2019-25):
  raw anomaly .006 / .159 / .057; year-detrended, network sensitivity .194 / .185 / .193;
  trait-set sensitivity .211 (2 seeds) / .161 / .152. With topography: raw -.061 / .169 / .010; trait-set
  .227 / .204 / .224. Diagnosis: raw T' tracks the 2019-25 warming (+0.59 C); detrended, the network
  over-learns a box-wide effect about twice its out-of-sample size.
- **T3 evaporative demand adds** (M4 - M3): 2 seeds fails on A (-0.002); **5 seeds passes**: +0.006 A
  (bar +0.005), +0.046 B 2019-25 (bar >= 0); ensembles +0.009 A, +0.050 B 2019-25. Physical check passes:
  k > 0 at every pixel, both boxes, all seeds (mean 1.3-2.6; seed-dependent size).
- Gate (Kratzert entity-aware) not tested: run stopped, one seed could not resolve it (addendum 3).

## Mechanism (M3, seed 0)
- Rain memory set mainly by elevation (lam coefficient 0.41; corr with elevation 0.89): half-life 8-17 days
  across pixels (base 9-13); higher ground holds rain longer.
- Rain Delta (2 steps, per 100 mm) vs elevation: corr -0.53 (base) -> -0.63 (M3); topographic position
  -0.09 -> -0.19 (valleys respond more).

## Attribution, Box A March-May 2022 (M4, mean of 5 fits; NDVI)
observed -0.168 = starting deficit -0.028 + rain -0.049 + evaporative demand -0.035 + not explained -0.055.
Shapley over the two forcings, residual held fixed. M2 for comparison: deficit 0.03, rain 0.07, residual 0.07.
Total weather attribution per seed: 0.090, 0.066, 0.104, 0.077, 0.084 (mean 0.084).
Per-pixel (2 km blocks) correlation with elevation: rain -0.53, demand -0.21, residual -0.24.
Figures: paper/figs/candidates/fig_2022_attribution_m4.png, fig_maps_split_2022_m4.png.

## Compute
M4: 1,002 parameters per fit. Full-box 2022 attribution (rain + demand shares) plus the rain Greek map on a
laptop CPU: 16 ms after compilation (0.2 s first call). One fit: ~80 min CPU, ~2.5 min H100.

## Not done / next
Superseded 2026-10-22 by the paper's test suite, revision/TESTS.md: four tests (skill, attribution, 2022 drought,
sensitivities) run on z-score, RESTREND, M1-M4, the M4 ensemble and Chronos-2 (revision/FM_HANDOFF.md).
Multi-step training: dropped for the workshop.
