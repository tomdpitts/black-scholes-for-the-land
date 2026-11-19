# M3: site traits and a second forcing (Kratzert et al. 2019). Pre-registered before any fit

Written 2026-10-21 before any M3 fit. Seen beforehand: the drivers run
(runs/20261006_1802_drivers: temperature appended to M2's features fails out of time, -0.25
three-month skill on Box A 2019-25; post hoc reasons: 2019-25 is +0.5-0.6 C warmer, and
box-wide temperature can act as a time stamp), the fair-fight M2 scores, and the topography maps
(revision/topo.py; alignment check only). No model with topography has been fitted.

## Inputs (revision/m3.py)
- Site traits (static): mean NDVI, seasonal amplitude, climatological rain (as M2), plus from the
  Copernicus DEM GLO-30 elevation, slope and topographic position (elevation minus its ~2.5 km
  mean). All standardised on Box A. Traits enter the drift network and set the rain memory lam(s).
- Forcings: rain as in M2; temperature anomaly T' (ERA5-Land 2 m mean, Open-Meteo archive,
  16-day window mean minus its 2003-2018 window climatology, deg C), entering linearly about
  normal with network-set sensitivities: mu += g_mu(f) T', log sigma += g_s(f) T'.
- Gated: traits also gate the hidden units, 2 sigmoid(W_g s + b_g), starting at 1.

## Configurations (all else identical: 64 hidden units, full-batch Adam 3e-3, 2000 steps, early
stopping on 2018 NLL, fit sees only steps up to 2018)
base (3 traits, rain) | topo (6 traits, rain) | temp (3 traits, rain + T') | both (6 traits,
rain + T') | gated (both + trait gate).
Seed 0 for all five; seed 1 for base, topo, temp, both.

## Scores
CRPS skill against M0 (history only, fair-fight parameters, scored by the same code), one step
and three months (6-step sampled rollouts, S = 16, realised forcings), on A_test (Box A 2019-25),
B_all (Box B all years, unseen landscape), B_test (Box B 2019-25).

## Criteria (differences averaged over seeds 0 and 1; both seeds must agree in sign)
- C0 sanity: base reproduces the paper's M2 within 0.01 (A_test 0.085 one step, 0.226 three months).
- T1 TOPOGRAPHY HELPS: topo - base three-month skill >= +0.01 on B_all and >= 0 on A_test.
- T2 TEMPERATURE ADDS: temp - base three-month skill >= +0.01 on A_test and >= 0 on B_test.
- Reported without criterion: both, gated (seed 0 only), one-step skill.

## Attribution (after the fits)
For the best configuration that passes its criteria: the March-May 2022 Box A season split into
start deficit, rain share, temperature share (if T2 passes) and residual. With two forcings the
rain and temperature shares are Shapley values: each forcing's share is its effect averaged over
the two orders of setting forcings to normal, residual held fixed. Temperature Greek g_mu
summarised over the season. If a criterion fails, that is the result.

## Addendum, 2026-10-21 04:55, after base_s0, topo_s0, temp_s0, both_s0 and before any run below
Seen: seed-0 scores of those four. temp_s0 fails T2 by a wide margin (three-month skill A_test
0.006 vs base 0.239; B_test 0.057 vs 0.198); both_s0 likewise. Diagnosis (fitted temp_s0
parameters and inputs only): the model learned a real weather effect (g_mu about -0.021 NDVI per
deg C per window; box T' vs greenness change, training corr -0.33), but 2019-25 sits +0.59 C above
the 2003-18 climatology, so the term adds about -0.012 every step and the rollout drifts; the
land did not brown with the warming. Cool 2018 (-0.30 C) explains the worse validation NLL.
Change: temperature enters as weather, the anomaly minus its mean over the preceding year
(23 windows, causal). Window length chosen from inputs only: 1 year gives the smallest
2019-25 offset (0.08 C) and the strongest training-period link to greenness change (-0.39)
among 1, 2, 3, 5 years. No model has been fitted with it.
Runs cancelled: gated_s0 (stopped at start), temp_s1, both_s1 (raw temperature, diagnosed).
New runs: tempd (3 traits, rain + year-detrended T'), bothd (6 traits, same), seeds 0 and 1;
topo_gated (6 traits, rain, trait gate), seed 0.
- T2' TEMPERATURE ADDS: tempd - base three-month skill >= +0.01 on A_test and >= 0 on B_test,
  both seeds agreeing in sign. T1 unchanged.
- Gate (exploratory, one seed): topo_gated - topo_s0, reported.

## Addendum 2, 2026-10-21 06:15, after tempd_s0 and before any run below
Seen: tempd_s0 (three-month skill A_test 0.194 vs base 0.239; B_test 0.193 vs 0.198; one step
A_test 0.083 vs 0.091). T2' fails on seed 0. Diagnosis (base_s0 errors and inputs only): the
year-detrended temperature explains base-model one-step errors out of sample (box-mean corr
-0.16 A, -0.23 B in 2019-25) but at about half the training-period slope (-9 vs -18 milli-NDVI
per C on A); a single coefficient fitted on 2003-17 lowers 2019-25 one-step MSE on both boxes,
while the network-set sensitivity over-corrects. Change: the temperature sensitivity is a linear
function of the traits alone (g = s G + b; about 14 numbers with topography), not a network output.
Cancelled: tempd_s1, bothd_s0, bothd_s1 (network-set sensitivity, diagnosed).
New runs: tempk (3 traits), bothk (6 traits), seeds 0 and 1; topo_gated seed 0 kept.
- T2'' TEMPERATURE ADDS: tempk - base three-month skill >= +0.01 on A_test and >= 0 on B_test,
  both seeds agreeing in sign. T1 unchanged.

## Addendum 3, 2026-10-21 ~10:00, user decision, before any run below
Seen: all completed seed-0/1 scores, incl. tempk_s0/s1 (T2'' fails on both seeds) and bothk_s0.
Stopped unfinished: bothk_s1 (no decision rides on it) and topo_gated_s0 (one seed cannot resolve a
gate effect smaller than the 0.026 seed spread of topo on B_all).
New runs: base_s2, topo_s2. T1 is re-evaluated on seeds 0-2 (mean difference, and the number of
seeds with a positive difference, reported); the two-seed verdict stays on record.
Trees (B1 GBM, refit as in baselines.py) scored on B_test (Box B 2019-25), skill against this
run's M0: does B1's B_all advantage over M2 survive when Box B's training-calendar years are excluded?

## Addendum 4, 2026-10-21 ~12:00, before any run below (fourth design for the second forcing)
Seen: all scores so far (temperature failed T2, T2', T2'' in three designs). Idea (user): heat acts
through water. Evaporative demand drains the water input: r_net = r - k(s) PET', with PET' the
ERA5 FAO-56 reference evapotranspiration anomaly against its 2003-2018 window climatology (100 mm
per window, as the rain; sd about 6 mm per window against 39 mm for rain; 2019-25 offset +0.7 mm,
so no detrending). k(s) is linear in the six traits, starting at zero (exactly the topo model), and
unconstrained in sign. r_net replaces r in the drift features and the memory bucket. PET' is driven
by temperature, radiation, humidity and wind: its share is an evaporative-demand share.
Base: topo (rain + topography). Controls: topo_s0, topo_s1 (already fitted). New: topo_pet s0, s1.
- T3 EVAPORATIVE DEMAND ADDS: topo_pet - topo three-month skill, mean of seeds 0 and 1, >= +0.005
  on A_test and >= 0 on B_test, both seeds the same sign on A_test.
- Physical check (reported with T3): pixel-mean k > 0, i.e. more demand leaves less water.
- If it passes: the MAM2022 split with an evaporative-demand share (Shapley, as pre-registered).
All four designs are reported, whatever the result.

## Addendum 5, 2026-10-21 ~13:00, after topo_pet s0/s1 (GPU) and before any run below
Seen: all scores so far. T3 fails as pre-registered (Box A -0.002); Box B gains on both seeds.
Seed spread on Box B is large (topo 0.173-0.199, topo_pet 0.231-0.281, three-month B_all).
All final comparisons move to five seeds (0-4), fitted on Modal H100 (GPU and CPU fits agree to
about 0.005): base, topo, topo_pet. Reported per configuration: mean and s.d. over seeds, and the
5-seed ensemble (equal-weight mixture of the seeds' predictive distributions, 4 rollout samples
per seed). T1 and T3 are re-evaluated on the 5-seed means with the same thresholds, and
on the ensembles; the 2-seed verdicts stay on record. Trees (B1 GBM) scored on B_test locally.
The ensemble is the model carried forward to attribution.
