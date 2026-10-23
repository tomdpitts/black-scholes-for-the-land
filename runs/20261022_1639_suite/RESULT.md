# AgriAI test suite: results (local run + Chronos-2 GPU cells 1, 3, 4)

Tables generated from the saved metrics by `revision/suite.py summary` (summary.md). M4 is reported as one row, the
5-seed ensemble. Blank cells were not run: Chronos-2 in 2a and in the 2c simulated world.

### Overview: all completed tests (blank = not yet run)

| Method | 1 Skill: 3-month, Box A / B | 2a Weather lean (per 100 mm) | 2c Error finding hidden damage, drought / other | 3 2022: NDVI lost to weather (of −0.168) | 4 Sensitivity, Box A / B | Compute: 2022 split; parameters |
|---|---|---|---|---|---|---|
| z-score (current practice) | 0 (the reference) | +0.043 | 297% / 295% | 0 | – (no rain response) | 0.1 ms; 24 |
| RESTREND (standard research method) | 0.170 / 0.190 | – | 92% / 131% | -0.075 | n/a: one fixed slope per pixel | <0.1 ms; 76,800 |
| M1 | 0.107 / 0.099 | – | 262% / 181% | -0.003 | n/a: same value everywhere | 4 ms; 24 |
| M2 | 0.241 / 0.177 | – | 118% / 54% | -0.069 | pass (5/5 fits) / mixed (3/5 fits) | 11 ms; 794 |
| M3 | 0.254 / 0.204 | – | 126% / 48% | -0.066 | pass (5/5 fits) / mixed (1/5 fits) | 11 ms; 995 |
| **M4 ensemble** | **0.273 / 0.274** | **+0.004** | **83% / 28%** | **-0.084** | **pass (+0.060) / fail (+0.018)** | **~71 ms; 5,010** |
| Trees | 0.222 / 0.197 | – | – | – | – | – |
| Chronos-2, with weather | 0.271 / 0.284 (4,000 px) | – |  | -0.101 | fail (-0.047) / fail (-0.015) | 645 s H100; 119.5M |
| Chronos-2, history only | 0.052 / 0.082 (4,000 px) | – | – | – | – |  |

Skill: how much better than history alone (0 = no better). Weather lean: how much the unexplained part tracks the season's rain (0 = none; run for the M4 ensemble with the z-score as yardstick). Error: % of the true hidden damage, 0% = perfect, averaged over five simulated worlds. Sensitivity: do pixels the model calls more rain-sensitive really respond more (Box A pre-registered). M2, M3: mean of 5 fits. Chronos-2 skill and sensitivity are on 4,000-pixel samples (M4 ensemble on the same pixels: skill 0.268 / 0.274).

### Claims (pre-registered)

| Test | Claim | Result | Verdict |
|---|---|---|---|
| 1 Skill | M4 ensemble beats Chronos-2 at 3 months (same 4,000 px) | M4 0.268 / 0.274; Chronos-2 with weather 0.271 / 0.284; history only 0.052 / 0.082 | **Chronos-2 better on both boxes** |
| 2a Weather lean | M4 ensemble's leftover does not lean with rain | +0.004 per 100 mm [-0.006, +0.025]; z-score +0.043 | **pass** |
| 2c Hidden damage | M4 ensemble's drought error below the z-score's and RESTREND's | 83% vs 297% and 92% | **pass** |
| 3 2022 drought | Chronos-2's weather share within 0.066–0.104 NDVI (M4 0.084) | Chronos-2 0.101 (120% of M4's) | **agree** |
| 4 Sensitivity | M4 ensemble passes on Box A | +0.060 [+0.027, +0.085] | **pass** |

### Test 1. Forecast skill on unseen years (2019–25), better than history alone

| Method | Box A: 16 days | Box A: 3 months | Box B (unseen landscape): 16 days | Box B: 3 months | Drought starts, 3 months, A / B |
|---|---|---|---|---|---|
| RESTREND (standard research method) | – | 0.170 | – | 0.190 | 0.167 / 0.268 |
| M1 | 0.013 | 0.107 | 0.040 | 0.099 | 0.025 / 0.174 |
| M2 | 0.092 | 0.241 | 0.075 | 0.177 | 0.243 / 0.317 |
| M3 | 0.099 | 0.254 | 0.083 | 0.204 | 0.250 / 0.307 |
| **M4 ensemble** | **0.111** | **0.273** | **0.131** | **0.274** | **0.273 / 0.359** |
| Trees | 0.046 | 0.222 | 0.003 | 0.197 | – |
| Chronos-2, with weather (4,000 px) | 0.058 | 0.271 | 0.071 | 0.284 | 0.260 / 0.340 |
| Chronos-2, history only (4,000 px) | -0.034 | 0.052 | -0.024 | 0.082 | -0.024 / -0.002 |
| M4 ensemble (same 4,000 px) | 0.109 | 0.268 | 0.130 | 0.274 | 0.269 / 0.359 |

Drought starts: forecast windows with box rain more than 60 mm below normal (33 on A, 20 on B, of 141). Chronos-2 and the M4 ensemble are compared on the same 4,000-pixel samples (M4 ensemble there: 0.268 / 0.274).

### Test 2a. Does what the model leaves unexplained lean with the weather? (13 seasons, 2019–25, 2 km blocks)

| Method | Lean with rain, NDVI per 100 mm [95% interval] | Lean with evaporative demand, per 100 mm |
|---|---|---|
| z-score (current practice) | +0.043 [+0.030, +0.068] | -0.287 [-0.384, -0.226] |
| **M4 ensemble** | +0.004 [-0.006, +0.025] | -0.050 [-0.156, +0.017] |
| Chronos-2 | | |

M4's lean is 8% of current practice's. Assumes real management does not follow the weather.

### Test 2c. Finding hidden land damage in simulated worlds (error as % of the true damage; 0% = perfect)

| Method | Drought seasons (4) | Other seasons (9) | All 13 | Drought, when damage worsens in droughts |
|---|---|---|---|---|
| z-score (current practice) | 297% | 295% | 296% | 321% |
| RESTREND (standard research method) | 92% | 131% | 119% | 105% |
| M1 | 262% | 181% | 206% | 287% |
| M2 | 118% | 54% | 74% | 138% |
| M3 | 126% | 48% | 72% | 148% |
| **M4 ensemble** | **83%** | **28%** | **45%** | **97%** |
| Chronos-2 | | | | |

Five worlds, each built from one M4 fit; a known damage of −0.01 NDVI per 16 days is added to a random half of 2 km blocks. Chronos-2 is scored in a sixth world it builds itself (GPU).

### Test 3. The March–May 2022 drought, Box A: splitting the observed -0.168 NDVI

| Method | Starting deficit | Rain | Evaporative demand | **Weather total** | Not explained (the land) |
|---|---|---|---|---|---|
| z-score (current practice) | – | +0.000 | – | +0.000 | -0.168 |
| RESTREND (standard research method) | – | -0.075 | – | -0.075 | -0.092 |
| M1 | -0.022 | -0.003 | – | -0.003 | -0.143 |
| M2 | -0.032 | -0.069 | – | -0.069 | -0.067 |
| M3 | -0.028 | -0.066 | – | -0.066 | -0.074 |
| **M4 ensemble** | **-0.028** | **-0.049** | **-0.035** | **-0.084** | **-0.055** |
| Chronos-2 | +0.002 | -0.016 | -0.085 | -0.101 | -0.069 |

M4 ensemble weather total across its 5 fits: -0.104 to -0.067.

### Test 4. Do pixels the model calls more rain-sensitive really respond more? (unseen years, gap-free starts)

| Method | Box A (pre-registered) | Box B (unseen landscape) |
|---|---|---|
| RESTREND (standard research method) | n/a: one fixed slope per pixel | |
| M1 | n/a: same value everywhere | |
| M2 | pass (5/5 fits) | mixed (3/5 fits) |
| M3 | pass (5/5 fits) | mixed (1/5 fits) |
| **M4 ensemble** | **pass: +0.060 [+0.027, +0.085]** | **fail: +0.018 [-0.046, +0.083]** |
| Chronos-2 (4,000 px) | fail: -0.047 [-0.074, -0.001] | fail: -0.015 [-0.054, +0.035] |
| M4 ensemble (same 4,000 px) | pass: +0.061 [+0.029, +0.085] | fail: +0.018 [-0.046, +0.082] |

Mean sensitivity (NDVI per 100 mm, Box A, same pixels): Chronos-2 0.014, M4 ensemble 0.071.

Numbers: extra greenness per 100 mm of rain in the most- vs least-sensitive third, with 95% interval over years; pass = interval above zero.

### Compute (laptop CPU, Apple M4 Max)

| Method | Parameters | 2022 split, full box | Sensitivity map, full box | Training |
|---|---|---|---|---|
| M1 | 24 | 4 ms | 77 ms | – |
| **M4 ensemble** | 5,010 | ~71 ms | ~28 ms | 5 × ~2.5 min H100 (~80 min CPU each) |
| Chronos-2 | 119,477,664 | 11 min H100 (CPU est. ~2.4 h) | – | none (pretrained) |

### Chronos-2 GPU run, estimated cost (Modal H100)

| Test | Central | Pessimistic |
|---|---|---|
| 1 Skill | $0.31 | $1.12 |
| 3 2022 drought | $0.19 | $0.69 |
| 4 Sensitivity | $0.33 | $1.20 |
| 2c Chronos-2 world (128×128 pixels) | $4.23 | $15.36 |
| **Total** | **$5.05** | **$18.38** |

**Actual H100 time** (inside the container; Modal also bills start-up): skill 19.5 min ($1.28), drought 10.8 min ($0.71), sens 11.8 min ($0.78); total $2.77.
Measured H100 time was 3.3x the central estimate. At that rate the 2c Chronos-2 world would cost about $14.

Estimates, not measurements: laptop CPU time measured, H100 throughput assumed (120 / 33 TFLOP/s), $3.95 per H100-hour. 2a for Chronos-2 would add about $0.4 / $1.5.

## Outcome (FM_HANDOFF.md, pre-registered): **C**
"Chronos-2 better skill, agree: method validated; M4 gives the same answer at a tiny fraction of the compute, on a
laptop, with terrain." Chronos-2's 2022 weather share (0.101) lies inside M4's range (0.066–0.104), and it edges M4 on
3-month skill on both boxes. Not outcome E (Chronos-2 does use the weather: 0.27 skill with it, 0.05 without) and not F.

## Caveats
- **Test 1 margin is tiny**: Chronos-2 leads by 0.003 (A) and 0.010 (B) in 3-month skill, with no interval. M4 leads
  at 16 days (0.109 vs 0.058, 0.130 vs 0.071) and on drought starts (0.269 vs 0.260, 0.359 vs 0.340). The
  pre-registered referee is 3-month skill on all starts, so the verdict stands, but "about equal" is the fair reading.
- **Test 3**: agreement is at the top of M4's range. How Chronos-2 divides weather (mostly evaporative demand, −0.085;
  rain −0.016) is descriptive only: rain and demand move together.
- **Test 4**: Chronos-2's sensitivity ranking fails on both boxes (negative on Box A); its mean rain sensitivity is a
  fifth of M4's (0.014 vs 0.071 per 100 mm). Reported, not deciding.
- **Compute**: the 2022 split took 11 min on an H100 for Chronos-2, ~0.07 s on a laptop CPU for the M4 ensemble.
  Measured H100 time was 3.3x the central estimate; the GPU run cost $2.77 (plus start-up). An earlier attempt failed
  in seconds (out of GPU memory at a 2048-task batch; rerun at 256, the CPU-tested size; outputs do not depend on it).
- **2a** rests on 13 seasons and assumes real management does not follow the weather; **2c** uses worlds built from
  M4's own fits and its drought win over RESTREND is narrow (83% vs 92%).
- **4** passes on Box A (pre-registered) and does not replicate on Box B.

## Pre-registration record
PREREG.md with three dated addenda, each hashed before the runs it governs (PREREG.sha256):
1. after the cost test: Chronos-2 world size (128 x 128, not run); M1's sensitivity is one value everywhere (test 4
   n/a); compiled timing for the 2022 split.
2. after the first test 2 results (user decision): five M4 worlds with the 5-seed ensemble as the estimator, replacing
   the single seed-0 world. The seed-0 result stays on record in tables.md ("seed 0 passes, seeds 1-4 do not").
3. (user decision) test 2a added, written before it ran.
Chronos-2 GPU cells run after the user's Go: 1, 3, 4. Not run: the 2c Chronos-2 world (~$14 at measured speed) and 2a.

## Files
summary.md (these tables), tables.md (full detail), skill.json, attrib_m4x5.json, attrib_m4.json, lean.json,
drought.json, sens.json, timing.json, c2_cost.json, c2_skill.npz, c2_drought.npz, c2_sens.npz, c2_check.json,
gpu_seconds_*.txt; figures fig_suite_skill, _lean, _attribution, _2022, _sens (.png/.pdf, also in paper/figs/candidates/).
