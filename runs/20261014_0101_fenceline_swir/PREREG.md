# Pre-registration: can a dry-vegetation signal see the 2017 invasions?

Run: `runs/20261014_0101_fenceline_swir` · written 2026-10-14, **before any reflectance data is downloaded** · its sha256 is logged before download (`LOG.md`, step `prereg`) and frozen again with the code in `FREEZE.json` before any outcome is computed.

## Why

The first test (`runs/20261013_2150_fenceline`) found no NDVI-residual signal of the 2017 invasions: rank 13 of 21, a pre-registered fail. Its leading explanation was that NDVI is blind to grazing of **senescent** grass. Shortwave-infrared reflectance does register dry plant material: cellulose and lignin absorb near 2.1 µm. This test asks whether a dry-vegetation index on the same pixels, same matched pairs and same placebo years shows the invasion that NDVI did not.

- **If it does,** sensor blindness is supported, and adding a dry-vegetation channel to the model is the fix.
- **If it does not,** sensor blindness is not the main explanation for the null. Exposure error, or grazing on both arms, becomes more likely.

## Design (reused, read-only)

From `runs/20261013_2150_fenceline`, verified against its `FREEZE.json` at every step:
- the matched design `design/design.npz`: 2,697 treated interior pixels, 5 matched community or unmapped controls each, caliper 0.5;
- the treated-property ids and dose coding (`design_meta.json`, `exposure.csv`);
- the primary placebo years: the 20 MAM years 2003–2025, excluding 2017, 2015 (coded invasion) and 2023 (MODIS gap);
- the clean placebo years: the 12 coded 0 in `years.csv`.

Nothing in the design is re-specified.

## Data

- **Product:** MODIS MOD09A1 v061 (Terra), 8-day 500 m surface reflectance, from Microsoft Planetary Computer.
- **Bands:** b01 (red), b02 (NIR), b06 (1.64 µm), b07 (2.13 µm) and `sur_refl_state_500m`.
- **Period:** 2003–2025.
- **Regridding:** warped onto the Box B 0.0025° grid by nearest neighbour, the same route as `ingest.py` uses for MOD13Q1. Each 500 m pixel feeds about 2×2 grid pixels; the treated interior lies at least 3 grid pixels (about 830 m) inside its boundary.
- **Replicate:** MYD09A1 (Aqua), processed identically.

**Valid observation:** state bits 0–1 = 00 (clear), bit 2 = 0 (no cloud shadow) and bit 10 = 0 (internal cloud flag clear), with b06 and b07 in (0, 1] after scaling by 1e-4.

## Outcome

- **NDTI** = (b06 − b07) / (b06 + b07). It is high for dry plant litter and low for bare soil; green vegetation also scores high, but little is green in a drought.
- **Climatology:** per pixel and per 8-day slot (composite start day), the mean of valid observations over 2003–2018 (the NDVI climatology years). A slot needs at least 5 valid years, otherwise it is NaN.
- **Anomaly:** NDTI minus the climatology.
- **Window statistic Y_y(i):** the mean anomaly over composites starting on day 57–105 (26 Feb – 22 Apr, the main invasion period: Suyian from about 29 Jan, Sosian from 17 Feb, Loisaba Feb–May; largely cleared by 22 Apr).
- **Valid pixel-years:** at least 3 valid composites in the window, and no MCD64 burn in 1 Jan – 30 Apr of year y (from `data/B/built.npz` `burn`).
- **Contrast:** D_y = mean over treated pixels i of [Y_y(i) − mean of i's valid matches Y_y(j)], exactly as in the first run.

## Tests

| ID | What | Prediction | Pass |
|---|---|---|---|
| **Q1 (primary)** | NDTI, Terra, D₂₀₁₇ ranked among the primary placebo years; p = (1 + #{D_y ≤ D₂₀₁₇}) / (1 + \|P\|) | litter removed from invaded ranches, so D₂₀₁₇ is negative and lowest | rank 1 = pass; rank 2 = supportive |
| Q2 | Q1 adjusted for rain: OLS of D_y on the Box B CHIRPS departure from 1 Oct of the previous year to 22 Apr (the rains that grew the litter, from `built.npz`), with 2017's studentised prediction error, one-sided | 2017 below its prediction | p < 0.05 |
| Q3 | b07 (SWIR2) reflectance anomaly, same construction | bare soil exposed, so D₂₀₁₇ is positive and highest | rank 1 from the top = pass |
| Q4 | NDVI from b01/b02, same construction | none (a cross-check of the NDVI null at 500 m) | descriptive only |
| Q5 | Q1 restricted to the heavy-dose properties (Loisaba, Suyian, Sosian, Segera) | as Q1 | rank 1 or 2 |
| Q6 | Q1 on Aqua (MYD09A1), the independent replicate | as Q1 | rank 1 or 2 |
| Q7 | Q1 per property: D₂₀₁₇ and rank | — | descriptive |
| Q8 | Q1 with the clean placebo years only (12) | as Q1 | rank 1 or 2 |
| Q9 | NDTI over the post-invasion window (day 113–177, 23 Apr – 30 Jun) | litter still depleted, so D₂₀₁₇ negative | descriptive |

Years where the box-wide median count of valid window composites is below 3 are excluded from the placebo set, and each exclusion is logged. If 2017 itself fails that rule, the test is void.

## Interpretation, fixed now

| Q1 | Q3 | Reading |
|---|---|---|
| pass or supportive | consistent (top 2) | Dry-vegetation loss on invaded ranches is visible in SWIR. NDVI blindness is the likely reason for the first test's null, so a dry-vegetation channel should go into the model. |
| pass or supportive | not consistent | Litter signal without bare-soil brightening: partial support. |
| fail | any | Sensor blindness is not supported as the main explanation. Exposure misclassification or grazing on both arms is more likely. |

## Procedure and audit

As in the first run:
- append-only `LOG.md` and `log.jsonl`;
- every script copied under `code/` by hash, and outputs written once;
- `FREEZE.json` (this file, the code and the hashes of the previous run's design) before any outcome;
- deviations in `DEVIATIONS.md`.

Raw reflectance is downloaded **before** the freeze but not inspected: the download step logs only counts and coverage. The code is dry-run on synthetic arrays, never on real Box B outcomes, before the freeze.
