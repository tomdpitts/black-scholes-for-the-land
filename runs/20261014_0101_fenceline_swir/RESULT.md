# Result: can a dry-vegetation (SWIR) signal see the 2017 invasions?

**Verdict (pre-registered): FAIL, in the opposite direction.** From 26 February to 22 April 2017, invaded ranches kept relatively *more* dry vegetation than matched community land, against each pixel's own normal. 2017 ranks **20th of 21** for the predicted drop (D = +0.013 NDTI, p = 0.95). No bare-soil signal appears. The pre-registered reading: **sensor blindness is not supported as the main explanation** for the first test's NDVI null. Exposure error, or grazing on both arms (here, community land more depleted than the ranches), is more likely.

Run `runs/20261014_0101_fenceline_swir` · PREREG hashed before download · frozen 2026-10-13T14:32Z (`FREEZE.json`) · design reused read-only from `runs/20261013_2150_fenceline` and hash-verified at every step · data: MODIS MOD09A1 v061 (Terra), 8-day 500 m, 1,034 composites, 2003–2025 · 2,697 treated pixels, 5 matches each · 20 placebo years (none excluded for data).

## Pre-registered results

| Test | 2017 | Rank / p | Verdict |
|---|---|---|---|
| **Q1** NDTI (dry litter), primary | D = **+0.0126** (placebo median −0.0014, range −0.010 to +0.016) | **20 / 21** from the bottom, p = 0.95 | **fail** (opposite sign) |
| Q2 rain-adjusted (Oct–Apr departure: −175 mm in 2017) | t = +1.54, df 18 | p = 0.93 | fail |
| Q3 SWIR2 b07 (bare soil), predicted high | D = −0.0002 | 11 / 21 from the top | fail |
| Q4 NDVI at 500 m (descriptive) | D = +0.0035 | 14 / 21 | reproduces the first null |
| Q5 heavy dose (Loisaba, Suyian, Sosian, Segera) | D = +0.0066 | 20 / 21 | fail (opposite sign) |
| Q6 Aqua replicate | — | — | **void**: Aqua band 6 detector failure; no year passes the data rule (`DEVIATIONS.md` D1) |
| Q7 per property | Mpala +0.024 (19/21), Suyian +0.025 (19/21), Loisaba +0.007 (19/21), Sosian +0.020 (20/21, n = 34), Ol Jogi +0.012 (14/21), Segera −0.009 (9/21), Ngorare −0.027 (2/21, n = 123), Ol Malo (n = 2) | — | descriptive |
| Q8 clean placebo years (12) | D = +0.0126 | 13 / 13 | fail |
| Q9 post-invasion window (23 Apr – 30 Jun) | D = +0.0073 | 15 / 21 | descriptive |

Figure: `outputs/fig_swir_by_year.png`.

## Reading

- **Two signals now agree.** Neither the NDVI residual (first run) nor shortwave-infrared dry matter (this run) shows a ranch-specific loss on the matched pixels in 2017. Dry matter is the variable that should have registered grazing of senescent grass. It moved the other way: relative to normal, the community land was more depleted than the invaded ranches. That fits why herders invaded (the ranches still had grass) and the reports that community group ranches were themselves occupied and heavily grazed.
- **Spatial caveat.** Matching keeps only the less-green third of ranch interiors. Herds concentrate near water and bomas, which may lie outside the matched pixels. No boma locations are available to test this.
- **Ngorare** is the only property whose sign matches the prediction in both runs (rank 2 in each). It has only 123 pixels and its exposure is inferred, so read nothing into it alone.

## Post-hoc observation (not tested; same design and pixels, so not independent)

The NDTI contrast is mostly negative in 2003–2013 and mostly positive from 2014. That matches the shift in the NDVI-residual contrast found post hoc in the first run (E3), and the reported occupation of community group ranches by outside herders from 2013. Two spectral signals agreeing on the same pixels strengthens the hypothesis, *chronic* degradation of community land relative to ranches, but does not test it. A test would need pre-registration on held-out data, for example Box A's ranch–community boundaries (Lewa/Borana against Il Ngwesi and Mukogodo) with a change point fixed at 2013.

## Deviations

D1, post-freeze: void handling for Q6 in `s03_analysis.py`. Q1–Q5 had already been computed when the crash happened and are unchanged. See `DEVIATIONS.md`.

## Audit

- `LOG.md` and `log.jsonl` (append-only, UTC); `PREREG.md` was hashed before any download.
- Raw reflectance: `inputs/raw/*.npz`, unmodified, hashed.
- The code was dry-run on synthetic reflectance only (logged) before the freeze.
- `FREEZE.json` covers PREREG, raw data, the previous run's design files, Box B data and every script.
- `code/` holds every script version that ran; outputs were written once.
- The previous run was only read, and its files verified against its own freeze, at start and at every step.
