# Result: fence-line / invasion test

**Verdict (pre-registered): FAIL.** In MAM 2017, the implied residual on Laikipia ranches invaded by herders' livestock was no lower than on matched community land. 2017 ranks 13th of 21 seasons (p = 0.62). Every secondary test agrees. By the pre-registered interpretation table, this is **weak evidence against** the residual detecting this management event. NDVI's insensitivity to grazing of dry grass, and grazing on the comparison land, cannot be excluded.

Run `runs/20261013_2150_fenceline` · frozen 2026-10-13T11:39:26Z (`FREEZE.json`) · outcomes computed after the freeze · model is the paper's M2 (`runs/20260914_1708_pipeline_h100_fairfight`), unchanged · data: Box B, held out.

## Design as executed

- **Treated (2017), per the exposure coding:**
  - heavy: Loisaba, Suyian, Sosian, Segera;
  - light: Mpala and Ol Jogi (one 6 March 2017 list; Ol Jogi contradicted by a second source), Ol Malo, Ngorare (presence inferred).
- **Pixels:** 8,386 interior pixels (at least 3 px, about 830 m, inside the boundary).
- **Controls:** 7,371 eligible pixels: community conservancies (Naibunga ×3, Narupa, Nanapa) plus unmapped land.
- **Matching:** at caliper 1.0, amplitude SMD was 0.104, so the pre-registered rule halved the caliper to 0.5. That left **2,697 treated pixels** with matches, and every |SMD| < 0.04.
  - Before matching, ranch interiors were far greener than community land (mean-NDVI SMD 1.37), so common support covers the less-green third of the ranches.
- **Outcome:** season-end attribution residual R = a(t₀+6) − F₆, rolled out from the 18 Feb 2017 state over 6 March to 25 May.
- **Placebo seasons:** 20 MAM seasons, 2003–2025, excluding 2017, 2015 (coded invasion) and 2023 (gap).

## Pre-registered results

| Test | 2017 statistic | Rank / p | Verdict |
|---|---|---|---|
| **P1** placebo-in-time | D = **+0.0097** NDVI (placebo median +0.0049, range −0.019 to +0.026) | 13 / 21, p = 0.62 | **fail** |
| **S1** rain-adjusted | MAM 2017 was the driest season on record (−109 mm); predicted D +0.006, t = 0.26 (df 18) | p = 0.60 | fail |
| **S2** per property | Suyian +0.059 (19/21); Ol Jogi +0.043 (17/21); Loisaba +0.009 (13/21); Mpala +0.008 (11/21); Sosian +0.004 (12/21, n = 34); Segera −0.023 (7/21); Ngorare −0.055 (2/21, n = 123); Ol Malo n = 2 | 1 of 8 properties ranks ≤ 2 | — |
| **S3** dose | heavy +0.012 (14/21, p = 0.67); light +0.008 (11/21, p = 0.52) | — | fail |
| **S4** placebo-in-space (Amendment 2) | fake median −0.016 | p = 0.995 | flawed: see D1 |
| **S5** Jan–Feb window | D = +0.0034 | 12 / 19, p = 0.63 | fail |
| **S6** community-polygon controls only (418 matched) | D = −0.0015 | 8 / 21, p = 0.38 | fail |
| **S7** sum of one-step residuals (method check) | D = +0.019 | 13 / 21 | same conclusion |
| **S8** 2015 (coded invasion) | D = +0.0065 | 13 / 21 | no signal |
| **S10** clean placebo years only (12) | D = +0.0097; rain-adjusted t = 0.38 | 8 / 13, p = 0.62 / 0.64 | fail |

**Detectable effect.** The placebo D has an s.d. of 0.014. For 2017 to rank first it would have needed a further −0.029 NDVI. So the design could detect a relative season-end effect of roughly **0.03 NDVI or more** on the matched pixels, and none appeared. For scale, the paper's 2022 box residual is 0.07.

## Post-hoc (declared in `DEVIATIONS.md` after the results; not confirmatory)

- **D1 / E1. S4 orientation flaw.** Angles drawn on [0, π) always made the northern side fake-treated. Corrected (θ on [0, 2π), seed 20172): fake median +0.002, 5–95% from −0.028 to +0.026, p = 0.60, consistent with P1. The registered p = 0.995 reflects a north–south gradient, not the hypothesis.
- **E2. Composite event study.** During the main invasion (29 January – 22 April 2017), the invaded-minus-control one-step residual stays inside the placebo-year 5–95% band, close to zero. There is no transient dip masked by recovery. The only excursions come after the invasion and point upwards: ranches greener in August and November 2017. One composite falls below the 5% band (19 December 2017), as expected by chance among 30.
- **E3. Hypothesis only: D shifts after 2013.** Mean D is −0.004 for 2003–12 and +0.011 for 2013–25 (Welch p = 0.006; slope +0.0013 per year, p = 0.001). That would fit community land degrading relative to ranches once outside herders occupied group ranches from 2013. But neither arm's trend is significant on its own (treated −0.0004 per year, p = 0.86; control −0.0017 per year, p = 0.48). The pattern was noticed after seeing the data, and regional drift between the two areas (controls lie to the north-east) is an alternative explanation. It needs its own pre-registered test before any claim.

Figures: `outputs/fig_D_by_year.png` (P1, S1) and `outputs/exploratory/fig_exploratory.png` (E2, E3). Design map: `design/design_map.png`.

## Reading

The residual did not register the best-documented management shock available in the study area. The pre-registered caveats cannot be separated here:

1. **Sensor and season.** The invasion peaked from February to mid-April, in the driest MAM on record. Cattle ate senescent grass, which NDVI barely registers, and max-value compositing keeps the greenest day in each 16-day window. Herders had largely left by late April, before most green-up.
2. **Both arms grazed.** Community group ranches (Koija, Morupusi) were themselves occupied from 2013 to 2017, and the unmapped land west of Mpala was a staging area. The contrast measures ranch against heavily used community land, not grazed against ungrazed.
3. **Exposure.** The two largest treated areas (Mpala, Ol Jogi) rest on one list. But the heavy-dose group fails too, and Suyian, the best-documented invasion, comes out *greener* than its controls.
4. **Common support.** Only the less-green third of ranch interiors could be matched.

**For the paper.** Do not claim external validation of the residual as a management detector. A defensible sentence would be:

> "A pre-registered test against the documented 2017 livestock invasions of Laikipia ranches found no signal (2017 ranked 13th of 21 seasons). NDVI's insensitivity to grazing of senescent grass and grazing on the comparison land are plausible reasons; the residual should be read as rain-conditioned, not as management."

This supports the C1 reframing ("rain-conditioned residual") and the caution on the fairness framing.

## Audit trail

- `PREREG.md`: drafted, then two amendments, all before the freeze. Each edit's hash is in `LOG.md`.
- `inputs/exposure/`:
  - `exposure_report_agent.md`, the agent's report verbatim;
  - 562 evidence files, hashed in `evidence_MANIFEST.sha256`;
  - `exposure.csv` and `years.csv`, coded before any outcome.
- `inputs/osm/`: Overpass failed on three servers (logged); polygons came from Nominatim (raw response saved).
- `inputs/covariates/`: WorldCover 2021 fractions and SoilGrids clay (raw GeoTIFF saved).
- `design/`, `FREEZE.json`, `outputs/` (written once each), `code/` (every script version that ran, named by hash).
- Three dry runs in scratch tested the code. The outcome code ran on **Box A** data only, so Box B stayed blind. Each dry run is logged; its outputs were discarded.
- Read-only inputs (model, data, `model.py`) were hashed at the start and verified unchanged at every post-freeze step.
- Repo: HEAD 56c7d66. No tracked file was modified. New files are only under `revision/fenceline/` and this run directory.
