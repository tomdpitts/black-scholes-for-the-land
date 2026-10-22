# Pre-registration: does the implied residual see real management? (fence-line / invasion test)

Run: `runs/20261013_2150_fenceline` · drafted 2026-10-13 · frozen by `FREEZE.json` (sha256 of this file) before any outcome is computed.
Model: fitted M2 from `runs/20260914_1708_pipeline_h100_fairfight/params.npy`, unchanged (the paper's model). No refitting.
Data: `data/B/built.npz` (Box B, never used for fitting; held out in every year). Read-only.

## Question

R1 asked for independent management data. Herders drove large numbers of cattle onto private ranches and conservancies in Laikipia during the 2016–17 drought. If the implied residual measures what the land did beyond the rain, it should fall on invaded properties, relative to comparable community land, in that season more than in any other.

**Scope.** This tests the residual under the rain that fell (the inversion). It does **not** test the normal-rain counterfactual: this management responds to rain, which is the case where the counterfactual is known to be biased (`runs/20261006_1753_identify`).

## Exposure (fixed before outcomes)

- `inputs/exposure/exposure.csv`: one row per property in or touching Box B, giving tenure, a 2017 invasion code, a dose code and sources. It is compiled from published reports by a researcher who has no access to any vegetation data. It is frozen, with its hash recorded in `FREEZE.json`, before step 05.
- **Treated (2017):** a private ranch or private conservancy that at least one named source reports invaded by herders' livestock at any time from 1 January to 31 May 2017.
- **Dose:** *heavy* if two or more independent sources report it, or herd sizes in the thousands, or a presence lasting more than a month. *Light* otherwise.
- **Control pool:** community land, meaning group ranches, community conservancies and land outside every private-property polygon.
- **Excluded from both groups:** private properties not coded as invaded, government land, and land of unknown tenure inside a polygon.
- **Placebo years:** Box B MAM seasons, 2003–2025, minus 2017, minus any year the exposure coding marks as an invasion of a treated property in January–May, minus seasons excluded for gaps (below).

## Pixels (static design, fixed before outcomes; step 03)

Grid: the Box B 160×160 grid at 0.0025°.
- **Polygons:** OpenStreetMap. The Overpass API was unavailable, so polygons were fetched by name through Nominatim (step 01b, logged). They are rasterised by pixel centre.
- **Interior rule:** a pixel enters either group only if its centre lies at least 3 pixels (about 830 m) from any private-property boundary, measured by Euclidean distance transform. This keeps M2's 5×5 neighbourhood input, which reaches ±2 pixels, entirely on one side of a fence.
- **Static masks** (a pixel failing any one is excluded):
  - WorldCover 2021 cropland fraction ≥ 0.2;
  - built-up fraction ≥ 0.1;
  - water plus wetland ≥ 0.1;
  - pixel mean NDVI (x₀) above the Box B 95th percentile, which catches riparian forest and irrigation;
  - within 2 pixels of the box edge.
- **Matching:**
  - Each treated pixel gets k = 5 nearest control-pool pixels, with replacement.
  - Distance is Euclidean on standardised covariates: mean NDVI, seasonal amplitude, climatological annual rain (the model's x), elevation, slope, and SoilGrids clay at 5–15 cm.
  - Covariates are standardised over all eligible pixels.
  - Caliper: a treated pixel is dropped if its nearest match lies more than 1.0 away.
- **Balance:** report the standardised mean difference of each covariate before and after matching. If any |SMD| exceeds 0.1, halve the caliper once (to 0.5) before the freeze and keep that design whatever balance it reaches. Matching is never re-specified after outcomes are seen.
- **Per-year masks:**
  - a pixel is dropped for year y if MCD64 recorded a burn from 6 composites before t₀ to t₀+6;
  - it is also dropped if the target composite a(t₀+6) is unobserved.
- Where masks remove some of a pixel's matches, the remaining ones are used. A treated pixel with no remaining match is dropped for that year.

## Outcome

For each MAM season y, t₀ is the composite immediately before the first MAM window, as in `residual.py`'s `season_starts`, and H = 6 (about 3 months).

- **R_y(i) = a_{t₀+6}(i) − F₆(i)**, where F₆ is M2's mean rollout from the observed state (a(t₀), w(t₀)) along the observed rain with zero innovation. Units are NDVI anomaly.
- This is the paper's season-level attribution residual. Using a rollout from the pre-season state stops M2's own state inputs (a, neighbourhood mean) from absorbing a sustained decline, which a sum of one-step residuals would partly do.

## Statistic and primary test (P1)

D_y = mean over treated pixels i of [ R_y(i) − mean over i's matches j of R_y(j) ]

- **P1, placebo-in-time:** p = (1 + #{y ∈ P : D_y ≤ D₂₀₁₇}) / (1 + |P|), one-sided, predicting that D₂₀₁₇ is unusually negative.
- **Pass:** D₂₀₁₇ is the lowest of all seasons (p = 1/(|P|+1)).
- **Supportive:** D₂₀₁₇ ranks second.
- **Fail:** any other rank.

## Secondary tests (pre-specified)

- **S1, the drought confound (key secondary):**
  - Fit an OLS of D_y on the Box B mean MAM rain departure (mm, the six windows) across P.
  - Score 2017 by its studentised prediction error, t = (D₂₀₁₇ − ŷ) / (s·√(1 + 1/n + (x−x̄)²/Sxx)), against t with n−2 degrees of freedom, one-sided.
  - Pass if p < 0.05.
  - Also report D₂₀₁₇'s rank among the five driest placebo seasons.
- **S2, per property:** D₂₀₁₇ for each treated property, and its rank among that property's own placebo years.
- **S3, dose:** heavy versus light properties, D₂₀₁₇ for each group.
- **S4, placebo-in-space:**
  - Take 200 random integer translations of the treated pixel mask (seed 20171).
  - Keep a translation only if at least 80% of its shifted pixels fall in the eligible control pool.
  - Re-match each one against the rest of the pool with the same rule.
  - p = (1 + #{D_fake ≤ D_true}) / (1 + number kept).
- **S5, dry-season window (exploratory):**
  - t₀ is the last composite of the preceding year, and H = 4 (January–February).
  - Same statistic and placebo rank.
  - Expected weak, because NDVI barely registers grazing of senescent grass.
- **S6, clean control:** restrict controls to pixels inside mapped community conservancy or group-ranch polygons, if at least 300 eligible pixels remain.
- **S7, method check:** the same contrast on the sum of one-step residuals in NDVI units, Σₜ σₜ·ε̂ₜ over the same six windows. Report beside P1, with no pass or fail.
- **S8, replication (exploratory):** any other Box B season the exposure coding marks as invaded (for example 2021–22), scored against the same placebo set.

## Seasons excluded

- 2002: rain-memory spin-up.
- Any season whose composites t₀ … t₀+6 include a spacing over 16 days (MODIS gaps).

## Interpretation, fixed now

| P1 | S1 | Reading |
|---|---|---|
| pass or supportive | pass | The residual detects management (invasion grazing) beyond drought-related model error. |
| pass or supportive | fail | Inconclusive: drought × land-type model error could produce the same contrast. |
| fail | any | Weak evidence against. NDVI's insensitivity to grazing of dry grass and errors in the exposure coding cannot be excluded. Reported in the paper either way. |

## Audit

- Every step appends to `LOG.md` and `log.jsonl` with UTC timestamps, input and output sha256, and a copy of the exact script under `code/`.
- Outputs are written once; scripts refuse to overwrite.
- Any change after `FREEZE.json` goes in `DEVIATIONS.md` with its reason, and affected results are reported under both versions.

## Amendment 1: before the freeze, after exposure coding, before any outcome (2026-10-13)

The exposure research (`inputs/exposure/exposure_report_agent.md`) found evidence of mixed quality. These rules make the coding operational. None depends on outcomes, and none had been computed.

1. **Tenure.** Land held privately by a trust, foundation or company counts as private: Mpala, Loisaba, Suyian, Ol Jogi, Sosian. A private–community partnership on group-ranch land (Ol Lentille), a settlement polygon, government land and unknown tenure all count as "other", excluded from both arms.
2. **Ambiguous years.** `years.csv` codes each year 1 (documented Jan–May invasion of a treated property), A (ambiguous: unnamed ranches, year-only reports, or uncertain identity) or 0 (no report found).
   - The primary placebo set (P1, S1, S2, S4 reference) excludes only years coded 1. That keeps A years, which is conservative: an invasion in a placebo year pushes that year's D down and makes 2017 harder to rank first.
   - **S10 (new):** P1 and S1 recomputed with placebo years restricted to those coded 0.
3. **S3 (extended).** For each dose group, report D₂₀₁₇ and its placebo-in-time rank.
   - Heavy, by the PREREG rule: Loisaba, Suyian, Sosian, Segera.
   - Light: Mpala, Ol Jogi (conflicting sources), Ol Malo, Ngorare (presence inferred).
   - The heavy group holds the best-documented properties. The two largest treated areas, Mpala and Ol Jogi, rest on one list from 6 March 2017.
4. **Control pool unchanged from registration**, including unmapped land. Two caveats are recorded:
   - The unmapped gap west of Mpala is probably land-buying-company plots used as staging areas in 2017.
   - Community group ranches (Koija, Morupusi) were themselves invaded in 2013–17.

   Both contaminate the controls in the direction of a null. S6 (community polygons only) is unchanged.

## Amendment 2: before the freeze, after the static design, before any outcome (2026-10-13)

- **S4 replaced.** The registered placebo-in-space (integer translations of the treated mask with ≥80% inside the control pool) is infeasible. The treated interior covers 8,386 pixels against a 7,371-pixel control pool, and 0 of 20,000 draws qualified (`LOG.md`, 03_design).
  - Replacement: 200 random straight-line splits of the control pool (seed 20171).
  - Angle θ is uniform on [0, π). The offset is uniform between the 30th and 70th percentiles of the pool's projection.
  - Pixels at least 3 px from the line on one side are fake-treated; those at least 3 px from it on the other side are the fake control pool.
  - Matching uses the final design's rule (k = 5, caliper 0.5, same standardisation). A split is kept if at least 100 fake-treated pixels match.
  - p_space = (1 + #{D_fake ≤ D_true}) / (1 + 200), for MAM 2017. Design file: `design/s4_line_splits.npz`.
- **Balance rule outcome (no change, for the record).** At caliper 1.0, amplitude SMD was 0.104, so the rule halved the caliper to 0.5. Treated pixels with a match fell from 7,301 to 2,697, and every |SMD| is now below 0.04. Ranch interiors are greener than almost all community land, so common support is thin. Per-property matched counts are reported under S2.
