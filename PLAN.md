# Plan: Laikipia savanna world model, three products, one transfer

Companion to `PROJECT.md`. This is the execution plan; `FINDINGS.md` holds the
pre-registered criteria and every number. Nothing below is a result.

## 0. Decisions fixed up front

| Decision | Choice | Reason |
|---|---|---|
| Model grid | 100 m cells, dekadal (36 steps/yr) | 1.3 km neighbourhood = 13×13 window (appendix 3); dekadal matches CHIRPS products and the weeks-scale grass response; sub-monthly is the untested cadence the appendix flags |
| Substrate | Sentinel-2 L2A 2017–2025 aggregated 20 m → 100 m; MODIS MOD13Q1 250 m 2000–2025 for the long record | S2 for the model; MODIS for RESTREND, VCI, z-score incumbents that need 20+ years |
| State vector | kNDVI anomaly (phase A); green / dry / woody fractions (phase B) | Phase A gets the pipeline and products working on one channel; phase B adds the per-fraction delta the carbon reading needs |
| Woody proxy | Dry-season minimum NDVI per year (persistent evergreen signal); grass = wet-season amplitude above it | Standard savanna trick; no field calibration needed to start; phase B replaces with 3-endmember unmixing if the audit says it separates |
| Driver | CHIRPS daily → dekadal totals; ERA5-Land T2m and 0–7 cm soil water dekadal | CHIRPS is the one the insurers and NDMA use; ERA5-Land soil water is the bucket state check |
| Framework | JAX | Per-pixel Jacobians of a rollout via `vmap(jacfwd)` are the sensitivity product; inversion is gradient descent on the driver path |
| Compute | Modal: CPU fan-out for ingestion; one A100 (or A10G) for training; Volume-backed zarr | Everything fits in a few GB; the cost is wall-clock on ingestion, not on training |

## 1. Area of interest

Two boxes in Laikipia and its northern edge, each about 45 × 45 km (≈2,000 km²,
≈200k cells at 100 m). Small enough that every run is minutes, large enough for
ranch-level products, several conservancy–communal boundaries, and two droughts.

- **Box A (development): Lewa – Borana – Il Ngwesi – Mukogodo.** Approx 0.05–0.45°N,
  37.15–37.55°E. Private conservancies, group ranches and communal grazing on shared
  soils and rain. Natural State's plots are expected here; confirm coordinates and
  re-centre the box on them.
- **Box B (held-out landscape): Mpala – Ol Jogi – Loisaba – Koija.** Approx
  0.25–0.65°N, 36.70–37.10°E. Contains the Kenya Long-term Exclosure Experiment (KLEE,
  Mpala, since 1995), which gives an independent exclosure utility test on a landscape
  the model never trained on.

If the partner plots turn out to sit in Box B, swap roles; the protocol is symmetric.
Box B is touched only in Week 4 scoring.

Rainfall regime in both: long rains MAM, short rains OND, weaker Jul–Aug continental
rains in the west. Two response experiments per year, 18 seasons in the S2 record.

**Transfer site (Week 5):** one 45 × 45 km box on the North Australian Tropical
Transect around an OzFlux savanna tower (Howard Springs, Dry River or Sturt Plains),
DEA fractional cover at 30 m as the substrate, SILO as the driver, NAFI as fire truth.

## 2. Data acquisition (Week 1, days 1–3)

All ingestion runs as Modal CPU functions fanned out per (box, month), writing dekadal
composites to a Modal Volume as zarr. One `ingest.py` per source; every source has a
`--check` mode that prints coverage and a thumbnail.

| Source | Access | Product built |
|---|---|---|
| Sentinel-2 L2A | Planetary Computer STAC (`sentinel-2-l2a`), free token | Dekadal max-NDVI composite of kNDVI, NDVI, NBR, bands B2–B12 at 20 m, SCL cloud mask; aggregate to 100 m mean and observed-fraction. **Subtract the 1000 DN offset for processing baseline ≥ 04.00 (scenes from 25 Jan 2022)** or 2022 onward is biased |
| MODIS MOD13Q1 v061 | Planetary Computer | 16-day NDVI 250 m, 2000–2025, for incumbents |
| MODIS MOD17A2H v061 | Planetary Computer | 8-day GPP 500 m, productivity incumbent |
| MODIS MCD64A1 v061 + FIRMS active fire CSV | Planetary Computer; FIRMS download | Dated burned pixels and fire points |
| CHIRPS v2 daily 0.05° | UCSB CHC HTTP | Dekadal totals, 1981–2025 |
| ERA5-Land hourly | CDS API (needs key; queue is slow, submit day 1) | Dekadal mean T2m, soil water layer 1, PET |
| ECMWF SEAS5 | CDS `seasonal-original-single-levels` | Daily precipitation, 51 members, forecasts issued 1 Feb and 1 Sep each year 2017–2025, leads 1–4 months |
| FAO Locust Hub | ArcGIS Hub CSV (CC-BY) | Dated swarm, band, hopper, adult points, 2019–2021 |
| Copernicus DEM 30 m; SoilGrids 250 m | Planetary Computer; ISRIC WCS | Static covariates |
| Conservancy / group-ranch boundaries | Laikipia Wildlife Forum, WDPA, OSM | Management covariate and contrast polygons |
| Natural State plots; KLEE plot data | Partner request (send day 1); KLEE via Mpala/published | Utility ground truth |

Dynamic World is Earth Engine only; skip unless GEE access is easy. Not required.

## 3. Substrate audit (Week 1, day 4, one day only)

Per pixel and pooled, on Box A, phase-A channel (kNDVI at 100 m dekadal):

1. Fraction of variance that is temporal; of that, the pixel's own seasonal cycle.
2. Anomaly autocorrelation at 1, 3, 6, 12 dekads; memory length in dekads.
3. Noise fraction: lag-1 anomaly-change structure and S2-vs-MODIS agreement at 250 m.
4. Linear driver-explained variance: distributed-lag regression of the anomaly on CHIRPS
   dekadal anomalies at lags 0–9, R² out of sample by year.
5. Gamma sniff test: binned response of seasonal greenness to seasonal rain total;
   is it concave?
6. Grass–woody separability: year-to-year stability of dry-season minimum NDVI, and
   its spatial agreement with a visual woody-cover check on a few tiles.
7. Cloud coverage: fraction of dekads observed per pixel, by season.

**Gate:** if item 4 gives R² below 0.10 at any lag out of sample, stop and reconsider
the driver or the cadence before building anything. Write the numbers into
`FINDINGS.md` §0.

## 4. Incumbents and baselines (Week 1, days 5–7)

Fitted on the same grid and splits as the model, scored the same way. These supply the
numbers that go into the kill criteria before any training.

| Name | Product it serves | Implementation |
|---|---|---|
| Pixel seasonal climatology + 2-parameter shrink filter | Null for all | From appendix 1–2 |
| VCI and NDVI z-score (MODIS, 2000–2025) | Anomaly | Operational definitions |
| RESTREND residuals | Anomaly, sensitivity | Annual NDVI vs annual rain regression per pixel, residual trend |
| Rain-use efficiency | Sensitivity (delta) | Seasonal ΔNDVI / seasonal rain per pixel |
| Distributed-lag regression | Sensitivity, impact | Linear anomaly on rain anomalies lags 0–9 + AR(1); this is also the model's initialisation prior |
| Soil-water bucket with saturating response | Impact, sensitivity | Minimal mechanistic forage stand-in (PHYGROW outputs are closed): 5 shared parameters + per-pixel scale, fitted by gradient descent in JAX |
| MOD17 GPP | Productivity | Reference only, compared at 500 m |
| Static historical baseline | Intervention effect | Pre-period mean per pixel, as the Verra methods use |

## 5. The model

**Bucket-residual mixture model.** Per 100 m cell, dekadal step.

- **Physics prior:** soil-water bucket `w_{t+1} = w_t + P_t − ET(w_t, T_t) − drain(w_t)`;
  greenness anomaly `a_{t+1} = ρ a_t + g(w_t)` with `g` saturating. Bucket parameters
  come from a shared MLP over a learned per-pixel static embedding plus DEM, soil and
  management covariates. Initialised to reproduce the fitted distributed-lag regression
  (appendix 8).
- **Learned residual:** GRU over a latent `h_t` with inputs `[s_t, d_t, w_t,
  conv(13×13 neighbourhood of s_t and a_t), static]`. Output head zero-initialised so
  the model starts exactly at the prior.
- **Output:** 3-component Gaussian mixture over next-step anomaly per channel
  (appendix 10). Phase B: one head per fraction.
- **Management:** conservancy / communal category enters as a static covariate in the
  main model and is withheld in the "intervention-null" variant used for product 3b.
- **Loss:** NLL at one step plus CRPS on sampled rollouts at horizons 1–9 dekads,
  observed dekads only (cloud mask in the loss, state carried through gaps).
- **Checkpoints:** early stopping on NLL and on error kept separately (appendix 9).

Size: 200k cells × 324 dekads × ~10 channels, float16, under 2 GB. A full training
run on one A100 is tens of minutes; leave-one-year-out CV is an afternoon.

Ablations, all pre-registered: prior only; prior + residual without neighbourhood;
without driver (history-only, the appendix-2 model); without management covariate;
single Gaussian instead of mixture.

## 6. Splits and protocol

- **Landscape:** Box B held out entirely until Week 4.
- **Years:** development is leave-one-year-out CV over 2017–2022 in Box A. Held-out
  seasons are all of 2023 and 2024 (four seasons: two long, two short; 2023 is
  post-drought recovery, the hard case). Nothing from 2023–2024 or Box B is looked at
  before Week 4.
- **Anomaly scoring uses the fold that excludes the event year**, so the 2020 locust
  and the 2021–22 drought are scored by models that never saw those years.
- **Calibration gate:** spread–skill correlation and PIT histogram on held-out years
  must be reported before any product; if the spread does not track error, the
  products are reported with that caveat in the title, not the footnote.
- Every number in `FINDINGS.md` with `runs/<date>_<slug>/`. Scratch outside git.
  Commit per result.

## 7. Products and pre-registered criteria

Incumbent numbers are filled in `FINDINGS.md` from §4 before training. The learned
model must clear the stated margin on the held-out years in Box A **and** on Box B.

### P1 Sensitivities

- **Computation:** `delta[pixel, season, lag, h] = ∂E[s_{t+h}] / ∂P_{t−lag}` via
  `jacfwd` of the mean rollout with respect to the driver path. Gamma: second
  derivative along the seasonal-total direction. Theta: decay rate fitted to the mean
  rollout from a unit anomaly under the climatological driver. Vega: derivative with
  respect to the scale of a fixed intraseasonal perturbation pattern.
- **Validation:** held-out response prediction: CRPS and MSE skill versus climatology
  of the model's response to the realised driver, against RUE and distributed-lag
  regression. Kill: learned skill not above distributed-lag by ≥ 0.03 on both held-out
  sets.
- **Utility:** AUC of the delta and theta maps for (a) conservancy versus adjoining
  communal pixels, (b) KLEE exclosure versus control plots in Box B (scored on the
  20 m substrate since plots are 200 m), (c) Natural State plot degradation classes.
  Kill: AUC not above RESTREND on (a) and (b).
- **Permanence map:** rank pixels by delta × theta exposure; check the ranking picks the
  pixels whose productivity collapsed in 2021–22, scored as precision at k versus VCI.

### P2 Impact

- **Computation:** SEAS5 members quantile-mapped to the CHIRPS climatology,
  disaggregated to dekads by climatological dekadal fractions, fed as the future driver
  from the issue date; output the ranch-level distribution of seasonal greenness and
  productivity proxy. (Both choices are crude and pre-registered as such.)
- **Validation:** hindcasts 2017–2024 issued 1 Feb and 1 Sep, CRPS at ranch level
  against (i) SEAS5 through distributed-lag and through the bucket model, (ii) the
  learned model with a climatological driver. Kill: no gain over both (i) and (ii).
  Report separately the value of the response (vs i) and of the forecast (vs ii).
- **Utility:** rank correlation of the ranch-level distribution with NDMA drought phase
  by ward and with plot productivity where available, against VCI.

### P3 Anomaly

- **Computation:** log-likelihood of each observed dekad under the predictive
  distribution conditioned on the realised driver; CUSUM of the log-likelihood ratio
  against the mixture's own null for detection.
- **3a Disturbance:** two curves fixed now. Detection lag at 5 % false-alarm rate on
  event-free pixels for FAO locust pixels (2020) and MCD64A1 burned pixels; false-alarm
  rate in the two highest-driver-anomaly seasons (OND 2021, MAM 2022). Pass only if
  both curves beat both the z-score and RESTREND.
- **3b Intervention effect:** cumulative driver-conditioned residual over each
  conservancy versus matched communal land under the same rain, from the
  intervention-null model variant. Pass if it explains more of the plot-measured
  difference than the static baseline on held-out years. If no plot data arrive, report
  the conservancy–communal residual as an unvalidated map and say so.

### Inversion (diagnostic)

Gradient descent on the dekadal driver path with a CHIRPS-climatology prior to
reproduce held-out observed greenness; report correlation of implied versus realised
rain out of sample and the spatial pattern of the residual against known CHIRPS
convective-rain error.

### Implied-variance hypothesis

Back out implied intraseasonal rain variance for fixed seasonal totals; compare with
dekadal CHIRPS variance out of sample. Report as failed if the audit's gamma is near
zero or the out-of-sample correlation is not above 0.3.

## 8. Repository and Modal layout

```
ProjectB/
  PROJECT.md  LESSONS.md  PLAN.md  FINDINGS.md
  aoi/            box_a.geojson  box_b.geojson  transect_au.geojson  ranches.geojson
  ingest/         s2.py  modis.py  chirps.py  era5.py  seas5.py  locust.py  fire.py  static.py
  audit/          substrate.py
  baselines/      climatology.py  vci_zscore.py  restrend.py  rue.py  dlag.py  bucket.py
  model/          prior.py  residual.py  mixture.py  train.py  rollout.py
  products/       sensitivities.py  impact.py  anomaly.py  inversion.py  implied_var.py
  eval/           splits.py  scores.py  utility.py
  runs/           (gitignored)  <date>_<slug>/config.json  metrics.json  figs/
  modal_app.py    volume "projectb-data"; CPU image for ingest; GPU image for train
```

Secrets on Modal: CDS API key, Planetary Computer token, Earthdata login (only if
FIRMS needs it).

## 9. Schedule

| Week | Deliverable | Gate |
|---|---|---|
| 1 | Box A zarr for all sources; audit numbers; incumbents scored; `FINDINGS.md` with criteria and incumbent numbers filled | Audit gate (§3); criteria committed before any `train.py` run |
| 2 | Prior-only and prior+residual models, LOYO CV in Box A; calibration diagnostics | Model beats distributed-lag on CV years; spread tracks error |
| 3 | Ablations; phase-B fractions if separability passed; inversion calibration check | Report both early-stopping variants |
| 4 | Box B and 2023–24 unsealed; P1, P2, P3a, P3b scored; implied-variance test | Pass/kill per product written the same day |
| 5 | Australian transect: DEA fractional cover + SILO + OzFlux; does productivity → carbon input hold, do sensitivity maps survive a different regime | Transfer scored against the same criteria |
| 6 | Four-page memo to one named user with numbers and non-claims | |

## 10. Risks and fallbacks

- **CDS queue latency** for ERA5-Land and SEAS5: submit all requests day 1; ERA5-Land
  is only a covariate, the model can run on CHIRPS alone until it arrives.
- **Partner plot data** may not arrive in time: P1 utility falls back to KLEE and
  conservancy contrasts; P3b is reported as unvalidated.
- **Cloud in MAM** in Laikipia: the audit's coverage number decides whether MAM
  products are reported at dekadal or monthly resolution.
- **S2 processing-baseline offset** (Jan 2022): handled in ingest; verify with a
  2021-vs-2022 dry-season histogram before anything else.
- **Woody proxy too weak** for phase B: report per-fraction delta as not identifiable
  and keep phase A's single channel.
- **Locust points are sparse in Box A**: if fewer than ~200 observed pixels, widen the
  locust evaluation to a third box in Samburu (FAO density is highest there) as a
  scoring-only region, never for training.
- **Driver R² below the audit gate**: the honest outcome is "the response is not
  linearly identifiable at this cadence"; test monthly before abandoning.

## 11. First actions

1. Email Natural State for plot coordinates and any biomass/carbon trajectories;
   request KLEE plot data or locate the published series.
2. Register CDS key, Planetary Computer token; submit ERA5-Land and SEAS5 requests.
3. `git init`, commit `PROJECT.md`, `LESSONS.md`, `PLAN.md`, empty `FINDINGS.md` with
   the §7 criteria table and blank incumbent cells.
4. Draw `box_a.geojson`, `box_b.geojson`; run `ingest/s2.py --check` on one month.
