# Brief: a savanna and rangeland world model for carbon, built like a pricing model

Cold start. This file is the whole inheritance. Do not read or import code, findings or
methods from any earlier project; the appendix carries the only facts worth keeping.

## Thesis

Weather is the underlying; the land is the option. (Rain is the race, pasture is the bet:
pasture depends on rain, rain does not depend on pasture. The weather forecast is the
bookmaker's odds on the race; this model turns them into odds on the bet.) A world model
of the savanna surface is useful in the way an option-pricing model is useful: not for
forecasting the underlying (weather models do that), but as a consistent, differentiable
map from the driver to the land's response, and as a null against which deviations mean
something. The response function is heterogeneous, stateful and nonlinear, and no
atmospheric model contains it.

The products are **sensitivities**, **impact** (a driver forecast composed with the land
response) and **anomaly detection**; inversion is a diagnostic. The emphasis is carbon:
rangeland carbon projects live or die on a counterfactual ("what would this land have
done without the management, given the rain it actually got") and on permanence ("how
exposed is the stored carbon to the next drought or fire"). Those are the anomaly and
sensitivity products under other names. Success is one landscape worked end to end
(Laikipia, Kenya), transfer shown on at least one other savanna, and one user who would
change a decision because of the output.

The label "world model" is earned only if the counterfactual uses are the products: a
sensitivity is a derivative of the rollout with respect to the input, an impact map is a
rollout under a driver scenario, an anomaly is the likelihood of an observation under the
rollout. If the deliverable degenerates into one-step forecasts scored on error, call it
a learned response model and move on.

## Why savanna and rangeland

- Water-limited: rainfall is the underlying in fact, not by assumption. Grass production
  tracks rain within weeks and saturates at high rainfall (the gamma that makes the
  distribution of rain matter, not only its total). In mesic savannas fire and herbivory
  take over from rain; the model should fail there, and that boundary is a result.
- Two components with different clocks: grass responds within a season, woody cover over
  years and to fire and browsing. Bush encroachment and bare-ground expansion, the two
  dominant degradation processes, are slow shifts in that balance and show up first as
  changes in sensitivity, not in mean state.
- The non-driver anomalies are the savanna's defining processes and are dated: fire
  (MODIS/VIIRS burned area), locusts (FAO point observations, dense in Kenya in 2020),
  grazing pressure (management records, conservancy boundaries), clearing.
- Rangelands are about half the world's land and most of Africa's; the incumbent
  methods (rain-use efficiency, RESTREND, static sensitivity and resilience maps) are
  linear, annual and unconditioned, and the carbon market built on top of them is under
  public challenge for exactly the counterfactual this model produces.

## The carbon question, stated honestly

Satellites do not see soil carbon, and rangeland carbon is mostly soil carbon. The model
never claims a stock. What it can claim, and what a carbon project needs from the
surface:

1. **Carbon inputs.** Grass and woody productivity (from fractional cover or class
   probabilities, calibrated against flux towers where they exist) is the input term of
   every soil-carbon model used for crediting (Century/DayCent, RothC, the Verra VM0032
   family). The world model supplies that term per pixel, with uncertainty, conditioned
   on the realised driver.
2. **The counterfactual baseline.** A project's additionality is observed productivity
   and cover minus what the land would have done under the same rain without the
   management. Current baselines are static histories or matched controls. The
   driver-conditioned null is a dynamic baseline at every pixel. This is product 3
   with the intervention as the anomaly.
3. **Permanence risk.** How much stored or accruing carbon is exposed to a failed season
   or a fire is a sensitivity question: delta and theta of productivity to rainfall,
   and the fire-fuel anomaly. Buffer pools are sized on this today by rule of thumb.
4. **Disturbance record.** Fire, clearing and grazing-pressure anomalies dated and
   located, as inputs to the accounting.

What it cannot do: measure soil carbon, resolve below-ground allocation, or replace
the soil model. It feeds the soil model and audits the baseline.

## Landscape choice

**Key example: Laikipia and the adjoining northern Kenyan counties.** A mosaic of
private conservancies, group ranches and communal grazing on shared soils and rain, so
management differences are a natural experiment; two droughts in the record (2016–17,
2021–22) and the 2020 locust upsurge as dated shocks; Natural State's research station,
carbon laboratory, long-term carbon and biodiversity plots, camera traps and acoustic
arrays as ground truth; the Leverhulme Centre for Nature Recovery partnership (Malhi;
Stevens on the ecological indicators) as the academic link; the region's grazing-based
soil-carbon project as the live dispute a dynamic baseline would inform. Sentinel-2
had a clear scene in 67–72 of 72 months at the sites checked; CHIRPS, ERA5-Land and the
FAO locust CSVs are open.

**Transfer sites, in order of value.** (i) Australian tropical and semi-arid savanna
along the North Australian Tropical Transect: OzFlux towers give carbon-flux ground
truth for the productivity-to-carbon-input step, DEA fractional cover gives green, dry
and bare fractions at 30 m since 1987, SILO gives 5 km daily climate since 1889, NAFI
gives fire. (ii) Natural State's Karoo site in Namibia (same partner, different
rainfall regime). (iii) US Great Plains rangeland, where the Rangeland Analysis
Platform's 30 m biomass and NPP product is the strongest public incumbent to beat and
the PRF insurance payouts are public.

## Incumbents (what already exists and must be beaten)

- Anomaly indices relative to the pixel's own history (VCI, NDVI z-scores) in drought
  bulletins and insurance triggers. No driver conditioning.
- Rain-use efficiency and RESTREND degradation maps: linear, annual, no uncertainty.
- Static sensitivity and resilience maps: Seddon et al. (2016) vegetation sensitivity
  index; autocorrelation-based resilience (Forzieri et al. 2022, Smith et al. 2022).
- Weather-driven forage models (PHYGROW/LEWS in East Africa, AussieGRASS in Australia):
  driver-conditioned, coarse, hand-parameterised, outputs largely closed.
- Productivity products: MODIS GPP/NPP (MOD17), RAP NPP (US only).
- Carbon-project baselines: static historical or matched-control baselines under the
  Verra rangeland methodologies, with process soil models calibrated at sites.

The kill criterion for each product is written against the strongest available
incumbent for that product, not against the linear baselines.

## The three products

Notation: s_t land-surface state at a pixel (fractional cover, class probabilities, or
an index), d_t driver; the model gives p(s_{t+1..t+h} | s_{≤t}, d_{≤t}, d_{t+1..t+h}).

### 1. Sensitivities (the Greeks); carbon reading: permanence and degradation

Concept: how much does this patch of land green up when it rains, and how fast does it
settle back down afterwards.

- **Delta** ∂E[s_{t+h}]/∂d_t: marginal rain-use efficiency per pixel, season and lag,
  separately for green, dry and woody fractions. Falling grass delta with rising woody
  delta is encroachment; falling delta overall is degradation.
- **Gamma** ∂²E[s]/∂d²: saturation or threshold of the response.
- **Theta**: decay of an anomaly with no new driver: recovery rate, the resilience
  indicator, here driver-conditioned.
- **Vega**: sensitivity to driver variability rather than level; exists where gamma is
  nonzero.

Validation: predict the response to the realised driver on held-out years and a
held-out landscape, against rain-use efficiency and lagged regression. Utility test:
do delta and theta maps agree with plot-based degradation and recovery indicators
(Natural State plots, exclosures, conservancy versus communal contrasts), and does a
permanence-risk map built from them rank the seasons and pixels where productivity
actually collapsed?

### 2. Impact (driver forecast composed with land response); carbon reading: expected inputs

Concept: if the weather people say the rains will be poor, what will each ranch actually
look like in three months.

Given s_{≤t}, d_{≤t} and an external forecast ensemble or scenario for d_{t+1..t+h},
the distribution of the land response at field scale: a rainfall outlook becomes a
forage and productivity distribution per ranch, months ahead, and hence an expected
carbon-input distribution for the season.

Validation: hindcasts with archived seasonal-forecast ensembles (ECMWF SEAS5, ICPAC)
scored against the realised surface, versus (i) the same forecast through the linear
response and through a forage model where obtainable, and (ii) the learned model with a
climatological driver. Utility test: does the ranch-level impact distribution track
drought phase, livestock condition or plot productivity better than the operational
index?

### 3. Anomaly detection (the null); carbon reading: the dynamic baseline

Concept: given the rain that actually fell, this land should look like X; it looks like Y;
the difference is something other than rain, such as fire, locusts, cattle, or the
project's management.

Anomaly = observation against the model's expectation conditioned on the realised
driver, scored as a likelihood. Two uses:

- **Disturbance**: fire, locusts, clearing, grazing-pressure events. Ground truth: FAO
  locust observations 2019–2021, MODIS/VIIRS burned area, management records. Metric:
  detection lag at fixed false-alarm rate on event-free pixels, and false-alarm rate in
  the highest-driver-anomaly seasons. Pass only if both beat RESTREND and the z-score.
- **Intervention effect**: the cumulative driver-conditioned residual over a managed
  area, against matched unmanaged land under the same rain. Ground truth: plot carbon
  and biomass trajectories, exclosure and conservancy contrasts. Pass if the dynamic
  baseline explains more of the plot-measured difference than the static baseline the
  project methodology uses, on held-out years.

### Inversion (diagnostic, not a product)

The driver path that makes the model reproduce the observed surface, with uncertainty.
Used for calibration (implied versus realised driver out of sample) and for diagnosing
driver-product error (CHIRPS on convective rain). Disagreement that is not driver error
is product 3.

## The implied-volatility analogue (a hypothesis)

If the response to rain is concave, then for a fixed seasonal total the outcome depends
on how the rain was distributed within the season, so the observed state is informative
about intraseasonal rainfall variability given its mean. Test: back out implied
intraseasonal variance from the surface, compare with dekadal gauge variance out of
sample; report failure if the response is near-linear. Secondary bookkeeping: the
model's predictive dispersion by horizon and season against realised dispersion.

## Model requirements (interface, not architecture)

- Conditioned on the driver, dekadal or monthly, per pixel, with the neighbourhood
  available; state that separates grass, dry matter and woody fractions where the
  substrate allows.
- Probabilistic, sampleable, scored with a proper scoring rule.
- Differentiable in the driver (sensitivities by autodiff).
- Accepts an external driver ensemble or scenario for the future window; invertible in
  practice for the calibration check.
- Calibrated: spread tracks error on held-out years and the held-out landscape before
  any product is reported.
- Management enters as a covariate or as the anomaly, never as a driver it learns to
  predict.

Architecture is open. Whatever is used is compared against the incumbents and the
linear versions of each product, fitted on the same data, and starts from them where it
can.

## What this does not provide

Soil carbon stocks; better rainfall forecasts; anything beyond a few months ahead for
grass and beyond the record for woody cover; answers where the optical record is
cloud-gapped; causes beyond "not the driver". Say so in the memo.

## Data

- Surface: Sentinel-2 at 10–20 m (indices or fractional cover unmixed locally), Dynamic
  World probabilities, DEA fractional cover for Australia, MODIS/eMODIS 250 m for the
  long record. One substrate for the model; the operational index kept as reference.
- Driver: CHIRPS daily rainfall; ERA5-Land soil moisture and temperature; SMAP;
  archived seasonal-forecast ensembles (ECMWF SEAS5 via CDS, ICPAC) for hindcasts.
- Carbon and biomass truth: Natural State plots and carbon laboratory; OzFlux savanna
  towers (Australia); published exclosure and grazing-trial sites; project monitoring
  reports where public.
- Disturbance truth: FAO Locust Watch CSVs (CC-BY); MODIS/VIIRS burned area; NAFI
  (Australia); conservancy boundaries and grazing plans.
- Region: contiguous landscapes (Laikipia and neighbours; a transect in Australia),
  not scattered sites, so neighbourhood structure and ranch-level products exist.

## Protocol

- Split by year and by landscape, never only by pixel. Hold out at least two full
  seasons and one landscape from all development.
- Pre-register pass/kill criteria per product in `FINDINGS.md` before training
  anything, with the incumbent's number filled in first.
- Audit the substrate for one day: linear driver-explained variance, memory length,
  noise fraction, grass versus woody separability. Then build.
- Every number in `FINDINGS.md` with its run directory. Faithful reporting. Scratch out
  of git. Commit per result.

## Milestones

1. Week 1: data for Laikipia; incumbents and baselines fitted and scored (VCI and
   z-score; rain-use efficiency, lagged regression, RESTREND; MOD17 productivity);
   kill criteria written against them.
2. Weeks 2–3: driver-conditioned probabilistic model; calibration on held-out seasons.
3. Week 4: the three products on held-out seasons; locust and fire anomaly curves;
   conservancy-versus-communal intervention residuals against plot data.
4. Week 5: transfer to one Australian savanna transect with OzFlux towers: does the
   productivity-to-carbon-input step hold, and do the sensitivity maps survive a
   different rainfall regime?
5. Week 6: a four-page memo addressed to one named user (a restoration or carbon
   project developer, a registry, or a credit-rating agency) stating what a dynamic
   baseline and a permanence-risk map change about a decision they make, with the
   numbers, and what they do not.

## Deliverable

One landscape worked end to end, one transfer, one user. Not a benchmark, not a model
family, not a survey.

## Appendix: facts carried over from two prior audits (measured, not advice)

From audits of learned land-surface dynamics on annual 64-d satellite embeddings
(2017–2025, 400 global sites) and monthly Dynamic World class probabilities (2016–2025,
same sites), history-only, no driver. Re-measure on the new domain before relying on any.

1. Annual cadence carried almost no learnable dynamics: 6.5 % of variance temporal, 83 %
   of year-to-year change noise-like; a two-parameter shrink-to-running-mean filter was
   never beaten on unseen sites. Dating and attribution on annual data failed against raw
   change magnitude.
2. Monthly cadence did: 29 % of variance temporal, 65 % of that the pixel's own seasonal
   cycle; anomaly autocorrelation 0.41 at one month, 0.03 at twelve; 70 % of monthly
   anomaly change noise-like. A learned model beat the seasonal filter by 0.07 skill and
   linear AR(12) by 0.04 on unseen sites; rollouts beat climatology for about three
   months, then converged to it. Sub-monthly cadence untested.
3. About half of a pixel's change (52–59 %) was shared with its 1.3 km neighbourhood;
   knowing the neighbourhood's realised change roughly doubled attainable skill. Its
   forecastable part from history was small.
4. Annual weather aggregates explained about 3 % of the annual shock beyond mean
   reversion. Driver conditioning with a sub-annual target was never tested.
5. Optical monthly coverage: median site observed 78 % of months, tropics 64 %; 11 % of
   sites unusable at a 50 % threshold. Coverage selection biases toward drylands.
6. Change was anti-persistent at both cadences (lag-1 cosine −0.24 to −0.38);
   trend-continuation baselines were negative or zero; no multi-year pixel trends found.
7. Class-probability outputs were soft (mean max probability 0.51); label-level Markov
   chains scored no better than persistence against them.
8. A learned model initialised exactly at the best simple filter was the difference
   between negative and positive skill on unseen sites at annual cadence; at monthly
   cadence the prior stopped mattering one step ahead but still mattered for rollouts.
9. Early stopping on likelihood chose checkpoints with lower point skill than early
   stopping on error, consistently; keep and report both.
10. Mixture (multi-modal) outputs improved likelihood, energy score and spread–skill
    correlation (0.55 annual, 0.74 monthly) without changing point skill; the mean of
    sampled rollouts equalled the deterministic rollout to within 0.01 at every horizon.
    The value of sampling was the spread, never the point.
11. Learned carried memory added nothing over a running mean at annual cadence and +0.02
    skill at monthly cadence.
12. A ConvLSTM trained from scratch on patches scored below the seasonal filter.
13. Unstructured models memorised site-year shocks and did not transfer; only a
    site-level split with unseen sites and unseen years exposed this.
14. Testing attribution before establishing that the null beat a trivial baseline cost two
    weeks and produced nothing.
15. Globally scattered sites made neighbourhood and regional structure untestable.
16. Eight annual steps gave three forecast horizons; cadence determined what could be
    asked at all.
