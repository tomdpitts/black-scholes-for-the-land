# Revision plan: FMTS reviews -> next workshop (4-page limit)

Reviews: R1 = 6 (conf 4), R2 = 5 (conf 3), R3 = 5 (conf 4). `paper/main.tex` is untouched;
every proposed change below is a draft for you to accept, edit or reject.
Every number cites a run directory; every new test was pre-registered (`PREREG.md`) before it ran.

Ordered by expected effect on the score.

## Scorecard: new pre-registered tests

| Test | Result |
|---|---|
| Counterfactual identified, management ignores rain | near-normal seasons yes; droughts **no** (+18 %, spread scaling) |
| Counterfactual identified, management responds to rain | **no** (+47 % in drought); a stated assumption |
| Counterfactual identified, rain-product error 25 % / 50 % | yes / narrowly no, at box scale; no at pixel scale |
| M2 competitive with strong baselines at 3 months | A yes; **B no** (GBM 12 % lower CRPS) |
| Markov state sufficient (GRU lead < 0.02) | A yes; **B no** (+0.036); diagnostics say drift, not memory |
| Rain dominant over temperature | **yes** |
| Temperature adds to rain | **no**; it subtracts out of time (warming shift) |
| Δ primary G1, gap-free (correction, not pre-registered as such) | **passes** (+.046 [+.014, +.078]) |

Bottom line: the failures make the paper more credible, not less, provided they are reported.
Every one sharpens what the paper claims: a rain-conditioned residual and an invertible,
differentiable density, valid at project scale, under a stated exogeneity assumption.

---

## 1. The residual is not management, and the counterfactual's assumption was untested  (R1, R3)

**Weakness.** R3's whole objection, and two of R1's main points. The residual absorbs everything
the model leaves out (grazing, fire, clearing, *and* CHIRPS error, model error, satellite noise),
yet the abstract and conclusion read it as management. The counterfactual of Eq. 3 holds the
residual fixed while changing the rain. That is an exogeneity assumption: it fails if what is in
the residual responds to rain (herders move stock onto rangeland in drought). The paper never
states it or tests it. R3: "this assumption is what turns a conditional predictive model into a
causal counterfactual mechanism."

**What we did.** `runs/20261006_1753_identify/` (PREREG, RESULT, scripts `revision/identify*.py`).
1. *Semi-synthetic identification test.* Take fitted M2 as the true world, so the model is exactly
   right and any error is the assumption's alone. Inject a known management signal (-0.01 NDVI per
   16-day window on a random half of 2 km blocks) into all 13 held-out seasons, invert, re-run
   under normal rain, compare with the true counterfactual.
2. *Real-data diagnostics.* Residual against rain departure (with year-cluster CIs, both boxes),
   against MODIS quality flag, and against CHIRPS-minus-ERA5 rain disagreement.
3. *Sensitivity of the paper's 2022 numbers* to every contaminant found.

**Results.**

| If management... | Error in the management effect, drought seasons | near-normal seasons |
|---|---|---|
| ignores rain (S1) | **+18 %** (max 30 %; MAM2022 +28 %) | +1 % (max 7 %) |
| intensifies when rain fails (S2a, incursions) | **+47 %** (max 81 %) | 4 % |
| eases when rain fails (S2b, destocking) | -7 % (max 24 %) | 11 % (max 72 %, wet seasons) |

| Rain-product error (no management) | box-mean counterfactual error | single-pixel RMSE |
|---|---|---|
| 25 % multiplicative per CHIRPS cell | <= 0.005 NDVI | 0.03 |
| 50 % | <= 0.013 NDVI | 0.05-0.11 |

- **New finding: a bias even when the assumption holds.** The paper holds the residual in
  *spread units*, and M2's spread rises steeply with rain (σ = 0.018 / 0.043 / 0.062 at
  -100 / 0 / +100 mm). Re-running a drought under normal rain therefore scales each residual by
  σ'/σ. That inflates a rain-independent management effect by about a fifth in droughts. Holding
  the residual in NDVI units instead mis-scales the noise and is worse; the two do not bracket
  the truth (5/13 seasons), so no interval is claimed.
- **Rain-responsive management is not identified** from greenness and rain alone (+47 % in
  drought under incursions). This is a stated assumption, not something more modelling can fix.
- **Rain-product error averages out at box scale but not at pixel scale.** Attribution is a
  project-scale product.
- **On real data the residual is not quite rain-free.** Box A: -0.24σ in the driest windows
  (CI [-0.37, -0.04]); Box B: +0.13 to +0.34σ in moderately wet windows (CIs exclude 0).
  Overall corr 0.05-0.06. Marginal-quality MODIS pixels (16 %) carry residual mean +0.13σ and
  26 % more spread. CHIRPS and ERA5 agree only at corr 0.52-0.56 per 16-day window; where they
  disagree most the residual is 10-20 % noisier.
- **The 2022 headline survives.** The counterfactual end-of-season box mean (paper: -0.099) is
  -0.096 with the rain-dependent bias removed, -0.094 on clean-quality pixels only, and -0.087
  with an additive residual. "0.10 below" holds to within about 0.01.

**Proposed text** (+10 lines; see the page budget).

*Abstract, replace "Inverting the model ... does not explain."*
> Inverting the model on observed greenness yields an implied residual: everything the land did
> that the rain does not explain, management and disturbance but also rain-product and model error.

*Approach, replace the first "Residual" paragraph (`main.tex` line 30; it duplicates the
Setup paragraph of the same name) with:*
> \paragraph{What the counterfactual assumes} Holding $\hat\varepsilon$ fixed while the rain
> changes is the exogeneity step of abduction. It assumes (A1) nothing in the residual responds to
> rain; (A2) rain-product and satellite error average out over the area reported; (A3) the state
> $(a,w)$ carries what rain does. A1 fails if herders move stock onto rangeland when rain fails, as
> in Laikipia in 2017 and 2022. The residual is a rain-conditioned residual; reading it as
> management requires A1. We measure what each failure costs (Section 3).

*Results, new paragraph after "Residual and counterfactual":*
> \paragraph{Is the counterfactual identified?} We take the fitted model as the true world, inject
> a known management signal ($-0.01$ NDVI per window on half the 2\,km blocks) into every held-out
> season, and invert. If management ignores rain, its effect is recovered to $1\%$ in near-normal
> seasons but overstated by $18\%$ in droughts: the residual is held in spread units and the
> spread rises with rain. If grazing pressure rises when rain fails, the drought overstatement is
> $47\%$. That is not identifiable from greenness and rain, and it is the assumption a
> credit must state. Rain-product error of $25\%$ moves the box-mean counterfactual by at most
> $0.005$ NDVI but single pixels by $0.03$: attribution is a project-scale quantity. On real data
> the residual leans $-0.24\sigma$ in the driest windows; removing that lean, or dropping
> marginal-quality pixels, moves the 2022 counterfactual by at most $0.005$ (Appendix).

*Conclusion, replace "attribute the whole 0.17 to management" with* "attribute the whole $0.17$
to the land", and add: "The residual is what the rain does not explain; reading it as management
assumes management does not itself respond to rain."

---

## 2. Baselines are weak  (R2)

**Weakness.** R2: history-only, linear and a hand-written S-curve are not strong rain-conditioned
temporal baselines. That is half of R2's review.

**What we did.** `runs/20261006_1759_baselines/` (PREREG; `revision/baselines.py`).
- B1 **GBM**: histogram gradient-boosted trees, non-Markovian, three lags of greenness, six windows
  of rain, neighbourhood, season, pixel covariates; second GBM for the variance.
- B2 **GRU-M2**: M2 plus a 32-unit GRU memory over each pixel's history, zero-initialised, so it
  has M2's capacity and more. It also answers R1's "model simplicity" question: does the Markov
  state suffice?

**Results** (`RESULT.md` there). R2 was right that the baselines were weak.

| CRPS skill vs M0 | A 1-step | A 3-month | B 1-step | B 3-month |
|---|---|---|---|---|
| M2 (paper) | **.085** | **.226** | **.104** | .170 |
| GBM, lagged greenness + rain | .046 | .222 | .090 | **.260** |
| GRU-M2, M2 + memory | .052 | .205 | .097 | .206 |

- M2 is the best one-step density on both sets, and the best or tied at three months on
  unseen years.
- **On the unseen landscape, both strong baselines beat M2 at three months.** Both pre-registered
  criteria fail on Box B: M2 is not competitive there (GBM's CRPS is 12 % lower), and memory
  "helps" (the GRU leads by 0.036).
- **Diagnosis** (post hoc). Give GBM *exactly M2's state* (current greenness, rain bucket $w$,
  this window's rain, no lags) and it still beats M2 on B (h=6 MAE 0.0586 vs 0.0630). So it is
  not memory and not the one-parameter rain bucket. The difference is drift: M2's rollout mean
  goes browner by 0.02 (A) and 0.011 (B) NDVI over three months; the trees' does not
  (|bias| <= 0.003). The gap is in how M2 is fitted (a one-step likelihood, early-stopped on
  likelihood), not in the Markov state.

**How to use this in the paper.** Report it; don't hide it. The paper's claim is not "best
forecaster". It is "an invertible, differentiable transition density", and trees cannot give
$\Delta$ (piecewise constant in rain). Proposed text (Fit paragraph, +3 lines):
> Against stronger baselines, gradient-boosted trees on lagged greenness and rain and M2 with a
> recurrent memory, M2 is the best one-step density on both landscapes and ties at three months
> on unseen years. On the unseen landscape both overtake it at three months ($0.26$ and $0.21$
> against $0.17$). Given M2's own state the trees still do, so the gap is not memory: M2's
> rollout drifts browner by $0.01$ NDVI over three months. Trees give no derivative in rain, and
> so no $\Delta$.

Model unchanged by decision. One line for the conclusion's limitations:
> Fitting the drift on multi-step rollouts rather than one step should remove it; we leave that
> to future work.

---

## 3. Only rain as a driver  (R2; R1 "omitted drivers")

**Weakness.** R2 asks whether the framework takes other drivers (temperature) and how omitted
weather changes the residual.

**What we did.** `runs/20261006_1802_drivers/` (PREREG; `revision/drivers.py`). ERA5-Land
temperature and ERA5 PET and VPD (Open-Meteo archive, `data/era5/`), as 16-day anomalies on the
grid. M2 refit as temperature only, rain + temperature, and rain + temperature + PET + VPD.

**Results** (`RESULT.md` there).

| 3-month skill vs M0 | A 2019-25 | B all years | B 2003-18 | B 2019-25 |
|---|---|---|---|---|
| M2 rain (paper) | **.226** | .170 | .163 | **.188** |
| temperature only | -.430 | .047 | | |
| rain + temperature | -.024 | .146 | .241 | -.061 |
| rain + temp + PET + VPD | .075 | **.220** | **.287** | .074 |

- **Rain is the driver** (pre-registered D1 passes): temperature alone is worse than history.
- **Adding temperature does not help out of time** (D2 fails). The cause is a warming trend:
  2019-25 is +0.5 to +0.6 °C against the 2003-18 climatology, and a network that learned
  temperature effects from training-year variation extrapolates. Where the weather is
  in-distribution (Box B 2003-18) the extra drivers help, but that gain is confounded: those are
  Box A's training years, and a four-variable regional weather vector can identify the composite.
  The clean out-of-time tests both say no. Rain-only M2 does not inflate in-period.
- **The residual's meaning depends on the driver set** (R2's actual question). Residuals
  correlate 0.91 across driver sets, but the MAM2022 counterfactual moves from -0.099 to -0.158:
  with heat modelled, "normal rain" keeps the heat, and the heat leaves the residual.

**Proposed text** (Fit paragraph, +2 lines; full table in the appendix):
> Nothing in Eq. 1 is specific to rain. Adding ERA5 temperature, evaporative demand and vapour-
> pressure deficit raises skill in years inside the climatology but lowers it in 2019-25, which
> run $0.5$-$0.6\,^\circ$C warmer: a driver must be stationary over the baseline to earn its place.
> The choice matters for attribution: what the residual holds is whatever the drivers leave out,
> and a counterfactual must say which drivers it sets to normal (Appendix).

---

## 4. Headline numbers are not in any table  (R1)

**Weakness.** The abstract's 23 % / 13 % are three-month numbers. Table 2 shows only one-step
skill (0.085), so the reviewer could not find them.

**What we did.** Re-derived them from the saved fair-fight run, and reproduced them exactly
(`runs/20261006_1801_gaps`, "all" columns). 23 % = 3-month CRPS skill vs M0, 0.226; 13 % = M2's
3-month CRPS (0.0535) below M1.5's (0.0617). On Box B: 17 % and 8 %.

**Proposed table** (main text, replaces Table 1, which moves to the appendix):

| Model | Box A unseen years: 1-step | 3-month | Box B unseen land: 1-step | 3-month |
|---|---|---|---|---|
| M0 history only | 0 | 0 | 0 | 0 |
| M1 linear | .013 | .107 | .052 | .099 |
| M1.5 S-curve | .009 | .107 | .052 | .097 |
| GBM, lagged greenness + rain (new) | .046 | .222 | .090 | **.260** |
| GRU-M2, with memory (new) | .052 | .205 | .097 | .206 |
| M2 without rain | .016 | -.006 | .023 | -.034 |
| M2 fed climatological rain | | -.036 | | -.099 |
| **M2 learned** | **.085** | **.226** | **.104** | .170 |
| M2 + temp + PET + VPD (new; appendix) | .070 | .075 | .124 | .220 |

CRPS skill vs M0, along the rain that fell. Caption to add: "No model sees a rain forecast:
these are attribution scores conditional on realised rain, not forecasts" (R1's
forecasting-vs-attribution point).

---

## 5. Δ claims outrun the evidence  (R1)

**Weakness.** The pre-registered primary Δ test (G1, Box A) fails, +.026 [-.026, +.070], while
the abstract says Δ "can be hedged, allowing efficient pricing".

**What we did.** Re-ran the pre-registered tests with the 3 starts whose "two-step" outcome
spans the 61-day 2021 MODIS gap removed (`runs/20261006_1803_greekstest_nogap`).

**Results.**

| Test | Pre-registered | Gap-free |
|---|---|---|
| G1 within-pixel, Box A (PRIMARY) | +.026 [-.026, +.070] FAIL | **+.046 [+.014, +.078] PASS** |
| G2 sorting, Box A | +.047 [+.005, +.084] | +.063 |
| G1, Box B | +.064 [+.014, +.116] | +.064 |
| G3 hedge-error skill | fail both boxes | fail both boxes |

Handle with care. The pre-registered failure stays the primary result. The gap-free pass is a
data-validity correction R1 asked for, decided before it was run. G3 still fails, so
"can be hedged" must go.

**Proposed text.** *Abstract, replace the last two sentences with:*
> $\Delta$, the derivative of greenness with respect to rain, averages $0.075$ NDVI per extra
> $100$\,mm; patches it ranks most exposed respond about twice as much as the least, on held-out
> years and an unseen landscape. Our pre-registered within-pixel test on the development
> landscape missed significance, and passes once steps spanning a satellite gap are removed.

*Sensitivities paragraph:* add the G1 result and interval in one sentence. Drop "that can be
hedged, allowing efficient pricing of contracts" (abstract) and keep "exposure".

---

## 6. MODIS gaps not excluded  (R1)

**Weakness.** The appendix admits "steps spanning a gap are not yet excluded". R1 worries this
drives the 2021-22 drought results.

**What we did.** `runs/20261006_1801_gaps` (RESULT). Re-scored everything with gap-spanning
transitions removed. Also found a fifth gap the paper does not list: **144 days from
2025-06-26**. It removes OND2025, so there are **13** held-out seasons, not the 14 the Setup
section claims.

**Results.** Gaps only ever hurt the model. Without them: Box A 3-month skill 0.226 -> 0.269,
lead over the S-curve 13 % -> 16 %; Box B 0.17 -> 0.18, 8 % -> 9 %; anomaly table unchanged to
0.01; MAM2022 contains no gap; the Δ primary test passes (item 5).

**Proposed text.** Keep the paper's numbers as the headline (they are the conservative ones).
Appendix: replace "Steps spanning a gap are not yet excluded from any analysis" with
> Excluding steps that span a gap raises every score (three-month skill $0.226\to0.269$ on Box A)
> and changes no conclusion; reported numbers keep them.

Add the 144-day 2025 gap to the list; Setup: "14 seasons" -> "13 seasons".

---

## 7. Housekeeping  (R1)

| Point | Fix |
|---|---|
| $\bar a_t$ (neighbourhood) next to $\tilde a_t$ (counterfactual) in Eq. 3 reads as a clash | Rename the neighbourhood mean $n_t$ throughout (the code already calls it `n`). Eq. 3 becomes $\mu_\theta(\tilde a_t, \tilde n_t, \tilde w_t, \ldots)$ |
| Table 1 hard to parse | Move to the appendix with columns "Real response by Δ tercile (low / mid / high)", "Model Δ (low / mid / high)", "High minus low" |
| "Figure 1 missing" | Cite Fig. 1 in Approach so it floats onto page 2, next to the method it shows |
| Black-Scholes analogy oversold | Add to the BSF paragraph: "The analogy is one of use, not foundation: implied volatility is unique under no-arbitrage; our residual is unique only given the fitted model." |
| Anomaly detection post hoc | Prefix the sentence with "Exploratory:" and say "post hoc" in the main text, not only the appendix |
| Attribution, not forecasting | Table caption, item 4 |
| λ interpretable? | One clause: M2's rain memory has an 11-day half-life (10-13 across pixels), longer on greener, wetter pixels: a memory proxy, not soil moisture (`runs/20261006_1753_identify/lambda.txt`) |
| CHIRPS-error diagnostics | Item 1 (appendix line) |
| Independent management data (R1) | Decline: none exists for these boxes. Name it as a limitation; the semi-synthetic test is the substitute |

---

## Page budget (main body is exactly 4 pages; references start at line 133 of p. 4)

| Change | Lines |
|---|---|
| Abstract: cut "Two capabilities drop out..." restatement and the hedging sentence | -5 |
| Table 1 -> appendix (numbers kept in one sentence) | -11 |
| ∂σ/∂r paragraph -> 3 lines (keep: spread rises with rain, which explains the scaling bias) | -4 |
| Θ paragraph -> 3 lines | -3 |
| Conclusion: drop restated 2022 numbers | -3 |
| Approach "Residual" -> "What the counterfactual assumes" | +2 |
| "Is the counterfactual identified?" paragraph | +8 |
| Three-month table | +11 |
| Baseline and driver sentences in Fit | +5 |
| Δ, analogy and forecasting sentences | +3 |
| Fit: drop the S-curve-amplitude sentence (now in the baselines story) | -2 |
| Fire probe sentence -> "Exploratory: ..." one line, details already in the appendix | -1 |
| **Net** | **0** |
