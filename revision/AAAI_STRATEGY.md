# AAAI-27 workshop strategy

Checked 2026-10-12 against the two CFP pages. Acceptance rates are not published for either workshop.

## 1. The two venues

| | **AgriAI** (2nd Workshop on AI in Agriculture) | **AI4TS** (4th Workshop on AI for Time Series) |
|---|---|---|
| CFP | agriai-workshop.github.io/aaai2027 | uconn-dsis.github.io/ai4ts-aaai2027/aaai2027.html |
| Deadline | **20 Nov 2026 AoE**; notification 2 Dec | **TBD**. Past AAAI deadlines: 9 Dec 2023, 8 Dec 2024, 2 Nov 2025 |
| Length | ≤ 6 pages **including** references, AAAI-27 template | 4–7 pages **excluding** references, AAAI-27 template; appendix of any length allowed |
| Review | **Single-blind**: names and affiliations in the paper | **Double-blind** |
| Parallel submission | **Forbidden**: "not under concurrent review elsewhere" | Allowed only if the other venue allows it, so not alongside AgriAI |
| Size | 18 papers in 2026 (8 oral) | 37 papers in 2026 (12 oral) |
| Attendance | One author in person, Montréal, 22–23 Feb 2027 | Poster; assume in person |
| Precedent for this topic | No rangeland, NDVI, insurance or carbon paper. Closest: LAI forecasting with a time-series foundation model, causal agro-ecosystem dynamics | No agriculture or Earth-observation papers. 1–3 climate papers and 2–4 finance papers per edition |

## 2. Who will judge it

**AgriAI.** Agricultural engineering, vision and IoT people.
- **Sarkar** (Iowa State; associate director of AIIRA) is the closest fit. His work covers weather-conditioned crop-yield models, interpretable and Granger-causal time series, and digital twins.
- **Arjunan** (IISc) works on time-series foundation models for energy and anomaly detection. His agriculture work is vision.
- **Das** (Missouri S&T) works on IoT sensing, smart farms and digital twins.
- None of the three works on rangeland, livestock, insurance or carbon. **The paper has to supply that context itself.** These reviewers are less likely to press on identifiability, which was R3's attack. They are more likely to ask "is this validated on the ground?" and "who uses it?"

**AI4TS.** Time-series ML methods people, much like the FMTS reviewers who gave 6/5/5.
- **Nevmyvaka** (Morgan Stanley; Lag-Llama) will get the Black–Scholes framing. He will also spot it if it is oversold.
- **Xie** (Georgia Tech) works on conformal prediction and change points: uncertainty quantification.
- **Luo** (SMU) works on time-series explanation and attribution, plus flood AI.
- **Song** and **Chen** work on anomaly detection.
- **Purushotham** has done satellite domain adaptation.
- Expect a repeat of R2: "baselines are weak, why not a foundation model?"

## 3. Strategy

1. **Primary: AgriAI, 20 Nov.**
   - The pool is smaller and application-friendly, and no paper there covers this topic.
   - The CFP names "counterfactual reasoning", satellite and weather fusion, and transfer across regions. The paper does all three.
   - Next to vision papers, the ML here (a calibrated density, inversion, autodiff sensitivities) reads as substantial. At AI4TS it reads as "a one-hidden-layer MLP".
2. **Fallback: AI4TS**, only if its deadline lands after the AgriAI notification on 2 Dec. Past AAAI deadlines say it might, but not certainly. If AgriAI rejects, anonymise and resubmit with the AI4TS framing (§5). If the AI4TS deadline is set before 2 Dec, choose one; the choice is still AgriAI.
3. **One core manuscript for both:**
   - Write it in the AAAI-27 two-column template: about 5 pages of text plus under 1 page of references.
   - That stays under AgriAI's 6-page total and over AI4TS's 4-page minimum.
   - Switching venues then takes a framing pass, not a rewrite.
4. **Email samy@iisc.ac.in this week.** Ask whether the arXiv preprint and the NeurIPS FMTS workshop submission (non-archival) conflict with "not published previously / not under concurrent review". Get it in writing.

## 4. Paper changes (both venues)

The NeurIPS 4-page limit was the binding constraint. AAAI's 5–6 two-column pages free about a page, which is enough to land `revision/REVISION.md` without the trade-offs in its page budget.

| Review point | Status | Change |
|---|---|---|
| R1, R3: residual ≠ management; untested exogeneity | Tested: `runs/20261006_1753_identify` | REVISION §1: an "What the counterfactual assumes" paragraph (A1–A3), plus "Is the counterfactual identified?" (+18 % in drought, +47 % if management responds to rain). Call it a "rain-conditioned residual". |
| R2: weak baselines | Tested: `runs/20261006_1759_baselines` | REVISION §2 and §4: a three-month table in the main text with GBM and GRU-M2. Say plainly that GBM wins on Box B at three months, and why (drift). |
| R2, R1: rain is the only driver | Tested: `runs/20261006_1802_drivers` | REVISION §3: two sentences, full table in the appendix. |
| R1: headline numbers in no table | Fixed by the new table | Caption: "attribution given realised rain, not forecasting". |
| R1: Δ overclaimed | Tested: `runs/20261006_1803_greekstest_nogap` | Drop "can be hedged / efficient pricing". Pre-registered G1 fails; it passes once gap steps are removed. Report both. |
| R1: MODIS gaps; the 14-seasons count | Tested: `runs/20261006_1801_gaps` | 13 seasons. Add the 144-day 2025 gap. One appendix line. |
| R1: housekeeping | Text only | Rename n_t; Table 1 to the appendix; analogy is "one of use, not foundation"; anomaly result labelled "exploratory". |

New, not raised by reviewers but likely from these pools:

- **Related work. The paper has none.** It has no ML citations, and the CHIRPS, MOD13Q1, MCD64 and Seddon entries are listed but never cited in the text. Add one paragraph:
  - **RESTREND** (Evans & Geerken 2004; Wessels et al. 2007). This is the land-degradation incumbent: NDVI residuals after a linear annual rain regression. It is the obvious prior a remote-sensing reviewer will name, and M1 is a 16-day generalisation of it. Positioning against it sharpens the novelty claim.
  - Pearl's abduction–action–prediction counterfactual.
  - Probabilistic time-series and foundation-model forecasting.
  - Time-series anomaly detection.
- **Abstract: about 280 words now; cut to about 180.**
  - Keep: the problem, the method, one 2022 number, Δ with its caveat.
  - Drop: the "two capabilities drop out" restatement and the last sentence.
- **Page-1 figure.** Two-column reviewers skim. Put the 2022 attribution figure, or a schematic of inversion → residual → counterfactual, at the top of page 2 at the latest.
- **NDVI is not soil carbon.** One sentence: the counterfactual is on productivity. It is a baseline input to a carbon model, not a carbon estimate. An agriculture reviewer will ask.

## 5. Venue framing

**AgriAI (primary)**
- **Title:** lead with the use. For example: *"Separating drought from management on rangelands: a rain-conditioned world model"*. Move Black–Scholes into the text as the guiding analogy. Keep the current title if you prize the hook; it costs clarity with this pool.
- **Introduction:** pastoralists → index-based livestock insurance (IBLI) pays on NDVI → carbon baseline (VM0032 / NKRCP) → both need drought separated from management.
- **CFP vocabulary, used where the paper earns it:**
  - "counterfactual reasoning"
  - "satellite and weather fusion"
  - "transfer across regions" (Box B as the unseen landscape)
  - "rain-conditioned digital twin"
- **"How it would be used" paragraph:**
  - The insurer pays on the rain's share.
  - The verifier uses the normal-rain run as the baseline.
  - A fit takes 190 s on one GPU for 25,600 pixels.
- Add authors, affiliation and contact (single-blind).
- **Optional, about 2 days: fence-line validation.**
  - This is the one result that would most move an agriculture reviewer. It is independent management data, which R1 asked for and REVISION.md declined.
  - Lewa sits in Box A and Mpala in Box B (from memory; verify with WDPA or NRT boundary data).
  - Test: in 2017 and 2022, does the residual differ across conservancy versus community-land boundaries, matched on pixel traits?
  - Pre-register it as usual. Report it whatever the outcome.
- **Optional, about 1 day: zero-shot foundation-model baseline** (Chronos-Bolt or Lag-Llama, history only).
  - It answers "why not a foundation model?", which the CFP invites and Arjunan works on.
  - Expected result: strong at one step, but no driver, so no counterfactual and no Δ.

**AI4TS (fallback)**
- Anonymise. Don't cite or link the arXiv version.
- Keep the Black–Scholes title. The finance reader is on the committee.
- **Lead with the general method:** an invertible, driver-conditioned transition density gives implied residuals and sensitivities for any exogenously driven series. Laikipia is the case study.
- Use 7 pages plus an appendix: full baseline and driver tables in the main text.
- **Hooks:**
  - Coverage of the predictive intervals (Xie).
  - Δ as a model-intrinsic, driver-level attribution, not post-hoc saliency (Luo).
  - The residual as an anomaly score, with the fire probe marked exploratory (Song, Chen).
  - The foundation-model baseline matters more here than at AgriAI.

## 6. Timeline (12 Oct → 20 Nov)

| Week | Task |
|---|---|
| 13–19 Oct | Email AgriAI. Port `paper/main.tex` to the AAAI-27 kit. Decide on the two optional experiments and pre-register them. |
| 20–26 Oct | Land REVISION.md §1–7. Related work. New three-month table. Cut the abstract. |
| 27 Oct–2 Nov | Optional experiments. |
| 3–9 Nov | AgriAI framing pass. Cold read by someone outside ML (an ecologist or agronomist). |
| 10–18 Nov | Page check (≤ 6 including references), then submit **18 Nov**, two days early. |
| From 2 Dec | If rejected and AI4TS is still open: anonymise, AI4TS framing, submit. |

After any revision, regenerate `paper/arxiv/` from the new source (two edits: the `preprint` option and the author line).
