# Reviewer Triage

## Summary

| # | Fix | Raised by | Cost | Score impact |
|---|-----|-----------|------|--------------|
| 1 | [Reframe the claims](#1-reframe-the-claims) | R1, R3 | Text, ~2 h | **High** |
| 2 | [Test that $\hat\varepsilon$ doesn't depend on rain](#2-test-that-hatvarepsilon-doesnt-depend-on-rain) | R1, R3 | ~½ day | **High** |
| 3 | [Three-month skill table](#3-three-month-skill-table) | R1 | ~1 h | Medium |
| 4 | [Soften the Δ claims](#4-soften-the-δ-claims) | R1 | Text | Medium |
| 5 | [Stronger rain-conditioned baseline](#5-stronger-rain-conditioned-baseline) | R2 | ~½ day + GPU | **High** (for R2) |
| 6 | [Temperature ablation](#6-temperature-ablation) | R2 (+ R1) | ~1 day | **High** (for R2) |
| 7 | [Rerun without gap steps](#7-rerun-without-gap-steps) | R1 | ~1 h | Low–medium |
| 8 | [Housekeeping](#8-housekeeping) | R1 | Text, ~2 h | Low each, cheap together |

---

## 1. Reframe the claims

**Raised by:** R1, R3 · **Cost:** Text, ~2 h · **Impact:** High

- Call the residual a **"rain-conditioned residual"**, not "management".
- Add an explicit **Assumptions** paragraph, with the ways each one fails (e.g. grazing that responds to rain):
  - residual unchanged when rain changes;
  - measurement errors stationary;
  - enough state in the model.
- Rework the abstract and conclusion to match.

## 2. Test that $\hat\varepsilon$ doesn't depend on rain

**Raised by:** R1, R3 · **Cost:** ~half a day · **Impact:** High — turns an unstated assumption into a measured one.

- **(a)** Move the existing residual-vs-rain diagnostic from the appendix into the main text as a direct test. It already shows the failure below −60 mm, and stating that honestly helps.
- **(b)** New semi-synthetic test: inject a known fake management signal into the held-out years — once independent of rain and once dependent on it — rerun the inversion and counterfactual, and report how much is recovered and how much bias the rain-dependent case adds. No refitting needed.

## 3. Three-month skill table

**Raised by:** R1 · **Cost:** ~1 h · **Impact:** Medium — fixes a credibility gap in the abstract.

- Table for **M0, M1, M1.5, M2** and **M2 without rain**, on **Box A** and **Box B**.
- The numbers already exist (run `20260914_1500_pipeline_h100`); only need to recompute the S-curve comparison behind the "13%".

## 4. Soften the Δ claims

**Raised by:** R1 · **Cost:** Text · **Impact:** Medium

- State in the abstract that G1 fails on Box A.
- Lead with G2 and Box B.

## 5. Stronger rain-conditioned baseline

**Raised by:** R2 · **Cost:** ~half a day + GPU · **Impact:** High for R2 — it's half their review.

- **LightGBM** on lagged rain, $w$, state and season.
- A small **GRU** over the sequence (also tests whether the Markovian state is enough).

## 6. Temperature ablation

**Raised by:** R2 (and R1's point about missing drivers) · **Cost:** ~1 day · **Impact:** High for R2

- Pull ERA5-Land temperature (or PET) through the existing Modal / Planetary Computer ingest.
- Compare **rain only**, **temperature only**, and **both**.

## 7. Rerun without gap steps

**Raised by:** R1 · **Cost:** ~1 h · **Impact:** Low–medium — also stops the paper saying "not yet excluded".

- Exclude transitions that span the MODIS gaps.
- Report how much the 2022 numbers move.

## 8. Housekeeping

**Raised by:** R1 · **Cost:** Text, ~2 h · **Impact:** Low each, cheap together

- [ ] Rename the neighbourhood mean to stop the bar/tilde confusion in Eq. 3.
- [ ] Relabel Table 1.
- [ ] Place Fig. 1 next to its first mention.
- [ ] Label anomaly detection "exploratory".
- [ ] One line saying this is attribution given the realised rain, not forecasting.
- [ ] One paragraph on the limits of the Black–Scholes analogy (no no-arbitrage argument, residual depends on the model, nothing makes it unique).
