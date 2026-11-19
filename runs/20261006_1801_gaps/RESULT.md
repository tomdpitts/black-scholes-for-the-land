# MODIS gap exclusion: result

Scripts: revision/gaps.py (this run), revision/greeks_nogap.py (runs/20261006_1803_greekstest_nogap).
Saved fair-fight fits, no refit (no gap falls in 2003-2018). Gap steps (> 16 days between
composites): 2021-11-01 (61 d), 2023-02-02 (32 d), 2024-07-27 (32 d), 2024-12-02 (30 d),
2025-06-26 (144 d; not listed in the paper). "all" reproduces the paper's numbers exactly.

| | Box A 2019-25 all | A no-gap | Box B all | B no-gap |
|---|---|---|---|---|
| rollout starts | 141 | 115 | 531 | 505 |
| M2 one-step skill | 0.085 | 0.094 | 0.104 | 0.106 |
| M2 3-month skill vs history (M0) | 0.226 | 0.269 | 0.171 | 0.179 |
| M1.5 3-month skill | 0.107 | 0.129 | 0.097 | 0.102 |
| M2 3-month CRPS below M1.5 | 13.3 % | 16.1 % | 8.1 % | 8.6 % |

Anomaly table (one-sided, post hoc; 206 of 208 burn events survive): M2 hit 0.279 -> 0.286, drought
FAR 0.051 -> 0.054, AUC 0.773 -> 0.771; z-score FAR 0.244 -> 0.249. Two-sided pre-registered: M2
hit 0.192 -> 0.194. No conclusion changes.

Delta tests, starts whose two-step outcome spans a gap removed (67 -> 64 per box):

| Test | Box | Pre-registered | Gap-free |
|---|---|---|---|
| G1 within pixel (PRIMARY) | A | +.026 [-.026, +.070] FAIL | **+.046 [+.014, +.078] PASS** |
| G2 raw sorting | A | +.047 [+.005, +.084] | +.063 |
| G1 | B | +.064 [+.014, +.116] | +.064 |

Reading: the gaps hurt the model, never help it. Three of 67 Box A starts have a "two-step"
outcome that in fact spans 77 days across the 2021 gap; removing them takes the primary Delta
test from fail to pass. The pre-registered result remains the primary one; the gap-free result
is a data-validity correction requested by reviewer 1, decided before it was run.
