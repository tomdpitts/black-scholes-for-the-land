# Latent residual test: result against PREREG.md

Run 2026-09-15 on a Modal H100 (`latent.py`, fair-fight M2, theta frozen, no refit). Seen before the full run:
a smoke test on a 32 x 32 crop with 40 Adam steps; PREREG.md unchanged after it.

Prior scales chosen s0 = s1 = 0.003, the smallest on the grid. Every scale on the grid raised 2018 one-step NLL
above M2's (best -1.4648 against -1.4653 at z = 0): on the selection year the lagged latent did not help, and the
chosen prior sits on the grid edge.

Skill = 1 - CRPS(variant) / CRPS(reference), pooled over years; [95% bootstrap over years].
Box A: years 2019-25 (7 clusters). Box B: years 2004-25 (22 clusters).

| Test | Box | Skill | Verdict |
|---|---|---|---|
| L1 lag vs M2, horizon 6 (PRIMARY) | A | +.0060 [-.0025, +.0156] | FAIL |
| L2 lag vs M2, one-step | A | +.0033 [+.0025, +.0045] | PASS |
| L3 lag vs lag_shuffled, horizon 6 | A | +.0058 [+.0048, +.0068] | PASS |
| L4 lag vs M2, horizon 6 | B | +.0017 [-.0072, +.0109] | FAIL |

## Reported, no pass/fail
- lag vs M2 by horizon 1..6, A: .0033 .0051 .0062 .0066 .0066 .0060; interval above zero at h1-4, not h5-6.
  B: .0015 .0023 .0024 .0020 .0019 .0017, no horizon above zero. B one-step +.0014 [-.0003, +.0033].
- Per-year lag vs M2 at h6, A: 2019 -.015, 2020 +.007, 2021 +.009, 2022 +.003, 2023 +.004, 2024 +.010, 2025 +.043.
- lag_shuffled vs M2 at h6: A +.0002 [-.0085, +.0103]; B -.0016 [-.0102, +.0077]. Box-wide shift adds nothing.
- lag vs lag_shuffled at h6, B: +.0034 [+.0013, +.0056].
- lag_level vs M2 at h6: A +.0036 [-.0003, +.0085]; B +.0003. lag vs lag_level at h6: A +.0024 [-.0023, +.0076].
- same_year (in-sample ceiling) vs M2 at h6: A +.027 [+.019, +.036]; B +.031 [+.023, +.040].
- Persistence, corr across pixels of z_y and z_{y-1}: A z0 .33, z1 .32; B z0 .28, z1 .24 (B years range -.31 to .64).
- Regime effect E[a_{t+6} | z_{y-1}] - E[a_{t+6} | z = 0], A 2019-25: pixel sd .004-.006 NDVI, p5/p95 about -.01/+.01.
- Fire, mean z0 burned vs unburned pixel-years: A same year +.00015 vs +.00071 (n = 208 burned pixel-composites);
  B same year +.0001 vs +.0007, following year -.0010 vs -.0002 (n = 1299). Lower where burned; small; untested.

## Verdict
Primary fails. What the land does beyond the rain has a pixel pattern that carries a year (year-to-year r of
about 0.3; the lagged latent beats the same values shuffled across pixels on both boxes, with tight intervals; a
one-step gain on A). It is worth 0.3-0.6% of M2's error, is not reliably positive at three months on A (2019
reverses it; 2025 carries much of the pooled gain) and is not detectable on the unseen landscape. Under the
interpretation fixed in advance: the latent is not usable for forecasting at this cadence and the carbon
counterfactual cannot be built from it. The regime it would remove moves three-month greenness by about
+-0.01 NDVI at the 5th/95th percentile, against a 0.2 NDVI seasonal swing. The in-sample ceiling (0.03) is five
times the lagged gain: most of what z absorbs within a year does not survive to the next.

L3 passing while L1 fails says the pixel pattern helps and the year's box-wide level hurts in some years. A
pixel-demeaned latent is the obvious next variant; it would be post hoc and is not run.

## Limits
Theta frozen (joint fit of theta and z not tested); one-year lag on calendar-year blocks; prior scale on the grid
edge; nothing here shows z is management.
