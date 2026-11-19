# Greeks utility test: result against PREREG.md

Run 2026-09-15 on CPU, 12 s, saved fair-fight params, Delta along climatological rain.
Held-out 2019-25 rainy-season starts (67 per box), ~1.6M pixel-steps per box, 7 year-clusters.
Slopes: real NDVI-anomaly change per 100 mm rain anomaly, controlling for starting anomaly.

| Test | Box | Low / mid / high real slope | Model Delta low / mid / high | High - low [95% CI] | Verdict |
|---|---|---|---|---|---|
| G1 within-pixel (PRIMARY) | A | .037 / .062 / .063 | .043 / .086 / .118 | +.026 [-.026, +.070] | FAIL |
| G2 raw sorting | A | .031 / .066 / .078 | .036 / .088 / .124 | +.047 [+.005, +.084] | PASS |
| G1 within-pixel | B | .073 / .090 / .137 | .065 / .098 / .122 | +.064 [+.014, +.116] | pass (reported) |
| G2 raw sorting | B | .074 / .087 / .136 | .065 / .098 / .123 | +.062 [+.009, +.110] | pass (reported) |
| M1.5 G2 raw sorting | A | .095 / .067 / .044 | .020 / .038 / .054 | -.052 [-.122, -.001] | sorts backwards |

G3 hedge error, skill vs persistence (constant / frozen M2 map / M2 state / M1.5 state):
A .025 / .032 / .034 / .009; B .081 / .080 / .087 / .064. M2 state is best on both boxes, but no
difference clears its interval (A: +.010 [-.023, +.050] vs constant; B: +.006 [-.005, +.018]). FAIL.

corr(M2 Delta, rain anomaly): A -0.14, B -0.03 (small; no matching needed).
Split by sign of rain anomaly (not pre-registered): Box B dry-anomaly slopes rise .049 / .114 / .180;
Box A dry-anomaly slopes fall .086 / .058 / -.018.

Verdict: primary fails on Box A (right direction, interval spans zero). M2's Delta ranks real rain
exposure on both boxes (G2), including the landscape it never saw, where its magnitudes also match
the real slopes; state-dependence is significant on Box B only. M1.5's Delta ranks exposure backwards
on Box A. Error-score gains exist but are within noise.
