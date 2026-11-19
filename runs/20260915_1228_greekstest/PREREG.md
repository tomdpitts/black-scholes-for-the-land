# Greeks utility test: pre-registered before any result was computed

Written 2026-09-15 before running greekstest.py. Seen beforehand: the frozen-map results
(runs/20260914_2137_deltatest_h100), not any current-state result.

Question (carbon-credit framing): at the start of a rainy season, do patches the model calls
rain-sensitive actually gain or lose more greenness when rain departs from normal? That is the
per-patch exposure a credit buyer or insurer would price.

Models: saved fair-fight fits (runs/20260914_1708_pipeline_h100_fairfight/params.npy); no refit.

Delta (no leakage): d E[a_{t+2}] / d r_{t+1} from the state at t (a_t, w_t), rolled out along
CLIMATOLOGICAL rain. Realised rain enters only the outcome.

Starts: every 16-day step whose window t+1 is in MAM or OND. Outcome y = a_{t+2} - a_t, rain
anomaly x = r_{t+1} - rclim_{t+1}, pixel-steps with a_t and a_{t+2} observed. Held-out years
2019-2025. Uncertainty: bootstrap over years (7 clusters), 2000 draws, 95% percentile interval.

Sorting regression, within each Delta tercile: y ~ 1 + x + a_t; the slope on x is the real
rain response of that group. Tercile cut-points use Delta only, never outcomes.

## Primary (Box A)
G1 state test. Demean M2 Delta per pixel over held-out starts, sort into terciles. PASS if the
real slope of the top tercile exceeds the bottom tercile, interval of the difference above zero.

## Secondary (Box A)
G2 sorting test. Same with raw M2 Delta terciles (across pixels and seasons).
G3 hedge error. y ~ alpha + beta * Delta * x fitted on 2003-2017 starts, scored on held-out
years as skill vs persistence. PASS if M2 current-state Delta beats both a single constant and
the frozen M2 map (mean over 2003-2017 starts), interval of each skill difference above zero.

## Reported, no pass/fail
Box B under the same three tests (transfer check); M1.5 current-state Delta under all tests;
mean model Delta per tercile against the real slope; corr(Delta, x); slopes split by sign of x.
