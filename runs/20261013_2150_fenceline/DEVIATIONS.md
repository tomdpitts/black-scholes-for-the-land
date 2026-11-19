# Deviations and post-hoc analyses

Everything below was decided **after** seeing the pre-registered results (`outputs/results.json`, 2026-10-13T11:40Z). None of it changes any pre-registered verdict. Script: `revision/fenceline/s07_exploratory.py`. Outputs go to `outputs/exploratory/`.

## D1. S4 (Amendment 2) has an orientation flaw

The angle was drawn on [0, π), and the fake-treated side was always the s < c side. Because sin θ ≥ 0, that side is always the northern half of the split. The registered S4 p-value (0.995) therefore partly measures a north–south gradient in the 2017 residual across community land, not the test question. **E1** reruns S4 with θ on [0, 2π) under seed 20172, otherwise unchanged. The registered S4 result stays as reported.

## E2. Event study at composite resolution (exploratory)

For each 16-day composite from September 2016 to December 2017: the one-step residual in NDVI units (a(t+1) − μ_t), treated minus matched control, over the final matched design. This is set against the same calendar composites in the primary placebo years (median and 5–95% envelope). Purpose: to see whether a dip occurred *during* the invasion (Feb–Apr) and was then masked by recovery, which the season-end statistic cannot show.

## E3. Trend in D across years (exploratory, hypothesis-generating)

`fig_D_by_year.png` shows D mostly negative in 2003–2012 and mostly positive in 2013–2025. 2013 is when the exposure research reports outside herders occupying community group ranches (Morupusi, Koija) and not leaving (S15). E3 reports:
- mean D for 2003–12 against 2013–25 (Welch t, two-sided);
- the OLS slope of D on year;
- each arm's mean season residual by year, to show which arm moves.

This was noticed after seeing the data and is reported only as a hypothesis for a future pre-registered test.
