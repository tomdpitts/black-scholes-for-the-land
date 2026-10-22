# Latent residual test: pre-registered before any result was computed

Written 2026-09-15 02:50 UTC before running latent.py. Seen beforehand: the Theta decay of the
fitted M2 (0.70 at one step, 0.15 at six), the frozen Delta-map result (runs/20260914_2137_deltatest_h100)
and the Greeks utility test (runs/20260915_1228_greekstest). No latent has been fitted.

Question (carbon-credit framing): is what the land does beyond the rain a persistent, pixel-specific
regime (management, degradation) that carries forward and can be plugged back in to forecast, or is it
transient noise? If it persists, the model gives a counterfactual: the same rain with the regime removed.

Model: saved M2 from runs/20260914_1708_pipeline_h100_fairfight/params.npy, theta frozen, no refit.
Latent per pixel and calendar year, z = (z0, z1), added to the drift:
    mu <- mu + z0 + z1 * (w_t + r_{t+1})
z0 shifts the level the anomaly relaxes to; z1 changes greenness gained per unit of effective water.

Inference of z_y: MAP over transitions whose target falls in year y (observed target and start only),
Gaussian one-step NLL plus independent Gaussian priors z0 ~ N(0, s0^2), z1 ~ N(0, s1^2), Adam from zero.
Prior scales: grid s0, s1 in {0.003, 0.01, 0.03}, chosen on Box A only by inferring z_2017 and scoring
one-step NLL on 2018. Chosen once, used everywhere.

Freeze and forecast: year y is scored with z_{y-1}. Box A: y = 2019..2025. Box B: y = 2004..2025.
Scores: one-step Gaussian CRPS on observed transitions with target in y; CRPS of 16 sampled rollouts
along realised rain at horizons 1..6 from every start whose first target is in y (common random numbers
across variants). Skill of a variant = 1 - CRPS(variant)/CRPS(M2, z = 0), pooled over the year set.
Uncertainty: bootstrap over years (clusters), 2000 draws, 95% percentile interval.

Variants: M2 (z = 0); lag (z_{y-1}); lag_level (z0 only, inferred with z1 fixed at 0);
lag_shuffled (z_{y-1} permuted across pixels, one fixed permutation per box: keeps the box-wide shift,
destroys the pixel pattern); same_year (z_y, in-sample, leakage upper bound, labelled as such).

## Primary (Box A)
L1. Skill of lag over M2 at horizon 6 (three months). PASS if the interval is above zero.

## Secondary
L2. Same on Box A one-step. PASS if the interval is above zero.
L3. Pixel specificity, Box A horizon 6: skill of lag over lag_shuffled. PASS if the interval is above zero.
L4. Transfer: L1 on Box B (theta never saw B; z inferred on B's preceding year). PASS if above zero.

## Reported, no pass/fail
Year-to-year persistence: corr across pixels of z_y and z_{y-1}, per component, per box.
lag_level against lag (does the rain-gain term add anything). same_year skill (ceiling).
Mean z0 in pixel-years with a MCD64A1 burn against without, same year and following year.
Maps of z0, z1 and of the regime effect E[a_{t+6} | z_{y-1}] - E[a_{t+6} | z = 0] averaged over starts.

## Interpretation fixed in advance
L1 fail: the residual does not carry forward a year; the latent is not usable for forecasting at this
cadence and the carbon counterfactual cannot be built from it. L1 pass, L3 fail: the gain is a box-wide
shift (e.g. CHIRPS bias or post-2018 climatology drift), not a pixel regime. Nothing here shows the
latent is management; that needs a management record.
