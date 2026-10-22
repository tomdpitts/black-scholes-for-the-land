# Does removing rain get closer to the management effect than baselines do? Pre-registered before any result was computed

Written 2026-10-19 before writing or running revision/attribution.py. Item 1b of the AgriAI
revision: the paper claims that subtracting a learned rain response leaves a residual closer to
the true management effect than baselines that leave rain in it. This scores that claim where
the answer is known. Seen beforehand: the identification test (runs/20261006_1753_identify;
M2's own estimator, E1, overstates a rain-independent management effect by +18 % in drought
seasons in the M2 world), and the B1 GBM skill scores (runs/20261006_1759_baselines). No
baseline estimator has been scored in any synthetic world.

## Worlds (the truth; the estimators never see it)
Box A, the 13 held-out rainy seasons 2019-2025 (residual.season_starts), six 16-day windows each,
start state observed, as in the identification test.
- W1 M2 world: the fitted fair-fight M2 (runs/20260914_1708_pipeline_h100_fairfight). Truth
  a_{t+1} = mean_M2 + sigma_M2 e_t + m_t, with e_t the real M2-implied innovations of that
  season. Identical to the identification test. M2 is correctly specified here, so this world
  favours M2.
- W2 GBM world: B1 of revision/baselines.py refitted with its exact settings and seeds (mean and
  variance GBMs, distributed lags, Box A 2003-2017, variance rescaled on 2018). Truth
  a_{t+1} = a_t + mu_GBM + sd_GBM e_t + m_t, with e_t the real GBM-implied innovations of that
  season. Greenness lags before the season are observed; in-season lags are the synthetic path.
  Rain lags before the season are CHIRPS; in-season rain is the path being run. Every
  estimator is misspecified here.
The true counterfactual re-runs the same world under climatological rain r' for the season's
six windows, with the same e_t and the management policy evaluated at r'.

## Management scenarios (as the identification test)
Same managed mask: a fixed random half of 8x8-pixel (2 km) blocks, seed 0. delta = 0.01 NDVI
per window.
- S0 none. S1 rain-independent, m_t = -delta. S2a drought incursion,
  m_t = -delta clip(1 + (rclim - r)/rclim, 0, 3). S2b destocking, the same with k = -1.

## Estimators (each sees only the synthetic greenness and CHIRPS rain)
Each returns its counterfactual end-of-season (h = 6) greenness under climatological rain,
CF_hat. Whatever it does not attribute to rain it attributes to the land.
- Z, z-score / history only: CF_hat = observed end-of-season anomaly. Rain changes nothing.
  This is also what a baseline conditioned only on long-run mean annual rain (SNAP in the
  Northern Kenya project's VM0032 baseline) does with year-to-year rain. Identical to inverting
  M0.
- L, linear (M1, a 16-day RESTREND analogue): invert M1 to standardised residuals, re-run under
  r' with them fixed.
- S, S-curve (M1.5): the same with M1.5.
- M2, the paper's estimator (E1): the same with M2. Secondary: M2 with the additive residual (E2).
All fitted models are the fair-fight parameters; nothing is refitted to the synthetic data.

## Outcomes
- Relative error on managed pixels: box mean of (CF_hat - CF_true) over managed pixels, divided
  by the true management effect under normal rain (box mean over managed pixels of
  CF_true - CF_true without management). S1, S2a, S2b.
- Invented management on unmanaged pixels: box mean of (CF_hat - CF_true) over unmanaged pixels,
  in NDVI and as a share of that season's |true management effect|. All scenarios; in S0 over
  all pixels, in NDVI.
- Summaries: mean signed and mean absolute relative error, drought seasons (CHIRPS departure
  over the six windows < -60 mm) and other seasons separately, per world, scenario and
  estimator. Per-season values reported.

## Criteria
- P1 (primary): in S1, drought seasons, M2's mean absolute relative error on managed pixels is
  lower than Z's, in both worlds.
- P2: in S1, drought seasons, M2's mean absolute relative error is lower than both L's and S's,
  in both worlds.
- Reported without a criterion: other seasons, S0, S2a, S2b, invented management, M2 E2.
A failure of P1 or P2 in either world is reported as the result.
