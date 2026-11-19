# Does removing rain get closer to the management effect than baselines do? Result

Pre-registration: PREREG.md (sha256 in PREREG.sha256, hashed before the script was written).
Script: revision/attribution.py (B1 fit factored out of revision/baselines.py as `fit_gbm`, unchanged).
Numbers: metrics.json; per-season S1 values in run.log. GBM refit reproduces the B1 run
(600/300 iterations, variance scale 10.125). The W1 M2 row reproduces the identification test
(+18 % in drought seasons), so the two scripts agree.

**Verdict: P1 and P2 pass in both worlds.** Removing rain beats ignoring it in both worlds. How much
better depends on whether the rain model is right: about tenfold when it is, about a third better
than the z-score when it is not.

## S1, rain-independent management (true effect -0.027 to -0.030 NDVI at season end)
Mean absolute error as a share of the true management effect; signed mean in brackets.

| World | Seasons | Z z-score | L linear | S S-curve | **M2** | M2 additive |
|---|---|---|---|---|---|---|
| W1 M2 | drought (4) | 2.26 (+2.26) | 1.80 (+1.80) | 1.87 (+1.87) | **0.18** (+0.18) | 0.43 (+0.08) |
| W1 M2 | other (9) | 2.71 (-1.86) | 1.66 (-0.73) | 1.17 (-0.25) | **0.03** (+0.01) | 0.17 (+0.09) |
| W2 GBM | drought (4) | 1.37 (+1.00) | 1.14 (+0.66) | 1.22 (+0.73) | **0.97** (-0.91) | 1.40 (-1.10) |
| W2 GBM | other (9) | 2.23 (-2.16) | 1.39 (-1.06) | **1.09** (-0.61) | 1.34 (-0.32) | 1.24 (-0.25) |

Invented management on unmanaged pixels, as a share of the true effect, S1 drought: W1 Z 2.28,
L 1.73, S 1.80, M2 0.02; W2 Z 1.41, L 1.12, S 1.20, M2 1.03.
S0 (no management), mean |error| in NDVI, drought / other: W1 Z .059/.074, L .045/.042,
S .047/.029, M2 .000/.000; W2 Z .041/.065, L .031/.039, S .034/.030, M2 .029/.034.
S2a and S2b rank the estimators the same way in drought seasons in both worlds (metrics.json).

## Reading
1. **Ignoring rain is worst.** The z-score, which is also how a baseline on long-run mean annual
   rain (SNAP in VM0032) treats a given year's rain, overstates management by about 2.3 times its
   true size in droughts in the M2 world and 1.4 times in the GBM world, and calls rain wet-season
   greening negative management in wet seasons (-1.9, -2.2).
2. **With the right rain model, the residual recovers management.** In W1, M2 is within 18 % in
   droughts and 3 % otherwise, and invents almost none on untouched land; linear and S-curve rain
   models remove part of the rain and still overstate by 1.8 times.
3. **With the wrong rain model, every estimator is off by about the size of the effect.** In W2,
   M2 is best in droughts (0.97 vs 1.14-1.37) but understates rather than overstates, and the
   S-curve edges it in other seasons (1.09 vs 1.34). In S0 every estimator invents 0.03-0.07 NDVI
   of management, as large as the injected effect. At single-season box scale the residual's
   accuracy is limited by how well the rain response is known.
4. **Late rain drives the disagreement between worlds.** MAM2019 and OND2022 were dry overall but
   wet in their late windows (MAM2019 windows 4-6: 104/46/60 mm against 80/24/18 normal). GBM
   greens more from late rain than M2, so the worlds disagree on the sign of the rain effect at
   season end in those seasons.

## Caveats
W1 favours M2 by construction; W2 is the fairer test and the margin there is modest (4 drought
seasons). Management is a uniform -0.01 NDVI per window on random 2 km blocks; real management is
patchier and may follow rain (S2a). The SNAP equivalence holds only for year-to-year rain: SNAP
models soil carbon from grazing intensity, not an NDVI counterfactual.
