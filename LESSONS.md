# Lessons carried over (measured facts only)

From two audits of learned land-surface dynamics, 2026-09: one on annual 64-d satellite
embeddings (AlphaEarth, 2017–2025, 400 global sites), one on monthly Dynamic World class
probabilities (2016–2025, same sites). These are things that were measured, not
recommendations. Re-measure them on the new domain before relying on any of them.

## Substrate facts

1. Annual cadence carries almost no learnable dynamics. Temporal variance was 6.5 % of
   total; 83 % of year-to-year change was observation-like noise; the best predictor
   was a two-parameter shrink-to-running-mean filter, and no learned model beat it on
   unseen sites. Forensic dating and attribution on annual data failed against raw
   change magnitude.
2. Monthly cadence does carry dynamics. Temporal variance 29 % of total; 65 % of that was
   the pixel's own seasonal cycle; the anomaly had autocorrelation 0.41 at one month and
   0.03 at twelve; 70 % of month-to-month anomaly change was noise. A learned model beat
   the seasonal filter by 0.07 skill and a linear AR(12) by 0.04 on unseen sites.
   Rollouts beat climatology for about three months, then converged to it.
3. About half of a pixel's change (59 % annual, 52 % monthly) was shared with its 1.3 km
   neighbourhood. Knowing the neighbourhood's realised change would roughly double the
   skill (annual oracle 0.59–0.70 vs 0.30 achievable from history). Its forecastable
   part from history alone was small.
4. Annual weather aggregates (ERA5-Land rainfall, temperature, soil water, radiation)
   explained about 3 % of the annual shock beyond mean reversion. Monthly features did
   not change that at annual cadence. Driver conditioning at annual cadence is not
   informative; this was never tested at monthly cadence with a monthly target.
5. Optical monthly coverage: median site observed 78 % of months; tropics 64 %; 11 % of
   sites had no pixel observed in half the months. Coverage selection biases the sample
   toward drylands and mid-latitudes.
6. Change vectors are anti-persistent (lag-1 cosine −0.24 to −0.38). "Continue the
   trend" baselines were negative or near zero at both cadences. No multi-year trend
   structure was found at the pixel level in nine years.
7. Class-probability outputs are soft (mean max probability 0.51). Label-level Markov
   chains scored no better than persistence against them.

## Modelling facts

8. A learned model initialised exactly at the best simple filter (zero-initialised
   residual on a fitted prior) was the difference between negative and positive skill on
   unseen sites at annual cadence. At monthly cadence, with ten times more transitions
   per trajectory, the prior stopped mattering for one-step skill but still mattered
   for rollouts beyond one month.
9. Early stopping on likelihood selected checkpoints with lower point skill than early
   stopping on error, consistently. Keep both and report both.
10. Multiple possibilities (a mixture) improved likelihood and energy score and
    spread–skill correlation (0.55 annual, 0.74 monthly) without changing point skill.
    The mean of imagined rollouts equalled the deterministic mean rollout to within 0.01
    at every horizon; the value of sampling was the spread, never the point.
11. Learned carried memory added nothing over a running mean at annual cadence and
    +0.02 at monthly cadence.
12. A ConvLSTM trained from scratch on patches scored below the seasonal filter.
13. Free (unstructured) models memorised site-year shocks and did not transfer to unseen
    sites at annual cadence; site-level k-fold with unseen sites and unseen years was
    the only split that exposed this.

## Process facts

14. Testing attribution or dating before establishing that the null explains ordinary
    variation better than a trivial baseline cost two weeks and produced nothing.
15. Globally scattered sites made neighbourhood and regional structure untestable.
16. Eight annual steps gave three forecast horizons; the entire multi-step story was
    statistically thin. Cadence determines what can be asked.
