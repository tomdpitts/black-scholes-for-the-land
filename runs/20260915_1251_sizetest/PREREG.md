# Does Delta work for small rain surprises and fail for large ones? Pre-registered before running.

Written 2026-09-15 after greekstest (runs/20260915_1228_greekstest) and an exploratory per-year look,
before any size-split result. Hypothesis from finance: Delta is a local slope, so it should rank exposure
when rain is near normal and lose that when rain departs far from normal.

Setup identical to greekstest: saved fair-fight M2, Delta along climatological rain from the state at t,
held-out 2019-25 MAM+OND starts, y = a_{t+2} - a_t, x = r_{t+1} - rclim_{t+1}, both ends observed.
Delta tercile cut-points computed once on all held-out pixel-steps of the box (Delta only).
Size split: pixel-steps with |x| at or below the box's median |x| are SMALL, above are LARGE.
Within each tercile and size half: slope of y on x controlling for a_t. Gap = top-tercile slope minus
bottom-tercile slope. Bootstrap over years (7), 2000 draws, same draws for both halves.

## Primary, both boxes must pass
S1 raw Delta ranking: gap(SMALL) - gap(LARGE) > 0, 95% interval above zero, on Box A AND Box B.

## Secondary
S2 gap(SMALL) > 0 with 95% interval above zero, each box.
S3 same as S1 for within-pixel (demeaned) Delta terciles, each box.

## Reported
gap(LARGE) with interval; slopes per tercile and half; median |x| per box; model Delta mean per tercile.
A failure of S1 means the local-tool reading of the Box A failure is not supported.
