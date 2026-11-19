# Size split: result against PREREG.md

Primary S1 (Delta ranks exposure better on small rain surprises than large, both boxes): FAIL on both.
The local-tool reading of Box A's failure is NOT supported.

Raw Delta terciles; gap = top-third minus bottom-third real slope (NDVI per 100 mm), 95% year-bootstrap interval.

| Box | Median surprise | Small surprises: gap [CI] | Large surprises: gap [CI] | Small minus large [CI] |
|---|---|---|---|---|
| A | 30 mm | +.051 [-.060, +.242] | +.044 [+.009, +.086] | +.008 [-.102, +.176] FAIL |
| B | 21 mm | +.154 [+.006, +.287] | +.070 [+.010, +.112] | +.084 [-.060, +.221] FAIL |

Within-pixel version (S3): A small +.019 [-.138, +.243], large +.026 [-.017, +.070]; B small +.169 [+.030, +.293],
large +.070 [+.005, +.115]; neither difference clears zero.

Reading: on large surprises Delta's ranking holds on both boxes (reported, not a pre-registered pass). Small
surprises carry too little rain variation to measure a slope, so their intervals are wide; this test had low power
for the small half, a design weakness that should have been anticipated. Box A's 2021 failure is not explained by
surprise size.
