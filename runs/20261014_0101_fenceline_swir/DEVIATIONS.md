# Deviations

## D1. s03_analysis.py: void handling for Q6 (post-freeze bug fix, 2026-10-13T14:3xZ)

On first run, s03 crashed at Q6 (`ValueError: zero-size array`). By then Q1–Q5 had already been computed and appended to `log.jsonl`; their values are unchanged by this fix.

**Cause.** Aqua (MYD09A1) fails the pre-registered data rule in every year, including 2017: the box-wide median of valid window composites is 1–2, below the 3 required. Aqua MODIS band 6 (1.64 µm) has mostly non-functional detectors, a known hardware fault, so NDTI is rarely valid. PREREG already says Q6 is void when 2017 fails the rule. The code computed ranks on an empty placebo set instead of declaring Q6 void.

**Fix.** Q6 is reported as void when 2017 or the placebo set fails the data rule. No other line changes. Results are written once, on the corrected run.
