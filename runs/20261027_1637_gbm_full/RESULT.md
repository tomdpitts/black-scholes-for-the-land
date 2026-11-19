# Gradient-boosted trees with M4's inputs: result against PREREG.md

Run: `.venv/bin/python revision/baselines.py gbm_full runs/20261027_1637_gbm_full A_test B_test` (laptop CPU, low
power; fit 57 s, scoring about 58 min). Raw CRPS in metrics_gbm_full_A_test_B_test.json; skill against the suite's M0
(runs/20261022_1639_suite/skill.json, same starts and masks).

| Model | A 2019-25, 16 days | A 2019-25, 3 months | B 2019-25, 16 days | B 2019-25, 3 months |
|---|---|---|---|---|
| Trees, rain only (B1, before) | .046 | .222 | .003 | .197 |
| **Trees, M4's inputs (this run)** | **.083** | **.246** | **.047** | **.209** |
| M4 ensemble | .111 | .273 | .131 | .274 |

3-month CRPS, M4 ensemble / trees: A 0.0502 / 0.0521, B 0.0488 / 0.0532.

## Verdict
**M4 BETTER**: lower 3-month CRPS than the trees on both A_test and B_test. Evaporative demand and topography lift the
trees by 0.024 (A) and 0.012 (B) at 3 months, and the gap to M4 narrows on A (0.051 to 0.027) and B (0.077 to 0.065).
As pre-registered, this run replaces B1 as the paper's trees row.
