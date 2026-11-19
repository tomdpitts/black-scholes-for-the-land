"""Re-run the pre-registered Delta tests (greekstest.py) with starts whose two-step outcome
window spans a MODIS gap removed. Same params, same bootstrap seed; only the start list changes.

  .venv/bin/python revision/greeks_nogap.py   -> runs/<ts>_greekstest_nogap/metrics.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import greekstest as G  # noqa: E402
from run import run_dir  # noqa: E402

_orig = G.starts


def starts_nogap(d, years):
    dt = d["dates"].astype("datetime64[D]")
    gap = np.concatenate([np.diff(dt).astype(int) > 16, [False]])
    return [t for t in _orig(d, years) if not gap[t:t + G.H2].any()]


if __name__ == "__main__":
    G.starts = starts_nogap
    G.main("runs/20260914_1708_pipeline_h100_fairfight/params.npy", run_dir("greekstest_nogap"))
