"""GBM, Markov in greenness, with rain history given as M2's one-parameter bucket w_t instead of
six rain lags. Isolates whether GBM's Box B edge over M2 is the richer rain memory or the tree
function class. Post hoc diagnostic, no pass/fail.

  .venv/bin/python revision/gbm_diag_bucket.py   -> runs/20261006_1759_baselines/diag_bucket.json
"""
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines as BL  # noqa: E402
import model as M  # noqa: E402
from run import load  # noqa: E402

P = np.load("runs/20260914_1708_pipeline_h100_fairfight/params.npy", allow_pickle=True).item()
p2 = {k: jnp.asarray(v) for k, v in P["M2"].items()}
W = {}
for box in "AB":
    d = load(box)
    W[box] = np.asarray(M.w_path(p2, {k: jnp.asarray(d[k]) for k in ("r", "rclim", "x")}))
    W[box + "_shape"] = d["a"].shape

_orig = BL.gbm_features
BL.LAGS_R = 1  # r_{t+1} only


def feats(a_lags, n, D, t_next, x):
    box = "A" if D is _A else "B"
    f = _orig(a_lags, n, D, t_next, x)
    w = np.broadcast_to(W[box][t_next - 1], a_lags[0].shape).reshape(-1, 1)
    return np.concatenate([f, w.astype(np.float32)], 1)


# patch load so the diagnostic's A and B dicts are identifiable
import run  # noqa: E402
_load = run.load
_A = None


def load_tag(box, *a, **k):
    global _A
    d = _load(box, *a, **k)
    if box == "A":
        _A = d
    return d


run.load = load_tag
BL.gbm_features = feats
os.environ["LAGS_A"] = "1"
src = open(os.path.join(os.path.dirname(__file__), "gbm_diag_lags.py")).read()
src = src.replace("from run import load, H", "from run import H\nload = run.load")
src = src.replace("from baselines import gbm_features, nbhd, evalsets, starts_of", "from baselines import nbhd, evalsets, starts_of\ngbm_features = BL.gbm_features")
src = src.replace('diag_lagsA{LAGS_A}.json', 'diag_bucket.json')
exec(compile(src, "gbm_diag_lags.py", "exec"), {"__name__": "__main__", "__file__": __file__, "run": run, "BL": BL})
