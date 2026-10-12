"""GBM vs M2 mean-rollout diagnostic with the number of greenness lags set by LAGS_A (1 = Markov in greenness).
bias and MAE of the box-wide pixel error by horizon. Post hoc diagnostic, no pass/fail.

  LAGS_A=1 .venv/bin/python revision/gbm_diag_lags.py   -> runs/20261006_1759_baselines/diag_lagsA1.json
"""
import json
import os
import sys

import jax.numpy as jnp
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aoi  # noqa: E402
import model as M  # noqa: E402
from run import load, H  # noqa: E402
import baselines as BL  # noqa: E402
BL.LAGS_A = int(os.environ.get("LAGS_A", 3))
from baselines import gbm_features, nbhd, evalsets, starts_of  # noqa: E402
LAGS_A = BL.LAGS_A

A, B = load("A"), load("B")
rng = np.random.default_rng(0)
nA = nbhd(A["a"])
tgt = A["year"][1:]
T = A["a"].shape[0]
X, Y = [], []
tr = [t for t in range(2, T - 1) if tgt[t] in aoi.TRAIN_YEARS]
per = 3_000_000 // len(tr)
for t in tr:  # identical sampling to baselines.gbm
    idx = np.where((A["obs"][t] & A["obs"][t + 1]).ravel())[0]
    idx = rng.choice(idx, min(per, len(idx)), replace=False)
    X.append(gbm_features([A["a"][max(t - k, 0)] for k in range(LAGS_A)], nA[t], A, t + 1, A["x"])[idx])
    Y.append((A["a"][t + 1] - A["a"][t]).ravel()[idx])
mean = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                                     l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                     n_iter_no_change=20, random_state=0).fit(np.concatenate(X), np.concatenate(Y))
P = np.load("runs/20260914_1708_pipeline_h100_fairfight/params.npy", allow_pickle=True).item()
p2 = {k: jnp.asarray(v) for k, v in P["M2"].items()}
out = {}
for es, (D, mt) in evalsets(A, B).items():
    J = {k: jnp.asarray(D[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    w = M.w_path(p2, J)
    acc = {m: {"bias": np.zeros(H), "mae": np.zeros(H), "persist_mae": np.zeros(H), "n": np.zeros(H)} for m in ("M2", "GBM")}
    for t in starts_of(D, mt):
        ob = D["obs"][t + 1:t + 1 + H] & D["obs"][t][None]
        y = D["a"][t + 1:t + 1 + H]
        m2 = np.asarray(M.rollout_mean(p2, J["a"][t], w[t], J["r"][t + 1:t + 1 + H], J["phase"][t + 1:t + 1 + H], J["x"], True))
        lags = [D["a"][max(t - k, 0)] for k in range(LAGS_A)]
        g = []
        for h in range(H):
            new = lags[0] + mean.predict(gbm_features(lags, nbhd(lags[0]), D, t + 1 + h, D["x"])).reshape(lags[0].shape)
            lags = [new] + lags[:-1]
            g.append(new)
        for nm, pr in (("M2", m2), ("GBM", np.stack(g))):
            for h in range(H):
                e = (pr[h] - y[h])[ob[h]]
                acc[nm]["bias"][h] += e.sum()
                acc[nm]["mae"][h] += np.abs(e).sum()
                acc[nm]["n"][h] += ob[h].sum()
    out[es] = {nm: {k: (v / a["n"]).round(4).tolist() for k, v in a.items() if k in ("bias", "mae")} for nm, a in acc.items()}
    print(es, json.dumps(out[es]), flush=True)
json.dump(out, open(f"runs/20261006_1759_baselines/diag_lagsA{LAGS_A}.json", "w"), indent=1)
