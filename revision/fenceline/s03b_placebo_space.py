"""Step 03b (Amendment 2): replacement placebo-in-space design, built before the freeze.
The registered S4 (translations of the treated mask, 80% inside the control pool) is infeasible:
0 of 20,000 draws qualified (log, 03_design). Replacement: random straight-line splits of the
control pool. One side (>= 3 px from the line) is fake-treated, the other side (>= 3 px) is its
fake control pool; matching as in the final design (k=5, caliper 0.5, same standardisation).
No outcome is read."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C
import numpy as np
from scipy.spatial import cKDTree

C.start("03b_placebo_space", __file__)
assert not os.path.exists(C.out("outputs", "outcomes.npz")), "outcomes exist: design must precede outcomes"
dz = np.load(C.out("design", "design.npz"), allow_pickle=True)
meta = json.load(open(C.out("design", "design_meta.json")))
K, CAL, BUF, SEED, NKEEP, MINM = 5, meta["caliper"], 3, 20171, 200, 100
treated = dz["treated_id"] >= 0
pool = dz["control_pool"]
cov = dz["cov"]
elig = treated | pool
Z = ((cov - cov[elig].mean(0)) / cov[elig].std(0)).reshape(-1, cov.shape[-1])
ii, jj = np.mgrid[:C.N, :C.N]
pi = np.flatnonzero(pool.ravel())
rng = np.random.default_rng(SEED)
kept_params, kept_t, kept_m, draws = [], [], [], 0
while len(kept_params) < NKEEP and draws < 5000:
    draws += 1
    th = rng.uniform(0, np.pi)
    s = (jj * np.cos(th) + ii * np.sin(th)).ravel()
    c = rng.uniform(np.percentile(s[pi], 30), np.percentile(s[pi], 70))
    ft = pi[s[pi] < c - BUF]
    fp = pi[s[pi] > c + BUF]
    if len(ft) < MINM or len(fp) < 5 * K:
        continue
    d, idx = cKDTree(Z[fp]).query(Z[ft], k=K)
    keep = d[:, 0] <= CAL
    if keep.sum() < MINM:
        continue
    kept_params.append((float(th), float(c))); kept_t.append(ft[keep]); kept_m.append(fp[idx[keep]])
out = C.out("design", "s4_line_splits.npz")
assert not os.path.exists(out)
np.savez_compressed(out, params=np.array(kept_params), t=np.array(kept_t, dtype=object), m=np.array(kept_m, dtype=object))
C.log("03b_placebo_space", "S4 replacement: random line splits of the control pool", seed=SEED, kept=len(kept_params),
      draws=draws, caliper=CAL, min_matched=MINM,
      median_fake_treated_matched=int(np.median([len(t) for t in kept_t])) if kept_t else 0,
      file=os.path.relpath(out, C.ROOT), sha256=C.sha256(out))
