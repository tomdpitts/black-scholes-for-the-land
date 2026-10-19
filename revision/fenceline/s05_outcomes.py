"""Step 05: outcomes. Runs only after FREEZE.json, and verifies it first.
For every Box-B season (MAM, H=6; Jan-Feb, H=4) computes, per pixel:
  R   = a(t0+H) - F_H,  F = M2 mean rollout from the observed pre-season state along observed rain
  S7  = sum of one-step residuals a(t+1) - mu_t over the same windows (NDVI units)
plus per-season validity masks (burn window, observed target), gap flags and rain departure.
Writes RUN/outputs/outcomes.npz once. No statistics here."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C

C.start("05_outcomes", __file__)
C.verify_freeze("05_outcomes")
sys.path.insert(0, C.ROOT)
import numpy as np, jax, jax.numpy as jnp
import model as M
from run import load, season_of

d = load("B")
params = np.load(os.path.join(C.MODEL_RUN, "params.npy"), allow_pickle=True).item()
p = {k: jnp.asarray(v) for k, v in params["M2"].items()}
D = {k: jnp.asarray(d[k]) for k in ("a", "r", "rclim", "x", "phase")}
w = M.w_path(p, D)
x = D["x"]
dates = d["dates"].astype("datetime64[D]")
T = len(dates)


@jax.jit
def roll_zero(a0, w0, r_path, ph_path):
    def body(c, inp):
        a, wv = c
        r, ph = inp
        mu, sig, wv = M.step(p, a, wv, r, ph, x, True)
        return (mu, wv), mu
    _, path = jax.lax.scan(body, (a0, w0), (r_path, ph_path))
    return path


@jax.jit
def one_step(t):
    mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], x, True)
    return D["a"][t + 1] - mu


def starts(kind):
    out = []
    for y in range(2003, 2026):
        if kind == "MAM":
            c = [t for t in range(T - 1) if int(d["year"][t + 1]) == y and season_of(d["doy"][t + 1]) == "MAM"
                 and season_of(d["doy"][t]) != "MAM"]
            H = 6
        else:                                              # Jan-Feb: from the last composite of y-1
            c = [t for t in range(T - 1) if int(d["year"][t]) == y - 1 and int(d["year"][t + 1]) == y]
            H = 4
        if c and c[0] + H < T:
            out.append((y, c[0], H))
    return out


res = {}
for kind in ("MAM", "JF"):
    rows = starts(kind)
    R, S7, valid, gap, dep, t0s, rmap = [], [], [], [], [], [], []
    for (y, t0, H) in rows:
        a0, w0 = D["a"][t0], w[t0]
        F = np.asarray(roll_zero(a0, w0, D["r"][t0 + 1:t0 + 1 + H], D["phase"][t0 + 1:t0 + 1 + H]))
        R.append(np.asarray(d["a"][t0 + H]) - F[-1])
        S7.append(np.sum([np.asarray(one_step(t)) for t in range(t0, t0 + H)], axis=0))
        burn = d["burn"][max(t0 - 6, 0):t0 + H + 1].any(0)
        valid.append(d["obs"][t0 + H] & ~burn)
        gap.append(bool((np.diff(dates[t0:t0 + H + 1]).astype(int) > 16).any()))
        rd = (d["r"][t0 + 1:t0 + 1 + H] - d["rclim"][t0 + 1:t0 + 1 + H]).sum(0) * 100.0
        rmap.append(rd); dep.append(float(rd.mean())); t0s.append(t0)
    res[kind] = dict(years=np.array([r[0] for r in rows]), t0=np.array(t0s), H=rows[0][2],
                     R=np.stack(R).astype(np.float32), S7=np.stack(S7).astype(np.float32),
                     valid=np.stack(valid), gap=np.array(gap), rain_dep_mm=np.array(dep),
                     rain_dep_map=np.stack(rmap).astype(np.float32),
                     start_dates=np.array([str(dates[t]) for t in t0s]))
    C.log("05_outcomes", f"{kind} seasons computed", n=len(rows),
          years=res[kind]["years"].tolist(), gap_flagged=res[kind]["years"][res[kind]["gap"]].tolist(),
          start_dates=res[kind]["start_dates"].tolist())

# sanity: confirm the MAM 2017 window dates
y17 = int(np.where(res["MAM"]["years"] == 2017)[0][0]); t0 = int(res["MAM"]["t0"][y17])
C.log("05_outcomes", "sanity: MAM 2017 start", start=str(dates[t0]), first_window=str(dates[t0 + 1]),
      last_window=str(dates[t0 + 6]))
fo = C.out("outputs", "outcomes.npz")
assert not os.path.exists(fo)
np.savez_compressed(fo, **{f"{k}_{q}": v for k, r in res.items() for q, v in r.items()})
C.log("05_outcomes", "outcomes saved", file=os.path.relpath(fo, C.ROOT), sha256=C.sha256(fo))
