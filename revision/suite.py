"""The AgriAI test suite (revision/TESTS.md): every cell of the cast except Chronos-2's, the scoring of Chronos-2's
outputs (revision/c2.py), the timing pass and the tables. Pre-registration: runs/20261022_1639_suite/PREREG.md.

  .venv/bin/python revision/suite.py plan <out> [--debug]     -> plan.npz: samples, starts, seasons, mask (c2.py reads it)
  .venv/bin/python revision/suite.py skill <out> [--debug]    -> skill.json      test 1
  .venv/bin/python revision/suite.py attrib <out> [--debug] [--tile L]
                                                              -> attrib_m4.json  test 2, M4 world (+ the tile, descriptive)
                                                                 attrib_c2.json  the Chronos-2 world, if <out>/c2_world.npz
  .venv/bin/python revision/suite.py lean <out> [--debug]     -> lean.json       test 2a, weather lean (addendum 3)
  .venv/bin/python revision/suite.py drought <out> [--debug]  -> drought.json    test 3
  .venv/bin/python revision/suite.py sens <out> [--debug]     -> sens.json       test 4 (+ Chronos-2 if <out>/c2_sens.npz)
  .venv/bin/python revision/suite.py timing <out>             -> timing.json     compute
  .venv/bin/python revision/suite.py report <out>             -> tables.md       Status table, detail tables, claims
  .venv/bin/python revision/suite.py summary <out>            -> summary.md      readable tables, one M4 row (the ensemble)
--debug: 2015-17 starts and seasons (in-sample; for building and testing only).
"""
import functools
import json
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aoi  # noqa: E402
import model as M  # noqa: E402
import greekstest as G  # noqa: E402
import m3  # noqa: E402
from run import load, H  # noqa: E402
from residual import season_starts  # noqa: E402
from identify import DELTA, block_mask, make_fns  # noqa: E402
from attribution import management  # noqa: E402
from greeks_nogap import starts_nogap  # noqa: E402
from m3_attribution import rerun, implied, season_split  # noqa: E402

GPU = "runs/20261021_0147_m3/gpu"
FF = "runs/20260914_1708_pipeline_h100_fairfight/params.npy"
CONFIG = {"M2": "base", "M3": "topo", "M4": "topo_pet"}
SEEDS = range(5)
TEST, DEBUG = range(2019, 2026), range(2015, 2018)
DRY = -60.0  # mm over six windows: drought start / season
SCEN = {"S0": (0.0, 0.0), "S1": (0.0, DELTA), "S2a": (1.0, DELTA), "S2b": (-1.0, DELTA)}
NPX = 4000
N = aoi.N


# ----------------------------------------------------------------------------- the cast
@functools.lru_cache(None)
def data(box, model):
    """(numpy dict, jax dict) a model reads: model.py layout for M0/M1, m3.load_box for M2-M4."""
    if model in ("M0", "M1"):
        d = load(box)
        return d, {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
    topo, temp = m3.CONFIGS[CONFIG[model]][:2]
    d = m3.load_box(box, topo, temp)
    return d, m3.jaxify(d)


@functools.lru_cache(None)
def params(model, seed=0):
    P = (np.load(FF, allow_pickle=True).item()[model] if model in ("M0", "M1")
         else np.load(f"{GPU}/params_{CONFIG[model]}_s{seed}.npy", allow_pickle=True).item())
    return {k: jnp.asarray(v) for k, v in P.items()}


def stepper(model, box, seed=0):
    """(step_fn(a, w, t_next) -> (mean, sigma, next w), w path), as m3.score takes them."""
    _, D = data(box, model)
    p = params(model, seed)
    if model in ("M0", "M1"):
        u = model == "M1"
        return (lambda a, w, t: M.step(p, a, w, D["r"][t], D["phase"][t], D["x"], u)), M.w_path(p, D)
    return m3.m3_step_fn(p, D), m3.w_path(p, D)


def r6(d):
    """Rain departure summed over windows t+1..t+H (100 mm), [T-H,N,N]."""
    c = np.concatenate([np.zeros((1,) + d["r"].shape[1:]), np.cumsum(d["r"] - d["rclim"], 0, dtype=np.float64)])
    return c[H + 1:] - c[1:-H]


@functools.lru_cache(None)
def restrend(box):
    """Per-pixel OLS a_{t+H} = alpha + beta R6 on Box A training starts (target year 2003-17, both ends observed);
    spread = residual sd. Box B is never fitted: its alpha, beta and spread come from Box A linear regressions of
    each on the three standardised traits."""
    d = load("A")
    R, y = r6(d), d["a"][H:].astype(np.float64)
    ok = d["obs"][:-H] & d["obs"][H:] & np.isin(d["year"][H:], list(aoi.TRAIN_YEARS))[:, None, None]
    n = ok.sum(0)
    mx, my = (R * ok).sum(0) / n, (y * ok).sum(0) / n
    beta = ((R - mx) * (y - my) * ok).sum(0) / ((R - mx) ** 2 * ok).sum(0)
    alpha = my - beta * mx
    f = {"alpha": alpha, "beta": beta, "sd": np.sqrt(((y - alpha - beta * R) ** 2 * ok).sum(0) / (n - 2))}
    if box == "B":
        X = [np.c_[np.ones(N * N), m3.load_box(b, False, None)["s"].reshape(-1, 3)] for b in "AB"]
        f = {k: (X[1] @ np.linalg.lstsq(X[0], v.ravel(), rcond=None)[0]).reshape(N, N) for k, v in f.items()}
        f["sd_clipped_px"] = int((f["sd"] < 1e-3).sum())
        f["sd"] = np.maximum(f["sd"], 1e-3)
    return f


def score_restrend(box, mask_t, px=None):
    """3-month Gaussian CRPS of RESTREND on m3.score's starts and masks."""
    d, f = load(box), restrend(box)
    R = r6(d)
    pm = np.ones((N, N), bool) if px is None else px
    num, den = 0.0, 0
    for t in [t for t in range(len(R)) if mask_t[t]]:
        ob = d["obs"][t] & d["obs"][t + H] & pm
        c = np.asarray(M.crps_gauss(f["alpha"] + f["beta"] * R[t], f["sd"], d["a"][t + H]))
        num, den = num + float(c[ob].sum()), den + int(ob.sum())
    return num / den


def npx(px):
    m = np.zeros(N * N, bool)
    m[px] = True
    return m.reshape(N, N)


# ----------------------------------------------------------------------------- plan
def plan(out, debug):
    """Everything the pre-registration fixes by rule: samples, starts, drought flags, seasons, mask."""
    years = DEBUG if debug else TEST
    n = int(sys.argv[sys.argv.index("--npx") + 1]) if debug and "--npx" in sys.argv else NPX  # smaller sample: debug only
    P = {"debug": debug, "years": np.array(list(years)), "px": np.sort(np.random.default_rng(0).choice(N * N, NPX, replace=False)[:n]),
         "mask": block_mask(N, 8, np.random.default_rng(0))}
    for box in "AB":
        d = load(box)
        T = d["a"].shape[0]
        dep = r6(d).mean((1, 2)) * 100
        st = [t for t in range(T - H) if int(d["year"][t + 1]) in years]
        P[f"{box}_skill_starts"], P[f"{box}_skill_dry"] = np.array(st), dep[st] < DRY
        st = G.starts(d, set(years))
        P[f"{box}_sens_starts"], P[f"{box}_sens_nogap"] = np.array(st), np.isin(st, starts_nogap(d, set(years)))
    d = load("A")
    ss = season_starts(d, years)
    dep = r6(d).mean((1, 2)) * 100
    P["season_t0"] = np.array([t for _, _, t in ss])
    P["season_name"] = np.array([f"{s}{y}" for y, s, _ in ss])
    P["season_dep"] = dep[P["season_t0"]]
    # management of every season and scenario, real and normal weather (attribution.management), for c2.py
    P["mgmt_real"] = np.stack([[management(d["r"][t + 1:t + 1 + H], d["rclim"][t + 1:t + 1 + H], k, dl, P["mask"])
                                for k, dl in SCEN.values()] for t in P["season_t0"]]).astype(np.float32)
    P["mgmt_normal"] = np.stack([[management(d["rclim"][t + 1:t + 1 + H], d["rclim"][t + 1:t + 1 + H], k, dl, P["mask"])
                                  for k, dl in SCEN.values()] for t in P["season_t0"]]).astype(np.float32)
    if debug:  # test 3 stand-in: the driest 2003-17 long-rains season
        mam = [(dep[t], t, f"MAM{y}") for y, s, t in season_starts(d, aoi.TRAIN_YEARS) if s == "MAM"]
        _, P["t0_2022"], P["name_2022"] = min(mam)
    else:
        P["t0_2022"], P["name_2022"] = [t for y, s, t in ss if (y, s) == (2022, "MAM")][0], "MAM2022"
    np.savez(f"{out}/plan.npz", **P)
    print("plan:", {k: (v.shape if hasattr(v, "shape") and np.ndim(v) else v) for k, v in P.items()})
    return P


def get_plan(out):
    return dict(np.load(f"{out}/plan.npz"))


# ----------------------------------------------------------------------------- test 1: skill
def skill(out, debug):
    P = get_plan(out)
    years = set(P["years"].tolist())
    px = npx(P["px"])
    res = {}
    if not debug:  # reproduce one existing number before the first 2019-25 score
        ref = json.load(open(f"{GPU}/metrics_topo_pet_s0.json"))["A_test"]
        dA = data("A", "M4")[0]
        r = m3.score(*stepper("M4", "A", 0), data("A", "M4")[1], np.isin(dA["year"][1:], list(years)))
        diff = max(abs(r["onestep_crps"] - ref["onestep_crps"]), abs(r["rollout_crps"][-1] - ref["rollout_crps"][-1]))
        print(f"reproduce M4 s0 A_test: max |diff| {diff:.2e}", flush=True)
        assert diff < 1e-3, "reproduction check failed"
        res["reproduction_check"] = {"max_abs_diff": diff, "new": r, "existing": ref}
    for box in "AB":
        es = f"{box}_test"
        d = load(box)
        T = d["a"].shape[0]
        mt = np.isin(d["year"][1:], list(years))
        dry = np.zeros(T - 1, bool)
        dry[:T - H] = r6(d).mean((1, 2)) * 100 < DRY
        V = {"all": (mt, None), "drought": (mt & dry, None), "px": (mt, px), "px_drought": (mt & dry, px)}
        res[es] = {v: {"n_starts": int(V[v][0][:T - H].sum())} for v in V}
        cells = [("M0", 0, v) for v in V] + [("M1", 0, v) for v in ("all", "drought")]
        cells += [(m, s, v) for m in ("M2", "M3", "M4") for s in SEEDS for v in (V if debug else ("drought",))
                  if m == "M4" or v in ("all", "drought")]
        if not debug:
            cells += [("M4", s, v) for s in SEEDS for v in ("px", "px_drought")]
        for m, s, v in cells:
            t1 = time.time()
            r = m3.score(*stepper(m, box, s), data(box, m)[1], V[v][0], px=V[v][1])
            res[es][v][m if m in ("M0", "M1") else f"{m}_s{s}"] = {**r, "seconds": time.time() - t1}
            print(es, v, m, s, f"1-step {r['onestep_crps']:.5f} 3-month {r['rollout_crps'][-1]:.5f}", flush=True)
        ps = [params("M4", s) for s in SEEDS]
        for v in V:
            if v == "all" and not debug:
                continue
            r = m3.score_ensemble(ps, data(box, "M4")[1], V[v][0], px=V[v][1])
            res[es][v]["M4_ens"] = r
            print(es, v, "M4 ensemble", f"3-month {r['rollout_crps'][-1]:.5f}", flush=True)
            res[es][v]["RESTREND"] = {"rollout_crps": [None] * (H - 1) + [score_restrend(box, V[v][0], V[v][1])]}
        if not debug:  # existing full-grid numbers (TESTS.md: reuse)
            res[es]["all"]["RESTREND"] = {"rollout_crps": [None] * (H - 1) + [score_restrend(box, mt)]}
            for m in ("M2", "M3", "M4"):
                for s in SEEDS:
                    res[es]["all"][f"{m}_s{s}"] = {**json.load(open(f"{GPU}/metrics_{CONFIG[m]}_s{s}.json"))[es], "source": "existing"}
            res[es]["all"]["M4_ens"] = {**json.load(open(f"{GPU}/metrics_ens_topo_pet.json"))[es], "source": "existing"}
        res[es]["restrend_box_B_sd_clipped_px"] = restrend("B")["sd_clipped_px"] if box == "B" else None
        json.dump(res, open(f"{out}/skill.json", "w"), indent=1)
    print("wrote", f"{out}/skill.json")


# ----------------------------------------------------------------------------- test 2: attribution
def world(p, s, a0, w0, r, ex, ph, eps, m):
    """The M4 world: m3_attribution.rerun plus management m [H,N,N] added to each step's value before it is fed back."""
    def body(c, i):
        a, w_ = c
        mu, sg, w_, _ = m3.step(p, a, w_, r[i], ex[i], ph[i], s)
        a = mu + sg * eps[i] + m[i]
        return (a, w_), a
    return jax.lax.scan(body, (a0, w0), jnp.arange(H))[1]


world_j = jax.jit(world)


@jax.jit
def _normal(p, s, a0, w0, y, r, rc, ex, ph):
    """An M2-M4 estimator: invert the season y, re-run it under normal weather, season end."""
    eps = implied(p, s, a0, w0, y, r, ex, ph)[0]
    return rerun(p, s, a0, w0, rc, jnp.zeros_like(ex), ph, eps)[-1]


def estimators():
    """name -> f(t0, y [H,N,N]) -> the estimator's season-end greenness under normal weather [N,N]."""
    d1, D1 = data("A", "M1")
    R, beta = r6(d1), restrend("A")["beta"]
    E = {"z-score": lambda t0, y: np.asarray(y[-1]), "RESTREND": lambda t0, y: np.asarray(y[-1]) - beta * R[t0]}
    p1 = params("M1")
    _, inv1, cf1, _ = make_fns(p1, D1["x"])
    w1 = M.w_path(p1, D1)

    def m1(t0, y):
        sl = slice(t0 + 1, t0 + 1 + H)
        a0, ph = D1["a"][t0], D1["phase"][sl]
        return np.asarray(cf1(a0, w1[t0], D1["rclim"][sl], ph, inv1(a0, w1[t0], y, D1["r"][sl], ph)[0], False)[-1])
    E["M1"] = m1

    def m3est(p, D, w, t0, y):
        sl = slice(t0 + 1, t0 + 1 + H)
        return np.asarray(_normal(p, D["s"], D["a"][t0], w[t0], jnp.asarray(y), D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]))
    for m in ("M2", "M3", "M4"):
        D = data("A", m)[1]
        for s in SEEDS:
            p = params(m, s)
            E[f"{m}_s{s}"] = functools.partial(m3est, p, D, m3.w_path(p, D))
    return E


def add_ensembles(cf):
    cf["M4_ens"] = np.mean([cf[f"M4_s{s}"] for s in SEEDS], 0)
    cf["M4_ens14"] = np.mean([cf[f"M4_s{s}"] for s in SEEDS if s], 0)
    return cf


def metrics(cf, cf_true, cf_true0, managed, region):
    """Season metrics of each estimator's counterfactual on `region`."""
    mg, um = managed & region, ~managed & region
    eff = float((cf_true - cf_true0)[mg].mean())
    out = {"true_effect": eff}
    for k, c in cf.items():
        e = c - cf_true
        out[k] = {"err_managed": float(e[mg].mean()), "err_unmanaged": float(e[um].mean()), "err_all": float(e[region].mean()),
                  "px_rmse": float(np.sqrt((e[region] ** 2).mean())),
                  "rel": float(e[mg].mean() / eff) if abs(eff) > 1e-6 else None,
                  "invented": float(e[um].mean() / abs(eff)) if abs(eff) > 1e-6 else None}
    return out


def tile_of(L):
    o = (N - L) // 2
    t = np.zeros((N, N), bool)
    t[o:o + L, o:o + L] = True
    return t


DEP_BINS = [-np.inf, -60, -30, 0, 30, 60, 120, np.inf]   # identify.BINS, mm per window
ET0_BINS = [-np.inf, -10, -5, 0, 5, 10, np.inf]           # mm per window


def binned(z, dep, et0):
    """Mean standardised residual by the window's rain and ET0 departure (pixel-steps)."""
    out = {}
    for name, v, b in (("rain", dep, DEP_BINS), ("et0", et0, ET0_BINS)):
        k = np.digitize(v, b[1:-1])
        out[name] = {"edges_mm": [float(e) for e in b[1:-1]], "n": [int((k == i).sum()) for i in range(len(b) - 1)],
                     "mean": [float(z[k == i].mean()) if (k == i).any() else None for i in range(len(b) - 1)]}
    return out


def attrib(out, debug, L=None, worlds=(0,)):
    """Test 2 in M4 worlds, one per seed in `worlds` (addendum 2: seeds 0-4; the first run used seed 0 only)."""
    P = get_plan(out)
    mask = P["mask"]
    mb = mask.astype(bool)
    E = estimators()
    d4, D4 = data("A", "M4")
    s = D4["s"]
    e5 = np.load("data/A/era5.npz")
    pet_dep = (e5["pet"] - e5["pet_clim"]) * 100
    regions = {"full": np.ones((N, N), bool)} | ({"tile": tile_of(L)} if L else {})
    rows, Z, DEP, ET = [], [], [], []
    for ws in worlds:
        p = params("M4", ws)
        w = m3.w_path(p, D4)
        for t0, name, dep in zip(P["season_t0"].tolist(), P["season_name"], P["season_dep"]):
            sl = slice(t0 + 1, t0 + 1 + H)
            a0, w0 = D4["a"][t0], w[t0]
            r, rc, ex, ph = D4["r"][sl], D4["rclim"][sl], D4["ex"][sl], D4["phase"][sl]
            ex0, zero = jnp.zeros_like(ex), jnp.zeros((H, N, N))
            eps = implied(p, s, a0, w0, D4["a"][sl], r, ex, ph)[0]
            if ws == worlds[0]:
                Z.append(np.asarray(eps))
                DEP.append(np.asarray(r - rc) * 100)
                ET.append(pet_dep[sl])
            ct0 = np.asarray(world_j(p, s, a0, w0, rc, ex0, ph, eps, zero)[-1])
            rd = jnp.maximum(rc - 1 / 6, 0)
            meta = {"world_seed": ws, "season": str(name), "t0": t0, "rain_departure_mm": float(dep), "drought": bool(dep < DRY),
                    "difficulty": float(np.asarray(world_j(p, s, a0, w0, rd, ex0, ph, eps, zero)[-1]).mean() - ct0.mean()),
                    "difficulty_rain_departure_mm": float((rd - rc).mean((1, 2)).sum() * 100)}
            for sc, (k, delta) in SCEN.items():
                y = world_j(p, s, a0, w0, r, ex, ph, eps, jnp.asarray(management(np.asarray(r), np.asarray(rc), k, delta, mask)))
                ct = np.asarray(world_j(p, s, a0, w0, rc, ex0, ph, eps, jnp.asarray(management(np.asarray(rc), np.asarray(rc), k, delta, mask)))[-1])
                cf = add_ensembles({e: f(t0, y) for e, f in E.items()})
                rows.append({**meta, "scenario": sc, **{g: metrics(cf, ct, ct0, mb, reg) for g, reg in regions.items()}})
            r1 = rows[-3]["full"]
            print(f"world s{ws} {name} {dep:+.0f} mm S1 rel: " + " ".join(f"{e} {r1[e]['rel']:+.2f}" for e in ("z-score", "RESTREND", "M1", f"M4_s{ws}", "M4_ens")), flush=True)
    fn = "attrib_m4.json" if tuple(worlds) == (0,) else "attrib_m4x5.json"
    res = {"world": f"M4 seeds {list(worlds)}", "n_managed_px": int(mb.sum()), "tile_L": L, "rows": rows,
           "eps_binned": binned(np.concatenate(Z).ravel(), np.concatenate(DEP).ravel(), np.concatenate(ET).ravel())}
    json.dump(res, open(f"{out}/{fn}", "w"), indent=1)
    print("wrote", f"{out}/{fn}")
    if os.path.exists(f"{out}/c2_world.npz"):
        attrib_c2(out, P, E)


def attrib_c2(out, P, E):
    """Score every estimator in the Chronos-2 world: the tile is Chronos-2's, outside it every scenario is the real season."""
    W = np.load(f"{out}/c2_world.npz")
    L, o = int(W["L"]), int(W["offset"])
    tile = np.zeros((N, N), bool)
    tile[o:o + L, o:o + L] = True
    mb = P["mask"].astype(bool)
    dA = load("A")
    R6 = r6(dA).mean((1, 2)) * 100
    emb = lambda full, part: np.where(tile, np.pad(part, ((0, 0),) * (part.ndim - 2) + ((o, N - o - L),) * 2), full)
    rows = []
    for i, t0 in enumerate(W["season_t0"].tolist()):
        sl = slice(t0 + 1, t0 + 1 + H)
        dep = float(R6[t0])
        meta = {"season": str(W["season_name"][i]), "t0": t0, "rain_departure_mm": dep, "drought": bool(dep < DRY),
                "difficulty": float(W["difficulty"][i]), "difficulty_rain_departure_mm": float(W["difficulty_dep"][i]),
                "reconstruction_max_abs": float(W["recon"][i])}
        ct0 = emb(np.zeros((N, N)), W["cf_true0"][i])
        for j, sc in enumerate(SCEN):
            y = emb(dA["a"][sl], W["yobs"][i, j]).astype(np.float32)
            cf = add_ensembles({e: f(t0, y) for e, f in E.items()})
            cf["Chronos-2"] = emb(np.zeros((N, N)), W["own"][i, j])
            rows.append({**meta, "scenario": sc, "tile": metrics(cf, emb(np.zeros((N, N)), W["cf_true"][i, j]), ct0, mb, tile)})
        print(f"c2 world {meta['season']}: S1 rel z {rows[-3]['tile']['z-score']['rel']:+.2f} M4_s0 {rows[-3]['tile']['M4_s0']['rel']:+.2f} "
              f"Chronos-2 {rows[-3]['tile']['Chronos-2']['rel']:+.2f}", flush=True)
    json.dump({"world": "Chronos-2 with weather", "tile_L": L, "offset": o, "debug": bool(W["debug"]), "rows": rows,
               "z_binned": binned(W["z"].ravel(), W["dep"].ravel(), W["et0"].ravel())}, open(f"{out}/attrib_c2.json", "w"), indent=1)
    print("wrote", f"{out}/attrib_c2.json")


# ----------------------------------------------------------------------------- test 2a: weather lean
@jax.jit
def forecast_m3(p, s, a0, w0, r, ex, ph):
    """Season-end forecast from the observed start along the weather that fell, residual zero (season_split's F)."""
    return rerun(p, s, a0, w0, r, ex, ph, jnp.zeros_like(r))[-1]


def blocks(v, ok, b=8):
    """2 km block means [N/b,N/b] over pixels with ok; NaN where under half the block is observed."""
    n = N // b
    tot = np.where(ok, v, 0.0).reshape(n, b, n, b).sum((1, 3))
    cnt = ok.reshape(n, b, n, b).sum((1, 3))
    return np.where(cnt >= b * b / 2, tot / np.maximum(cnt, 1), np.nan)


def slope(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / (xm ** 2).sum())


def lean(out, debug):
    """Test 2a (PREREG addendum 3): does the leftover (observed - forecast along the weather that fell) lean with rain?"""
    P = get_plan(out)
    d1, D1 = data("A", "M1")
    p0 = params("M0")
    w0 = M.w_path(p0, D1)
    D4 = data("A", "M4")[1]
    fits = [(params("M4", s), m3.w_path(params("M4", s), D4)) for s in SEEDS]
    e5 = np.load("data/A/era5.npz")
    R = r6(d1)
    S = {k: [] for k in ("rain", "et0", "M4_ens", "z-score")}
    table = []
    for t0, name in zip(P["season_t0"].tolist(), P["season_name"]):
        sl = slice(t0 + 1, t0 + 1 + H)
        obs, ok = d1["a"][t0 + H], d1["obs"][t0] & d1["obs"][t0 + H]
        F4 = np.mean([np.asarray(forecast_m3(p, D4["s"], D4["a"][t0], w[t0], D4["r"][sl], D4["ex"][sl], D4["phase"][sl])) for p, w in fits], 0)
        F0 = np.asarray(M.rollout_mean(p0, D1["a"][t0], w0[t0], D1["r"][sl], D1["phase"][sl], D1["x"], False)[-1])
        v = {"rain": blocks(R[t0] * 100, ok), "et0": blocks((e5["pet"][sl] - e5["pet_clim"][sl]).sum(0) * 100, ok),
             "M4_ens": blocks(obs - F4, ok), "z-score": blocks(obs - F0, ok)}
        keep = np.isfinite(v["M4_ens"])
        for k in S:
            S[k].append(v[k][keep])
        table.append({"season": str(name), "rain_departure_mm": float(v["rain"][keep].mean()), "et0_departure_mm": float(v["et0"][keep].mean()),
                      "leftover_M4_ens": float(v["M4_ens"][keep].mean()), "leftover_z-score": float(v["z-score"][keep].mean()),
                      "n_blocks": int(keep.sum())})
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in table[-1].items()}), flush=True)
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(table), len(table)) for _ in range(2000)]
    res = {"seasons": table, "n_blocks": int(sum(len(x) for x in S["rain"]))}
    for m in ("M4_ens", "z-score"):
        for wx in ("rain", "et0"):
            f = lambda idx: slope(np.concatenate([S[wx][i] for i in idx]) / 100, np.concatenate([S[m][i] for i in idx]))
            bs = [f(ix) for ix in draws]
            res[f"{m}_{wx}"] = {"slope_per_100mm": f(range(len(table))), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["M4_share_of_z_lean_rain"] = res["M4_ens_rain"]["slope_per_100mm"] / res["z-score_rain"]["slope_per_100mm"]
    lo, hi = res["M4_ens_rain"]["ci95"]
    res["claim_pass"] = bool(lo <= 0 <= hi)
    res["outcome"] = ("no lean: weather removed" if res["claim_pass"] else
                      "leftover browner in dry seasons: M4 underplays the weather, or management worsened in droughts" if lo > 0 else
                      "leftover greener in dry seasons: M4 overplays the weather, or management eased in droughts")
    json.dump(res, open(f"{out}/lean.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "seasons"}, indent=1))
    print("wrote", f"{out}/lean.json")


# ----------------------------------------------------------------------------- test 3: the 2022 drought
def drought(out, debug):
    P = get_plan(out)
    t0, name = int(P["t0_2022"]), str(P["name_2022"])
    sl = slice(t0 + 1, t0 + 1 + H)
    d1, D1 = data("A", "M1")
    obs = float(d1["a"][t0 + H].mean())
    res = {"season": name, "t0": t0, "obs": obs, "rows": {}}
    res["rows"]["z-score"] = [{"obs": obs, "start_deficit": None, "rain": 0.0, "temp": 0.0, "residual": obs}]
    rain = float((restrend("A")["beta"] * r6(d1)[t0]).mean())
    res["rows"]["RESTREND"] = [{"obs": obs, "start_deficit": None, "rain": rain, "temp": 0.0, "residual": obs - rain}]
    p1 = params("M1")
    _, inv1, cf1, _ = make_fns(p1, D1["x"])
    w1 = M.w_path(p1, D1)
    a0, r, rc, ph = D1["a"][t0], D1["r"][sl], D1["rclim"][sl], D1["phase"][sl]
    eh = inv1(a0, w1[t0], D1["a"][sl], r, ph)[0]
    F = float(cf1(a0, w1[t0], r, ph, jnp.zeros_like(eh), False)[-1].mean())
    rain = obs - float(cf1(a0, w1[t0], rc, ph, eh, False)[-1].mean())
    res["rows"]["M1"] = [{"obs": obs, "residual": obs - F, "rain": rain, "temp": 0.0, "start_deficit": F - rain}]
    for m in ("M2", "M3", "M4"):
        D = data("A", m)[1]
        res["rows"][m] = []
        for s in SEEDS:
            p = params(m, s)
            res["rows"][m].append(season_split(p, D, m3.w_path(p, D), t0))
    if not debug:  # M4 against the existing per-seed files
        ex = [[r_ for r_ in json.load(open(f"{GPU}/attribution_topo_pet_s{s}.json")) if r_["season"] == name][0] for s in SEEDS]
        res["m4_check_max_abs_diff"] = max(abs(a[k] - b[k]) for a, b in zip(res["rows"]["M4"], ex)
                                           for k in ("obs", "residual", "rain", "temp", "start_deficit"))
        assert res["m4_check_max_abs_diff"] < 1e-4, res["m4_check_max_abs_diff"]
    if os.path.exists(f"{out}/c2_drought.npz"):  # Chronos-2: box means of its maps, split by season_split's formula
        C = np.load(f"{out}/c2_drought.npz")
        o, F, cr, ct, crt = (float(C[k].mean()) for k in ("obs", "F", "cf_r", "cf_t", "cf_rt"))
        rain, temp = 0.5 * ((o - cr) + (ct - crt)), 0.5 * ((o - ct) + (cr - crt))
        res["rows"]["Chronos-2"] = [{"obs": o, "residual": o - F, "rain": rain, "temp": temp, "start_deficit": F - rain - temp,
                                     "n_px": int(len(C["px"])), "recon_max_abs": float(np.abs(C["recon"] - C["obs"]).max())}]
        wt = rain + temp
        res["agreement"] = {"chronos2_weather": wt, "m4_seed_range": [min(r_["rain"] + r_["temp"] for r_ in res["rows"]["M4"]),
                                                                     max(r_["rain"] + r_["temp"] for r_ in res["rows"]["M4"])]}
        res["agreement"]["within"] = res["agreement"]["m4_seed_range"][0] <= wt <= res["agreement"]["m4_seed_range"][1]
        res["agreement"]["minus_m4_mean"] = wt - float(np.mean([r_["rain"] + r_["temp"] for r_ in res["rows"]["M4"]]))
    for m, rr in res["rows"].items():
        wt = [r_["rain"] + r_["temp"] for r_ in rr]
        print(f"{name} {m}: weather {np.mean(wt):+.3f} [{min(wt):+.3f}, {max(wt):+.3f}] residual {np.mean([r_['residual'] for r_ in rr]):+.3f}", flush=True)
    json.dump(res, open(f"{out}/drought.json", "w"), indent=1)
    print("wrote", f"{out}/drought.json")


# ----------------------------------------------------------------------------- test 4: sensitivities
@jax.jit
def _delta(p, s, a0, w0, rc, ex, ph):
    """dE[a_{t+2}]/dr_{t+1}: jvp of the mean rollout (zero residual) along normal weather, +1 (100 mm) in window 1."""
    e1, zero = jnp.zeros_like(rc).at[0].set(1.0), jnp.zeros_like(rc)
    return jax.jvp(lambda r: rerun(p, s, a0, w0, r, jnp.zeros_like(ex), ph, zero), (rc,), (e1,))[1][G.H2 - 1]


def deltas(model, box, st, seed=0):
    if model == "M1":
        return G.deltas(params("M1"), data(box, "M1")[1], st)
    D = data(box, model)[1]
    p = params(model, seed)
    w = m3.w_path(p, D)
    return np.stack([np.asarray(_delta(p, D["s"], D["a"][t], w[t], D["rclim"][t + 1:t + 1 + H], D["ex"][t + 1:t + 1 + H],
                                       D["phase"][t + 1:t + 1 + H])) for t in st]).astype(np.float64)


def sort_all(Dl, y, x, a0, ok, yr, ctrl, sets):
    """G1 (within pixel) and G2 (raw) tercile tests, primary and with the ET0 control, per start set. A test is
    undefined if its Delta has no variation to sort on (RESTREND's fixed map, G1; M1, linear, one number, G1 and G2)."""
    res = {}
    for name, Dv in Dl.items():
        res[name] = {}
        for sn, idx in sets.items():
            res[name][sn] = {}
            for g in ("G1", "G2"):
                Gv = Dv[idx] - Dv[idx].mean(0) if g == "G1" else Dv[idx]
                if np.ptp(Gv[ok[idx]]) <= 1e-6 * np.abs(Dv).max():
                    res[name][sn][g] = {"undefined": "Delta does not vary " + ("within pixels" if g == "G1" else "at all")}
                    continue
                res[name][sn][g] = {c: G.sort_test(Gv, Dv[idx], y[idx], x[idx], a0[idx], ok[idx], yr[idx], np.random.default_rng(0),
                                                   ctrl=None if c == "primary" else ctrl[idx]) for c in ("primary", "secondary")}
    return res


def sens(out, debug):
    P = get_plan(out)
    c2 = dict(np.load(f"{out}/c2_sens.npz")) if os.path.exists(f"{out}/c2_sens.npz") else None
    res = {}
    for box in "AB":
        d = load(box)
        st = P[f"{box}_sens_starts"].tolist()
        sets = {"nogap": np.where(P[f"{box}_sens_nogap"])[0], "all": np.arange(len(st))}
        y, x, a0, ok, yr = G.outcomes(d, st)
        e = np.load(f"data/{box}/era5.npz")
        ctrl = np.stack([e["pet"][t + 1] - e["pet_clim"][t + 1] for t in st]).astype(np.float64)
        Dl = {"M1": deltas("M1", box, st)}
        for m in ("M2", "M3", "M4"):
            for s in SEEDS:
                Dl[f"{m}_s{s}"] = deltas(m, box, st, s)
                print(box, m, s, "Delta mean", float(Dl[f"{m}_s{s}"].mean()), flush=True)
        Dl["M4_ens"] = np.mean([Dl[f"M4_s{s}"] for s in SEEDS], 0)
        Dl["RESTREND"] = np.broadcast_to(restrend(box)["beta"], Dl["M1"].shape)
        res[box] = {"n_starts": {k: len(v) for k, v in sets.items()}, "delta_mean": {k: float(v.mean()) for k, v in Dl.items()},
                    "tests": sort_all(Dl, y, x, a0, ok, yr, ctrl, sets)}
        if c2 is not None:  # Chronos-2 and the M4 ensemble on the 4,000-px sample
            assert (c2[f"{box}_starts"] == P[f"{box}_sens_starts"]).all()
            f = lambda v: v.reshape(len(st), -1)[:, P["px"]]
            Dc = {"Chronos-2": c2[f"{box}_delta"].astype(np.float64), "M4_ens": f(Dl["M4_ens"])}
            res[box]["px"] = {"delta_mean": {k: float(v.mean()) for k, v in Dc.items()},
                              "tests": sort_all(Dc, f(y), f(x), f(a0), f(ok), yr, f(ctrl), sets)}
        g = res[box]["tests"]["M4_ens"]["nogap"]["G1"]["primary"]
        print(box, "M4 ensemble G1 gap-free:", g["high_minus_low"], g["high_minus_low_ci95"], g["pass"], flush=True)
        json.dump(res, open(f"{out}/sens.json", "w"), indent=1)
    print("wrote", f"{out}/sens.json")


# ----------------------------------------------------------------------------- compute
@jax.jit
def split_maps(p, s, a0, w0, y, r, rc, ex, ph):
    """season_split's five season-end maps (observed, F, rain-normal, demand-normal, both) as one compiled call."""
    eps = implied(p, s, a0, w0, y, r, ex, ph)[0]
    R = lambda r_, e_, ep: rerun(p, s, a0, w0, r_, e_, ph, ep)[-1]
    ex0 = jnp.zeros_like(ex)
    return y[-1], R(r, ex, jnp.zeros_like(eps)), R(rc, ex, eps), R(r, ex0, eps), R(rc, ex0, eps)


def clock(f, warm=5):
    """(first call incl. compilation, median warm call) in seconds; JAX caches cleared first."""
    jax.clear_caches()
    t = time.perf_counter()
    jax.block_until_ready(f())
    first = time.perf_counter() - t
    ws = []
    for _ in range(warm):
        t = time.perf_counter()
        jax.block_until_ready(f())
        ws.append(time.perf_counter() - t)
    return first, statistics.median(ws)


def timing(out):
    import platform
    import subprocess
    P = get_plan(out)
    years = set(P["years"].tolist())
    t0 = int(P["t0_2022"])
    res = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "cores": os.cpu_count(), "jax": jax.__version__, "python": platform.python_version(), "params": {}, "test1_s": {},
           "test2_s_per_season": {}, "test3_s": {}, "test4_s": {}}
    for m in ("M0", "M1", "M2", "M3", "M4"):
        res["params"][m] = int(sum(np.size(v) for v in params(m).values()))
    res["params"]["M4_ens"] = 5 * res["params"]["M4"]
    res["params"]["RESTREND"] = 3 * N * N
    # test 1: scoring per set, full grid, all starts (one call each, compilation included)
    for box in "AB":
        d = load(box)
        mt = np.isin(d["year"][1:], list(years))
        for m in ("M0", "M1", "M2", "M3", "M4"):
            t = time.perf_counter()
            m3.score(*stepper(m, box), data(box, m)[1], mt)
            res["test1_s"][f"{m} {box}_test"] = time.perf_counter() - t
        t = time.perf_counter()
        m3.score_ensemble([params("M4", s) for s in SEEDS], data(box, "M4")[1], mt)
        res["test1_s"][f"M4_ens {box}_test"] = time.perf_counter() - t
        t = time.perf_counter()
        score_restrend(box, mt)
        res["test1_s"][f"RESTREND {box}_test"] = time.perf_counter() - t
        print(res["test1_s"], flush=True)
    # test 2: one season (the 2022 one), the M4 world's four scenarios and each estimator's four inversions
    d4, D4 = data("A", "M4")
    p, s = params("M4", 0), D4["s"]
    w = m3.w_path(p, D4)
    sl = slice(t0 + 1, t0 + 1 + H)
    a0, w0, r, rc, ex, ph = D4["a"][t0], w[t0], D4["r"][sl], D4["rclim"][sl], D4["ex"][sl], D4["phase"][sl]
    eps = implied(p, s, a0, w0, D4["a"][sl], r, ex, ph)[0]
    mg = jnp.asarray(management(np.asarray(r), np.asarray(rc), 0.0, DELTA, P["mask"]))

    def sim():
        out_ = [world_j(p, s, a0, w0, r, ex, ph, eps, mg) for _ in SCEN] + [world_j(p, s, a0, w0, rc, 0 * ex, ph, eps, mg) for _ in SCEN]
        return out_ + [world_j(p, s, a0, w0, rc, 0 * ex, ph, eps, 0 * mg)]
    res["test2_s_per_season"]["M4 world simulation"] = clock(sim)
    y = sim()[0]
    E = estimators()
    for e in ("z-score", "RESTREND", "M1", "M2_s0", "M3_s0", "M4_s0"):
        res["test2_s_per_season"][e] = clock(lambda: [E[e](t0, y) for _ in SCEN])
    print(res["test2_s_per_season"], flush=True)
    # test 3: full-box 2022 attribution
    D1 = data("A", "M1")[1]
    p1 = params("M1")
    _, inv1, cf1, _ = make_fns(p1, D1["x"])
    w1 = M.w_path(p1, D1)

    def m1_split():
        eh = inv1(D1["a"][t0], w1[t0], D1["a"][sl], D1["r"][sl], D1["phase"][sl])[0]
        return (cf1(D1["a"][t0], w1[t0], D1["r"][sl], D1["phase"][sl], 0 * eh, False)[-1].mean(),
                cf1(D1["a"][t0], w1[t0], D1["rclim"][sl], D1["phase"][sl], eh, False)[-1].mean())
    R6 = r6(load("A"))[t0]
    res["test3_s"]["z-score"] = clock(lambda: jnp.asarray(D1["a"][t0 + H].mean()))
    res["test3_s"]["RESTREND"] = clock(lambda: jnp.asarray((restrend("A")["beta"] * R6).mean()))
    res["test3_s"]["M1"] = clock(m1_split)
    for m in ("M2", "M3", "M4"):
        D = data("A", m)[1]
        pm = params(m)
        wm = m3.w_path(pm, D)
        res["test3_s"][m] = clock(lambda: jnp.asarray(season_split(pm, D, wm, t0)["rain"]))
        res["test3_s"][f"{m} compiled maps"] = clock(lambda: split_maps(pm, D["s"], D["a"][t0], wm[t0], D["a"][sl], D["r"][sl], D["rclim"][sl], D["ex"][sl], D["phase"][sl]))
    print(res["test3_s"], flush=True)
    # test 4: full-box Delta map at one start, autodiff; M4 also by finite differences
    res["test4_s"]["M1"] = clock(lambda: jnp.asarray(G.deltas(p1, D1, [t0])))
    for m in ("M2", "M3", "M4"):
        D = data("A", m)[1]
        pm = params(m)
        wm = m3.w_path(pm, D)
        res["test4_s"][m] = clock(lambda: _delta(pm, D["s"], D["a"][t0], wm[t0], D["rclim"][sl], D["ex"][sl], D["phase"][sl]))
    fd = jax.jit(lambda: (rerun(pm, D["s"], D["a"][t0], wm[t0], D["rclim"][sl].at[0].add(0.01), 0 * D["ex"][sl], D["phase"][sl], 0 * D["rclim"][sl])
                          - rerun(pm, D["s"], D["a"][t0], wm[t0], D["rclim"][sl], 0 * D["ex"][sl], D["phase"][sl], 0 * D["rclim"][sl]))[G.H2 - 1] / 0.01)
    res["test4_s"]["M4 finite differences"] = clock(fd)
    res["test4_fd_vs_autodiff_max_abs"] = float(jnp.abs(fd() - _delta(pm, D["s"], D["a"][t0], wm[t0], D["rclim"][sl], D["ex"][sl], D["phase"][sl])).max())
    print(res["test4_s"], flush=True)
    json.dump(res, open(f"{out}/timing.json", "w"), indent=1)
    print("wrote", f"{out}/timing.json")


# ----------------------------------------------------------------------------- tables
TREES = ("runs/20261006_1759_baselines/metrics_gbm.json", "runs/20261021_0147_m3/metrics_gbm_B_test.json")
ROWS = [("z-score", "M0 / z-score"), ("RESTREND", "RESTREND"), ("M1", "M1"), ("M2", "M2 (`base` refit)"), ("M3", "M3"),
        ("M4", "M4"), ("M4_ens", "M4 ensemble"), ("Trees", "Trees (GBM)"), ("Chronos-2", "Chronos-2")]


def fmt(v, d=3, sign=False):
    return "–" if v is None else (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def seeds(f, model, d=3, sign=False):
    """'mean [min, max]' over the 5 seeds of f(name)."""
    v = [f(f"{model}_s{s}") for s in SEEDS]
    return f"{fmt(np.mean(v), d, sign)} [{fmt(min(v), d, sign)}, {fmt(max(v), d, sign)}]"


def summarise(rows, region, scen, est, drought):
    """Season-class summary of one estimator: S1-type errors, invented share, S0 error, pixel RMSE, wins."""
    rr = [r for r in rows if r["scenario"] == scen and (drought is None or r["drought"] == drought)]
    g = lambda k: np.array([r[region][est][k] for r in rr], float)
    rel = g("rel")
    out = {"n": len(rr), "signed": rel.mean(), "size": np.abs(rel).mean(), "worst": rel[np.argmax(np.abs(rel))],
           "invented": np.abs(g("invented")).mean(), "px_rmse": g("px_rmse").mean(), "box_err": g("err_managed").mean(),
           "box_err_size": np.abs(g("err_managed")).mean()}
    for ref in ("z-score", "RESTREND"):
        out[f"wins_vs_{ref}"] = int((np.abs(rel) < np.abs([r[region][ref]["rel"] for r in rr])).sum())
    rr0 = [r for r in rows if r["scenario"] == "S0" and (drought is None or r["drought"] == drought)]
    e0 = np.array([r[region][est]["err_all"] for r in rr0])
    out["s0_signed"], out["s0_size"] = e0.mean(), np.abs(e0).mean()
    return out


def world_avg(rows, est, drought, key, region="full", scen="S1"):
    """A season-class metric per world (averaged over the names est(world seed) gives), then over worlds: (mean, min, max)."""
    v = []
    for ws in sorted({r.get("world_seed", 0) for r in rows}):
        rw = [r for r in rows if r.get("world_seed", 0) == ws]
        names = est(ws) if callable(est) else [est]
        v.append(float(np.mean([summarise(rw, region, scen, n, drought)[key] for n in names])))
    return float(np.mean(v)), min(v), max(v)


SEEDS_OF = lambda m: (lambda ws: [f"{m}_s{s}" for s in SEEDS])
EST5 = [("z-score", "z-score"), ("RESTREND", "RESTREND"), ("M1", "M1"), ("M2 (5 seeds)", SEEDS_OF("M2")), ("M3 (5 seeds)", SEEDS_OF("M3")),
        ("M4 single fit, own world", lambda ws: [f"M4_s{ws}"]), ("M4 single fit, other worlds", lambda ws: [f"M4_s{s}" for s in SEEDS if s != ws]),
        ("**M4 ensemble, 5 seeds**", "M4_ens")]
EST_SEED0 = EST5[:5] + [("M4 seed 0 (own world)", "M4_s0"), ("M4 seeds 1-4", lambda ws: [f"M4_s{s}" for s in range(1, 5)]),
                        ("M4 ensemble, 5 seeds", "M4_ens"), ("M4 ensemble, seeds 1-4", "M4_ens14")]
EST_C2 = EST5[:5] + [("M4 (5 seeds)", SEEDS_OF("M4")), ("**M4 ensemble, 5 seeds**", "M4_ens"), ("Chronos-2 (own world)", "Chronos-2")]


def t2_table(rows, est, region="full", rng=True):
    """Test 2 detail table: S1 by season class (worlds averaged; [range over worlds] when there are several)."""
    multi = rng and len({r.get("world_seed", 0) for r in rows}) > 1
    def c(e, dro, key, sign=False, d=2, r=False):
        m, lo, hi = world_avg(rows, e, dro, key, region)
        return fmt(m, d, sign) + (f" [{fmt(lo, d, sign)}, {fmt(hi, d, sign)}]" if r and multi else "")
    n = lambda dro: summarise(rows, region, "S1", "z-score", dro)["n"] // max(1, len({r.get("world_seed", 0) for r in rows}))
    L = [f"| Estimator | Drought (n={n(True)}): mean size | signed | worst | wins vs z / RESTREND | Other (n={n(False)}): mean size | signed | "
         f"All {n(None)}: mean size | signed | wins vs RESTREND | Invented (dr / oth) | S0 found, NDVI, size (dr / oth) | Pixel RMSE, NDVI (dr / oth) | Box error, NDVI, size (dr / oth) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, e in est:
        wz, wr, wa = (world_avg(rows, e, True, "wins_vs_z-score", region)[0], world_avg(rows, e, True, "wins_vs_RESTREND", region)[0],
                      world_avg(rows, e, None, "wins_vs_RESTREND", region)[0])
        wins = "–" if e in ("z-score", "RESTREND") else f"{wz:.1f}/{n(True)} / {wr:.1f}/{n(True)}"
        L.append(f"| {lab} | {c(e, True, 'size', r=True)} | {c(e, True, 'signed', True)} | {c(e, True, 'worst', True)} | {wins} | "
                 f"{c(e, False, 'size')} | {c(e, False, 'signed', True)} | {c(e, None, 'size', r=True)} | {c(e, None, 'signed', True)} | "
                 f"{'–' if e == 'RESTREND' else f'{wa:.1f}/{n(None)}'} | {c(e, True, 'invented')} / {c(e, False, 'invented')} | "
                 f"{c(e, True, 's0_size', d=4)} / {c(e, False, 's0_size', d=4)} | {c(e, True, 'px_rmse', d=4)} / {c(e, False, 'px_rmse', d=4)} | "
                 f"{c(e, True, 'box_err_size', d=4)} / {c(e, False, 'box_err_size', d=4)} |")
    return "\n".join(L)


def report(out):
    J = lambda f: json.load(open(f"{out}/{f}")) if os.path.exists(f"{out}/{f}") else None
    sk, at, dr, se, tm, ac2 = J("skill.json"), J("attrib_m4.json"), J("drought.json"), J("sens.json"), J("timing.json"), J("attrib_c2.json")
    cost, at5 = J("c2_cost.json"), J("attrib_m4x5.json")
    c2s = dict(np.load(f"{out}/c2_skill.npz")) if os.path.exists(f"{out}/c2_skill.npz") else None
    L = []
    w = L.append

    # ---- test 1
    def sk1(es, v, name, h):
        m0, r = sk[es][v]["M0"]["rollout_crps" if h else "onestep_crps"], sk[es][v].get(name)
        if r is None:
            return None
        if h:
            return 1 - r["rollout_crps"][h - 1] / m0[h - 1] if r["rollout_crps"][h - 1] is not None else None
        return 1 - r["onestep_crps"] / m0 if "onestep_crps" in r else 1 - r["rollout_crps"][0] / sk[es][v]["M0"]["rollout_crps"][0]
    gA, gB = json.load(open(TREES[0]))["A_test"], json.load(open(TREES[1]))["B_test"]
    m0B = json.load(open(f"{GPU}/metrics_M0.json"))["B_test"]
    trees = {("A_test", 1): gA["onestep_skill_vs_M0"], ("A_test", 6): gA["rollout_skill_vs_M0"][-1],
             ("B_test", 1): 1 - gB["onestep_crps"] / m0B["onestep_crps"], ("B_test", 6): 1 - gB["rollout_crps"][-1] / m0B["rollout_crps"][-1]}

    def c2skill(box, mode, h, dry=False):
        if c2s is None:
            return None
        sel = c2s[f"{box}_dry"] if dry and f"{box}_dry" in c2s else (get_plan(out)[f"{box}_skill_dry"] if dry else slice(None))
        v = "px_drought" if dry else "px"
        crps = c2s[f"{box}_{mode}_num"][sel, h - 1].sum() / c2s[f"{box}_{mode}_den"][sel, h - 1].sum()
        return 1 - crps / sk[f"{box}_test"][v]["M0"]["rollout_crps"][h - 1]

    def cell1(model, es, v, h):
        if model == "z-score":
            return "0 (reference)"
        if model == "Trees":
            return fmt(trees[(es, h)]) if v == "all" else "–"
        if model in ("M2", "M3", "M4"):
            if f"{model}_s0" not in sk[es][v]:
                return "–"
            return seeds(lambda n: sk1(es, v, n, h), model)
        if model == "Chronos-2":
            return "GPU"
        return fmt(sk1(es, v, model, h)) if model in sk[es][v] else "–"

    # ---- test 2
    rows = at["rows"]
    S = lambda est, dro, region="full", rr=rows, scen="S1": summarise(rr, region, scen, est, dro)

    def cell2(model, dro, key="size", rr=rows, region="full", sign=False, d=2):
        if model in ("z-score", "RESTREND", "M1", "M4_s0", "M4_ens", "M4_ens14", "Chronos-2"):
            return fmt(S(model, dro, region, rr)[key], d, sign)
        ss = [s for s in SEEDS if s or model != "M4" or rr is not rows]
        v = [S(f"{model}_s{s}", dro, region, rr)[key] for s in ss]
        return f"{fmt(np.mean(v), d, sign)} [{fmt(min(v), d, sign)}, {fmt(max(v), d, sign)}]"

    # ---- test 3
    wt = lambda r: r["rain"] + r["temp"]
    def cell3(model):
        if model == "Chronos-2" and "Chronos-2" not in dr["rows"]:
            return "GPU"
        if model == "Trees":
            return "–"
        rr = dr["rows"]["M4" if model == "M4_ens" else model]
        if len(rr) == 1:
            return fmt(wt(rr[0]), sign=True)
        v = [wt(r) for r in rr]
        return f"{fmt(np.mean(v), sign=True)} [{fmt(min(v), sign=True)}, {fmt(max(v), sign=True)}]"

    # ---- test 4
    def g(box, name, sn="nogap", G_="G1", c="primary"):
        return se[box]["tests"][name][sn].get(G_, {}).get(c, se[box]["tests"][name][sn].get(G_))

    def cell4(box, model, sn="nogap", G_="G1", c="primary"):
        if model in ("M2", "M3", "M4"):
            v = [g(box, f"{model}_s{s}", sn, G_, c) for s in SEEDS]
            h = [x["high_minus_low"] for x in v]
            return f"{fmt(np.mean(h))} [{fmt(min(h))}, {fmt(max(h))}], {sum(x['pass'] for x in v)}/5 pass"
        x = g(box, model, sn, G_, c)
        if x is None or "undefined" in x:
            return "undefined"
        lo, hi = x["high_minus_low_ci95"]
        return f"{fmt(x['high_minus_low'])} [{fmt(lo)}, {fmt(hi)}] {'pass' if x['pass'] else 'fail'}"

    # ---- status table
    warm = lambda m: tm["test3_s"].get(f"{m} compiled maps", tm["test3_s"][m])[1] * 1e3 if tm else None
    par = lambda m: tm["params"][m] if tm else None
    w("| Model | 1 Skill: 3-month, A_test / B_test | 2 Attribution (5 M4 worlds): S1 drought / other, mean size (signed mean) | "
      "3 2022: total weather share, NDVI | 4 Sensitivities: G1 high-minus-low, gap-free, A / B | Compute: 2022 full box warm, parameters |")
    w("|---|---|---|---|---|---|")
    for key, lab in ROWS:
        c1 = (f"{cell1(key, 'A_test', 'all', 6)} / {cell1(key, 'B_test', 'all', 6)}" if key not in ("z-score", "Chronos-2")
              else cell1(key, "A_test", "all", 6))
        if key == "RESTREND":
            c1 += " (3-month only)"
        if key in ("Trees",):
            c2 = c3 = c4 = c5 = "–"
        elif key == "Chronos-2":
            c2, c3, c4 = "not scored (own world, GPU)", "GPU", "GPU (reported, not deciding)"
            c5 = (f"CPU est. {cost['cells']['test 3 2022 full box']['cpu_h']:.1f} h; {cost['params']:,}" if cost else "🆕 CPU estimate")
        else:
            def c2f(dro):
                rw = at5["rows"] if at5 else rows
                v = lambda e, k="size", sg=False: fmt(world_avg(rw, e, dro, k)[0], 2, sg)
                e = {"M2": SEEDS_OF("M2"), "M3": SEEDS_OF("M3"), "M4_ens": "M4_ens"}.get(key, key)
                if key == "M4":
                    return f"own world {v(lambda ws: [f'M4_s{ws}'])}; other worlds {v(lambda ws: [f'M4_s{s}' for s in SEEDS if s != ws])}"
                return f"{v(e)} ({v(e, 'signed', True)})"
            c2 = f"{c2f(True)} / {c2f(False)}"
            c3 = "0 (all −0.168 to the land)" if key == "z-score" else cell3(key)
            if key == "z-score":
                c4 = "— (no rain response)"
            elif key == "RESTREND":
                c4 = f"G2 {cell4('A', key, G_='G2')} / {cell4('B', key, G_='G2')}"
            else:
                c4 = f"{cell4('A', key)} / {cell4('B', key)}"
            mk = {"z-score": "z-score", "M4_ens": "M4"}.get(key, key)
            pk = {"z-score": "M0"}.get(key, key)
            c5 = f"{fmt(warm(mk), 1)} ms; {par(pk):,}" + (" (5 x 1,002)" if key == "M4_ens" else "") if tm else "🆕"
        w(f"| {lab} | {c1} | {c2} | {c3} | {c4} | {c5} |")
    status = "\n".join(L)
    L.clear()

    # ---- detail tables
    w("### Test 1: skill (CRPS skill vs M0; single fits: 5-seed mean [range])")
    for v, title in (("all", "All starts, full grid"), ("drought", "Drought starts only, full grid"),
                     ("px", "4,000-px samples, all starts"), ("px_drought", "4,000-px samples, drought starts")):
        nA, nB = sk["A_test"][v]["n_starts"], sk["B_test"][v]["n_starts"]
        w(f"\n**{title}** (starts: A {nA}, B {nB})\n")
        w("| Model | A 1-step | A 3-month | B 1-step | B 3-month |\n|---|---|---|---|---|")
        names = [("M0", "z-score"), ("RESTREND", "RESTREND"), ("M1", "M1"), ("M2", "M2"), ("M3", "M3"), ("M4", "M4"),
                 ("M4_ens", "M4 ensemble"), ("Trees", "Trees")]
        for k, lab in names:
            if k == "M0":
                continue
            cells = []
            for es in ("A_test", "B_test"):
                for h in (1, 6):
                    cells.append("–" if k == "RESTREND" and h == 1 else cell1(k, es, v, h))
            if all(c == "–" for c in cells):
                continue
            w(f"| {lab} | " + " | ".join(cells) + " |")
        if v.startswith("px"):
            for mode, lab in (("weather", "Chronos-2, with weather"), ("history", "Chronos-2, history only")):
                cells = [fmt(c2skill(b, mode, h, v == "px_drought")) if c2s is not None else "GPU" for b in "AB" for h in (1, 6)]
                w(f"| {lab} | " + " | ".join(cells) + " |")
    if "reproduction_check" in sk:
        w(f"\nReproduction check (M4 seed 0, A_test, extended score vs gpu/metrics_topo_pet_s0.json): max |diff| "
          f"{sk['reproduction_check']['max_abs_diff']:.1e} (bar 1e-3).")
    m0e = json.load(open(f"{GPU}/metrics_M0.json"))
    w(f"M0 re-scored, all starts, vs gpu/metrics_M0.json: max |diff| 3-month "
      f"{max(abs(sk[es]['all']['M0']['rollout_crps'][-1] - m0e[es]['rollout_crps'][-1]) for es in ('A_test', 'B_test')):.1e}.")

    rw = at5["rows"] if at5 else rows
    w("\n### Test 2: attribution, scenario S1 (error in the recovered management as a share of the true effect; + = overstated)")
    w("\n**Five M4 worlds (addendum 2): one per seed; metrics per world, then averaged; [range over worlds].** Signed means of cross-world rows "
      "largely cancel by construction (in S0 each fit's estimate is its own world's truth), so read the sizes.\n")
    w(t2_table(rw, EST5))
    w("\nS2a (incursion: management follows rain) and S2b (destocking): mean size, drought / other / all (the stated limit)\n")
    w("| Estimator | S2a | S2b |\n|---|---|---|")
    for lab, e in EST5:
        cc = [" / ".join(fmt(world_avg(rw, e, dro, "size", scen=sc)[0], 2) for dro in (True, False, None)) for sc in ("S2a", "S2b")]
        w(f"| {lab} | {cc[0]} | {cc[1]} |")
    s1 = [r for r in rw if r["scenario"] == "S1"]
    dif = lambda rr, dro=None: np.mean([r["difficulty"] for r in rr if dro is None or r["drought"] == dro])
    w(f"\nDifficulty check (season-end box-mean change for a season 100 mm drier, spread evenly; achieved departure "
      f"{np.mean([r['difficulty_rain_departure_mm'] for r in s1]):.0f} mm): M4 worlds {dif(s1):+.4f} NDVI "
      f"(drought {dif(s1, True):+.4f}, other {dif(s1, False):+.4f}); Chronos-2 world: "
      + (f"{dif([r for r in ac2['rows'] if r['scenario'] == 'S1']):+.4f}" if ac2 else "GPU") + ".")
    w(f"Mean true effect of the injected management (S1, managed pixels, season end): drought "
      f"{np.mean([r['full']['true_effect'] for r in s1 if r['drought']]):+.4f}, other {np.mean([r['full']['true_effect'] for r in s1 if not r['drought']]):+.4f} NDVI.")
    if at5 and at5.get("tile_L"):
        w(f"\nOn the Chronos-2 world's tile (L = {at5['tile_L']}, descriptive), five M4 worlds, S1 mean size drought / other:\n")
        w("| Estimator | Full box | Tile |\n|---|---|---|")
        for lab, e in EST5:
            f5 = lambda reg: " / ".join(fmt(world_avg(rw, e, dro, "size", reg)[0], 2) for dro in (True, False))
            w(f"| {lab} | {f5('full')} | {f5('tile')} |")
    if ac2:
        w(f"\n**Chronos-2 world** (tile L = {ac2['tile_L']}{', DEBUG' if ac2['debug'] else ''}):\n")
        w(t2_table(ac2["rows"], EST_C2, "tile"))
    w("\n**On record: the first run, one M4 world (seed 0), as pre-registered before addendum 2.**\n")
    w(t2_table(rows, EST_SEED0))

    w(f"\n### Test 3: {dr['season']} Box A, split of the observed {dr['obs']:+.3f} NDVI (single fits: 5-seed mean [range])\n")
    w("| Model | Starting deficit | Rain | Evaporative demand | **Total weather** | Not explained |\n|---|---|---|---|---|---|")
    for k, lab in [("z-score", "z-score"), ("RESTREND", "RESTREND"), ("M1", "M1"), ("M2", "M2"), ("M3", "M3"), ("M4", "M4 (= ensemble)"),
                   ("Chronos-2", "Chronos-2")]:
        if k not in dr["rows"]:
            w(f"| {lab} | GPU | GPU | GPU | GPU | GPU |")
            continue
        rr = dr["rows"][k]
        f4 = lambda key: ("–" if rr[0][key] is None else (fmt(rr[0][key], sign=True) if len(rr) == 1 else
                          f"{fmt(np.mean([r[key] for r in rr]), sign=True)} [{fmt(min(r[key] for r in rr), sign=True)}, {fmt(max(r[key] for r in rr), sign=True)}]"))
        w(f"| {lab} | {f4('start_deficit')} | {f4('rain')} | {f4('temp') if k in ('M4', 'Chronos-2') else '–'} | **{cell3(k)}** | {f4('residual')} |")
    if "m4_check_max_abs_diff" in dr:
        w(f"\nM4 recomputed vs gpu/attribution_topo_pet_s*.json: max |diff| {dr['m4_check_max_abs_diff']:.1e}.")

    w("\n### Test 4: sensitivities (Delta tercile test: high-minus-low slope of greenness change on rain, per 100 mm; 95% CI over years)\n")
    w(f"Starts: Box A gap-free {se['A']['n_starts']['nogap']} of {se['A']['n_starts']['all']}, Box B {se['B']['n_starts']['nogap']} of {se['B']['n_starts']['all']}. "
      "Single fits: mean [range] of high-minus-low over seeds and seeds passing.\n")
    w("| Model | G1 A, gap-free | G1 B, gap-free | G2 A | G2 B | G1 A + ET0 control | G1 A, all starts | G1 B, all starts |\n|---|---|---|---|---|---|---|---|")
    for k, lab in [("M1", "M1"), ("M2", "M2"), ("M3", "M3"), ("M4", "M4"), ("M4_ens", "M4 ensemble"), ("RESTREND", "RESTREND")]:
        w(f"| {lab} | {cell4('A', k)} | {cell4('B', k)} | {cell4('A', k, G_='G2')} | {cell4('B', k, G_='G2')} | "
          f"{cell4('A', k, c='secondary')} | {cell4('A', k, 'all')} | {cell4('B', k, 'all')} |")
    if "px" not in se["A"]:
        w("| Chronos-2 | GPU | GPU | GPU | GPU | GPU | GPU | GPU |")
    else:  # 4,000-px samples, Chronos-2 beside the M4 ensemble on the same pixels
        for k, lab in (("Chronos-2", "Chronos-2 (4,000 px)"), ("M4_ens", "M4 ensemble (same 4,000 px)")):
            c = lambda box, sn="nogap", G_="G1", cc="primary": (lambda x: f"{fmt(x['high_minus_low'])} [{fmt(x['high_minus_low_ci95'][0])}, "
                                                               f"{fmt(x['high_minus_low_ci95'][1])}] {'pass' if x['pass'] else 'fail'}")(se[box]["px"]["tests"][k][sn][G_][cc])
            w(f"| {lab} | {c('A')} | {c('B')} | {c('A', G_='G2')} | {c('B', G_='G2')} | {c('A', cc='secondary')} | {c('A', 'all')} | {c('B', 'all')} |")
    w(f"\nMean Delta (NDVI per 100 mm, Box A): " + ", ".join(f"{k} {v:.3f}" for k, v in se["A"]["delta_mean"].items() if not k[-3:-1] == "_s" or k.endswith("s0")))

    if tm:
        w(f"\n### Compute ({tm['chip']}, {tm['cores']} cores, JAX {tm['jax']}, CPU)\n")
        w("| Model | Parameters | Test 1 scoring, A / B (s) | Test 2 per season, 4 scenarios (s) | Test 3 2022 first / warm | Test 4 Delta map first / warm |")
        w("|---|---|---|---|---|---|")
        for k, lab, p_, t2 in [("z-score", "M0 / z-score", "M0", "z-score"), ("RESTREND", "RESTREND", "RESTREND", "RESTREND"), ("M1", "M1", "M1", "M1"),
                               ("M2", "M2", "M2", "M2_s0"), ("M3", "M3", "M3", "M3_s0"), ("M4", "M4", "M4", "M4_s0")]:
            t1k = "M0" if k == "z-score" else k
            t3, t4 = tm["test3_s"].get(f"{k} compiled maps", tm["test3_s"][k]), tm["test4_s"].get(k)
            w(f"| {lab} | {tm['params'][p_]:,} | {tm['test1_s'][f'{t1k} A_test']:.0f} / {tm['test1_s'][f'{t1k} B_test']:.0f} | "
              f"{tm['test2_s_per_season'][t2][1]:.3f} | {t3[0]:.2f} s / {t3[1] * 1e3:.1f} ms | "
              + (f"{t4[0]:.2f} s / {t4[1] * 1e3:.1f} ms |" if t4 else "– |"))
        w(f"| M4 ensemble | {tm['params']['M4_ens']:,} | {tm['test1_s']['M4_ens A_test']:.0f} / {tm['test1_s']['M4_ens B_test']:.0f} | 5 x M4 | 5 x M4 | 5 x M4 |")
        w(f"\nM4 world simulation per season (4 scenarios + truths): {tm['test2_s_per_season']['M4 world simulation'][1]:.3f} s warm. "
          f"M4 Delta map by finite differences: {tm['test4_s']['M4 finite differences'][1] * 1e3:.1f} ms warm "
          f"(max |FD - autodiff| {tm['test4_fd_vs_autodiff_max_abs']:.1e}). Training (existing): M4 ~80 min CPU / ~2.5 min H100 per fit, x5. "
          "Test 3 for M2-M4 is the split's five maps as one compiled call; the uncompiled season_split (with its Python syncs and the "
          "demand Greek) takes " + ", ".join(f"{m} {tm['test3_s'][m][1] * 1e3:.0f} ms" for m in ("M2", "M3", "M4")) + " warm.")
    if cost:
        w("\n### Chronos-2 cost (CPU measured on 2003-17 contexts; H100 estimated)\n")
        w(f"Parameters {cost['params']:,}. Price ${cost['price_usd_per_h100_hour']}/H100-hour. GPU throughput assumed: central "
          f"{cost['gpu_flops_assumed']['central_tf32'] / 1e12:.0f} TFLOP/s (TF32), pessimistic {cost['gpu_flops_assumed']['pessimistic_fp32'] / 1e12:.0f} TFLOP/s (FP32).\n")
        w("| Cell | Row-forwards (M) | Laptop CPU (h) | H100 central (h) | $ central | H100 pessimistic (h) | $ pessimistic |\n|---|---|---|---|---|---|---|")
        tot = {k: 0.0 for k in ("cpu_h", "h100_h_central_tf32", "usd_central_tf32", "h100_h_pessimistic_fp32", "usd_pessimistic_fp32")}
        for name, c in cost["cells"].items():
            w(f"| {name} | {c['rows_M']:.1f} | {c['cpu_h']:.1f} | {c['h100_h_central_tf32']:.2f} | {c['usd_central_tf32']:.2f} | "
              f"{c['h100_h_pessimistic_fp32']:.2f} | {c['usd_pessimistic_fp32']:.2f} |")
            for k in tot:
                tot[k] += c[k]
        Lw = cost.get("world_L_chosen", cost["world_L_rule"])
        wl = cost["world_by_L"][str(Lw)] if str(Lw) in cost["world_by_L"] else None
        if wl:
            wp = cost["world_per_tile_px"]
            n2 = int(Lw) ** 2
            gh = {g: wp["pflop"] * n2 * 1e15 / fl / 3600 for g, fl in cost["gpu_flops_assumed"].items()}
            c = {"cpu_h": wp["cpu_h"] * n2, **{f"h100_h_{g}": v for g, v in gh.items()},
                 **{f"usd_{g}": v * cost["price_usd_per_h100_hour"] for g, v in gh.items()}}
            w(f"| test 2 Chronos-2 world, L = {Lw} ({n2:,} px) | {wp['rows_M'] * n2:.1f} | {c['cpu_h']:.1f} | {c['h100_h_central_tf32']:.2f} | "
              f"{c['usd_central_tf32']:.2f} | {c['h100_h_pessimistic_fp32']:.2f} | {c['usd_pessimistic_fp32']:.2f} |")
            for k in tot:
                tot[k] += c[k]
        w(f"| **Total** | | {tot['cpu_h']:.1f} | {tot['h100_h_central_tf32']:.2f} | **{tot['usd_central_tf32']:.2f}** | "
          f"{tot['h100_h_pessimistic_fp32']:.2f} | **{tot['usd_pessimistic_fp32']:.2f}** |")
    detail = "\n".join(L)

    # ---- claims
    d0, dz0, dR0, d14 = (S(k, True)["size"] for k in ("M4_s0", "z-score", "RESTREND", "M4_ens14"))
    c2old = ("Both beat both" if d0 < min(dz0, dR0) and d14 < min(dz0, dR0) else "Seed 0 passes, seeds 1-4 do not" if d0 < min(dz0, dR0) else
             "Beats z-score, not RESTREND" if d0 < dz0 and d14 < dz0 else "Beats neither")
    de, dz, dR = (world_avg(rw, k, True, "size")[0] for k in ("M4_ens", "z-score", "RESTREND"))
    c2t = "Beats both" if de < dz and de < dR else "Beats z-score, not RESTREND" if de < dz else "Beats neither"
    g4 = g("A", "M4_ens")
    claims = [
        ("1 Skill", "M4 ensemble vs Chronos-2 (with / without weather), 3-month, 4,000-px samples", "pending GPU (Chronos-2 cells)"),
        ("2 Attribution (five M4 worlds, addendum 2)", f"S1 drought mean error size: M4 5-seed ensemble {de:.2f}; z-score {dz:.2f}, RESTREND {dR:.2f}",
         f"**{c2t}**"),
        ("2 Attribution (seed-0 world, on record)", f"M4 seed 0 {d0:.2f}, seeds 1-4 ensemble {d14:.2f}; z-score {dz0:.2f}, RESTREND {dR0:.2f}",
         f"{c2old}"),
        ("3 2022", "Chronos-2 total weather share within 0.066-0.104 NDVI", "pending GPU" if "agreement" not in dr else
         f"Chronos-2 {dr['agreement']['chronos2_weather']:+.3f}: {'within' if dr['agreement']['within'] else 'outside'}"),
        ("4 Sensitivities", f"M4 ensemble G1, gap-free, Box A: {g4['high_minus_low']:.3f} [{g4['high_minus_low_ci95'][0]:.3f}, {g4['high_minus_low_ci95'][1]:.3f}]",
         f"**{'passes' if g4['pass'] else 'fails'}**")]
    ctab = "| Test | Measured | Verdict |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in claims)
    open(f"{out}/tables.md", "w").write(f"## Status\n\n{status}\n\n## Claims\n\n{ctab}\n\n{detail}\n")
    print(open(f"{out}/tables.md").read())


def summary(out):
    """Readable tables (one M4 row: the 5-seed ensemble; plain labels) from the saved metrics -> summary.md."""
    J = lambda f: json.load(open(f"{out}/{f}")) if os.path.exists(f"{out}/{f}") else None
    sk, x5, ln, dr, se, tm, cost = (J(f) for f in ("skill.json", "attrib_m4x5.json", "lean.json", "drought.json", "sens.json",
                                                     "timing.json", "c2_cost.json"))
    rw = x5["rows"]
    L = []
    w = L.append
    names = [("z-score", "z-score (current practice)"), ("RESTREND", "RESTREND (standard research method)"), ("M1", "M1"),
             ("M2", "M2"), ("M3", "M3"), ("M4_ens", "**M4 ensemble**"), ("Trees", "Trees"), ("c2w", "Chronos-2, with weather"),
             ("c2h", "Chronos-2, history only")]
    b = lambda v: f"**{v}**" if v and v != "–" else v
    P_ = get_plan(out)
    c2s = dict(np.load(f"{out}/c2_skill.npz")) if os.path.exists(f"{out}/c2_skill.npz") else None
    c2d = dr["rows"]["Chronos-2"][0] if "Chronos-2" in dr["rows"] else None
    c2g = lambda box: se[box]["px"]["tests"]["Chronos-2"]["nogap"]["G1"]["primary"] if "px" in se[box] else None
    gpu_s = {c: float(open(f"{out}/gpu_seconds_{c}.txt").read().split(":")[1].split()[0])
             for c in ("skill", "drought", "sens", "world") if os.path.exists(f"{out}/gpu_seconds_{c}.txt")}

    def c2sk(mode, box, h, dry=False):
        """Chronos-2 skill vs M0 on the 4,000-px sample (same starts and masks)."""
        if c2s is None:
            return ""
        sel = P_[f"{box}_skill_dry"] if dry else slice(None)
        crps = c2s[f"{box}_{mode}_num"][sel, h - 1].sum() / c2s[f"{box}_{mode}_den"][sel, h - 1].sum()
        return fmt(1 - crps / sk[f"{box}_test"]["px_drought" if dry else "px"]["M0"]["rollout_crps"][h - 1])

    def g4s(x):
        return f"{'pass' if x['pass'] else 'fail'}: {x['high_minus_low']:+.3f} [{x['high_minus_low_ci95'][0]:+.3f}, {x['high_minus_low_ci95'][1]:+.3f}]"

    def sk_(k, es, h, v="all"):
        """3-month (h=6) or 1-step (h=1) skill vs M0; 5-seed mean for M2/M3."""
        m0 = sk[es][v]["M0"]
        one = lambda n: (1 - sk[es][v][n]["rollout_crps"][h - 1] / m0["rollout_crps"][h - 1]) if h == 6 or "onestep_crps" not in sk[es][v][n] \
            else 1 - sk[es][v][n]["onestep_crps"] / m0["onestep_crps"]
        if k == "Trees":
            g = (json.load(open(TREES[0]))["A_test"], json.load(open(TREES[1]))["B_test"])
            if v != "all":
                return "–"
            m0B = json.load(open(f"{GPU}/metrics_M0.json"))["B_test"]
            return fmt({("A_test", 1): g[0]["onestep_skill_vs_M0"], ("A_test", 6): g[0]["rollout_skill_vs_M0"][-1],
                        ("B_test", 1): 1 - g[1]["onestep_crps"] / m0B["onestep_crps"],
                        ("B_test", 6): 1 - g[1]["rollout_crps"][-1] / m0B["rollout_crps"][-1]}[(es, h)])
        if k in ("M2", "M3"):
            return fmt(np.mean([one(f"{k}_s{s_}") for s_ in SEEDS])) if f"{k}_s0" in sk[es][v] else "–"
        if k == "RESTREND" and h == 1:
            return "–"
        return fmt(one(k)) if k in sk[es][v] else "–"

    e2 = lambda k: {"M2": SEEDS_OF("M2"), "M3": SEEDS_OF("M3")}.get(k, k)
    pct = lambda k, dro, sc="S1": f"{world_avg(rw, e2(k), dro, 'size', scen=sc)[0] * 100:.0f}%"
    wt = lambda r: r["rain"] + r["temp"]
    dk = lambda k: "M4" if k == "M4_ens" else k
    dmean = lambda k, key: (None if dk(k) not in dr["rows"] or dr["rows"][dk(k)][0][key] is None else
                            float(np.mean([r[key] for r in dr["rows"][dk(k)]])))

    def s4(k, box):
        if k in ("M2", "M3"):
            v = [se[box]["tests"][f"{k}_s{s_}"]["nogap"]["G1"]["primary"] for s_ in SEEDS]
            n = sum(x["pass"] for x in v)
            return f"{'pass' if n == 5 else 'fail' if n == 0 else 'mixed'} ({n}/5 fits)"
        x = se[box]["tests"][k]["nogap"]["G1"]["primary"]
        return f"{'pass' if x['pass'] else 'fail'} ({x['high_minus_low']:+.3f})"

    # ---- overview
    w("### Overview: all completed tests (blank = not yet run)\n")
    w("| Method | 1 Skill: 3-month, Box A / B | 2a Weather lean (per 100 mm) | 2c Error finding hidden damage, drought / other | "
      "3 2022: NDVI lost to weather (of −0.168) | 4 Sensitivity, Box A / B | Compute: 2022 split; parameters |")
    w("|---|---|---|---|---|---|---|")
    for k, lab in names:
        if k == "z-score":
            row = ["0 (the reference)", f"{ln['z-score_rain']['slope_per_100mm']:+.3f}", f"{pct(k, True)} / {pct(k, False)}", "0",
                   "– (no rain response)", f"0.1 ms; {tm['params']['M0']}"]
        elif k in ("c2w", "c2h"):
            mode = "weather" if k == "c2w" else "history"
            sk1 = f"{c2sk(mode, 'A', 6)} / {c2sk(mode, 'B', 6)} (4,000 px)" if c2s is not None else ""
            if k == "c2h":
                row = [sk1, "–", "–", "–", "–", ""]
            else:
                t4 = (f"{'pass' if c2g('A')['pass'] else 'fail'} ({c2g('A')['high_minus_low']:+.3f}) / {'pass' if c2g('B')['pass'] else 'fail'} "
                      f"({c2g('B')['high_minus_low']:+.3f})") if c2g("A") else ""
                gt = f"{gpu_s['drought']:.0f} s H100" if "drought" in gpu_s else f"~{cost['cells']['test 3 2022 full box']['cpu_h']:.1f} h CPU (est.)"
                row = [sk1, "–", "", f"{c2d['rain'] + c2d['temp']:+.3f}" if c2d else "", t4, f"{gt}; {cost['params'] / 1e6:.1f}M"]
        elif k == "Trees":
            row = [f"{sk_(k, 'A_test', 6)} / {sk_(k, 'B_test', 6)}", "–", "–", "–", "–", "–"]
        else:
            t3 = {"M4_ens": "M4", "z-score": "z-score"}.get(k, k)
            ms = tm["test3_s"].get(f"{t3} compiled maps", tm["test3_s"].get(t3))[1] * 1e3 * (5 if k == "M4_ens" else 1)
            par = tm["params"]["M4_ens" if k == "M4_ens" else k]
            row = [f"{sk_(k, 'A_test', 6)} / {sk_(k, 'B_test', 6)}",
                   f"{ln['M4_ens_rain']['slope_per_100mm']:+.3f}" if k == "M4_ens" else "–",
                   f"{pct(k, True)} / {pct(k, False)}", f"{dmean(k, 'rain') + dmean(k, 'temp'):+.3f}",
                   "n/a: one fixed slope per pixel" if k == "RESTREND" else "n/a: same value everywhere" if k == "M1" else f"{s4(k, 'A')} / {s4(k, 'B')}",
                   f"{'~' if k == 'M4_ens' else ''}{ms:.0f} ms; {par:,}" if ms >= 0.5 else f"<0.1 ms; {par:,}"]
            if k == "M4_ens":
                row = [b(v) for v in row]
        w(f"| {lab} | " + " | ".join(row) + " |")
    w("\nSkill: how much better than history alone (0 = no better). Weather lean: how much the unexplained part tracks the "
      "season's rain (0 = none; run for the M4 ensemble with the z-score as yardstick). Error: % of the true hidden damage, "
      "0% = perfect, averaged over five simulated worlds. Sensitivity: do pixels the model calls more rain-sensitive really "
      "respond more (Box A pre-registered). M2, M3: mean of 5 fits. Chronos-2 skill and sensitivity are on 4,000-pixel samples "
      f"(M4 ensemble on the same pixels: skill {sk_('M4_ens', 'A_test', 6, 'px')} / {sk_('M4_ens', 'B_test', 6, 'px')}).\n")

    # ---- verdicts
    de, dz, dR = (world_avg(rw, k, True, "size")[0] for k in ("M4_ens", "z-score", "RESTREND"))
    g4 = se["A"]["tests"]["M4_ens"]["nogap"]["G1"]["primary"]
    lo, hi = ln["M4_ens_rain"]["ci95"]
    w("### Claims (pre-registered)\n")
    w("| Test | Claim | Result | Verdict |\n|---|---|---|---|")
    if c2s is not None:
        mA, mB = (float(sk_("M4_ens", f"{x}_test", 6, "px")) for x in "AB")
        cA, cB = (float(c2sk("weather", x, 6)) for x in "AB")
        hA, hB = (float(c2sk("history", x, 6)) for x in "AB")
        sk_v = "M4 better on both boxes" if mA > cA and mB > cB else "Chronos-2 better on both boxes" if cA > mA and cB > mB else "split between boxes"
        w(f"| 1 Skill | M4 ensemble beats Chronos-2 at 3 months (same 4,000 px) | M4 {mA:.3f} / {mB:.3f}; Chronos-2 with weather {cA:.3f} / {cB:.3f}; "
          f"history only {hA:.3f} / {hB:.3f} | **{sk_v}** |")
    else:
        w("| 1 Skill | M4 ensemble beats Chronos-2 at 3 months | waiting for Chronos-2 | – |")
    w(f"| 2a Weather lean | M4 ensemble's leftover does not lean with rain | {ln['M4_ens_rain']['slope_per_100mm']:+.3f} per 100 mm "
      f"[{lo:+.3f}, {hi:+.3f}]; z-score {ln['z-score_rain']['slope_per_100mm']:+.3f} | **{'pass' if ln['claim_pass'] else 'fail'}** |")
    w(f"| 2c Hidden damage | M4 ensemble's drought error below the z-score's and RESTREND's | {de * 100:.0f}% vs {dz * 100:.0f}% and "
      f"{dR * 100:.0f}% | **{'pass' if de < min(dz, dR) else 'fail'}** |")
    if c2d:
        wt2 = c2d["rain"] + c2d["temp"]
        agree = -0.104 <= wt2 <= -0.066
        w(f"| 3 2022 drought | Chronos-2's weather share within 0.066–0.104 NDVI (M4 0.084) | Chronos-2 {-wt2:.3f} "
          f"({-wt2 / 0.084 * 100:.0f}% of M4's) | **{'agree' if agree else 'disagree'}** |")
    else:
        w("| 3 2022 drought | Chronos-2's weather share within 0.066–0.104 NDVI | waiting for Chronos-2 | – |")
    w(f"| 4 Sensitivity | M4 ensemble passes on Box A | {g4['high_minus_low']:+.3f} [{g4['high_minus_low_ci95'][0]:+.3f}, "
      f"{g4['high_minus_low_ci95'][1]:+.3f}] | **{'pass' if g4['pass'] else 'fail'}** |\n")

    # ---- test 1
    w("### Test 1. Forecast skill on unseen years (2019–25), better than history alone\n")
    w("| Method | Box A: 16 days | Box A: 3 months | Box B (unseen landscape): 16 days | Box B: 3 months | Drought starts, 3 months, A / B |")
    w("|---|---|---|---|---|---|")
    for k, lab in names:
        if k in ("z-score", "c2w", "c2h"):
            if k != "z-score":
                mode = "weather" if k == "c2w" else "history"
                row = ([c2sk(mode, "A", 1), c2sk(mode, "A", 6), c2sk(mode, "B", 1), c2sk(mode, "B", 6),
                        f"{c2sk(mode, 'A', 6, True)} / {c2sk(mode, 'B', 6, True)}"] if c2s is not None else [""] * 5)
                w(f"| {lab} (4,000 px) | " + " | ".join(row) + " |")
                if k == "c2h":
                    row = [sk_("M4_ens", "A_test", 1, "px"), sk_("M4_ens", "A_test", 6, "px"), sk_("M4_ens", "B_test", 1, "px"),
                           sk_("M4_ens", "B_test", 6, "px"), f"{sk_('M4_ens', 'A_test', 6, 'px_drought')} / {sk_('M4_ens', 'B_test', 6, 'px_drought')}"]
                    w("| M4 ensemble (same 4,000 px) | " + " | ".join(row) + " |")
            continue
        row = [sk_(k, "A_test", 1), sk_(k, "A_test", 6), sk_(k, "B_test", 1), sk_(k, "B_test", 6),
               f"{sk_(k, 'A_test', 6, 'drought')} / {sk_(k, 'B_test', 6, 'drought')}" if k != "Trees" else "–"]
        w(f"| {lab} | " + " | ".join(b(v) if k == "M4_ens" else v for v in row) + " |")
    w(f"\nDrought starts: forecast windows with box rain more than 60 mm below normal ({sk['A_test']['drought']['n_starts']} on A, "
      f"{sk['B_test']['drought']['n_starts']} on B, of {sk['A_test']['all']['n_starts']}). Chronos-2 and the M4 ensemble are compared on the same "
      f"4,000-pixel samples (M4 ensemble there: {sk_('M4_ens', 'A_test', 6, 'px')} / {sk_('M4_ens', 'B_test', 6, 'px')}).\n")

    # ---- test 2a
    w("### Test 2a. Does what the model leaves unexplained lean with the weather? (13 seasons, 2019–25, 2 km blocks)\n")
    w("| Method | Lean with rain, NDVI per 100 mm [95% interval] | Lean with evaporative demand, per 100 mm |\n|---|---|---|")
    for k, lab in (("z-score", "z-score (current practice)"), ("M4_ens", "**M4 ensemble**")):
        a_, e_ = ln[f"{k}_rain"], ln[f"{k}_et0"]
        w(f"| {lab} | {a_['slope_per_100mm']:+.3f} [{a_['ci95'][0]:+.3f}, {a_['ci95'][1]:+.3f}] | {e_['slope_per_100mm']:+.3f} "
          f"[{e_['ci95'][0]:+.3f}, {e_['ci95'][1]:+.3f}] |")
    w("| Chronos-2 | | |")
    w(f"\nM4's lean is {ln['M4_share_of_z_lean_rain'] * 100:.0f}% of current practice's. Assumes real management does not follow the weather.\n")

    # ---- test 2c
    w("### Test 2c. Finding hidden land damage in simulated worlds (error as % of the true damage; 0% = perfect)\n")
    w("| Method | Drought seasons (4) | Other seasons (9) | All 13 | Drought, when damage worsens in droughts |\n|---|---|---|---|---|")
    for k, lab in names[:6]:
        row = [pct(k, True), pct(k, False), pct(k, None), pct(k, True, "S2a")]
        w(f"| {lab} | " + " | ".join(b(v) if k == "M4_ens" else v for v in row) + " |")
    w("| Chronos-2 | | | | |")
    w("\nFive worlds, each built from one M4 fit; a known damage of −0.01 NDVI per 16 days is added to a random half of 2 km "
      "blocks. Chronos-2 is scored in a sixth world it builds itself (GPU).\n")

    # ---- test 3
    w(f"### Test 3. The March–May 2022 drought, Box A: splitting the observed {dr['obs']:+.3f} NDVI\n")
    w("| Method | Starting deficit | Rain | Evaporative demand | **Weather total** | Not explained (the land) |\n|---|---|---|---|---|---|")
    for k, lab in names[:6] + [("Chronos-2", "Chronos-2")]:
        if k == "Chronos-2":
            w(f"| {lab} | " + (" | ".join(f"{c2d[x]:+.3f}" for x in ("start_deficit", "rain", "temp")) +
                               f" | {c2d['rain'] + c2d['temp']:+.3f} | {c2d['residual']:+.3f} |" if c2d else " | | | | |"))
            continue
        f_ = lambda key: "–" if dmean(k, key) is None or (key == "temp" and k != "M4_ens") else f"{dmean(k, key):+.3f}"
        row = [f_("start_deficit"), f_("rain"), f_("temp"), f"{dmean(k, 'rain') + dmean(k, 'temp'):+.3f}", f_("residual")]
        w(f"| {lab} | " + " | ".join(b(v) if k == "M4_ens" else v for v in row) + " |")
    w(f"\nM4 ensemble weather total across its 5 fits: {min(wt(r) for r in dr['rows']['M4']):+.3f} to {max(wt(r) for r in dr['rows']['M4']):+.3f}.\n")

    # ---- test 4
    w("### Test 4. Do pixels the model calls more rain-sensitive really respond more? (unseen years, gap-free starts)\n")
    w("| Method | Box A (pre-registered) | Box B (unseen landscape) |\n|---|---|---|")
    for k, lab in names[1:6]:
        if k in ("RESTREND", "M1"):
            w(f"| {lab} | n/a: {'one fixed slope per pixel' if k == 'RESTREND' else 'same value everywhere'} | |")
            continue
        def c4(box):
            if k in ("M2", "M3"):
                return s4(k, box)
            x = se[box]["tests"][k]["nogap"]["G1"]["primary"]
            return f"{'pass' if x['pass'] else 'fail'}: {x['high_minus_low']:+.3f} [{x['high_minus_low_ci95'][0]:+.3f}, {x['high_minus_low_ci95'][1]:+.3f}]"
        row = [c4("A"), c4("B")]
        w(f"| {lab} | " + " | ".join(b(v) if k == "M4_ens" else v for v in row) + " |")
    if c2g("A"):
        w(f"| Chronos-2 (4,000 px) | {g4s(c2g('A'))} | {g4s(c2g('B'))} |")
        w(f"| M4 ensemble (same 4,000 px) | {g4s(se['A']['px']['tests']['M4_ens']['nogap']['G1']['primary'])} | "
          f"{g4s(se['B']['px']['tests']['M4_ens']['nogap']['G1']['primary'])} |")
        w(f"\nMean sensitivity (NDVI per 100 mm, Box A, same pixels): Chronos-2 {se['A']['px']['delta_mean']['Chronos-2']:.3f}, "
          f"M4 ensemble {se['A']['px']['delta_mean']['M4_ens']:.3f}.")
    else:
        w("| Chronos-2 | | |")
    w("\nNumbers: extra greenness per 100 mm of rain in the most- vs least-sensitive third, with 95% interval over years; "
      "pass = interval above zero.\n")

    # ---- compute
    w("### Compute (laptop CPU, Apple M4 Max)\n")
    w("| Method | Parameters | 2022 split, full box | Sensitivity map, full box | Training |\n|---|---|---|---|---|")
    for k, lab, tr in (("M1", "M1", "–"), ("M4_ens", "**M4 ensemble**", "5 × ~2.5 min H100 (~80 min CPU each)")):
        t3 = "M4" if k == "M4_ens" else k
        n = 5 if k == "M4_ens" else 1
        a3 = tm["test3_s"].get(f"{t3} compiled maps", tm["test3_s"][t3])[1] * 1e3 * n
        a4 = tm["test4_s"][t3][1] * 1e3 * n
        w(f"| {lab} | {tm['params']['M4_ens' if k == 'M4_ens' else k]:,} | {'~' if n > 1 else ''}{a3:.0f} ms | {'~' if n > 1 else ''}{a4:.0f} ms | {tr} |")
    c3, c4_ = cost["cells"]["test 3 2022 full box"], cost["cells"]["test 4 Delta (sample, both boxes)"]
    t3g = f"{gpu_s['drought'] / 60:.0f} min H100 (CPU est. ~{c3['cpu_h']:.1f} h)" if "drought" in gpu_s else f"~{c3['cpu_h']:.1f} h (est.)"
    w(f"| Chronos-2 | {cost['params']:,} | {t3g} | – | none (pretrained) |\n")

    # ---- GPU cost
    w("### Chronos-2 GPU run, estimated cost (Modal H100)\n")
    w("| Test | Central | Pessimistic |\n|---|---|---|")
    lab_c = {"test 1 skill (both modes, both boxes)": "1 Skill", "test 3 2022 full box": "3 2022 drought",
             "test 4 Delta (sample, both boxes)": "4 Sensitivity"}
    tot = [0.0, 0.0]
    for nm, c in cost["cells"].items():
        w(f"| {lab_c[nm]} | ${c['usd_central_tf32']:.2f} | ${c['usd_pessimistic_fp32']:.2f} |")
        tot = [tot[0] + c["usd_central_tf32"], tot[1] + c["usd_pessimistic_fp32"]]
    Lw = int(cost["world_L_rule"])
    wv = cost["world_by_L"][str(Lw)]
    w(f"| 2c Chronos-2 world ({Lw}×{Lw} pixels) | ${wv['usd_central_tf32']:.2f} | ${wv['usd_pessimistic_fp32']:.2f} |")
    w(f"| **Total** | **${tot[0] + wv['usd_central_tf32']:.2f}** | **${tot[1] + wv['usd_pessimistic_fp32']:.2f}** |")
    if gpu_s:
        w("\n**Actual H100 time** (inside the container; Modal also bills start-up): " +
          ", ".join(f"{c} {v / 60:.1f} min (${v / 3600 * 3.95:.2f})" for c, v in gpu_s.items()) +
          f"; total ${sum(gpu_s.values()) / 3600 * 3.95:.2f}.")
        cen = {"skill": "test 1 skill (both modes, both boxes)", "drought": "test 3 2022 full box", "sens": "test 4 Delta (sample, both boxes)"}
        ratio = sum(gpu_s[c] for c in cen if c in gpu_s) / sum(cost["cells"][cen[c]]["h100_h_central_tf32"] * 3600 for c in cen if c in gpu_s)
        w(f"Measured H100 time was {ratio:.1f}x the central estimate. At that rate the 2c Chronos-2 world would cost about "
          f"${wv['usd_central_tf32'] * ratio:.0f}.")
    w("\nEstimates, not measurements: laptop CPU time measured, H100 throughput assumed (120 / 33 TFLOP/s), $3.95 per H100-hour. "
      "2a for Chronos-2 would add about $0.4 / $1.5.")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    debug = "--debug" in sys.argv
    os.makedirs(out, exist_ok=True)
    if cmd == "plan":
        plan(out, debug)
    elif cmd == "skill":
        skill(out, debug)
    elif cmd == "attrib":
        attrib(out, debug, int(sys.argv[sys.argv.index("--tile") + 1]) if "--tile" in sys.argv else None,
               tuple(int(v) for v in sys.argv[sys.argv.index("--worlds") + 1].split(",")) if "--worlds" in sys.argv else (0,))
    elif cmd == "drought":
        drought(out, debug)
    elif cmd == "lean":
        lean(out, debug)
    elif cmd == "sens":
        sens(out, debug)
    elif cmd == "timing":
        timing(out)
    elif cmd == "report":
        report(out)
    elif cmd == "summary":
        summary(out)
