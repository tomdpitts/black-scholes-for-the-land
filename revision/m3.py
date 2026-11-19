"""M3: M2 with site traits and a second forcing, after Kratzert et al. (2019).

Two kinds of input:
  site traits s (static per pixel: mean NDVI, seasonal amplitude, climatological rain, and from
    the DEM elevation, slope, topographic position). Traits set how a pixel responds, and its
    rain memory lam(s). They never get a share of the anomaly: they do not differ from normal.
  forcings (vary in time): rain r, through the nonlinear drift and the memory bucket w, as in M2;
    temperature anomaly T' (deg C; "year" configs: minus its mean over the preceding year, so the
    input is weather and not the warming trend), entering linearly about normal with sensitivities that the
    network sets from state, rain and traits:

      a_{t+1} = a_t + mu(f) + g_mu(f) T'_{t+1} + sigma(f) exp(g_s(f) T'_{t+1}) eps,
      f = (a_t, n_t, phase, s, w_t, r_{t+1}).

    g_mu is the one-step temperature Greek. Linearity in T' keeps a regional, box-wide temperature
    series from acting as a time stamp the network can memorise.
Option gated: traits also gate the hidden units (Kratzert's entity-aware input gate).

All traits are standardised on Box A. Fit as M2: full-batch Adam on the one-step Gaussian
likelihood, Box A 2003-2017, early stopping on 2018.

  .venv/bin/python revision/m3.py ref <out>                 -> M0 reference scores
  .venv/bin/python revision/m3.py fit <config> <seed> <out> -> params_<config>_s<seed>.npy, metrics_*.json
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aoi  # noqa: E402
import model as M  # noqa: E402
from run import load, H, S  # noqa: E402

REF = "runs/20260914_1708_pipeline_h100_fairfight"
HIDDEN = 64
CONFIGS = {  # (topography traits, temperature forcing, trait gate, what sets the temperature sensitivity)
    "base": (False, None, False, None),
    "topo": (True, None, False, None),
    "temp": (False, "raw", False, "net"),
    "both": (True, "raw", False, "net"),
    "gated": (True, "raw", True, "net"),
    # addendum 1, 2026-10-21: temperature as weather, relative to the past year
    "tempd": (False, "year", False, "net"),
    "bothd": (True, "year", False, "net"),
    "topo_gated": (True, None, True, None),
    # addendum 2: the temperature sensitivity is set by the traits alone (linear), not by the network
    "tempk": (False, "year", False, "traits"),
    "bothk": (True, "year", False, "traits"),
    # addendum 4: evaporative demand (ERA5 FAO reference ET anomaly) drains the water bucket
    "topo_pet": (True, "pet", False, "bucket"),
}
TOPO = ("elev", "slope", "tpi")
YEAR = 23  # 16-day windows per year


# ----------------------------------------------------------------------------- data
def load_box(box, topo, temp):
    """built.npz plus standardised traits s [N,N,nx] and forcing ex [T,N,N,K] (K = 0 or 1)."""
    d = load(box)
    s = [d["x"][..., k] for k in range(3)]
    if topo:
        t = np.load(f"data/{box}/topo.npz")
        s += [t[k] for k in TOPO]
    ref = load("A") if box == "B" else d
    s_ref = [ref["x"][..., k] for k in range(3)]
    if topo:
        s_ref += [np.load("data/A/topo.npz")[k] for k in TOPO]
    d["s"] = np.stack([(v - r.mean()) / r.std() for v, r in zip(s, s_ref)], -1).astype(np.float32)
    T = d["a"].shape[0]
    if temp:
        e = np.load(f"data/{box}/era5.npz")
        assert (e["dates"] == d["dates"]).all()
        tp = (e["tmean"] - e["tmean_clim"]).astype(np.float64)
        if temp == "pet":  # evaporative-demand anomaly, 100 mm per window like the rain
            tp = (e["pet"] - e["pet_clim"]).astype(np.float64)
        if temp == "year":  # weather, not climate: subtract the mean anomaly of the preceding year
            c = np.concatenate([np.zeros((1,) + tp.shape[1:]), np.cumsum(tp, 0)])
            t = np.arange(T)
            lo = np.maximum(t - YEAR, 0)
            tp = tp - np.where((t > 0)[:, None, None], (c[t] - c[lo]) / np.maximum(t - lo, 1)[:, None, None], 0)
        d["ex"] = tp[..., None].astype(np.float32)
    else:
        d["ex"] = np.zeros((T, aoi.N, aoi.N, 0), np.float32)
    return d


def jaxify(d):
    return {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "s", "phase", "ex")}


# ----------------------------------------------------------------------------- model
def init(key, nx, K, gated, sens="net"):
    """How the second forcing enters. "net": sensitivities of mu and log sigma from the network and
    all features; "traits": a linear function of the traits alone; "bucket": evaporative demand
    drains the water input, r_net = r - k(s) PET' with k(s) linear in the traits."""
    F = 1 + 1 + 2 + nx + 2
    heads = 2 + 2 * K if sens == "net" else 2
    p = {"lin_mu": jnp.zeros(F), "b_mu": jnp.zeros(()), "lin_ls": jnp.zeros(F), "b_ls": jnp.log(0.05),
         "lam": jnp.zeros(1 + nx),
         "W1": jax.random.normal(key, (F, HIDDEN)) / jnp.sqrt(F), "b1": jnp.zeros(HIDDEN),
         "W2": jnp.zeros((HIDDEN, heads)), "b2": jnp.zeros(heads)}
    if sens == "bucket":
        p["k"] = jnp.zeros(1 + nx)  # starts at zero: exactly the model without the second forcing
    else:
        p["G"], p["bG"] = jnp.zeros((F if sens == "net" else nx, 2 * K)), jnp.zeros(2 * K)
    if gated:
        p["Wg"], p["bg"] = jnp.zeros((nx, HIDDEN)), jnp.zeros(HIDDEN)  # gate starts at 1 everywhere
    return p


def lam(p, s):
    return jax.nn.sigmoid(p["lam"][0] + s @ p["lam"][1:])


def net_water(p, r, ex, s):
    """Water input of a window: rain, less k(s) times the evaporative-demand anomaly ("bucket")."""
    return r - (p["k"][0] + s @ p["k"][1:]) * ex[..., 0] if "k" in p else r


def step(p, a, w, r, ex, phase, s):
    """One transition: (mean, sigma, next w, temperature sensitivities g [N,N,2K])."""
    r = net_water(p, r, ex, s)
    f = M.features(a, M.neighbourhood(a), w, r, phase, s, True)
    h = jax.nn.gelu(f @ p["W1"] + p["b1"])
    if "Wg" in p:
        h = h * 2 * jax.nn.sigmoid(s @ p["Wg"] + p["bg"])
    o = h @ p["W2"] + p["b2"]
    K = 0 if "k" in p else ex.shape[-1]
    if K == 0:
        g = jnp.zeros(a.shape + (0,))
    elif p["G"].shape[0] == s.shape[-1]:  # sensitivity set by the traits alone
        g = s @ p["G"] + p["bG"]
    else:
        g = f @ p["G"] + p["bG"] + o[..., 2:]
    mu = f @ p["lin_mu"] + p["b_mu"] + o[..., 0] + jnp.sum(g[..., :K] * ex[..., :K], -1)
    ls = f @ p["lin_ls"] + p["b_ls"] + o[..., 1] + jnp.sum(g[..., K:] * ex[..., :K], -1)
    return a + mu, jnp.exp(jnp.clip(ls, -6, 1)), lam(p, s) * w + r, g


def w_path(p, D):
    l = lam(p, D["s"])
    w0 = D["rclim"][0] / (1 - l)
    _, ws = jax.lax.scan(lambda w, x: (l * w + net_water(p, x[0], x[1], D["s"]),) * 2, w0, (D["r"][1:], D["ex"][1:]))
    return jnp.concatenate([w0[None], ws], 0)


def nll_sequence(p, D):
    w = w_path(p, D)

    def body(_, t):
        m, sg, _, _ = step(p, D["a"][t], w[t], D["r"][t + 1], D["ex"][t + 1], D["phase"][t + 1], D["s"])
        z = (D["a"][t + 1] - m) / sg
        return None, 0.5 * z ** 2 + jnp.log(sg) + 0.5 * jnp.log(2 * jnp.pi)

    return jax.lax.scan(body, None, jnp.arange(D["a"].shape[0] - 1))[1]


def fit(D, train_t, val_t, gated, sens, seed, steps=2000, lr=3e-3, log=print):
    p = init(jax.random.PRNGKey(seed), D["s"].shape[-1], D["ex"].shape[-1], gated, sens)
    obs = D["obs"][1:] & D["obs"][:-1]
    m_tr, m_va = obs & train_t[:, None, None], obs & val_t[:, None, None]
    n_tr, n_va = float(m_tr.sum()), float(m_va.sum())
    opt = optax.adam(lr)
    st = opt.init(p)

    @jax.jit
    def upd(p, st):
        l, g = jax.value_and_grad(lambda p: jnp.sum(nll_sequence(p, D) * m_tr) / n_tr)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    val = jax.jit(lambda p: jnp.sum(nll_sequence(p, D) * m_va) / n_va)
    best, best_p = np.inf, p
    for i in range(steps):
        p, st, l = upd(p, st)
        if i % 50 == 0 or i == steps - 1:
            v = float(val(p))
            log(f"  step {i:5d} train nll {float(l):.4f} val nll {v:.4f}")
            if v < best:
                best, best_p = v, p
    return best_p, best


# ----------------------------------------------------------------------------- scores
def score(step_fn, w, D, mask_t, key=jax.random.PRNGKey(1), px=None):
    """One-step Gaussian CRPS and 1..H-step sampled-rollout CRPS (S samples, realised forcings).
    step_fn(a, w, t_next) -> (mean, sigma, next w). px: optional [N,N] pixel mask; the grid is still run
    whole (neighbourhood features), only the scored pixels change."""
    T = D["a"].shape[0]
    pm = jnp.ones(D["a"].shape[1:], bool) if px is None else jnp.asarray(px, bool)

    def body(_, t):
        m, sg, _ = step_fn(D["a"][t], w[t], t + 1)
        return None, M.crps_gauss(m, sg, D["a"][t + 1])

    cr = np.asarray(jax.lax.scan(body, None, jnp.arange(T - 1))[1])
    ok = np.asarray(D["obs"][1:] & D["obs"][:-1] & pm) & mask_t[:, None, None]

    @jax.jit
    def roll(t, k):
        def one(c, h):
            a, w_, k_ = c
            k_, sub = jax.random.split(k_)
            m, sg, w_ = step_fn(a, w_, t + 1 + h)
            a = m + sg * jax.random.normal(sub, m.shape)
            return (a, w_, k_), a

        def sample(k_):
            return jax.lax.scan(one, (D["a"][t], w[t], k_), jnp.arange(H))[1]

        s = jax.vmap(sample)(jax.random.split(k, S))  # [S,H,N,N]
        y = jax.lax.dynamic_slice_in_dim(D["a"], t + 1, H, 0)
        ob = jax.lax.dynamic_slice_in_dim(D["obs"], t + 1, H, 0) & D["obs"][t][None] & pm[None]
        c = jax.vmap(M.crps_samples, in_axes=(1, 0))(s, y)
        return jnp.sum(c * ob, (1, 2)), jnp.sum(ob, (1, 2))

    num, den = np.zeros(H), np.zeros(H)
    for t in [t for t in range(T - H) if mask_t[t]]:
        key, sub = jax.random.split(key)
        n_, d_ = roll(t, sub)
        num += np.asarray(n_)
        den += np.asarray(d_)
    return {"onestep_crps": float(cr[ok].mean()), "rollout_crps": (num / den).tolist()}


def score_ensemble(ps, D, mask_t, per=4, key=jax.random.PRNGKey(1), px=None):
    """1..H-step rollout CRPS of the equal-weight mixture of several fits: `per` samples from each
    fit, drawn by the same sampler as score(), pooled, then scored. (Stacking the fits' parameters
    under one vmap gave different results once compiled; drawing per fit avoids that.) px as in score()."""
    T = D["a"].shape[0]
    pm = jnp.ones(D["a"].shape[1:], bool) if px is None else jnp.asarray(px, bool)
    rollers = []
    for p in ps:
        w, f = w_path(p, D), m3_step_fn(p, D)

        @jax.jit
        def roll(t, k, w=w, f=f):
            def one(c, h):
                a, w_, k_ = c
                k_, sub = jax.random.split(k_)
                m, sg, w_ = f(a, w_, t + 1 + h)
                a = m + sg * jax.random.normal(sub, m.shape)
                return (a, w_, k_), a
            return jax.vmap(lambda k_: jax.lax.scan(one, (D["a"][t], w[t], k_), jnp.arange(H))[1])(jax.random.split(k, per))
        rollers.append(roll)

    @jax.jit
    def crps(s, t):
        y = jax.lax.dynamic_slice_in_dim(D["a"], t + 1, H, 0)
        ob = jax.lax.dynamic_slice_in_dim(D["obs"], t + 1, H, 0) & D["obs"][t][None] & pm[None]
        c = jax.vmap(M.crps_samples, in_axes=(1, 0))(s, y)
        return jnp.sum(c * ob, (1, 2)), jnp.sum(ob, (1, 2))

    num, den = np.zeros(H), np.zeros(H)
    for t in [t for t in range(T - H) if mask_t[t]]:
        key, *subs = jax.random.split(key, len(ps) + 1)
        n_, d_ = crps(jnp.concatenate([r(t, k) for r, k in zip(rollers, subs)]), t)
        num += np.asarray(n_)
        den += np.asarray(d_)
    return {"rollout_crps": (num / den).tolist()}


def evalsets(dA, dB):
    tA, tB = dA["year"][1:], dB["year"][1:]
    return {"A_test": ("A", np.isin(tA, list(aoi.TEST_YEARS))),
            "B_all": ("B", np.ones(len(tB), bool)),
            "B_test": ("B", np.isin(tB, list(aoi.TEST_YEARS)))}


def m3_step_fn(p, D):
    def f(a, w, t):
        m, sg, w2, _ = step(p, a, w, D["r"][t], D["ex"][t], D["phase"][t], D["s"])
        return m, sg, w2
    return f


# ----------------------------------------------------------------------------- commands
def ref(out):
    """M0 (history only) scored on the same sets: the skill denominator."""
    P = np.load(f"{REF}/params.npy", allow_pickle=True).item()
    p0 = {k: jnp.asarray(v) for k, v in P["M0"].items()}
    dA, dB = load("A"), load("B")
    res = {}
    for es, (box, mt) in evalsets(dA, dB).items():
        d = dA if box == "A" else dB
        D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
        w = M.w_path(p0, D, False)
        res[es] = score(lambda a, w_, t: M.step(p0, a, w_, D["r"][t], D["phase"][t], D["x"], False), w, D, mt)
        print("M0", es, res[es], flush=True)
    json.dump(res, open(f"{out}/metrics_M0.json", "w"), indent=1)


def run_fit(config, seed, out):
    topo, temp, gated, sens = CONFIGS[config]
    tag = f"{config}_s{seed}"
    logf = open(f"{out}/progress_{tag}.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    dA, dB = load_box("A", topo, temp), load_box("B", topo, temp)
    JA, JB = jaxify(dA), jaxify(dB)
    tgt = dA["year"][1:]
    train_t, val_t = np.isin(tgt, list(aoi.TRAIN_YEARS)), np.isin(tgt, list(aoi.VAL_YEARS))
    aoi.assert_allowed_for_training("A", np.unique(tgt[train_t | val_t]))
    n = int(np.where(train_t | val_t)[0].max()) + 2  # the fit never sees a step past 2018
    Jfit = {k: (v if k == "s" else v[:n]) for k, v in JA.items()}
    t0 = time.time()
    log(f"fit {tag}: traits {JA['s'].shape[-1]}, forcings rain + {JA['ex'].shape[-1]} ({temp}, sensitivity {sens}), gated {gated}, steps 0..{n - 1}")
    p, best = fit(Jfit, jnp.asarray(train_t[:n - 1]), jnp.asarray(val_t[:n - 1]), gated, sens, seed, log=log)
    secs = time.time() - t0
    log(f"fit done {secs:.0f}s best val nll {best:.4f}")
    np.save(f"{out}/params_{tag}.npy", {k: np.asarray(v) for k, v in p.items()}, allow_pickle=True)
    m0 = json.load(open(f"{out}/metrics_M0.json"))
    res = {"config": config, "seed": seed, "best_val_nll": best, "fit_seconds": secs}
    for es, (box, mt) in evalsets(dA, dB).items():
        D = JA if box == "A" else JB
        r = score(m3_step_fn(p, D), w_path(p, D), D, mt)
        r["onestep_skill"] = 1 - r["onestep_crps"] / m0[es]["onestep_crps"]
        r["rollout_skill"] = (1 - np.array(r["rollout_crps"]) / np.array(m0[es]["rollout_crps"])).tolist()
        res[es] = r
        log(f"{tag} {es}: one-step skill {r['onestep_skill']:.3f}; 3-month skill {r['rollout_skill'][-1]:.3f}")
    json.dump(res, open(f"{out}/metrics_{tag}.json", "w"), indent=1)
    log(f"wrote {out}/metrics_{tag}.json")


def run_ensemble(R, config, seeds):
    """Score the equal-weight ensemble of <R>/params_<config>_s<seed>.npy; skill against <R>/metrics_M0.json."""
    topo, temp = CONFIGS[config][:2]
    dA, dB = load_box("A", topo, temp), load_box("B", topo, temp)
    JA, JB = jaxify(dA), jaxify(dB)
    ps = [{k: jnp.asarray(v) for k, v in np.load(f"{R}/params_{config}_s{s}.npy", allow_pickle=True).item().items()}
          for s in seeds]
    m0 = json.load(open(f"{R}/metrics_M0.json"))
    res = {"config": config, "seeds": seeds}
    for es, (box, mt) in evalsets(dA, dB).items():
        r = score_ensemble(ps, JA if box == "A" else JB, mt)
        r["rollout_skill"] = (1 - np.array(r["rollout_crps"]) / np.array(m0[es]["rollout_crps"])).tolist()
        res[es] = r
        print(f"ensemble {config} {es}: 1-step skill {r['rollout_skill'][0]:.3f}; 3-month skill {r['rollout_skill'][-1]:.3f}", flush=True)
    json.dump(res, open(f"{R}/metrics_ens_{config}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "ref":
        ref(sys.argv[2])
    elif sys.argv[1] == "ens":  # m3.py ens <run dir> <config> <seed,seed,...>
        run_ensemble(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(",")])
    else:
        run_fit(sys.argv[2], int(sys.argv[3]), sys.argv[4])
