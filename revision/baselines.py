"""Stronger rain-conditioned baselines for reviewer 2: a gradient-boosted distributed-lag model
(B1 GBM) and M2 with a learned per-pixel GRU memory (B2 GRU-M2).

Pre-registration: runs/20261006_1759_baselines/PREREG.md (written before this ran).
Same splits, masks and scores as run.py; M0 and M2 reference numbers from the fair-fight run.

  .venv/bin/python revision/baselines.py gbm   -> runs/20261006_1759_baselines/metrics_gbm.json
  .venv/bin/python revision/baselines.py gru   -> runs/20261006_1759_baselines/metrics_gru.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aoi  # noqa: E402
from run import load, H, S  # noqa: E402

OUT = "runs/20261006_1759_baselines"
REF = "runs/20260914_1708_pipeline_h100_fairfight"
LAGS_A = 3   # a_t, a_{t-1}, a_{t-2}
LAGS_R = 6   # r_{t+1} ... r_{t-4}


def log(msg):
    print(msg, flush=True)


def evalsets(A, B):
    tgt = A["year"][1:]
    return {"A_test": (A, np.isin(tgt, list(aoi.TEST_YEARS))), "B_all": (B, np.ones(B["a"].shape[0] - 1, bool))}


def starts_of(D, mt):
    return np.array([t for t in range(D["a"].shape[0] - H) if mt[t]])


def crps_gauss(mu, sig, y):
    from scipy.special import erf
    z = (y - mu) / sig
    return sig * (z * erf(z / np.sqrt(2)) + np.sqrt(2 / np.pi) * np.exp(-0.5 * z ** 2) - 1 / np.sqrt(np.pi))


def crps_samples(s, y):
    """s [S,...], y [...]"""
    n = s.shape[0]
    t1 = np.abs(s - y[None]).mean(0)
    t2 = np.abs(s[:, None] - s[None, :]).mean((0, 1)) * n / (n - 1)
    return t1 - 0.5 * t2


def summarise(res, name):
    ref = json.load(open(f"{REF}/metrics.json"))
    out = {}
    for es, r in res.items():
        m0_1 = ref["onestep"][es]["M0"]["crps"]
        m0_h = np.array(ref["rollout"][es]["M0"]["crps"])
        m2_h = np.array(ref["rollout"][es]["M2"]["crps"])
        out[es] = {"onestep_crps": r["onestep_crps"], "onestep_skill_vs_M0": 1 - r["onestep_crps"] / m0_1,
                   "rollout_crps": r["rollout_crps"], "rollout_skill_vs_M0": (1 - np.array(r["rollout_crps"]) / m0_h).tolist(),
                   "M2_rollout_skill_vs_M0": (1 - m2_h / m0_h).tolist(),
                   "M2_h6_crps_over_this_h6_crps": float(m2_h[-1] / r["rollout_crps"][-1])}
        log(f"{name} {es}: one-step skill {out[es]['onestep_skill_vs_M0']:.3f}; rollout skill "
            + " ".join(f"{v:.3f}" for v in out[es]["rollout_skill_vs_M0"])
            + f"; M2 " + " ".join(f"{v:.3f}" for v in out[es]["M2_rollout_skill_vs_M0"]))
    return out


# ============================================================================ B1: GBM
def nbhd(a):
    from scipy.ndimage import uniform_filter
    return uniform_filter(a, size=5, mode="nearest", axes=(-2, -1))


def gbm_features(a_lags, n, D, t_next, x):
    """a_lags: list of LAGS_A arrays [...,N,N] (a_t first); t_next: index of target step.
    Returns [M,F] float32 rows for every pixel in the leading dims."""
    shape = a_lags[0].shape
    T = D["r"].shape[0]
    cols = [al for al in a_lags] + [n]
    cols += [np.broadcast_to(D["phase"][t_next][k], shape) for k in range(2)]
    cols += [np.broadcast_to(x[..., k], shape) for k in range(3)]
    cols += [np.broadcast_to(D["r"][min(max(t_next - L, 0), T - 1)], shape) for L in range(LAGS_R)]
    cols += [np.broadcast_to(D["rclim"][t_next], shape)]
    return np.stack([c.reshape(-1) for c in cols], 1).astype(np.float32)


def gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    A, B = load("A"), load("B")
    rng = np.random.default_rng(0)
    nA = nbhd(A["a"])
    tgt = A["year"][1:]
    T, N = A["a"].shape[0], A["a"].shape[1]

    def rows(ts, max_rows):
        X, Y = [], []
        per = max(1, max_rows // len(ts))
        for t in ts:
            ok = (A["obs"][t] & A["obs"][t + 1]).ravel()
            idx = np.where(ok)[0]
            idx = rng.choice(idx, min(per, len(idx)), replace=False)
            lags = [A["a"][max(t - k, 0)] for k in range(LAGS_A)]
            X.append(gbm_features(lags, nA[t], A, t + 1, A["x"])[idx])
            Y.append((A["a"][t + 1] - A["a"][t]).ravel()[idx])
        return np.concatenate(X), np.concatenate(Y)

    tr = [t for t in range(2, T - 1) if tgt[t] in aoi.TRAIN_YEARS]
    va = [t for t in range(2, T - 1) if tgt[t] in aoi.VAL_YEARS]
    Xtr, Ytr = rows(tr, 3_000_000)
    Xva, Yva = rows(va, 300_000)
    log(f"gbm rows: train {len(Ytr)} val {len(Yva)}")
    t0 = time.time()
    mean = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=0).fit(Xtr, Ytr)
    res2 = np.log((Ytr - mean.predict(Xtr)) ** 2 + 1e-8)
    var = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=500,
                                        early_stopping=True, validation_fraction=0.1, random_state=0).fit(Xtr, res2)
    c = float(np.mean((Yva - mean.predict(Xva)) ** 2) / np.mean(np.exp(var.predict(Xva))))
    log(f"gbm fit {time.time() - t0:.0f}s; iters mean {mean.n_iter_} var {var.n_iter_}; variance scale {c:.3f}")

    def predict(X):
        return mean.predict(X), np.sqrt(np.exp(var.predict(X)) * c)

    res = {}
    for es, (D, mt) in evalsets(A, B).items():
        nD = nbhd(D["a"])
        num = den = 0.0
        for t in np.where(mt)[0]:
            m = (D["obs"][t] & D["obs"][t + 1]).ravel()
            if t < 0 or not m.any():
                continue
            lags = [D["a"][max(t - k, 0)] for k in range(LAGS_A)]
            mu, sg = predict(gbm_features(lags, nD[t], D, t + 1, D["x"]))
            cr = crps_gauss(D["a"][t].ravel() + mu, sg, D["a"][t + 1].ravel())
            num += cr[m].sum()
            den += m.sum()
        one = float(num / den)
        st = starts_of(D, mt)
        rnum, rden = np.zeros(H), np.zeros(H)
        t0 = time.time()
        for k, t in enumerate(st):
            cur = np.broadcast_to(D["a"][t], (S, N, N)).copy()
            lags = [cur] + [np.broadcast_to(D["a"][max(t - k_, 0)], (S, N, N)).copy() for k_ in range(1, LAGS_A)]
            samp = np.zeros((S, H, N, N), np.float32)
            g = np.random.default_rng(int(t))
            for h in range(H):
                mu, sg = predict(gbm_features(lags, nbhd(lags[0]), D, t + 1 + h, D["x"]))
                new = lags[0] + (mu + sg * g.standard_normal(mu.shape)).reshape(S, N, N)
                lags = [new] + lags[:-1]
                samp[:, h] = new
            ob = D["obs"][t + 1:t + 1 + H] & D["obs"][t][None]
            for h in range(H):
                c_ = crps_samples(samp[:, h], D["a"][t + 1 + h])
                rnum[h] += c_[ob[h]].sum()
                rden[h] += ob[h].sum()
            if k % 10 == 0:
                log(f"  {es} rollout {k}/{len(st)} {time.time() - t0:.0f}s")
        res[es] = {"onestep_crps": one, "rollout_crps": (rnum / rden).tolist()}
    out = summarise(res, "GBM")
    out["fit"] = {"train_rows": int(len(Ytr)), "mean_iters": int(mean.n_iter_), "var_iters": int(var.n_iter_), "var_scale": c}
    json.dump(out, open(f"{OUT}/metrics_gbm.json", "w"), indent=1)
    log(f"wrote {OUT}/metrics_gbm.json")


# ============================================================================ B2: GRU-M2
HG = 32


def gru():
    import jax
    import jax.numpy as jnp
    import optax
    import model as M

    A, B = load("A"), load("B")
    T, N = A["a"].shape[0], A["a"].shape[1]
    F = M.N_FEAT

    def init(key):
        k = jax.random.split(key, 4)
        p = M.init_params(k[0], mlp=True)
        s = 1 / np.sqrt(F + HG)
        p.update({"Wz": jax.random.normal(k[1], (F + HG, HG)) * s, "Wr": jax.random.normal(k[2], (F + HG, HG)) * s,
                  "Wh": jax.random.normal(k[3], (F + HG, HG)) * s, "bz": jnp.zeros(HG), "br": jnp.zeros(HG),
                  "bh": jnp.zeros(HG), "Wg": jnp.zeros((HG, 2))})  # zero head: starts exactly at M2's architecture
        return p

    def cell(p, h, f):
        xh = jnp.concatenate([f, h], -1)
        z = jax.nn.sigmoid(xh @ p["Wz"] + p["bz"])
        r = jax.nn.sigmoid(xh @ p["Wr"] + p["br"])
        hh = jnp.tanh(jnp.concatenate([f, r * h], -1) @ p["Wh"] + p["bh"])
        return (1 - z) * h + z * hh

    def head(p, f, h):
        mu, sig = M.drift_diffusion(p, f)
        g = h @ p["Wg"]
        return mu + g[..., 0], sig * jnp.exp(jnp.clip(g[..., 1], -3, 3))

    def feats(a, n, w, r, ph, x):
        P = a.shape
        return jnp.concatenate([a[..., None], n[..., None], jnp.broadcast_to(ph, P + (2,)), x,
                                w[..., None], r[..., None]], -1)

    def seq(p, a, n, r, rc0, ph, x):
        """Teacher-forced pass over flat pixels: a,n,r [T,P]; returns mu, sig [T-1,P] and h [T,P,HG]."""
        lam = M.lam(p, x)
        w0 = rc0 / (1 - lam)

        def body(c, t):
            w, h = c
            f = feats(a[t], n[t], w, r[t + 1], ph[t + 1], x)
            h = cell(p, h, f)
            mu, sig = head(p, f, h)
            return (lam * w + r[t + 1], h), (a[t] + mu, sig, h)

        _, (mu, sig, hs) = jax.lax.scan(body, (w0, jnp.zeros(a.shape[1:] + (HG,))), jnp.arange(a.shape[0] - 1))
        return mu, sig, hs

    def flat(D, idx=None):
        nn = np.asarray(jax.vmap(M.neighbourhood)(jnp.asarray(D["a"])))
        g = lambda v: v.reshape(v.shape[0], -1) if idx is None else v.reshape(v.shape[0], -1)[:, idx]
        xs = D["x"].reshape(-1, 3) if idx is None else D["x"].reshape(-1, 3)[idx]
        return {"a": jnp.asarray(g(D["a"])), "n": jnp.asarray(g(nn)), "r": jnp.asarray(g(D["r"])),
                "rc0": jnp.asarray(g(D["rclim"])[0]), "ph": jnp.asarray(D["phase"]), "x": jnp.asarray(xs),
                "obs": jnp.asarray(g(D["obs"]))}

    rng = np.random.default_rng(0)
    frac = float(os.environ.get("GRU_FRAC", 0.25))
    tag = os.environ.get("GRU_TAG", "gru")
    idx = rng.choice(N * N, int(N * N * frac), replace=False)
    FA = flat(A, idx)
    tgt = A["year"][1:]
    ok = FA["obs"][1:] & FA["obs"][:-1]
    mtr = ok & jnp.asarray(np.isin(tgt, list(aoi.TRAIN_YEARS)))[:, None]
    mva = ok & jnp.asarray(np.isin(tgt, list(aoi.VAL_YEARS)))[:, None]
    ntr, nva = float(mtr.sum()), float(mva.sum())

    def nll(p, m, n_):
        mu, sig, _ = seq(p, FA["a"], FA["n"], FA["r"], FA["rc0"], FA["ph"], FA["x"])
        z = (FA["a"][1:] - mu) / sig
        return jnp.sum((0.5 * z ** 2 + jnp.log(sig)) * m) / n_ + 0.5 * jnp.log(2 * jnp.pi)

    p = init(jax.random.PRNGKey(0))
    opt = optax.adam(3e-3)
    st = opt.init(p)

    @jax.jit
    def upd(p, st):
        l, g = jax.value_and_grad(lambda q: nll(q, mtr, ntr))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    val = jax.jit(lambda q: nll(q, mva, nva))
    best, best_p, bad, t0 = np.inf, p, 0, time.time()
    steps = int(os.environ.get("GRU_STEPS", 1500))
    for i in range(steps):
        p, st, l = upd(p, st)
        if i % 50 == 0 or i == steps - 1:
            v = float(val(p))
            log(f"  step {i:5d} train nll {float(l):.4f} val nll {v:.4f} ({time.time() - t0:.0f}s)")
            if v < best:
                best, best_p, bad = v, p, 0
            else:
                bad += 1
                if bad >= 6:
                    log("  early stop")
                    break
    p = best_p
    np.save(f"{OUT}/params_{tag}.npy", {k: np.asarray(v) for k, v in p.items()}, allow_pickle=True)

    res = {}
    for es, (D, mt) in evalsets(A, B).items():
        FD = flat(D)
        mu, sig, hs = jax.jit(seq)(p, FD["a"], FD["n"], FD["r"], FD["rc0"], FD["ph"], FD["x"])
        m = np.asarray((FD["obs"][1:] & FD["obs"][:-1])) & mt[:, None]
        cr = crps_gauss(np.asarray(mu), np.asarray(sig), np.asarray(FD["a"][1:]))
        one = float(cr[m].sum() / m.sum())
        lam = M.lam(p, FD["x"])
        wpath = np.asarray(M.w_path(p, {k: jnp.asarray(D[k]) for k in ("r", "rclim", "x")})).reshape(D["a"].shape[0], -1)
        # memory entering step t is the memory after step t-1 (zeros at t = 0)
        hs = jnp.concatenate([jnp.zeros((1,) + hs.shape[1:]), hs], 0)
        wpath = jnp.asarray(wpath)

        @jax.jit
        def roll(key, t, h0, w0):
            a0 = FD["a"][t]

            def one_s(k):
                def body(c, h_):
                    a, w, h, k = c
                    k, sub = jax.random.split(k)
                    nb = M.neighbourhood(a.reshape(N, N)).reshape(-1)
                    f = feats(a, nb, w, FD["r"][t + 1 + h_], FD["ph"][t + 1 + h_], FD["x"])
                    h = cell(p, h, f)
                    mu_, sg = head(p, f, h)
                    a = a + mu_ + sg * jax.random.normal(sub, a.shape)
                    return (a, lam * w + FD["r"][t + 1 + h_], h, k), a
                return jax.lax.scan(body, (a0, w0, h0, k), jnp.arange(H))[1]
            return jax.vmap(one_s)(jax.random.split(key, S))  # [S,H,P]

        st_ = starts_of(D, mt)
        rnum, rden = np.zeros(H), np.zeros(H)
        key = jax.random.PRNGKey(1)
        for k, t in enumerate(st_):
            key, sub = jax.random.split(key)
            smp = np.asarray(roll(sub, jnp.int32(t), hs[t], wpath[t]))
            ob = (D["obs"][t + 1:t + 1 + H] & D["obs"][t][None]).reshape(H, -1)
            for h_ in range(H):
                c_ = crps_samples(smp[:, h_], D["a"][t + 1 + h_].reshape(-1))
                rnum[h_] += c_[ob[h_]].sum()
                rden[h_] += ob[h_].sum()
        res[es] = {"onestep_crps": one, "rollout_crps": (rnum / rden).tolist()}
    out = summarise(res, f"GRU-M2 ({tag})")
    out["fit"] = {"best_val_nll": best, "pixels": int(len(idx)), "steps_run": i + 1}
    json.dump(out, open(f"{OUT}/metrics_{tag}.json", "w"), indent=1)
    log(f"wrote {OUT}/metrics_{tag}.json")


if __name__ == "__main__":
    {"gbm": gbm, "gru": gru}[sys.argv[1]]()
