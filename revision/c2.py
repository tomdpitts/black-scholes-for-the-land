"""Chronos-2 (amazon/chronos-2, zero-shot) as an engine in the suite's four tests (revision/TESTS.md, FM_HANDOFF.md).
Protocol: runs/20261022_1639_suite/PREREG.md, section "Chronos-2 protocol". Runs in .venv-chronos (torch). Reads
<out>/plan.npz (revision/suite.py plan); revision/suite.py scores what this writes.

  .venv-chronos/bin/python revision/c2.py check <out>               -> c2_check.json   quantile maths, API, autodiff (2003-17)
  .venv-chronos/bin/python revision/c2.py skill <out>               -> c2_skill.npz    test 1, with and without weather
  .venv-chronos/bin/python revision/c2.py world <out> --L 96 [--seasons k]
                                                                    -> c2_world.npz    test 2, the Chronos-2 world
  .venv-chronos/bin/python revision/c2.py drought <out> [--px n]    -> c2_drought.npz  test 3, season-end maps
  .venv-chronos/bin/python revision/c2.py drought0 <out> <plan dir> [--px n]
                                                                    -> c2_drought0.npz test 3, zero-residual split
  .venv-chronos/bin/python revision/c2.py sens <out> [--fd]         -> c2_sens.npz     test 4, Delta on the sample
  .venv-chronos/bin/python revision/c2.py cost <out> <plan dir>     -> c2_cost.json    CPU cost, GPU estimate
--seasons k, --px n: the first k seasons / n sample pixels (debug only).
"""
import json
import math
import os
import sys
import time
from statistics import NormalDist

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import load, H  # noqa: E402

H2 = 2  # greekstest.H2: Delta read two steps ahead
N = 160
SCEN = ("S0", "S1", "S2a", "S2b")  # suite.SCEN order; the management arrays come from the plan
LEVELS = np.array([0.01, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99])
Z = np.array([NormalDist().inv_cdf(v) for v in LEVELS])
TAU = (np.arange(1000) + 0.5) / 1000
KT = np.clip(np.searchsorted(Z, [NormalDist().inv_cdf(v) for v in TAU]) - 1, 0, len(Z) - 2)
WT = (np.array([NormalDist().inv_cdf(v) for v in TAU]) - Z[KT]) / (Z[KT + 1] - Z[KT])
MED = int(np.where(LEVELS == 0.5)[0][0])


# ----------------------------------------------------------------------------- the predictive distribution
def value(q, z):
    """Quantile function at standard-normal score z [B]: linear in z between the 21 knots q [B,21] (sorted),
    extended linearly beyond 0.01 and 0.99 by the outer segments (Gaussian tails)."""
    k = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
    i = np.arange(len(q))
    return q[i, k] + (q[i, k + 1] - q[i, k]) * (z - Z[k]) / (Z[k + 1] - Z[k])


def pit(q, y):
    """The percentile of y [B], as its standard-normal score: the inverse of value(), clipped to [-8, 8]."""
    k = np.clip((y[:, None] >= q).sum(1) - 1, 0, len(Z) - 2)
    i = np.arange(len(q))
    return np.clip(Z[k] + (y - q[i, k]) * (Z[k + 1] - Z[k]) / np.maximum(q[i, k + 1] - q[i, k], 1e-9), -8, 8)


def crps(q, y):
    """CRPS = 2 x mean pinball loss over 1,000 levels of the same quantile function. q [B,21], y [B]."""
    d = y[:, None] - (q[:, KT] + (q[:, KT + 1] - q[:, KT]) * WT)
    return 2 * np.mean(np.maximum(TAU * d, (TAU - 1) * d), 1)


# ----------------------------------------------------------------------------- the network
class Chronos2:
    """Direct calls to the network: per pixel a task of target row + covariate rows (one group each, so no
    cross-learning); known-future covariates in `future_covariates`, NaN on the target row."""

    def __init__(self, batch=256):
        from chronos import Chronos2Pipeline
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cuda":
            torch.set_float32_matmul_precision("high")  # TF32 matmuls
            # batch stays at 256 tasks: group attention spans every row of a batch, so memory grows with its square
        self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=self.dev)
        self.net, self.batch = self.pipe.model.eval(), batch
        assert np.allclose(self.pipe.quantiles, LEVELS)

    def _forward(self, a, cov, fut, h):
        """a [B,L], cov [B,K,L] or None, fut [B,K,h] (torch) -> quantile forecasts [B,21,h]."""
        B, L = a.shape
        K = 0 if cov is None else cov.shape[1]
        n_out = math.ceil(h / 16)
        ctx = (a[:, None] if K == 0 else torch.cat([a[:, None], cov], 1)).reshape(B * (1 + K), L)
        kw = {}
        if K:
            fc = torch.full((B, 1 + K, 16 * n_out), float("nan"), device=fut.device)
            fc[:, 1:, :h] = fut
            kw["future_covariates"] = fc.reshape(B * (1 + K), -1).to(self.dev)
        g = torch.arange(B).repeat_interleave(1 + K).to(self.dev)
        q = self.net(context=ctx.to(self.dev), group_ids=g, num_output_patches=n_out, **kw).quantile_preds
        return q.reshape(B, 1 + K, len(LEVELS), -1)[:, 0, :, :h]

    def quantiles(self, a, cov=None, fut=None, h=1):
        """Sorted quantile forecasts [B,21,h] (numpy) of series a [B,L] with covariates cov [B,K,L] / fut [B,K,h]."""
        T = lambda x, i: None if x is None else torch.as_tensor(np.ascontiguousarray(x[i:i + self.batch]), dtype=torch.float32)
        out = []
        with torch.inference_mode():
            for i in range(0, len(a), self.batch):
                out.append(self._forward(T(a, i), T(cov, i), T(fut, i), h).float().cpu())
        return np.sort(torch.cat(out).numpy(), axis=1)

    def delta(self, a, cov, fd=False):
        """d median(t+2) / d rain anomaly(t+1), from the history to t along normal weather (future covariates 0).
        Autodiff through the network, or finite differences (+0.01) if fd. [B]"""
        out, nb = [], min(self.batch, 256)  # backward keeps activations: smaller batches (outputs do not depend on batching)
        for i in range(0, len(a), nb):
            ai, ci = (torch.as_tensor(np.ascontiguousarray(x[i:i + nb]), dtype=torch.float32) for x in (a, cov))
            fut = torch.zeros(len(ai), cov.shape[1], H2)
            if fd:
                with torch.inference_mode():
                    up = fut.clone()
                    up[:, 0, 0] = 0.01
                    out.append(((self._forward(ai, ci, up, H2)[:, MED, H2 - 1] - self._forward(ai, ci, fut, H2)[:, MED, H2 - 1]) / 0.01).cpu())
            else:
                fut.requires_grad_(True)
                self._forward(ai, ci, fut, H2)[:, MED, H2 - 1].sum().backward()  # tasks are independent
                out.append(fut.grad[:, 0, 0].clone())
        return torch.cat(out).numpy()


def box_series(box):
    """(built dict, series [P,T], covariates [P,2,T]: rain anomaly, ET0 anomaly, 100 mm per window)."""
    d = load(box)
    e = np.load(f"data/{box}/era5.npz")
    T = d["a"].shape[0]
    cov = np.stack([d["r"] - d["rclim"], e["pet"] - e["pet_clim"]], 1).reshape(T, 2, -1)
    return d, d["a"].reshape(T, -1).T.copy(), cov.transpose(2, 1, 0).copy()


def invert(eng, a, cov, t0, y, fc):
    """Percentiles (as z) [B,H] of the season y [B,H] in the one-step forecasts given the history to t0 + y."""
    z = []
    for h in range(H):
        ctx = np.concatenate([a[:, :t0 + 1], y[:, :h]], 1)
        cc = np.concatenate([cov[:, :, :t0 + 1], fc[:, :, :h]], 2)
        z.append(pit(eng.quantiles(ctx, cc, fc[:, :, h:h + 1])[:, :, 0], y[:, h]))
    return np.stack(z, 1)


def roll(eng, a, cov, t0, z, fc, m=None):
    """Re-run a season step by step along weather fc [B,2,H]: each step takes percentile z[:, h] of the new
    predictive distribution, plus management m [B,H], and is fed back as history. [B,H]"""
    y = np.zeros((len(a), 0), np.float32)
    for h in range(H):
        ctx = np.concatenate([a[:, :t0 + 1], y], 1)
        cc = np.concatenate([cov[:, :, :t0 + 1], fc[:, :, :h]], 2)
        v = value(eng.quantiles(ctx, cc, fc[:, :, h:h + 1])[:, :, 0], z[:, h]) + (0 if m is None else m[:, h])
        y = np.concatenate([y, v[:, None].astype(np.float32)], 1)
    return y


def get_plan(out):
    return dict(np.load(f"{out}/plan.npz"))


def arg(name, default=None):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


# ----------------------------------------------------------------------------- tests
def skill(out):
    """Per start and horizon: summed quantile CRPS and pixel count, with weather and history only."""
    P, eng, res = get_plan(out), Chronos2(), {}
    px = P["px"]
    for box in "AB":
        d, a, cov = box_series(box)
        obs = d["obs"].reshape(len(d["obs"]), -1)
        st = P[f"{box}_skill_starts"]
        a, cov, obs = a[px], cov[px], obs[:, px]
        for mode in ("weather", "history"):
            num, den = np.zeros((len(st), H)), np.zeros((len(st), H))
            t1 = time.time()
            for i, t in enumerate(st):
                w = mode == "weather"
                q = eng.quantiles(a[:, :t + 1], cov[:, :, :t + 1] if w else None, cov[:, :, t + 1:t + 1 + H] if w else None, H)
                for h in range(H):
                    ob = obs[t] & obs[t + 1 + h]
                    num[i, h], den[i, h] = crps(q[:, :, h], a[:, t + 1 + h])[ob].sum(), ob.sum()
            res[f"{box}_{mode}_num"], res[f"{box}_{mode}_den"] = num, den
            print(box, mode, f"{len(st)} starts, {time.time() - t1:.0f} s, 3-month CRPS {num[:, -1].sum() / den[:, -1].sum():.5f}", flush=True)
        res[f"{box}_starts"] = st
    np.savez(f"{out}/c2_skill.npz", px=px, debug=P["debug"], **res)
    print("wrote", f"{out}/c2_skill.npz")


def world(out, L, k=None):
    """The Chronos-2 world on the centred L x L tile (PREREG test 2), every season and scenario."""
    P, eng = get_plan(out), Chronos2()
    o = (N - L) // 2
    tile = (np.arange(N)[:, None] * N + np.arange(N)[None, :])[o:o + L, o:o + L].ravel()
    d, a, cov = box_series("A")
    a, cov = a[tile], cov[tile]
    rclim = d["rclim"].reshape(len(d["a"]), -1)[:, tile]
    crop = lambda m: m.reshape(m.shape[:-2] + (-1,))[..., tile]  # [..., N, N] -> [..., B]
    R = {x: [] for x in ("yobs", "cf_true", "cf_true0", "own", "difficulty", "difficulty_dep", "recon", "z", "dep", "et0")}
    seasons = list(range(len(P["season_t0"])))[:k]
    for i in seasons:
        t0 = int(P["season_t0"][i])
        t1 = time.time()
        y, fc = a[:, t0 + 1:t0 + 1 + H], cov[:, :, t0 + 1:t0 + 1 + H]
        normal = np.zeros_like(fc)
        z = invert(eng, a, cov, t0, y, fc)
        ct0 = roll(eng, a, cov, t0, z, normal)[:, -1]
        dry = normal.copy()
        rc = rclim[t0 + 1:t0 + 1 + H].T
        dry[:, 0] = np.maximum(rc - 1 / 6, 0) - rc
        R["difficulty"].append(roll(eng, a, cov, t0, z, dry)[:, -1].mean() - ct0.mean())
        R["difficulty_dep"].append(dry[:, 0].mean(0).sum() * 100)
        yo, ct, own, cache = [], [], [], {}
        for j in range(len(SCEN)):
            yj = roll(eng, a, cov, t0, z, fc, crop(P["mgmt_real"][i, j]).T)
            mn = crop(P["mgmt_normal"][i, j]).T
            key = mn.tobytes()
            if key not in cache:
                cache[key] = ct0 if not mn.any() else roll(eng, a, cov, t0, z, normal, mn)[:, -1]
            yo.append(yj)
            ct.append(cache[key])
            own.append(roll(eng, a, cov, t0, invert(eng, a, cov, t0, yj, fc), normal)[:, -1])
        R["recon"].append(np.abs(yo[0] - y).max())  # S0 re-injects the real percentiles: must reproduce the season
        sq = lambda v: np.asarray(v).reshape(np.shape(v)[:-1] + (L, L))
        R["yobs"].append(sq(np.stack(yo).transpose(0, 2, 1)))
        R["cf_true"].append(sq(np.stack(ct)))
        R["cf_true0"].append(sq(ct0))
        R["own"].append(sq(np.stack(own)))
        R["z"].append(z)
        R["dep"].append(fc[:, 0] * 100)
        R["et0"].append(fc[:, 1] * 100)
        print(f"{P['season_name'][i]}: {time.time() - t1:.0f} s, recon {R['recon'][-1]:.1e}, difficulty {R['difficulty'][-1]:+.4f}", flush=True)
    np.savez(f"{out}/c2_world.npz", L=L, offset=o, debug=P["debug"], season_t0=P["season_t0"][seasons],
             season_name=P["season_name"][seasons], **{x: np.asarray(v) for x, v in R.items()})
    print("wrote", f"{out}/c2_world.npz")


def drought(out, n=None):
    """Season-end maps for the test 3 split: observed, median path (F), counterfactuals, re-injection."""
    P, eng = get_plan(out), Chronos2()
    t0 = int(P["t0_2022"])
    d, a, cov = box_series("A")
    px = np.arange(N * N) if n is None else P["px"][:n]
    a, cov = a[px], cov[px]
    y, fc = a[:, t0 + 1:t0 + 1 + H], cov[:, :, t0 + 1:t0 + 1 + H]
    t1 = time.time()
    z = invert(eng, a, cov, t0, y, fc)
    nr, nt = fc.copy(), fc.copy()
    nr[:, 0], nt[:, 1] = 0, 0
    M = {"obs": y[:, -1], "F": roll(eng, a, cov, t0, np.zeros_like(z), fc)[:, -1], "recon": roll(eng, a, cov, t0, z, fc)[:, -1],
         "cf_r": roll(eng, a, cov, t0, z, nr)[:, -1], "cf_t": roll(eng, a, cov, t0, z, nt)[:, -1],
         "cf_rt": roll(eng, a, cov, t0, z, np.zeros_like(fc))[:, -1]}
    print(f"{P['name_2022']} on {len(px)} px: {time.time() - t1:.0f} s, recon {np.abs(M['recon'] - M['obs']).max():.1e}", flush=True)
    np.savez(f"{out}/c2_drought.npz", px=px, t0=t0, name=P["name_2022"], debug=P["debug"], z=z, **M)
    print("wrote", f"{out}/c2_drought.npz")


def drought0(out, plan, n=None):
    """As drought, the four weather combinations as median paths (percentile 50 at every step), so the split is
    baseline (normal weather) + rain + demand (Shapley) + residual (observed - F0), each a direct run."""
    P, eng = get_plan(plan), Chronos2()
    t0 = int(P["t0_2022"])
    d, a, cov = box_series("A")
    px = np.arange(N * N) if n is None else P["px"][:n]
    a, cov = a[px], cov[px]
    y, fc = a[:, t0 + 1:t0 + 1 + H], cov[:, :, t0 + 1:t0 + 1 + H]
    z0 = np.zeros((len(px), H))
    nr, nt = fc.copy(), fc.copy()
    nr[:, 0], nt[:, 1] = 0, 0
    M, t1 = {"obs": y[:, -1]}, time.time()
    for k, f in (("F0", fc), ("cf_r0", nr), ("cf_t0", nt), ("cf_rt0", np.zeros_like(fc))):
        M[k] = roll(eng, a, cov, t0, z0, f)[:, -1]
        print(f"{k}: {time.time() - t1:.0f} s", flush=True)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/c2_drought0.npz", px=px, t0=t0, name=P["name_2022"], device=eng.dev, **M)
    print("wrote", f"{out}/c2_drought0.npz")


def sens(out, fd=False):
    P, eng, res = get_plan(out), Chronos2(), {"method": "autodiff"}
    for box in "AB":
        _, a, cov = box_series(box)
        a, cov = a[P["px"]], cov[P["px"]]
        st = P[f"{box}_sens_starts"]
        t1 = time.time()
        res[f"{box}_starts"] = st
        res[f"{box}_delta"] = np.stack([eng.delta(a[:, :t + 1], cov[:, :, :t + 1]) for t in st])
        print(box, f"{len(st)} starts, {time.time() - t1:.0f} s, mean Delta {res[f'{box}_delta'].mean():.4f}", flush=True)
        if fd:  # debug: the finite-difference version beside it
            t1 = time.time()
            f = np.stack([eng.delta(a[:, :t + 1], cov[:, :, :t + 1], fd=True) for t in st])
            res[f"{box}_fd_seconds"], res[f"{box}_ad_vs_fd_corr"] = time.time() - t1, np.corrcoef(f.ravel(), res[f"{box}_delta"].ravel())[0, 1]
            res[f"{box}_ad_vs_fd_max_abs"] = np.abs(f - res[f"{box}_delta"]).max()
            print(box, "autodiff vs finite differences: corr", res[f"{box}_ad_vs_fd_corr"], "max abs", res[f"{box}_ad_vs_fd_max_abs"], flush=True)
    np.savez(f"{out}/c2_sens.npz", px=P["px"], **res)
    print("wrote", f"{out}/c2_sens.npz")


# ----------------------------------------------------------------------------- checks and cost
def check(out):
    """Quantile maths against a Gaussian; network call against pipeline.predict; autodiff vs FD. 2003-17 data."""
    rng = np.random.default_rng(0)
    mu, sg, y = rng.normal(0, 1, 500), rng.uniform(0.05, 2, 500), rng.normal(0, 2, 500)
    q = mu[:, None] + sg[:, None] * Z[None]
    zz = (y - mu) / sg
    Phi = np.array([NormalDist().cdf(v) for v in zz])
    exact = sg * (zz * (2 * Phi - 1) + 2 * np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    inside = np.abs(zz) < 8
    res = {"crps_vs_gaussian_max_rel": float(np.max(np.abs(crps(q, y) - exact) / exact)),
           "pit_value_roundtrip_max_abs_within_8sd": float(np.abs(value(q, pit(q, y)) - y)[inside].max()),
           "share_beyond_8sd_clipped": float(1 - inside.mean())}
    eng = Chronos2()
    _, a, cov = box_series("A")
    P = get_plan(out)
    px, t = P["px"][:64], 300  # 2014
    a, cov = a[px], cov[px]
    q = eng.quantiles(a[:, :t + 1], cov[:, :, :t + 1], cov[:, :, t + 1:t + 7], 6)
    ref = eng.pipe.predict([{"target": a[i, :t + 1], "past_covariates": {"r": cov[i, 0, :t + 1], "e": cov[i, 1, :t + 1]},
                             "future_covariates": {"r": cov[i, 0, t + 1:t + 7], "e": cov[i, 1, t + 1:t + 7]}} for i in range(len(px))],
                           prediction_length=6)
    res["network_vs_pipeline_max_abs"] = float(np.abs(q - np.sort(np.stack([r_[0].numpy() for r_ in ref]), 1)).max())
    one = eng.quantiles(a[:1, :t + 1], cov[:1, :, :t + 1], cov[:1, :, t + 1:t + 7], 6)
    res["alone_vs_batch_max_abs"] = float(np.abs(one[0] - q[0]).max())
    ad, fd = eng.delta(a[:, :t + 1], cov[:, :, :t + 1]), eng.delta(a[:, :t + 1], cov[:, :, :t + 1], fd=True)
    res["delta_autodiff_vs_fd"] = {"corr": float(np.corrcoef(ad, fd)[0, 1]), "max_abs": float(np.abs(ad - fd).max()),
                                   "mean_autodiff": float(ad.mean()), "mean_fd": float(fd.mean())}
    q0 = eng.quantiles(a[:, :t + 1], cov[:, :, :t + 1], np.zeros_like(cov[:, :, t + 1:t + 7]), 6)
    res["median_change_rain_to_normal_6step"] = float((q[:, MED, -1] - q0[:, MED, -1]).mean())
    res["versions"] = {"torch": torch.__version__, "chronos": __import__("chronos").__version__}
    json.dump(res, open(f"{out}/c2_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def cost(out, plan_dir):
    """CPU time per row-forward against context length (2003-17 contexts; synthetic beyond 2017's length), the
    protocol's call counts for the run plan, and the H100 estimate (PREREG "Compute")."""
    eng = Chronos2()
    _, a, cov = box_series("A")
    px = get_plan(plan_dir)["px"][:256]
    a, cov = a[px], cov[px]
    rng = np.random.default_rng(0)
    nparam = sum(p.numel() for p in eng.net.parameters())
    tokens = lambda L, h=1: math.ceil(L / 16) + 1 + math.ceil(h / 16)
    meas = []
    for L in (128, 256, 383, 470, 537):
        aa = a[:, :L] if L <= 383 else rng.normal(0, 0.05, (len(px), L)).astype(np.float32)  # 383 = end of 2017
        cc = cov[:, :, :L] if L <= 383 else rng.normal(0, 0.3, (len(px), 2, L)).astype(np.float32)
        for K, kind in ((2, "fwd"), (0, "fwd"), (2, "delta")):
            f = (lambda: eng.delta(aa, cc)) if kind == "delta" else (lambda: eng.quantiles(aa, cc if K else None, cc[:, :, :6] if K else None, 6))
            f()
            ts = []
            for _ in range(3):
                t1 = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t1)
            rows = len(px) * (1 + K)
            sec = min(ts) / rows
            meas.append({"L": L, "rows_per_task": 1 + K, "kind": kind, "s_per_row": sec,
                         "cpu_tflops": 2 * nparam * tokens(L) * (3 if kind == "delta" else 1) / sec / 1e12})
            print(meas[-1], flush=True)
    fit = {kind: np.polyfit([tokens(m["L"]) for m in meas if m["kind"] == kind and m["rows_per_task"] == 3],
                            [m["s_per_row"] for m in meas if m["kind"] == kind and m["rows_per_task"] == 3], 1) for kind in ("fwd", "delta")}
    cpu_s = lambda L, kind="fwd": float(np.polyval(fit[kind], tokens(L)))
    flop = lambda L, kind="fwd": 2 * nparam * tokens(L) * (3 if kind == "delta" else 1)
    P = get_plan(plan_dir)
    T = 537
    n = len(P["px"])
    cells = {}
    # (rows, L, kind) per forward, summed: test 1 both modes, test 3 full box, test 4 the sample, test 2 per tile
    calls = lambda st, rows, Lf, kind="fwd": [(rows, Lf(t), kind) for t in st]
    c1 = []
    for box in "AB":
        st = P[f"{box}_skill_starts"]
        c1 += calls(st, 3 * n, lambda t: t + 1) + calls(st, n, lambda t: t + 1)
    cells["test 1 skill (both modes, both boxes)"] = c1
    t0 = int(P["t0_2022"])
    cells["test 3 2022 full box"] = [(3 * N * N, t0 + 1 + h, "fwd") for _ in range(6) for h in range(H)]
    cells["test 4 Delta (sample, both boxes)"] = sum((calls(P[f"{box}_sens_starts"], 3 * n, lambda t: t + 1, "delta") for box in "AB"), [])
    per_px = []  # test 2, per tile pixel: invert 6 + 4 scenarios + 2 truths + 4 x own (invert + roll) + difficulty, 6 steps each
    for t0 in P["season_t0"]:
        per_px += [(3, int(t0) + 1 + h, "fwd") for _ in range(1 + 4 + 2 + 8 + 1) for h in range(H)]
    summ = lambda cs: {"rows_M": sum(r for r, _, _ in cs) / 1e6, "cpu_h": sum(r * cpu_s(L, k) for r, L, k in cs) / 3600,
                       "pflop": sum(r * flop(L, k) for r, L, k in cs) / 1e15}
    res = {"params": nparam, "measured": meas, "cells": {k: summ(v) for k, v in cells.items()}, "world_per_tile_px": summ(per_px)}
    price, gpu = 3.95, {"central_tf32": 120e12, "pessimistic_fp32": 33e12}
    gh = lambda pf, g: pf * 1e15 / gpu[g] / 3600
    for c in res["cells"].values():
        c.update({f"h100_h_{g}": gh(c["pflop"], g) for g in gpu} | {f"usd_{g}": gh(c["pflop"], g) * price for g in gpu})
    wp = res["world_per_tile_px"]
    res["world_by_L"] = {L: {f"usd_{g}": gh(wp["pflop"] * L * L, g) * price for g in gpu} | {"cpu_h": wp["cpu_h"] * L * L}
                         for L in range(160, 31, -16)}
    ok = [L for L, v in res["world_by_L"].items() if v["usd_central_tf32"] <= 5.0]
    res["world_L_rule"] = max(ok) if ok else "32, drought seasons only"
    res["price_usd_per_h100_hour"], res["gpu_flops_assumed"] = price, gpu
    json.dump(res, open(f"{out}/c2_cost.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "measured"}, indent=1, default=float))


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    {"check": lambda: check(out), "skill": lambda: skill(out), "world": lambda: world(out, arg("--L"), arg("--seasons")),
     "drought": lambda: drought(out, arg("--px")), "sens": lambda: sens(out, "--fd" in sys.argv),
     "cost": lambda: cost(out, sys.argv[3]), "drought0": lambda: drought0(out, sys.argv[3], arg("--px"))}[cmd]()
