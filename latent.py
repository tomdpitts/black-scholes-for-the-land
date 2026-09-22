"""Latent residual: does what the land did beyond the rain carry forward a year? See PREREG.md.

A latent z = (z0, z1) per pixel and year enters M2's drift as mu + z0 + z1 * (w + r), theta frozen.
z_y is inferred by MAP on year y's transitions; year y is forecast with z_{y-1}.

  modal run --detach latent.py                        -> /vol/runs/latent/{metrics.json,maps.npz,progress.log}
  modal volume get projectb-data runs/latent runs/<dir>
  python latent.py <params.npy> <out dir> [crop]      -> local CPU run (crop: smoke test, few steps)
"""
import json
import sys

import modal

app = modal.App("projectb-latent")
vol = modal.Volume.from_name("projectb-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "scipy", "optax", "jax[cuda12]")
         .add_local_python_source("aoi", "model", "run"))

SCALES = (0.003, 0.01, 0.03)
NBOOT = 2000
VARIANTS = ("M2", "lag", "lag_level", "lag_shuffled", "same_year")
YEARS = {"A": list(range(2019, 2026)), "B": list(range(2004, 2026))}


def compute(data_dir, params_path, out, log=print, crop=None, S=16, adam_steps=300, chunk=8):
    import itertools
    import os
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import model as M
    from run import load, H

    os.makedirs(out, exist_ok=True)
    p = {k: jnp.asarray(v) for k, v in np.load(params_path, allow_pickle=True).item()["M2"].items()}
    res, maps = {}, {}

    def prep(box):
        d = load(box, data_dir, crop)
        D = {k: jnp.asarray(d[k]) for k in ("a", "obs", "r", "rclim", "x", "phase")}
        return d, D, M.w_path(p, D)

    def steps_of(d, y):
        return np.array([t for t in range(d["a"].shape[0] - 1) if int(d["year"][t + 1]) == y])

    # ------------------------------------------------------------ one-step NLL / CRPS with a latent
    def onestep(D, w, ts, z):
        def one(t):
            mu, sig, _ = M.step(p, D["a"][t], w[t], D["r"][t + 1], D["phase"][t + 1], D["x"], True, z)
            y = D["a"][t + 1]
            nll = 0.5 * ((y - mu) / sig) ** 2 + jnp.log(sig) + 0.5 * jnp.log(2 * jnp.pi)
            return nll, M.crps_gauss(mu, sig, y), D["obs"][t + 1] & D["obs"][t]
        return jax.vmap(one)(ts)

    onestep_j = jax.jit(onestep)

    # ------------------------------------------------------------ MAP inference of z_y, whitened: z = u * scale
    opt = optax.adam(optax.cosine_decay_schedule(0.1, adam_steps))

    def loss(u, D, w, ts, scale, gate):
        nll, _, m = onestep(D, w, ts, u * scale * gate)
        return jnp.sum(nll * m) + 0.5 * jnp.sum((u * gate) ** 2)

    @jax.jit
    def adam_step(u, st, D, w, ts, scale, gate):
        l, g = jax.value_and_grad(loss)(u, D, w, ts, scale, gate)
        upd, st = opt.update(g, st, u)
        return optax.apply_updates(u, upd), st, l, jnp.mean(jnp.abs(g))

    def infer(D, w, ts, s0, s1, level_only=False):
        N = D["x"].shape[0]
        scale, gate = jnp.array([s0, s1]), jnp.array([1.0, 0.0 if level_only else 1.0])
        u = jnp.zeros((N, N, 2))
        st = opt.init(u)
        ts = jnp.asarray(ts)
        for _ in range(adam_steps):
            u, st, l, g = adam_step(u, st, D, w, ts, scale, gate)
        return u * scale * gate, float(l), float(g)

    # ------------------------------------------------------------ rollouts along realised rain
    @jax.jit
    def roll_chunk(D, w, ts, z, z0):
        key0 = jax.random.PRNGKey(7)

        def one(t):
            k = jax.random.fold_in(key0, t)  # common random numbers across variants
            rp = jax.lax.dynamic_slice_in_dim(D["r"], t + 1, H, 0)
            pp = jax.lax.dynamic_slice_in_dim(D["phase"], t + 1, H, 0)
            s = M.rollout_sample(p, k, D["a"][t], w[t], rp, pp, D["x"], True, S=S, z=z)
            y = jax.lax.dynamic_slice_in_dim(D["a"], t + 1, H, 0)
            ob = jax.lax.dynamic_slice_in_dim(D["obs"], t + 1, H, 0) & D["obs"][t][None]
            c = jax.vmap(M.crps_samples, in_axes=(1, 0))(s, y)
            eff = (M.rollout_mean(p, D["a"][t], w[t], rp, pp, D["x"], True, z)[-1]
                   - M.rollout_mean(p, D["a"][t], w[t], rp, pp, D["x"], True, z0)[-1])
            return jnp.sum(c * ob, (1, 2)), jnp.sum(ob, (1, 2)), eff

        return jax.vmap(one)(ts)

    def boot(num_v, num_b, seed=1):
        pt = 1 - num_v.sum(0) / num_b.sum(0)
        rng = np.random.default_rng(seed)
        Y = len(num_v)
        dr = np.stack([1 - num_v[i].sum(0) / num_b[i].sum(0) for i in (rng.integers(0, Y, Y) for _ in range(NBOOT))])
        lo, hi = np.percentile(dr, [2.5, 97.5], axis=0)
        return {"skill": np.round(pt, 5).tolist(), "lo": np.round(lo, 5).tolist(), "hi": np.round(hi, 5).tolist()}

    # ------------------------------------------------------------ prior scales: Box A, z_2017 scored on 2018
    dA, DA, wA = prep("A")
    ts17, ts18 = steps_of(dA, 2017), jnp.asarray(steps_of(dA, 2018))
    N = DA["x"].shape[0]
    nll, _, m = onestep_j(DA, wA, ts18, jnp.zeros((N, N, 2)))
    sel = {"M2": float(jnp.sum(nll * m) / jnp.sum(m))}
    for s0, s1 in itertools.product(SCALES, SCALES):
        z, l, g = infer(DA, wA, ts17, s0, s1)
        nll, _, m = onestep_j(DA, wA, ts18, z)
        sel[f"{s0},{s1}"] = float(jnp.sum(nll * m) / jnp.sum(m))
        log(f"scale {s0},{s1}: 2018 nll {sel[f'{s0},{s1}']:.5f} (M2 {sel['M2']:.5f}); final loss {l:.1f} mean|grad| {g:.3g}")
    best = min((k for k in sel if k != "M2"), key=sel.get)
    s0, s1 = (float(v) for v in best.split(","))
    res["scale_selection"] = {"nll_2018": sel, "chosen": [s0, s1]}
    log(f"chosen scales {s0}, {s1}")

    # ------------------------------------------------------------ per box
    for box in ("A", "B"):
        d, D, w = (dA, DA, wA) if box == "A" else prep("B")
        T, N = d["a"].shape[0], d["x"].shape[0]
        years = YEARS[box]
        perm = np.random.default_rng(0).permutation(N * N)
        Z, ZL, conv = {}, {}, {}
        for y in sorted(set(years) | {y - 1 for y in years}):
            ts = steps_of(d, y)
            Z[y], l, g = infer(D, w, ts, s0, s1)
            ZL[y], _, gl = infer(D, w, ts, s0, s1, level_only=True)
            conv[y] = {"loss": l, "mean_abs_grad": g, "mean_abs_grad_level": gl, "n_steps": int(len(ts))}
            maps[f"{box}_z_{y}"] = np.asarray(Z[y], np.float32)
            log(f"{box} z_{y}: z0 mean {float(Z[y][..., 0].mean()):+.4f} sd {float(Z[y][..., 0].std()):.4f}; "
                f"z1 mean {float(Z[y][..., 1].mean()):+.4f} sd {float(Z[y][..., 1].std()):.4f}; mean|grad| {g:.3g}")

        V = len(VARIANTS)
        os_num, ro_num = np.zeros((len(years), V)), np.zeros((len(years), V, H))
        os_den, ro_den = np.zeros(len(years)), np.zeros((len(years), H))
        zero = jnp.zeros((N, N, 2))
        for iy, y in enumerate(years):
            zs = {"M2": zero, "lag": Z[y - 1], "lag_level": ZL[y - 1],
                  "lag_shuffled": jnp.asarray(np.asarray(Z[y - 1]).reshape(-1, 2)[perm].reshape(N, N, 2)),
                  "same_year": Z[y]}
            ts = steps_of(d, y)
            starts = ts[ts < T - H]
            for iv, v in enumerate(VARIANTS):
                _, crps, m = onestep_j(D, w, jnp.asarray(ts), zs[v])
                os_num[iy, iv] = float(jnp.sum(crps * m))
                os_den[iy] = float(jnp.sum(m))
                eff = 0.0
                for i in range(0, len(starts), chunk):
                    n_, d_, e_ = roll_chunk(D, w, jnp.asarray(starts[i:i + chunk]), zs[v], zero)
                    ro_num[iy, iv] += np.asarray(n_).sum(0)
                    if iv == 0:
                        ro_den[iy] += np.asarray(d_).sum(0)
                    eff = eff + np.asarray(e_).sum(0)
                if v == "lag":
                    maps[f"{box}_effect_h{H}_{y}"] = (eff / max(len(starts), 1)).astype(np.float32)
            log(f"{box} {y}: one-step skill vs M2 " + json.dumps({v: round(1 - os_num[iy, i] / os_num[iy, 0], 4) for i, v in enumerate(VARIANTS)})
                + f"; h{H} " + json.dumps({v: round(1 - ro_num[iy, i, -1] / ro_num[iy, 0, -1], 4) for i, v in enumerate(VARIANTS)}))

        r = {"years": years, "convergence": conv}
        iv = {v: i for i, v in enumerate(VARIANTS)}
        r["onestep_vs_M2"] = {v: boot(os_num[:, iv[v]], os_num[:, 0]) for v in VARIANTS[1:]}
        r["rollout_vs_M2"] = {v: boot(ro_num[:, iv[v]], ro_num[:, 0]) for v in VARIANTS[1:]}
        r["rollout_lag_vs_shuffled"] = boot(ro_num[:, iv["lag"]], ro_num[:, iv["lag_shuffled"]])
        r["rollout_lag_vs_lag_level"] = boot(ro_num[:, iv["lag"]], ro_num[:, iv["lag_level"]])
        r["crps_M2_onestep"] = float(os_num[:, 0].sum() / os_den.sum())
        r["crps_M2_rollout"] = (ro_num[:, 0].sum(0) / ro_den.sum(0)).tolist()
        # year-to-year persistence of the pixel pattern
        pers = {"z0": [], "z1": [], "z0_level_only": []}
        for y in years:
            a0, a1 = np.asarray(Z[y - 1]), np.asarray(Z[y])
            pers["z0"].append(float(np.corrcoef(a0[..., 0].ravel(), a1[..., 0].ravel())[0, 1]))
            pers["z1"].append(float(np.corrcoef(a0[..., 1].ravel(), a1[..., 1].ravel())[0, 1]))
            pers["z0_level_only"].append(float(np.corrcoef(np.asarray(ZL[y - 1])[..., 0].ravel(), np.asarray(ZL[y])[..., 0].ravel())[0, 1]))
        r["persistence"] = {k: {"by_year": v, "mean": float(np.mean(v))} for k, v in pers.items()}
        # fire: z0 in burned pixel-years, same year and following year
        if "burn" in d:
            fire = {"same_year": [], "next_year": []}
            for y in years:
                b = d["burn"][d["year"] == y].any(0)
                if not b.any():
                    continue
                for tag, yy in (("same_year", y), ("next_year", y + 1)):
                    if yy in Z:
                        z0 = np.asarray(Z[yy])[..., 0]
                        fire[tag].append({"year": y, "n_burned": int(b.sum()),
                                          "z0_burned": float(z0[b].mean()), "z0_unburned": float(z0[~b].mean())})
            r["fire"] = fire
        res[box] = r
        log(f"{box} summary: " + json.dumps({k: r[k] for k in ("rollout_vs_M2", "rollout_lag_vs_shuffled", "persistence")})[:1500])
        with open(f"{out}/metrics.json", "w") as f:
            json.dump(res, f, indent=1)
        np.savez_compressed(f"{out}/maps.npz", **maps)

    A, B = res["A"], res["B"]
    above = lambda b, h: b["lo"][h] > 0 if isinstance(b["lo"], list) else b["lo"] > 0
    res["verdicts"] = {
        "L1_primary_A_h6_lag_vs_M2": above(A["rollout_vs_M2"]["lag"], H - 1),
        "L2_A_onestep_lag_vs_M2": A["onestep_vs_M2"]["lag"]["lo"] > 0,
        "L3_A_h6_lag_vs_shuffled": above(A["rollout_lag_vs_shuffled"], H - 1),
        "L4_B_h6_lag_vs_M2": above(B["rollout_vs_M2"]["lag"], H - 1),
    }
    log(f"verdicts: {json.dumps(res['verdicts'])}")
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(res, f, indent=1)
    log("done")
    return res


@app.function(image=image, gpu="H100", memory=65536, timeout=3600 * 2, volumes={"/vol": vol})
def remote():
    import os
    out = "/vol/runs/latent"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(f"{out}/{f}")

    def log(msg):
        print(msg, flush=True)
        with open(f"{out}/progress.log", "a") as fh:
            fh.write(msg + "\n")
        vol.commit()

    compute("/vol/data", "/vol/runs/fairfight/params.npy", out, log)
    vol.commit()


@app.local_entrypoint()
def main():
    remote.remote()


if __name__ == "__main__":
    crop = int(sys.argv[3]) if len(sys.argv) > 3 else None
    kw = dict(S=4, adam_steps=40) if crop else {}
    compute("data", sys.argv[1], sys.argv[2], crop=crop, **kw)
