"""Implied residual surface: the land's analogue of the implied-volatility surface.

IV surface: implied vol against strike (moneyness) and maturity; flat if Black-Scholes were right.
Here:       mean residual (observed - forecast) against how abnormal the weather was (moneyness) and horizon h
            (maturity); flat at zero if the method has taken the weather out.

  moneyness  climatic water balance departure (rain minus ED, vs climatology) summed over windows t0+1..t0+h,
             2 km block mean, divided by its sd over 2003-17 starts at that h (the sqrt(T) scaling of moneyness)
  horizon    h = 1..6 16-day steps
  height     residual at t0+h, NDVI, 2 km block mean over pixels observed at t0 and t0+h
               z-score    observed anomaly (expects normal)
               RESTREND   observed - (alpha_h + beta_h * rain departure over t0+1..t0+h), per pixel, fitted per h
                          on 2003-17 as suite.restrend (which is h = 6)
               M4         observed - M4 5-fit ensemble rollout from the observed state at t0, residual zero

Starts: every 16-day start whose first window is in 2019-25 (suite plan A_skill_starts, 141). Box A.
Cells: moneyness bins of width 1 sd, -4 to +7; masked below 3 distinct years (the wet tail beyond +4 sd
at 64-96 days is mostly OND 2019; see eff_years in the JSON); interval: bootstrap
over the 7 years (starts overlap within a year). Spread: sd of the unmasked cell means (flatness; the level is not the test).

  .venv/bin/python revision/implied_surface.py compute runs/<dir>   -> surface.npz, surface.json
  .venv/bin/python revision/implied_surface.py summarise runs/<dir>
  figure: revision/implied_surface_plotly.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUITE = "runs/20261022_1639_suite"
METHODS = ("z-score", "RESTREND", "M4")
EDGES = np.arange(-4.0, 7.01, 1.0)
MIN_YEARS = 3  # cells with fewer distinct years are masked; effective years (block-weighted) are reported in the JSON


def bmean(v, b=8):
    """2 km block means of [..., N, N] (no mask: weather is complete)."""
    n = v.shape[-1] // b
    return v.reshape(v.shape[:-2] + (n, b, n, b)).mean((-3, -1))


def compute(out):
    import jax
    import jax.numpy as jnp
    import aoi
    import m3
    import suite as S
    from run import load, H
    from m3_attribution import rerun

    os.makedirs(out, exist_ok=True)
    P = S.get_plan(SUITE)
    starts = P["A_skill_starts"].astype(int)
    d = load("A")
    D4 = S.data("A", "M4")[1]
    fits = [(S.params("M4", s), m3.w_path(S.params("M4", s), D4)) for s in S.SEEDS]
    e5 = np.load("data/A/era5.npz")
    T = d["a"].shape[0]

    # cumulative departures, 100 mm: c[t] = sum over windows 1..t
    cum = lambda x: np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0, dtype=np.float64)])
    crain = cum(d["r"] - d["rclim"])
    cwb = crain - cum(e5["pet"] - e5["pet_clim"])
    over = lambda c, t, h: c[t + h + 1] - c[t + 1]  # windows t+1..t+h

    # moneyness scale per h: sd of block wb departure over 2003-17 starts
    train = [t for t in range(T - H) if int(d["year"][t + 1]) in aoi.TRAIN_YEARS]
    sd = np.array([bmean(np.stack([over(cwb, t, h) for t in train])).std() for h in range(1, H + 1)])

    # RESTREND per horizon (suite.restrend with H -> h)
    rt = []
    for h in range(1, H + 1):
        R = np.stack([over(crain, t, h) for t in range(T - h)])
        y = d["a"][h:].astype(np.float64)
        ok = d["obs"][:-h] & d["obs"][h:] & np.isin(d["year"][h:], list(aoi.TRAIN_YEARS))[:, None, None]
        n = ok.sum(0)
        mx, my = (R * ok).sum(0) / n, (y * ok).sum(0) / n
        beta = ((R - mx) * (y - my) * ok).sum(0) / ((R - mx) ** 2 * ok).sum(0)
        rt.append((my - beta * mx, beta))

    path = jax.jit(lambda p, s, a0, w0, r, ex, ph: rerun(p, s, a0, w0, r, ex, ph, jnp.zeros_like(r)))
    nb = (d["a"].shape[-1] // 8) ** 2
    res = {m: np.full((len(starts), H, nb), np.nan) for m in METHODS}
    mny = np.full((len(starts), H, nb), np.nan)
    wbmm = np.full((len(starts), H, nb), np.nan)
    for i, t0 in enumerate(starts):
        sl = slice(t0 + 1, t0 + 1 + H)
        F = np.mean([np.asarray(path(p, D4["s"], D4["a"][t0], w[t0], D4["r"][sl], D4["ex"][sl], D4["phase"][sl]))
                     for p, w in fits], 0)
        for h in range(1, H + 1):
            obs, ok = d["a"][t0 + h], d["obs"][t0] & d["obs"][t0 + h]
            pred = {"z-score": 0.0, "RESTREND": rt[h - 1][0] + rt[h - 1][1] * over(crain, t0, h), "M4": F[h - 1]}
            v = {m: S.blocks(obs - pred[m], ok).ravel() for m in METHODS}
            keep = np.isfinite(v["M4"])
            wb = bmean(over(cwb, t0, h)).ravel()
            for m in METHODS:
                res[m][i, h - 1] = np.where(keep, v[m], np.nan)
            wbmm[i, h - 1] = np.where(keep, wb * 100, np.nan)
            mny[i, h - 1] = np.where(keep, wb / sd[h - 1], np.nan)
        if i % 20 == 0:
            print(f"start {i + 1}/{len(starts)}", flush=True)

    year = d["year"][starts + 1].astype(int)
    np.savez(f"{out}/surface.npz", starts=starts, year=year, sd_wb_mm=sd * 100, moneyness=mny, wb_mm=wbmm,
             **{f"res_{m}": res[m] for m in METHODS})
    summarise(out)


def cells(z):
    """Per-cell sums and counts by year: [Y, H, B] for one method, plus distinct starts per cell."""
    f = np.load(z)
    yrs = np.unique(f["year"])
    mny, H = f["moneyness"], f["moneyness"].shape[1]
    b = np.digitize(mny, EDGES) - 1  # bin index, -1 / len-1 outside
    nbin = len(EDGES) - 1
    out = {}
    for m in METHODS:
        r = f[f"res_{m}"]
        s, c = np.zeros((len(yrs), H, nbin)), np.zeros((len(yrs), H, nbin))
        for yi, y in enumerate(yrs):
            for h in range(H):
                sel = f["year"] == y
                rr, bb = r[sel, h].ravel(), b[sel, h].ravel()
                ok = np.isfinite(rr) & (bb >= 0) & (bb < nbin)
                s[yi, h] += np.bincount(bb[ok], rr[ok], nbin)
                c[yi, h] += np.bincount(bb[ok], minlength=nbin)
        out[m] = (s, c)
    nst = np.zeros((H, nbin), int)
    for h in range(H):
        for i in range(len(f["starts"])):
            bb = b[i, h][np.isfinite(mny[i, h])]
            nst[h, np.unique(bb[(bb >= 0) & (bb < nbin)])] += 1
    return out, nst, yrs


def summarise(out):
    z = f"{out}/surface.npz"
    f = np.load(z)
    cs, nst, yrs = cells(z)
    cy = cs["M4"][1]  # blocks per year per cell [Y, H, B]; same for every method
    nyr = (cy > 0).sum(0)
    neff = cy.sum(0) ** 2 / np.maximum((cy ** 2).sum(0), 1)
    valid = nyr >= MIN_YEARS
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(yrs), (2000, len(yrs)))
    J = {"starts": int(len(f["starts"])), "years": yrs.tolist(), "edges_sd": EDGES.tolist(), "min_years": MIN_YEARS,
         "sd_wb_mm_by_h": f["sd_wb_mm"].round(1).tolist(), "n_starts": nst.tolist(), "n_years": nyr.tolist(), "eff_years": neff.round(2).tolist()}
    for m in METHODS:
        s, c = cs[m]
        mean = s.sum(0) / np.maximum(c.sum(0), 1)
        bs = np.stack([s[d_].sum(0) / np.maximum(c[d_].sum(0), 1) for d_ in draws])
        # slope on moneyness per h, and per 100 mm wb at h = 6 (comparable to the paper's lean), year bootstrap
        def sl(idx, h, x="moneyness", k=1.0):
            sel = np.isin(f["year"], yrs[idx])
            xx = np.concatenate([f[x][i, h] for i in np.where(sel)[0]]) / k
            yy = np.concatenate([f[f"res_{m}"][i, h] for i in np.where(sel)[0]])
            o = np.isfinite(xx) & np.isfinite(yy)
            return S_slope(xx[o], yy[o])
        allix = np.arange(len(yrs))
        slopes = [sl(allix, h) for h in range(f["moneyness"].shape[1])]
        wb6 = [sl(d_, 5, "wb_mm", 100.0) for d_ in draws[:500]]
        J[m] = {"cell_mean": np.where(valid, mean, np.nan).round(4).tolist(),
                "cell_ci95": [np.where(valid, np.percentile(bs, q, 0), np.nan).round(4).tolist() for q in (2.5, 97.5)],
                "spread_over_valid_cells": float(np.nanstd(np.where(valid, mean, np.nan))),  # sd of cell means: flatness
                "level_over_valid_cells": float(np.nanmean(np.where(valid, mean, np.nan))),
                "rms_over_valid_cells": float(np.sqrt(np.nanmean(np.where(valid, mean, np.nan) ** 2))),
                "max_abs_cell": float(np.nanmax(np.abs(np.where(valid, mean, np.nan)))),
                "slope_per_sd_by_h": np.round(slopes, 4).tolist(),
                "slope_per_100mm_wb_h6": [sl(allix, 5, "wb_mm", 100.0), float(np.percentile(wb6, 2.5)), float(np.percentile(wb6, 97.5))]}
    json.dump(J, open(f"{out}/surface.json", "w"), indent=1, default=lambda x: None if x != x else x)
    for m in METHODS:
        print(m, {k: v for k, v in J[m].items() if not k.startswith("cell")})
    print("effective years per cell (h x bin):\n", neff.round(1))
    print("wrote", f"{out}/surface.json")


def S_slope(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / (xm ** 2).sum())


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    {"compute": compute, "summarise": summarise}[cmd](out)
