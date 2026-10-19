"""Step 03: static design. Groups, masks, matches and placebo-in-space translations.
Reads boundaries (step 01), covariates (step 02), exposure coding (inputs/exposure/exposure.csv),
and from data/B/built.npz ONLY the static pixel traits x (no anomaly, no rain, no residual).
Writes RUN/design/*. Refuses to run if any outcome file exists."""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

C.start("03_design", __file__)
assert not os.path.exists(C.out("outputs", "outcomes.npz")), "outcomes exist: design must precede outcomes"
TR = from_origin(C.BOX_B[0], C.BOX_B[3], C.RES, C.RES)
BUF = 3          # pixels, interior rule
EDGE = 2         # pixels from grid edge
K, CALIPER = 5, 1.0
SEED, NSHIFT = 20171, 200

# ------------------------------------------------------------------ exposure coding and polygons
expo = list(csv.DictReader(open(C.out("inputs", "exposure", "exposure.csv"))))
fc = json.load(open(C.out("inputs", "osm", "polygons.geojson")))   # step 01b (Nominatim)
geom_by_osm = {f["properties"]["osm"]: f["geometry"] for f in fc["features"]}


def mask_of(osm_ids):
    """Union of the listed OSM polygons, rasterised by pixel centre (holes respected by rasterize)."""
    m = np.zeros((C.N, C.N), bool)
    for tok in filter(None, (t.strip() for t in osm_ids.split(";"))):
        m |= rasterize([(geom_by_osm[tok], 1)], out_shape=(C.N, C.N), transform=TR, fill=0, dtype="uint8").astype(bool)
    return m


props = {}
for r in expo:
    if not r["osm_ids"].strip():
        continue
    props[r["property"]] = {"mask": mask_of(r["osm_ids"]), **r}
private = np.zeros((C.N, C.N), bool); other = np.zeros((C.N, C.N), bool); community = np.zeros((C.N, C.N), bool)
for k, v in props.items():
    t = v["tenure"]
    if t in ("private_ranch", "private_conservancy", "company_ranch"):
        private |= v["mask"]
    elif t in ("community_conservancy", "group_ranch"):
        community |= v["mask"]
    else:
        other |= v["mask"]
treated_props = [k for k, v in props.items() if v["invaded_2017"] == "1"
                 and v["tenure"] in ("private_ranch", "private_conservancy", "company_ranch")]
C.log("03_design", "polygons rasterised", n_properties=len(props),
      pixels_by_property={k: int(v["mask"].sum()) for k, v in props.items()},
      treated_properties=treated_props)

# ------------------------------------------------------------------ static masks
x = np.load(C.DATA_B)["x"]                               # static traits only
wc = np.load(C.out("inputs", "covariates", "worldcover2021_fractions.npz"))
clay = np.load(C.out("inputs", "covariates", "clay_5-15cm_gkg.npy"))
elev = np.load(os.path.join(C.ROOT, "data/B/dem.npz"))["elev"].astype(np.float64)
gy, gx = np.gradient(elev, 278.0)                         # m per m; pixel ~278 m
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
ii, jj = np.mgrid[:C.N, :C.N]
static_ok = ((wc["crop"] < 0.2) & (wc["built"] < 0.1) & ((wc["water"] + wc["wetland"]) < 0.1)
             & (x[..., 0] <= np.percentile(x[..., 0], 95))
             & (ii >= EDGE) & (jj >= EDGE) & (ii < C.N - EDGE) & (jj < C.N - EDGE))
C.log("03_design", "static mask", eligible_pixels=int(static_ok.sum()),
      dropped={"crop": int((wc["crop"] >= 0.2).sum()), "built": int((wc["built"] >= 0.1).sum()),
               "water_wetland": int(((wc["water"] + wc["wetland"]) >= 0.1).sum()),
               "x0_above_p95": int((x[..., 0] > np.percentile(x[..., 0], 95)).sum()), "edge": int(((ii < EDGE) | (jj < EDGE) | (ii >= C.N - EDGE) | (jj >= C.N - EDGE)).sum())})

# interior rules
d_out_private = distance_transform_edt(~private)          # distance from non-private pixel to nearest private pixel
treated_id = np.full((C.N, C.N), -1, int)
for n_, k in enumerate(treated_props):
    m = props[k]["mask"]
    d_in = distance_transform_edt(m)                      # distance from inside pixel to nearest outside pixel
    treated_id[(m) & (d_in >= BUF) & static_ok & (treated_id < 0)] = n_
treated = treated_id >= 0
control_pool = (~private) & (~other) & (d_out_private >= BUF) & static_ok
community_ctrl = control_pool & community & (distance_transform_edt(community) >= BUF)
C.log("03_design", "groups", treated_pixels=int(treated.sum()),
      treated_by_property={k: int((treated_id == n_).sum()) for n_, k in enumerate(treated_props)},
      control_pool_pixels=int(control_pool.sum()), community_polygon_controls=int(community_ctrl.sum()))

# ------------------------------------------------------------------ matching
cov_names = ["mean_ndvi", "amplitude", "clim_rain", "elevation", "slope", "clay"]
cov = np.stack([x[..., 0], x[..., 1], x[..., 2], elev, slope, clay], -1).astype(np.float64)
elig = treated | control_pool
mu, sd = cov[elig].mean(0), cov[elig].std(0)
Z = (cov - mu) / sd


def match(tmask, pool):
    ti = np.flatnonzero(tmask.ravel()); pi = np.flatnonzero(pool.ravel())
    tree = cKDTree(Z.reshape(-1, 6)[pi])
    dist, idx = tree.query(Z.reshape(-1, 6)[ti], k=K)
    keep = dist[:, 0] <= CALIPER
    return ti[keep], pi[idx[keep]], dist[keep], int((~keep).sum())


t_idx, m_idx, m_dist, n_drop = match(treated, control_pool)


def smd(a, b):
    return [float((a[:, k].mean() - b[:, k].mean()) / np.sqrt((a[:, k].var() + b[:, k].var()) / 2)) for k in range(6)]


Zf = cov.reshape(-1, 6)
bal = {"before": dict(zip(cov_names, np.round(smd(Zf[np.flatnonzero(treated.ravel())], Zf[np.flatnonzero(control_pool.ravel())]), 3).tolist())),
       "after": dict(zip(cov_names, np.round(smd(Zf[t_idx], Zf[m_idx.ravel()]), 3).tolist()))}
C.log("03_design", "matching", k=K, caliper=CALIPER, treated_matched=int(len(t_idx)), dropped_by_caliper=n_drop,
      median_nn_distance=round(float(np.median(m_dist[:, 0])), 3), balance_smd=bal)
if max(abs(v) for v in bal["after"].values()) > 0.1:                  # PREREG balance rule: halve caliper once
    CALIPER = CALIPER / 2
    t_idx, m_idx, m_dist, n_drop = match(treated, control_pool)
    bal = {"before": bal["before"],
           "after": dict(zip(cov_names, np.round(smd(Zf[t_idx], Zf[m_idx.ravel()]), 3).tolist()))}
    C.log("03_design", "balance rule triggered: caliper halved once (final)", caliper=CALIPER,
          treated_matched=int(len(t_idx)), dropped_by_caliper=n_drop, balance_smd=bal)

# clean-community matches (S6)
if community_ctrl.sum() >= 300:
    t6, m6, d6, n6 = match(treated, community_ctrl)
else:
    t6 = m6 = np.zeros((0,), int); n6 = None
C.log("03_design", "S6 community-only matching", eligible=int(community_ctrl.sum()), matched=int(len(t6)), dropped=n6)

# ------------------------------------------------------------------ placebo-in-space translations (S4)
rng = np.random.default_rng(SEED)
ti, tj = np.nonzero(treated)
shifts, kept = [], []
tries = 0
while len(kept) < NSHIFT and tries < 20000:
    tries += 1
    di, dj = rng.integers(-C.N, C.N, 2)
    si, sj = ti + di, tj + dj
    ok = (si >= 0) & (sj >= 0) & (si < C.N) & (sj < C.N)
    if ok.mean() < 0.8:
        continue
    inpool = np.zeros_like(ok); inpool[ok] = control_pool[si[ok], sj[ok]]
    if inpool.mean() < 0.8:
        continue
    fake = np.zeros((C.N, C.N), bool); fake[si[inpool], sj[inpool]] = True
    ft, fm, _, _ = match(fake, control_pool & ~fake)
    kept.append((int(di), int(dj))); shifts.append((ft, fm))
C.log("03_design", "S4 placebo-in-space translations", seed=SEED, requested=NSHIFT, kept=len(kept), tries=tries)

# ------------------------------------------------------------------ save (write-once)
dz = C.out("design", "design.npz")
assert not os.path.exists(dz)
np.savez_compressed(dz, treated_id=treated_id, control_pool=control_pool, community_ctrl=community_ctrl,
                    private=private, other=other, community=community, static_ok=static_ok,
                    t_idx=t_idx, m_idx=m_idx, m_dist=m_dist, t6=t6, m6=m6,
                    shift_offsets=np.array(kept, int),
                    shift_t=np.array([s[0] for s in shifts], dtype=object), shift_m=np.array([s[1] for s in shifts], dtype=object),
                    cov=cov, cov_names=np.array(cov_names))
meta = {"treated_properties": treated_props, "buffer_px": BUF, "edge_px": EDGE, "k": K, "caliper": CALIPER,
        "balance_smd": bal, "property_rows": [{k: v for k, v in r.items()} for r in expo]}
mj = C.out("design", "design_meta.json")
C.write_new(mj, json.dumps(meta, indent=1))
C.log("03_design", "design saved", design=os.path.relpath(dz, C.ROOT), design_sha256=C.sha256(dz),
      meta=os.path.relpath(mj, C.ROOT), meta_sha256=C.sha256(mj))

# design map (no outcomes)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
img = np.full((C.N, C.N, 3), 0.92)
img[private] = (0.80, 0.80, 0.80); img[other] = (0.85, 0.85, 0.75)
img[control_pool] = (0.62, 0.78, 0.92); img[community_ctrl] = (0.35, 0.60, 0.85)
img[treated] = (0.90, 0.45, 0.25); img[~static_ok & ~private] = (1, 1, 1)
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img, extent=(C.BOX_B[0], C.BOX_B[2], C.BOX_B[1], C.BOX_B[3]))
for k, v in props.items():
    if v["mask"].any():
        ax.contour(np.linspace(C.BOX_B[0], C.BOX_B[2], C.N), np.linspace(C.BOX_B[3], C.BOX_B[1], C.N),
                   v["mask"].astype(float), levels=[0.5], colors="k", linewidths=0.5)
        yy, xx = np.nonzero(v["mask"]); ax.text(C.BOX_B[0] + (xx.mean() + .5) * C.RES, C.BOX_B[3] - (yy.mean() + .5) * C.RES, k, fontsize=5, ha="center")
ax.set_title("Design: orange treated interior, blue control pool (dark: community polygons)", fontsize=6)
fp = C.out("design", "design_map.png")
fig.savefig(fp, dpi=200, bbox_inches="tight")
C.log("03_design", "design map", file=os.path.relpath(fp, C.ROOT), sha256=C.sha256(fp))
