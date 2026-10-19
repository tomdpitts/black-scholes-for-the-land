"""Shared paths, hashing and the append-only audit log for the fence-line SWIR (dry-vegetation) test.

Every step writes one JSON line to <RUN>/log.jsonl and one human line to <RUN>/LOG.md.
Nothing here writes outside <RUN>. Inputs outside <RUN> are read-only and hashed.
"""
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "/Users/tompitts/dphil/ProjectB"
RUN = os.path.join(ROOT, "runs/20261014_0101_fenceline_swir")
PREV = os.path.join(ROOT, "runs/20261013_2150_fenceline")    # frozen design is read from here, read-only
MODEL_RUN = os.path.join(ROOT, "runs/20260914_1708_pipeline_h100_fairfight")  # the paper's M2
DATA_B = os.path.join(ROOT, "data/B/built.npz")
BOX_B = (36.70, 0.25, 37.10, 0.65)  # (west, south, east, north); must match aoi.BOXES["B"]
RES = 0.0025
N = 160


def now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(step, msg, **fields):
    """Append to both logs. Never rewrites earlier lines."""
    rec = {"utc": now(), "step": step, "msg": msg, **fields}
    with open(os.path.join(RUN, "log.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    extra = "".join(f"\n    - {k}: {v}" for k, v in fields.items())
    with open(os.path.join(RUN, "LOG.md"), "a") as f:
        f.write(f"- `{rec['utc']}` **{step}** {msg}{extra}\n")
    print(f"[{step}] {msg}")


def snapshot_code(script_path):
    """Copy the running script into <RUN>/code/ under its hash, and log it."""
    h = sha256(script_path)
    dst = os.path.join(RUN, "code", f"{os.path.basename(script_path)}.{h[:12]}")
    if not os.path.exists(dst):
        shutil.copy2(script_path, dst)
    common = os.path.join(os.path.dirname(script_path), "fs_common.py")
    hc = sha256(common)
    dstc = os.path.join(RUN, "code", f"fs_common.py.{hc[:12]}")
    if not os.path.exists(dstc):
        shutil.copy2(common, dstc)
    return h, hc


def start(step, script_path):
    """Call at the top of every script: snapshot code, log interpreter and argv."""
    h, hc = snapshot_code(script_path)
    log(step, "start", script=os.path.relpath(script_path, ROOT), script_sha256=h,
        fs_common_sha256=hc, python=sys.version.split()[0], argv=sys.argv[1:])


def out(*parts):
    p = os.path.join(RUN, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_new(path, data, mode="w"):
    """Refuse to overwrite: every output file is written once."""
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    with open(path, mode) as f:
        f.write(data)
    return sha256(path)


def git_state():
    g = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()
    return {"head": g("rev-parse", "HEAD"), "status_short": g("status", "--short")}


def pixel_centres():
    """Lat/lon of pixel centres on the Box B grid (row 0 = north, col 0 = west)."""
    import numpy as np
    w, s, e, n = BOX_B
    lat = n - (np.arange(N) + 0.5) * RES
    lon = w + (np.arange(N) + 0.5) * RES
    return lat, lon


def verify_freeze(step):
    """Later steps call this first. Data, model, PREREG, exposure and design must be unchanged.
    A script changed after the freeze is allowed only if DEVIATIONS.md names it."""
    fz = os.path.join(RUN, "FREEZE.json")
    if not os.path.exists(fz):
        raise RuntimeError("not frozen: run s04_freeze.py first")
    rec = json.load(open(fz))
    dev = open(os.path.join(RUN, "DEVIATIONS.md")).read() if os.path.exists(os.path.join(RUN, "DEVIATIONS.md")) else ""
    changed = [f for f, h in rec["sha256"].items() if sha256(os.path.join(ROOT, f)) != h]
    hard = [f for f in changed if not (f.startswith("revision/fenceline_swir/") and f.endswith(".py"))]
    soft = [f for f in changed if f not in hard]
    if hard:
        raise RuntimeError(f"frozen inputs changed since FREEZE.json: {hard}")
    undeclared = [f for f in soft if os.path.basename(f) not in dev]
    if undeclared:
        raise RuntimeError(f"scripts changed after freeze without a DEVIATIONS.md entry: {undeclared}")
    log(step, "freeze verified", changed_scripts_declared_in_deviations=soft)
    return rec


PREV_FILES = ["design/design.npz", "design/design_meta.json", "inputs/exposure/exposure.csv",
              "inputs/exposure/years.csv", "outputs/results.json", "FREEZE.json"]


def verify_prev():
    """The first run's frozen design must be byte-identical to what its FREEZE.json recorded."""
    rec = json.load(open(os.path.join(PREV, "FREEZE.json")))
    for f in ("design/design.npz", "design/design_meta.json", "inputs/exposure/exposure.csv", "inputs/exposure/years.csv"):
        key = os.path.relpath(os.path.join(PREV, f), ROOT)
        if sha256(os.path.join(PREV, f)) != rec["sha256"][key]:
            raise RuntimeError(f"previous run file changed: {key}")
    return {os.path.relpath(os.path.join(PREV, f), ROOT): sha256(os.path.join(PREV, f)) for f in PREV_FILES}
