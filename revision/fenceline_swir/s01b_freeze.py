"""Freeze: PREREG, raw downloads, the previous run's design files, Box B data, and every script.
After this, outcome steps verify these hashes and refuse to run if anything changed."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fs_common as C

C.start("01b_freeze", __file__)
fz = C.out("FREEZE.json")
assert not os.path.exists(fz), "already frozen"
assert not os.path.exists(C.out("outputs", "outcomes.npz")), "outcomes exist before freeze"
files = [C.out("PREREG.md"), C.out("inputs", "raw", "mod09a1_terra.npz"), C.out("inputs", "raw", "myd09a1_aqua.npz"), C.DATA_B]
files += [os.path.join(C.PREV, f) for f in ("design/design.npz", "design/design_meta.json", "inputs/exposure/exposure.csv",
                                            "inputs/exposure/years.csv", "outputs/results.json", "FREEZE.json")]
files += sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py")))
rec = {"frozen_utc": C.now(), "git": C.git_state(), "sha256": {os.path.relpath(f, C.ROOT): C.sha256(f) for f in files}}
C.write_new(fz, json.dumps(rec, indent=1))
C.log("01b_freeze", "FROZEN", file=os.path.relpath(fz, C.ROOT), freeze_sha256=C.sha256(fz),
      prereg_sha256=rec["sha256"][os.path.relpath(files[0], C.ROOT)])
