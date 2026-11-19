"""Step 04: freeze. Hash PREREG, exposure coding, design and every script. After this file
exists, outcome steps verify these hashes and refuse to run if anything changed."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C

C.start("04_freeze", __file__)
fz = C.out("FREEZE.json")
assert not os.path.exists(fz), "already frozen"
assert not os.path.exists(C.out("outputs", "outcomes.npz")), "outcomes exist before freeze"
files = [C.out("PREREG.md"), C.out("inputs", "exposure", "exposure.csv"), C.out("inputs", "exposure", "years.csv"),
         C.out("design", "design.npz"), C.out("design", "design_meta.json"), C.out("design", "s4_line_splits.npz"),
         os.path.join(C.MODEL_RUN, "params.npy"), C.DATA_B, os.path.join(C.ROOT, "model.py")]
files += sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py")))
rec = {"frozen_utc": C.now(), "git": C.git_state(),
       "sha256": {os.path.relpath(f, C.ROOT): C.sha256(f) for f in files}}
C.write_new(fz, json.dumps(rec, indent=1))
C.log("04_freeze", "FROZEN", file=os.path.relpath(fz, C.ROOT), freeze_sha256=C.sha256(fz),
      prereg_sha256=rec["sha256"][os.path.relpath(files[0], C.ROOT)])

