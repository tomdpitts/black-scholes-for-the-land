"""Step 00: record the starting state. Writes only inside RUN."""
import os, sys, json, platform
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C

C.write_new(os.path.join(C.RUN, "LOG.md"),
            "# Fence-line test: audit log\n\nAppend-only. One line per event, UTC. Machine-readable twin: `log.jsonl`.\n\n")
C.start("00_init", __file__)
import numpy as np, jax, scipy, rasterio
C.log("00_init", "environment", platform=platform.platform(), numpy=np.__version__, jax=jax.__version__,
      scipy=scipy.__version__, rasterio=rasterio.__version__, venv=sys.executable)
C.log("00_init", "git state at start", **C.git_state())
ins = {"model_params": os.path.join(C.MODEL_RUN, "params.npy"), "data_B": C.DATA_B,
       "model_py": os.path.join(C.ROOT, "model.py"), "run_py": os.path.join(C.ROOT, "run.py"),
       "aoi_py": os.path.join(C.ROOT, "aoi.py"), "fire_B": os.path.join(C.ROOT, "data/B/fire.npz"),
       "dem_B": os.path.join(C.ROOT, "data/B/dem.npz")}
C.log("00_init", "read-only inputs hashed (never written by this run)",
      **{k: f"{os.path.relpath(v, C.ROOT)} sha256={C.sha256(v)}" for k, v in ins.items()})
sys.path.insert(0, C.ROOT)
import aoi
assert tuple(aoi.BOXES["B"]) == C.BOX_B and aoi.RES == C.RES and aoi.N == C.N
C.log("00_init", "grid check: Box B bounds, resolution and size match aoi.py", box=C.BOX_B, res=C.RES, n=C.N)
C.log("00_init", "blinding statement",
      note=("Before this run, no property-level residual for Box B has been computed or viewed in this "
            "project: earlier runs used Box A for residual maps; runs/20261006_1753_identify reports only "
            "box-wide residual-vs-rain bins for Box B. Outcomes for this test are computed only after FREEZE.json."))
