"""Step 00: record starting state; hash read-only inputs incl. the first run's frozen design; log PREREG hash
before any data download."""
import os, sys, platform
sys.path.insert(0, os.path.dirname(__file__))
import fs_common as C

C.write_new(os.path.join(C.RUN, "LOG.md"),
            "# Fence-line SWIR test: audit log\n\nAppend-only. One line per event, UTC. Machine-readable twin: `log.jsonl`.\n\n")
C.start("00_init", __file__)
import numpy as np, scipy, rasterio
C.log("00_init", "environment", platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
      rasterio=rasterio.__version__, venv=sys.executable)
C.log("00_init", "git state at start", **C.git_state())
C.log("00_init", "previous run (design source) verified against its FREEZE.json, read-only", **C.verify_prev())
ro = {"data_B": C.DATA_B}
C.log("00_init", "read-only inputs hashed", **{k: f"{os.path.relpath(v, C.ROOT)} sha256={C.sha256(v)}" for k, v in ro.items()})
C.log("prereg", "PREREG.md written before any reflectance download", sha256=C.sha256(os.path.join(C.RUN, "PREREG.md")))
C.log("00_init", "blinding statement", note=("No MODIS surface-reflectance data for Box B has been downloaded or viewed in this project "
      "before this run. The download step logs counts and coverage only; indices are computed only after FREEZE.json."))
