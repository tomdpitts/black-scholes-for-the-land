"""3-month skill of M2 with rain removed (fair-fight fit "M2nd"), scored as suite.skill scores M1: m3.score on all
2019-25 starts, full grid, A_test and B_test, with M0 re-scored alongside as the reference. Box A should reproduce the
fair-fight number (rollout A_test skill -0.0056).

  .venv/bin/python revision/skill_m2nd.py   -> runs/20261022_1639_suite/skill_m2nd.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
import m3  # noqa: E402
import suite as S  # noqa: E402
from run import load  # noqa: E402

OUT = "runs/20261022_1639_suite"


def main():
    years = set(S.get_plan(OUT)["years"].tolist())
    FF = {k: {kk: S.jnp.asarray(v) for kk, v in P.items()} for k, P in np.load(S.FF, allow_pickle=True).item().items()}
    res = {}
    for box in "AB":
        d, D = S.data(box, "M1")
        mt = np.isin(d["year"][1:], list(years))
        res[f"{box}_test"] = {}
        for name, u in (("M0", False), ("M2nd", False)):
            p = FF[name]
            t1 = time.time()
            r = m3.score(lambda a, w, t: M.step(p, a, w, D["r"][t], D["phase"][t], D["x"], u), M.w_path(p, D, u), D, mt)
            res[f"{box}_test"][name] = {**r, "seconds": time.time() - t1}
        m0, m2 = res[f"{box}_test"]["M0"], res[f"{box}_test"]["M2nd"]
        res[f"{box}_test"]["M2nd_skill"] = {"onestep": 1 - m2["onestep_crps"] / m0["onestep_crps"],
                                            "3month": 1 - m2["rollout_crps"][-1] / m0["rollout_crps"][-1]}
        print(box, "M2 rain removed skill:", res[f"{box}_test"]["M2nd_skill"], flush=True)
    json.dump(res, open(f"{OUT}/skill_m2nd.json", "w"), indent=1, default=float)
    print("wrote", f"{OUT}/skill_m2nd.json")


if __name__ == "__main__":
    main()
