#!/usr/bin/env python3
"""
Blended cells: per transfer direction (split), number of test cells and MAE of the fallback, of the experts
and of the cell-level envelope (means over experts), for each wrap output given. Used to reconcile the
figures quoted in Sections 6.8 and 6.9.

  python blend_check.py old=$R/Tongji_blend/wrap/cells.csv,$R/seeds/Tongji_blend_s1/wrap/cells.csv,$R/seeds/Tongji_blend_s2/wrap/cells.csv \
                        review6=$R/review6/wrap/Tongji_blend_s0/cells.csv,$R/review6/wrap/Tongji_blend_s1/cells.csv,$R/review6/wrap/Tongji_blend_s2/cells.csv
"""
import sys

import pandas as pd

BASE = "base_dQ+T"
pd.set_option("display.width", 220)
for spec in sys.argv[1:]:
    name, files = spec.split("=", 1)
    rows = []
    for s, f in enumerate(files.split(",")):
        C = pd.read_csv(f)
        experts = sorted({m[9:-1] for m in C.method.unique() if m.startswith("GATEcell[")})
        for (proto, split), sub in C.groupby(["protocol", "split"]):
            M = sub.groupby("method").mae.mean()
            rows.append(dict(seed=s, protocol=proto, split=split, cells=sub[sub.method == BASE].cell.nunique(),
                             fallback=M.get(BASE), experts=M[[e for e in experts if e in M]].mean(),
                             envelope=M[[f"GATEcell[{e}]" for e in experts if f"GATEcell[{e}]" in M]].mean(),
                             env_min=M[[f"GATEcell[{e}]" for e in experts if f"GATEcell[{e}]" in M]].min(),
                             env_max=M[[f"GATEcell[{e}]" for e in experts if f"GATEcell[{e}]" in M]].max()))
    R = pd.DataFrame(rows)
    print(f"\n=== {name}: per split, mean over seeds ===")
    print(R.groupby(["protocol", "split"])[["cells", "fallback", "experts", "envelope", "env_min", "env_max"]]
          .mean().round(2).to_string())
    S = R.groupby("seed")[["fallback", "experts", "envelope"]].mean().mean()
    print(f"mean over splits then seeds: fallback {S.fallback:.2f}, experts {S.experts:.2f}, envelope {S.envelope:.2f}")
