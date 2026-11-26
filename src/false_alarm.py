#!/usr/bin/env python3
"""
Cell-level false alarms of the gate on in-domain test cells (protocol "cell"), with 95 % Wilson intervals
over cells. A cell raises an alarm at its r-th window if its cell-level weight w_cell < 1 (running median of
the score above the threshold q). Calibration uses training cells only; in-domain test cells are disjoint
from them by construction (GroupKFold by cell). Reads cycles.csv files (gated_wrapper.py --dump-cycles).

  python false_alarm.py Tongji=$R/wrap_dump/cycles.csv MICH=$R/ablation/full/MICH/cycles.csv ...
"""
import sys

import numpy as np
import pandas as pd

RANKS = [1, 2, 3, 5, 10]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)


rows = []
for spec in sys.argv[1:]:
    name, files = spec.split("=", 1)
    for s, f in enumerate(files.split(",")):
        C = pd.read_csv(f, usecols=["protocol", "split", "cell", "rank", "wc"])
        C = C[C.protocol == "cell"]
        alarm = C.wc < 1 - 1e-9
        for r in RANKS + ["any", "final"]:
            if r == "any":
                per = alarm.groupby([C.split, C.cell]).any()
            elif r == "final":
                last = C.groupby([C.split, C.cell])["rank"].transform("max") == C["rank"]
                per = alarm[last].groupby([C.split[last], C.cell[last]]).any()
            else:
                sel = C["rank"] == r
                per = alarm[sel].groupby([C.split[sel], C.cell[sel]]).any()
            rows.append(dict(dataset=name, seed=s, window=str(r), alarms=int(per.sum()), cells=int(per.size)))
R = pd.DataFrame(rows).groupby(["dataset", "window"], sort=False)[["alarms", "cells"]].sum().reset_index()
R[["lo", "hi"]] = [wilson(a, n) for a, n in zip(R.alarms, R.cells)]
R["rate"] = R.alarms / R.cells
R["txt"] = R.apply(lambda x: f"{x.alarms}/{x.cells} = {x.rate:.1%} [{x.lo:.1%}, {x.hi:.1%}]", axis=1)
pd.set_option("display.width", 250)
print("False alarms on in-domain test cells (cell-level weight < 1), pooled over seeds given; 95 % Wilson CI over cells.")
print("window = r-th window of the cell; any = at any window of its life; final = at its last window.")
print(R.pivot(index="dataset", columns="window", values="txt")[[str(r) for r in RANKS] + ["any", "final"]].to_string())
