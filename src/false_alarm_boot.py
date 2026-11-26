#!/usr/bin/env python3
"""
False alarms on in-domain test cells with a cluster bootstrap over cells: each resampled cell keeps all its
seeds, so the interval accounts for the same cell being counted once per seed (cell-seed pairs are not
independent). Alarm at window r: cell-level weight w_cell < 1. Reads cycles.csv (gated_wrapper --dump-cycles).

  python false_alarm_boot.py Tongji=a.csv,b.csv,c.csv MICH=... ISU=...
"""
import sys

import numpy as np
import pandas as pd

RANKS = [1, 2, 3, 5, 10]
NB = 2000
rng = np.random.default_rng(0)
rows = []
for spec in sys.argv[1:]:
    name, files = spec.split("=", 1)
    per = []
    for s, f in enumerate(files.split(",")):
        C = pd.read_csv(f, usecols=["protocol", "cell", "rank", "wc"])
        C = C[C.protocol == "cell"].copy()
        C["alarm"] = C.wc < 1 - 1e-9
        C["seed"] = s
        last = C.groupby("cell")["rank"].transform("max") == C["rank"]
        for r in RANKS + ["any", "final"]:
            if r == "any":
                a = C.groupby("cell").alarm.any()
            elif r == "final":
                a = C[last].groupby("cell").alarm.any()
            else:
                a = C[C["rank"] == r].groupby("cell").alarm.any()
            per.append(pd.DataFrame(dict(window=str(r), seed=s, cell=a.index, alarm=a.values)))
    P = pd.concat(per)
    for r, sub in P.groupby("window", sort=False):
        by_cell = sub.groupby("cell").alarm.agg(["sum", "size"])
        k, n = by_cell["sum"].values, by_cell["size"].values
        idx = rng.integers(0, len(k), size=(NB, len(k)))
        boot = k[idx].sum(1) / n[idx].sum(1)
        rows.append(dict(dataset=name, window=r, alarms=int(k.sum()), pairs=int(n.sum()), cells=len(k),
                         rate=k.sum() / n.sum(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
R = pd.DataFrame(rows)
if R.empty:
    raise SystemExit("no in-domain (protocol cell) windows in the files given")
R["txt"] = R.apply(lambda x: f"{x.rate:.1%} [{x.lo:.1%}, {x.hi:.1%}] ({x.cells} cells)", axis=1)
pd.set_option("display.width", 250)
print("False alarms on in-domain test cells (w_cell < 1); 95 % cluster-bootstrap CI over cells (seeds kept together).")
print(R.pivot(index="dataset", columns="window", values="txt")[[str(r) for r in RANKS] + ["any", "final"]].to_string())
