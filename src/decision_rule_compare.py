#!/usr/bin/env python3
"""
Decision-rule ablation: progressive weight (GATEcell, the paper's envelope) vs hard switch with the same
score and threshold (SWITCHcell: expert if the cell's running-median score is below q, fallback otherwise).
Also the cycle-level versions (GATE vs SWITCH). Reads wrap/results.csv files (gated_wrapper.py), one per
seed, grouped by dataset; MAE = mean over folds, then over seeds.

  python decision_rule_compare.py Tongji=$R/wrap,$R/seeds/Tongji_s1/wrap,$R/seeds/Tongji_s2/wrap MICH=... 
"""
import argparse
import os

import pandas as pd

NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./formation/ageing", "chemtemp": "double",
         "chemcv": "unseen current"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="NAME=wrapdir1,wrapdir2,...")
    ap.add_argument("--out", default="decision_rule.csv")
    args = ap.parse_args()
    rows = []
    for gspec in args.groups:
        name, dirs = gspec.split("=", 1)
        for s, d in enumerate(dirs.split(",")):
            f = os.path.join(d, "results.csv")
            if not os.path.exists(f):
                print(f"  missing: {f}"); continue
            R = pd.read_csv(f).groupby(["protocol", "method"]).MAE_cell.mean().reset_index()
            R["dataset"], R["seed"] = name, s
            rows.append(R)
    A = pd.concat(rows)
    A = A.groupby(["dataset", "protocol", "method"]).MAE_cell.mean().reset_index()
    out = []
    for (ds, proto), sub in A.groupby(["dataset", "protocol"]):
        m = sub.set_index("method").MAE_cell
        experts = [k[9:-1] for k in m.index if k.startswith("GATEcell[") and f"SWITCHcell[{k[9:-1]}]" in m.index]
        if not experts:
            continue
        g = pd.Series({e: m[f"GATEcell[{e}]"] for e in experts})
        sw = pd.Series({e: m[f"SWITCHcell[{e}]"] for e in experts})
        gc = pd.Series({e: m[f"GATE[{e}]"] for e in experts})
        swc = pd.Series({e: m[f"SWITCH[{e}]"] for e in experts})
        d = g - sw
        out.append(dict(dataset=ds, protocol=proto, experts=len(experts), progressive_cell=g.mean(),
                        switch_cell=sw.mean(), diff_mean=d.mean(), diff_min=d.min(), diff_max=d.max(),
                        progressive_better=f"{int((d < -0.005).sum())}/{len(d)}",
                        switch_better=f"{int((d > 0.005).sum())}/{len(d)}",
                        progressive_cycle=gc.mean(), switch_cycle=swc.mean()))
    T = pd.DataFrame(out)
    T.to_csv(args.out, index=False)
    T["protocol"] = T.protocol.map(NAMES).fillna(T.protocol)
    pd.set_option("display.width", 250)
    print("MAE (SOH points), mean over experts and seeds. diff = progressive - switch (negative: progressive better); "
          "counts use a 0.005 tolerance.")
    print(T.round(3).to_string(index=False))
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
