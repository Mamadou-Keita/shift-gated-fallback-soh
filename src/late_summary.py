#!/usr/bin/env python3
"""
Summary of late_init.py: MAE (SOH points) per dataset, protocol and start level, on the observed windows.
Columns: expert alone and envelopes are means over expert variants; everything is a mean over seeds.

  python late_summary.py Tongji=$R/review6/late/Tongji_s0/cells.csv,... Blend=... MICH=... ISU=...
"""
import argparse

import pandas as pd

NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./form./ageing", "chemtemp": "double shift"}
COLS = ["expert", "fallback_nominal", "fallback_selfref_new", "fallback_selfref_late",
        "ENV_nominal", "ENV_selfref_new", "ENV_selfref_late"]


def family(m):
    for k in ("ENV_nominal", "ENV_selfref_new", "ENV_selfref_late"):
        if m.startswith(k + "["):
            return k
    return m if m.startswith("fallback_") else "expert"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--out", default="late_summary.csv")
    args = ap.parse_args()
    out = []
    for grp in args.groups:
        name, files = grp.split("=", 1)
        for s, f in enumerate(files.split(",")):
            C = pd.read_csv(f)
            C["family"] = C.method.map(family)
            # split-mean MAE per method, then mean over the methods of a family (experts)
            M = C.groupby(["level", "protocol", "split", "method", "family"]).mae.mean().reset_index()
            M = M.groupby(["level", "protocol", "method", "family"]).mae.mean().reset_index()
            F = M.groupby(["level", "protocol", "family"]).mae.mean().unstack("family")
            n = C[C.method == "fallback_nominal"].groupby(["level", "protocol"]).cell.size()
            F["cells"] = n
            F = F.reset_index()
            F["dataset"], F["seed"] = name, s
            out.append(F)
    A = pd.concat(out)
    A.to_csv(args.out, index=False)
    S = A.groupby(["dataset", "protocol", "level"])[COLS + ["cells"]].mean()
    order = {"new": 0, "95": 1, "90": 2, "85": 3, "80": 4}
    S = S.reset_index()
    S["o"] = S.level.astype(str).map(order).fillna(9)
    S["protocol"] = S.protocol.map(NAMES).fillna(S.protocol)
    S = S.sort_values(["dataset", "protocol", "o"]).drop(columns="o")
    pd.set_option("display.width", 250)
    print("MAE on the observed windows (SOH points), mean over seeds; expert and ENV = mean over experts.")
    print("level = SOH at which test cells are first observed (new = from the start). selfref_new = reference from "
          "the true first windows (assumption met); selfref_late = reference from the first OBSERVED windows.")
    print(S.round(2).to_string(index=False))
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
