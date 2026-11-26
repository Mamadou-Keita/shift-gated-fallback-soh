#!/usr/bin/env python3
"""
Per-model tables with mean ± sample standard deviation (ddof = 1) over seeds, for EVERY model and protocol,
alone and inside the cell-level envelope (alone → enveloped). Same aggregation as seed_summary.py and as
Tables 3 (Tongji) and 4 (MICH): MAE per cell averaged over the folds of a protocol, then mean ± SD over seeds.
Also: the in-domain and unseen-T cells that the paper prints without ±, and the largest SD per column.

  python model_tables.py --name Tongji $R/wrap $R/seeds/Tongji_s1/wrap $R/seeds/Tongji_s2/wrap
  python model_tables.py --name MICH $R/MICH/wrap $R/seeds/MICH_s1/wrap $R/seeds/MICH_s2/wrap
Writes model_table_<name>.md (markdown, all variants) and model_table_<name>.csv.
"""
import argparse
import os

import pandas as pd

PROTOS = ["cell", "temp", "chem", "chemtemp", "chemcv"]
FAMILIES = ["GPR", "HGB", "MLP", "CNN", "LSTM", "CPMLP", "CPTransformer", "iTransformer", "PatchTST", "DLinear",
            "ConvTimeNet", "PatchMLP"]
DISPLAY = {"CNN": "1D-CNN"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wrap_dirs", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gate", default="GATEcell")
    args = ap.parse_args()
    frames = []
    for i, d in enumerate(args.wrap_dirs):
        R = pd.read_csv(os.path.join(d, "results.csv"))
        R = R.groupby(["protocol", "method"]).MAE_cell.mean().reset_index()
        R["seed"] = i
        frames.append(R)
    A = pd.concat(frames)
    S = A.groupby(["method", "protocol"]).MAE_cell.agg(["mean", lambda v: v.std(ddof=1), "count"]).reset_index()
    S.columns = ["method", "protocol", "mean", "sd", "n"]
    S.to_csv(f"model_table_{args.name}.csv", index=False)
    get = S.set_index(["method", "protocol"])
    fmt = lambda m, p: (f"{get.loc[(m, p), 'mean']:.2f} ± {get.loc[(m, p), 'sd']:.2f}"
                        if (m, p) in get.index else "n/a")
    protos = [p for p in PROTOS if p in set(S.protocol)]
    head = {"cell": "In-domain", "temp": "Unseen temperature", "chem": "Unseen chemistry / formation",
            "chemtemp": "Double shift", "chemcv": "Unseen current"}
    lines = [f"| {args.name}, model | " + " | ".join(head[p] for p in protos) + " |",
             "|---" * (len(protos) + 1) + "|"]
    for fam in FAMILIES:
        for t in ("+T", ""):
            m = fam + t
            if not any((m, p) in get.index for p in protos):
                continue
            cells = []
            for p in protos:
                g = f"{args.gate}[{m}]"
                cells.append("n/a" if (m, p) not in get.index else f"{fmt(m, p)} → {fmt(g, p)}")
            lines.append(f"| {DISPLAY.get(fam, fam)}{t} | " + " | ".join(cells) + " |")
    for m, lab in (("base_dQ+T", "Base ΔQ/Q_nom + T (fallback)"), ("FUS_pondérée", "Ridge fusion FUS")):
        if any((m, p) in get.index for p in protos):
            lines.append(f"| {lab} | " + " | ".join(fmt(m, p) for p in protos) + " |")
    open(f"model_table_{args.name}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSeeds: {A.seed.nunique()}; SD with ddof = 1. Largest SD per protocol (experts alone / enveloped):")
    ex = S[S.method.isin([f + t for f in FAMILIES for t in ("", "+T")])]
    en = S[S.method.str.startswith(args.gate + "[")]
    for p in protos:
        a, b = ex[ex.protocol == p], en[en.protocol == p]
        print(f"  {p:9s} {a.sd.max():.2f} ({a.loc[a.sd.idxmax(), 'method']}) / {b.sd.max():.2f} "
              f"({b.loc[b.sd.idxmax(), 'method']})")
    print(f"\nmodel_table_{args.name}.md, model_table_{args.name}.csv")


if __name__ == "__main__":
    main()
