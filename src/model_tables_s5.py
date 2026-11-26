#!/usr/bin/env python3
"""
Supplementary Section S5: per-model tables, every expert variant (with and without T, with current input +I),
alone -> inside the cell-level envelope, mean +/- sample SD (ddof = 1) over seeds.
Same aggregation as the paper tables: MAE per cell, mean within each split, mean over the splits of a protocol,
then mean +/- SD over seeds.

  python model_tables_s5.py --name Tongji $R/wrap $R/seeds/Tongji_s1/wrap $R/seeds/Tongji_s2/wrap
  python model_tables_s5.py --name MICH   $R/MICH/wrap $R/seeds/MICH_s1/wrap $R/seeds/MICH_s2/wrap
  python model_tables_s5.py --name ISU    $R/dr/ISU_c5_s0 $R/dr/ISU_c5_s1 $R/dr/ISU_c5_s2
  python model_tables_s5.py --name ISUrate $R/dr/ISU_rate_s0 $R/dr/ISU_rate_s1 $R/dr/ISU_rate_s2
Writes S5_<name>.md and S5_<name>.csv in the current directory.
"""
import argparse
import os
import re

import pandas as pd

FAMILIES = ["GPR", "HGB", "MLP", "CNN", "LSTM", "CPMLP", "CPTransformer", "iTransformer", "PatchTST", "DLinear",
            "ConvTimeNet", "PatchMLP"]
DISPLAY = {"CNN": "1D-CNN"}
PROTO_ORDER = ["cell", "temp", "chem", "chemtemp", "chemcv"]
NON_EXPERT = ("base_dQ+T", "FUS_pondérée", "tout+T")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="one gated_wrapper output directory (results.csv) per seed")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gate", default="GATEcell")
    args = ap.parse_args()

    frames = []
    for i, d in enumerate(args.dirs):
        R = pd.read_csv(os.path.join(d, "results.csv"))
        R = R.groupby(["protocol", "split", "method"]).MAE_cell.mean().groupby(["protocol", "method"]).mean()
        R = R.reset_index(); R["seed"] = i
        frames.append(R)
    A = pd.concat(frames)
    S = A.groupby(["method", "protocol"]).MAE_cell.agg(mean="mean", sd=lambda v: v.std(ddof=1), n="count")
    S = S.reset_index()
    S.to_csv(f"S5_{args.name}.csv", index=False)
    get = S.set_index(["method", "protocol"])
    protos = [p for p in PROTO_ORDER if p in set(S.protocol)] + sorted(set(S.protocol) - set(PROTO_ORDER))

    def fmt(m, p):
        if (m, p) not in get.index:
            return None
        r = get.loc[(m, p)]
        return f"{r['mean']:.2f}" if r["n"] < 2 or r["sd"] < 0.005 else f"{r['mean']:.2f} ± {r['sd']:.2f}"

    experts = sorted({m for m in S.method if not m.startswith(("GATE", "SWITCH")) and m not in NON_EXPERT})

    def key(m):
        fam = re.match(r"[A-Za-z]+", m).group(0)
        return (FAMILIES.index(fam) if fam in FAMILIES else 99, m)

    head = {"cell": "In-domain", "temp": "Unseen temperature", "chem": "Unseen chemistry / formation / ageing",
            "chemtemp": "Double shift", "chemcv": "Unseen current"}
    lines = [f"| {args.name}, model | " + " | ".join(head.get(p, p) for p in protos) + " |",
             "|---" * (len(protos) + 1) + "|"]
    for m in sorted(experts, key=key):
        cells = []
        for p in protos:
            a, b = fmt(m, p), fmt(f"{args.gate}[{m}]", p)
            cells.append("–" if a is None else f"{a} → {b}")
        fam = re.match(r"[A-Za-z]+", m).group(0)
        lines.append(f"| {DISPLAY.get(fam, fam)}{m[len(fam):]} | " + " | ".join(cells) + " |")
    if any(("base_dQ+T", p) in get.index for p in protos):
        lines.append("| Fallback (ridge on ΔQ/Q_nom + T) | " + " | ".join(fmt("base_dQ+T", p) or "–" for p in protos) + " |")
    out = "\n".join(lines) + "\n"
    open(f"S5_{args.name}.md", "w").write(out)
    print(out)
    print(f"seeds: {A.seed.nunique()} | variants: {len(experts)} | protocols: {protos}")


if __name__ == "__main__":
    main()
