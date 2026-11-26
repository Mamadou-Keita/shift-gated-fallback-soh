#!/usr/bin/env python3
"""
Ablation of the shift score: which components detect which shift, and what it does to the envelope.

Each variant directory holds one gated_wrapper.py output per dataset (results.csv + cycles.csv, run with
--score <variant> --dump-cycles), in sub-folders Tongji/, MICH/, ISU_c5/, ISU_rate/.

Detection: for each test cell, the median of s/q over its windows (s = score, q = calibrated threshold,
so 1 = threshold). AUROC separates in-domain test cells (protocol "cell" of the same dataset; for
ISU_rate, the in-domain cells of ISU_c5) from the cells of each shift protocol. "Alarm" = share of
cells with median s/q > 1, i.e. a weight below 1.
Effect: mean MAE of GATEcell[X] over the expert variants, per dataset and protocol.

  python score_ablation.py --variant full $R/ablation/full --variant maha $R/ablation/maha \
      --variant recon $R/ablation/recon --variant maha_shape $R/ablation/maha_shape
"""
import argparse
import os

import numpy as np
import pandas as pd

DATASETS = ["Tongji", "MICH", "ISU_c5", "ISU_rate"]
REF = {"ISU_rate": ("ISU_c5", "cell")}
NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./formation/ageing", "chemtemp": "double",
         "chemcv": "unseen current"}


def auroc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    allv = np.concatenate([neg, pos])
    ranks = pd.Series(allv).rank().values
    return (ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def cell_scores(d):
    f = os.path.join(d, "cycles.csv")
    if not os.path.exists(f):
        return None
    C = pd.read_csv(f, usecols=["protocol", "split", "cell", "sq"])
    return C.groupby(["protocol", "split", "cell"]).sq.median().reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
    ap.add_argument("--out", default="score_ablation.csv")
    args = ap.parse_args()
    det, eff = [], []
    for name, root in args.variant:
        S = {ds: cell_scores(os.path.join(root, ds)) for ds in DATASETS}
        for ds in DATASETS:
            if S[ds] is None:
                continue
            ref_ds, ref_p = REF.get(ds, (ds, "cell"))
            if S.get(ref_ds) is None:
                continue
            neg = S[ref_ds].query("protocol == @ref_p").sq.values
            for proto, sub in S[ds].groupby("protocol"):
                row = dict(variant=name, dataset=ds, protocol=proto, alarm=float((sub.sq > 1).mean()),
                           median_sq=float(sub.sq.median()))
                row["auroc"] = np.nan if (ds, proto) == (ref_ds, ref_p) else auroc(neg, sub.sq.values)
                det.append(row)
            R = pd.read_csv(os.path.join(root, ds, "results.csv"))
            g = R[R.method.str.startswith("GATEcell[")].groupby("protocol").MAE_cell.mean()
            for proto, v in g.items():
                eff.append(dict(variant=name, dataset=ds, protocol=proto, mae_gatecell=v))
    Dt, Ef = pd.DataFrame(det), pd.DataFrame(eff)
    M = Dt.merge(Ef, on=["variant", "dataset", "protocol"], how="outer")
    M.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    M["case"] = M.dataset + " · " + M.protocol.map(NAMES).fillna(M.protocol)
    order = [v for v, _ in args.variant]
    for col, title, fmt in (("auroc", "AUROC: in-domain cells vs shifted cells (1 = perfect, 0.5 = chance)", "{:.2f}"),
                            ("alarm", "Share of test cells whose gate closes at least partly (median s/q > 1)", "{:.0%}"),
                            ("mae_gatecell", "MAE of the cell-level envelope, mean over experts (SOH points)", "{:.2f}")):
        P = M.pivot_table(index="case", columns="variant", values=col)[order]
        print(f"\n=== {title} ===")
        print(P.map(lambda v: "" if pd.isna(v) else fmt.format(v)).to_string())
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
