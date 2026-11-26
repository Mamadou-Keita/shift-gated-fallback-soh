#!/usr/bin/env python3
"""
Is the NCA / NCM shape separation an artefact of different SOH ranges? Compare the two populations at
matched SOH.

For each training chemistry A (and its counterpart B) and each SOH band, the gate (same score, PCA blocks
and 0.99 threshold as in the paper) is fitted on A cells of 5 grouped folds and applied to the held-out A
cells and to all B cells, restricted to windows inside the band. Per cell, the median s/q over its windows
in the band. Reported per band: number of cells, median s/q of held-out A and of B, AUROC (B vs held-out A)
and the share of B cells with median s/q > 1. Optionally restricted to one temperature (--temp 25) so that
temperature does not contribute.

  python shape_matched_soh.py --npz $R/benchT_Tongji/samples.npz --include-regex '^Tongji[12]' \
      --temp-regex 'CY(\\d+)-' --chem-regex '^(Tongji\\d)' --temp 25
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cond_models import load
from gated_fusion import Gate, calibrate_gate, pca_blocks

BANDS = [(97, 101), (94, 97), (91, 94), (88, 91), (85, 88), (0, 85)]


def auroc(neg, pos):
    allv = np.concatenate([neg, pos])
    r = pd.Series(allv).rank().values
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", required=True)
    ap.add_argument("--temp", type=float, default=None, help="Keep only this temperature (e.g. 25)")
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--gate-q", type=float, default=0.99)
    ap.add_argument("--out", default="shape_matched_soh.csv")
    args = ap.parse_args()
    D = load(args)
    if args.temp is not None:
        keep = D["T"] == args.temp
        D = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in D.items()}
    y, g, chem = D["soh"], D["groups"], D["chem"]
    chems = sorted(set(chem))
    print(f"chemistries {chems}; cells {[len(set(g[chem == c])) for c in chems]}; "
          f"SOH range per chemistry: " + ", ".join(f"{c} {y[chem == c].min():.0f}–{y[chem == c].max():.0f}" for c in chems))
    rows = []
    for a in chems:
        b = [c for c in chems if c != a][0]
        A = np.flatnonzero(chem == a)
        Bi = np.flatnonzero(chem == b)
        sqA = np.full(len(y), np.nan)
        sqB = []
        for k, (i, j) in enumerate(GroupKFold(n_splits=5).split(A, A, g[A])):
            tr, ho = A[i], A[j]
            q = calibrate_gate(D, tr, args.n_pca, args.gate_q)
            te = np.concatenate([ho, Bi])
            Ztr, Zte, Rtr, Rte = pca_blocks(D, tr, te, args.n_pca)
            s = Gate().fit(Ztr, Rtr).score(Zte, Rte) / q
            sqA[ho] = s[: len(ho)]
            sqB.append(s[len(ho):])
        sqB = np.mean(sqB, axis=0)  # B scored by each fold's gate: average over folds
        for lo, hi in BANDS:
            inA = A[(y[A] >= lo) & (y[A] < hi)]
            inB = Bi[(y[Bi] >= lo) & (y[Bi] < hi)]
            if len(inA) == 0 or len(inB) == 0:
                continue
            cA = pd.Series(sqA[inA]).groupby(g[inA]).median()
            cB = pd.Series(sqB[np.searchsorted(Bi, inB)]).groupby(g[inB]).median()
            if len(cA) < 2 or len(cB) < 2:
                continue
            rows.append(dict(train=a, test=b, band=f"{lo}–{hi if hi < 101 else 100}", cells_A=len(cA),
                             cells_B=len(cB), median_sq_A=cA.median(), median_sq_B=cB.median(),
                             auroc=auroc(cA.values, cB.values), B_above_threshold=(cB > 1).mean(),
                             A_above_threshold=(cA > 1).mean()))
    R = pd.DataFrame(rows)
    R.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print("\nPer SOH band: held-out training-chemistry cells (A) vs other chemistry (B), cell-level median s/q.")
    print(R.round(2).to_string(index=False))
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
