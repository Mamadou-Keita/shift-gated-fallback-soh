#!/usr/bin/env python3
"""
Late initialisation of the self-referenced fallback: what if a test cell is first observed already aged?

Training cells are observed from new (historical data). For each test cell and each start level S
(e.g. 95, 90, 85 % SOH), the cell is "first observed" at its first window with SOH <= S; earlier windows
are discarded for every method (the SOH label is used ONLY to choose this start point). The self-referenced
amplitude of a test cell is then referenced to its first observed windows (causal: at the r-th observed
window, median of the first min(r, k) observed windows).

Methods, evaluated on the same observed windows:
  expert X; fallback nominal (ΔQ/Q_nom); fallback self-ref "from new" (reference from the cell's true first
  windows, i.e. the assumption satisfied, an upper bound); fallback self-ref "late" (the realistic case);
  and the cell-level envelope with each fallback (running median of w restarted at the first observed window).

  python late_init.py --npz ... --include-regex ... --temp-regex ... --chem-regex ... --lit-root ... \
      --levels 95 90 85 --out $R/review6/late/Tongji_s0
Writes cells.csv (level, protocol, split, cell, method, mae, n_windows). Summarise with late_summary.py.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from cond_models import design, load, splits
from gated_fusion import Gate, calibrate_gate, causal_cell_median, pca_blocks, ridge_T, weights
from gated_wrapper import label_of


def causal_ref(dqn, cells, cyc, mask, k):
    """Per window in mask: median of the first min(r, k) masked windows of its cell (chronological)."""
    ref = np.full(len(dqn), np.nan)
    for c in np.unique(cells[mask]):
        idx = np.flatnonzero((cells == c) & mask)
        idx = idx[np.argsort(cyc[idx], kind="stable")]
        for j, i in enumerate(idx):
            ref[i] = np.median(dqn[idx[: min(j + 1, k)]])
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--lit-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", default=None)
    ap.add_argument("--protocols", nargs="+", default=["cell", "temp", "chem", "chemtemp"])
    ap.add_argument("--levels", type=float, nargs="+", default=[95, 90, 85])
    ap.add_argument("--ref-k", type=int, default=3)
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--gate-q", type=float, default=0.99)
    ap.add_argument("--gate-k", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = load(args)
    y, g, T = D["soh"], D["groups"], D["T"]
    cyc = D["cycle"] if "cycle" in D else np.arange(len(y))
    dqn0 = D["dqn"].copy()
    allmask = np.ones(len(y), bool)
    ref_new = causal_ref(dqn0, g, cyc, allmask, args.ref_k)

    # start mask per level: windows from the cell's first window with SOH <= S onwards
    starts = {}
    for S in [None] + list(args.levels):
        m = np.zeros(len(y), bool)
        for c in np.unique(g):
            idx = np.flatnonzero(g == c)
            idx = idx[np.argsort(cyc[idx], kind="stable")]
            if S is None:
                m[idx] = True
                continue
            hit = np.flatnonzero(y[idx] <= S)
            if len(hit):
                m[idx[hit[0]:]] = True
        starts["new" if S is None else f"{S:g}"] = m

    rows = []
    for proto in args.protocols:
        for i, (name, tr, te) in enumerate(splits(D, proto)):
            P = {}
            for d in glob.glob(os.path.join(args.lit_root, f"lit_{proto}_*")):
                if not re.fullmatch(rf"lit_{proto}_[A-Za-z]+", os.path.basename(d)):
                    continue
                for f in glob.glob(os.path.join(d, "preds", f"s{i:02d}__*.npy")):
                    P[label_of(os.path.basename(f)[5:-4])] = np.load(f)
            if "base_dQ+T" not in P:
                print(f"  {proto} / {name}: no cached predictions, skipped")
                continue
            experts = [k for k in P if k not in ("base_dQ+T", "tout+T", "FUS_pondérée")]
            tmu, tsd = (T[tr].mean(), 1.0) if len(set(T[tr])) < 2 else (T[tr].mean(), T[tr].std() + 1e-9)
            q = calibrate_gate(D, tr, args.n_pca, args.gate_q)
            Ztr, Zte, Rtr, Rte = pca_blocks(D, tr, te, args.n_pca)
            w = weights(Gate().fit(Ztr, Rtr).score(Zte, Rte), q, args.gate_k)

            def selfref_pred(ref_te):
                D2 = dict(D)
                dq = dqn0 / ref_new
                dq[te] = dqn0[te] / ref_te
                D2["dqn"] = dq
                m, _, _, Xte = ridge_T("dQ", D2, tr, te, args, np.logspace(-3, 4, 15), tmu, tsd)
                return m.predict(design(Xte, T[te], "M1", tmu, tsd))

            b_new = selfref_pred(ref_new[te])
            for lev, mask in starts.items():
                keep = mask[te]
                if not keep.any():
                    continue
                ref_late = causal_ref(dqn0, g, cyc, mask, args.ref_k)[te]
                ref_late[~keep] = 1.0  # unused
                b_late = selfref_pred(ref_late)
                gk, ck, yk = g[te][keep], cyc[te][keep], y[te][keep]
                wc = causal_cell_median(w[keep], gk, ck)
                bases = {"nominal": P["base_dQ+T"][keep], "selfref_new": b_new[keep], "selfref_late": b_late[keep]}
                preds = {f"fallback_{k}": v for k, v in bases.items()}
                for e in experts:
                    preds[e] = P[e][keep]
                    for k, b in bases.items():
                        preds[f"ENV_{k}[{e}]"] = b + wc * (P[e][keep] - b)
                for mth, p in preds.items():
                    err = pd.Series(np.abs(p - yk)).groupby(gk)
                    for c, v in err.mean().items():
                        rows.append(dict(level=lev, protocol=proto, split=name, method=mth, cell=c, mae=v,
                                         n_windows=int(err.size()[c])))
            print(f"  {proto} / {name}: done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "cells.csv"), index=False)
    print(f"{args.out}/cells.csv")


if __name__ == "__main__":
    main()
