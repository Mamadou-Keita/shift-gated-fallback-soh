#!/usr/bin/env python3
"""
Start-up of the cell-level gate: how many windows of a new cell are needed before the envelope
protects it? Reads cycles.csv written by gated_wrapper.py --dump-cycles (one file per seed).

For the r-th observed window of each test cell (r = 1, 2, 3, 5, 10, 20): MAE of the expert alone,
of the cell-level envelope, of the fallback, and the median cell-level weight. Then the variant
"fallback only for the first K windows, envelope afterwards", over all windows, for K = 0, 1, 3, 5.

  python startup_analysis.py $R/wrap_dump/cycles.csv $R/seeds/Tongji_s1/wrap_dump/cycles.csv ... \
      --protocols cell chem chemtemp
"""
import argparse

import numpy as np
import pandas as pd

RANKS = [1, 2, 3, 5, 10, 20]
NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chemistry", "chemtemp": "double shift",
         "chemcv": "unseen current"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--protocols", nargs="+", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--out", default="startup.csv")
    args = ap.parse_args()
    frames = []
    for s, f in enumerate(args.files):
        C = pd.read_csv(f)
        C["seed"] = s
        frames.append(C)
    C = pd.concat(frames)
    if args.protocols:
        C = C[C.protocol.isin(args.protocols)]
    experts = [c[5:-1] for c in C.columns if c.startswith("pred[")]
    rows, rows_k = [], []
    for e in experts:
        p = C[f"pred[{e}]"]
        C["err_x"] = (p - C.y).abs()
        C["err_b"] = (C.base - C.y).abs()
        C["err_env"] = (C.base + C.wc * (p - C.base) - C.y).abs()
        for proto, sub in C.groupby("protocol"):
            for r in RANKS:
                at = sub[sub["rank"] == r]
                if len(at) == 0:
                    continue
                rows.append(dict(expert=e, protocol=proto, rank=r, n=len(at), mae_expert=at.err_x.mean(),
                                 mae_envelope=at.err_env.mean(), mae_fallback=at.err_b.mean(),
                                 median_wc=at.wc.median()))
            for k in args.ks:
                wk = np.where(sub["rank"] <= k, 0.0, sub.wc)
                err = (sub.base + wk * (sub[f"pred[{e}]"] - sub.base) - sub.y).abs()
                # per-cell MAE, then mean over cells (as in the paper tables)
                m = err.groupby([sub.seed, sub.split, sub.cell]).mean().groupby(level=[0, 1]).mean().mean()
                rows_k.append(dict(expert=e, protocol=proto, K=k, mae=m))
    R = pd.DataFrame(rows)
    R.to_csv(args.out, index=False)
    K = pd.DataFrame(rows_k)
    K.to_csv(args.out.replace(".csv", "_fallback_first_k.csv"), index=False)
    pd.set_option("display.width", 220)
    for proto, sub in R.groupby("protocol"):
        print(f"\n=== {NAMES.get(proto, proto)} : error at the r-th window of each test cell "
              f"(mean over experts {', '.join(experts)} and seeds) ===")
        T = sub.groupby("rank")[["mae_expert", "mae_envelope", "mae_fallback", "median_wc"]].mean().round(2)
        print(T.T.to_string())
    print("\n=== Fallback only for the first K windows, envelope afterwards: MAE over all windows ===")
    print(K.groupby(["protocol", "K"]).mae.mean().unstack().round(3).rename(index=NAMES).to_string())
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
