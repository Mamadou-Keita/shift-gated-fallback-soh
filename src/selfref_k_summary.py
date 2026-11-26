#!/usr/bin/env python3
"""
Sensitivity of the self-referenced fallback to the number of initial windows (ref_k) and to causality.

Reads strategy_per_expert.csv written by strategy_table.py for each configuration, and prints:
  1. fallback-alone MAE per dataset x protocol (rows) and configuration (columns);
  2. envelope MAE (mean over experts);
  3. regret of the envelope over all cases (mean and worst), the same cases for every configuration.

  python selfref_k_summary.py nominal=$R/selfref/strategy_nominal fixed3=$R/selfref/strategy_selfref \
      fixed1=$R/selfref_k/fixed1/strategy ...
"""
import argparse
import os

import pandas as pd

NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./formation/ageing",
         "chemtemp": "double shift", "chemcv": "unseen current"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="NAME=strategy_dir")
    ap.add_argument("--out", default="selfref_k_summary.csv")
    args = ap.parse_args()
    frames = []
    for c in args.configs:
        name, d = c.split("=", 1)
        f = os.path.join(d, "strategy_per_expert.csv")
        if not os.path.exists(f):
            print(f"  missing: {f}")
            continue
        P = pd.read_csv(f)
        P["config"] = name
        frames.append(P)
    A = pd.concat(frames)
    A.to_csv(args.out, index=False)
    order = [c.split("=", 1)[0] for c in args.configs if c.split("=", 1)[0] in set(A.config)]
    A["case"] = A.dataset + " · " + A.protocol.map(NAMES).fillna(A.protocol)
    pd.set_option("display.width", 250)
    for col, title in (("mae_fallback", "Fallback alone, MAE (SOH points, mean over seeds)"),
                       ("mae_envelope", "Envelope (GATEcell), MAE, mean over experts and seeds")):
        T = A.pivot_table(index="case", columns="config", values=col, aggfunc="mean")[order]
        print(f"\n=== {title} ===")
        print(T.round(2).to_string())
    print("\n=== Envelope regret over all cases (vs the better of expert and that configuration's own fallback) ===")
    R = A.groupby("config").agg(cases=("regret_envelope", "size"), mean=("regret_envelope", "mean"),
                                worst=("regret_envelope", "max"),
                                fallback_mean=("regret_fallback", "mean"))[["cases", "mean", "worst", "fallback_mean"]]
    print(R.loc[order].round(2).to_string())
    print("\n=== Same, initial benchmark only (without the blended cells) ===")
    B = A[A.dataset != "Blend"]
    R0 = B.groupby("config").agg(cases=("regret_envelope", "size"), mean=("regret_envelope", "mean"),
                                 worst=("regret_envelope", "max"), fallback_mean=("regret_fallback", "mean"))
    print(R0.loc[[o for o in order if o in R0.index]].round(2).to_string())
    w = B.sort_values("regret_envelope").groupby("config").tail(1)
    print("\nWorst case per configuration (initial benchmark):")
    print(w[["config", "dataset", "protocol", "expert", "regret_envelope"]].round(2).to_string(index=False))
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
