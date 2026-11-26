#!/usr/bin/env python3
"""
Envelope vs fixed blend with α chosen by inner validation on the training cells (alpha_cv.py), plus the
hindsight oracle α of fixed_alpha.py for reference. Same MAE (split mean, then seed mean) and regret
definitions as strategy_table.py.

  python alpha_cv_summary.py \
     Tongji=$A/Tongji_s0.csv:$W/Tongji_s0/cells.csv,$A/Tongji_s1.csv:$W/Tongji_s1/cells.csv,... \
     Blend=... MICH=... ISU=... ISUrate=... --oracle $R/review6/fixed_alpha/fixed_alpha_per_seed.csv
Each seed is given as <alpha_cv csv>:<gated_wrapper cells.csv> (the review6 wrap outputs).
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = "base_dQ+T"
NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./formation/ageing",
         "chemtemp": "double shift", "chemcv": "unseen current"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--out", default="alpha_cv_summary.csv")
    args = ap.parse_args()
    rows = []
    for grp in args.groups:
        name, specs = grp.split("=", 1)
        for s, spec in enumerate(specs.split(",")):
            fa, fc = spec.split(":", 1)
            if not (os.path.exists(fa) and os.path.exists(fc)):
                print(f"  missing: {spec}")
                continue
            A = pd.read_csv(fa)
            C = pd.read_csv(fc)
            for (proto, x), sub in A.groupby(["protocol", "expert"]):
                cc = C[(C.protocol == proto)]
                W = cc.pivot_table(index=["split", "cell"], columns="method", values="mae")
                M = W.groupby(level="split").mean()
                env = f"GATEcell[{x}]"
                if x not in M or env not in M:
                    continue
                splits_ = sub.split.values
                Mx = M.loc[M.index.intersection(splits_)]
                rows.append(dict(dataset=name, seed=s, protocol=proto, expert=x,
                                 alpha_cv=sub.alpha_cv.mean(), fixed_cv=sub.mae_fixcv.mean(),
                                 expert_mae=Mx[x].mean(), fallback_mae=Mx[BASE].mean(), envelope=Mx[env].mean(),
                                 n_splits=len(Mx)))
    O = pd.DataFrame(rows)
    O["best_of_two"] = np.minimum(O.expert_mae, O.fallback_mae)
    for c in ("fixed_cv", "envelope"):
        O[f"regret_{c}"] = O[c] - O.best_of_two
    if args.oracle and os.path.exists(args.oracle):
        Q = pd.read_csv(args.oracle)[["dataset", "seed", "protocol", "expert", "fixed_oracle", "fixed_in"]]
        O = O.merge(Q, on=["dataset", "seed", "protocol", "expert"], how="left")
        O["regret_fixed_oracle"] = O.fixed_oracle - O.best_of_two
    O.to_csv(args.out.replace(".csv", "_per_seed.csv"), index=False)
    P = O.groupby(["dataset", "protocol", "expert"]).mean(numeric_only=True).reset_index()
    P.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    cols = ["alpha_cv", "expert_mae", "fallback_mae", "fixed_cv", "envelope"] + (["fixed_oracle"] if "fixed_oracle" in P else [])
    T = P.groupby(["dataset", "protocol"])[cols].mean().reset_index()
    T["protocol"] = T.protocol.map(NAMES).fillna(T.protocol)
    print("MAE (SOH points), mean over experts and seeds. alpha_cv = mean of the α chosen by inner validation on "
          "the training cells of each split (grid 0, 0.25, 0.5, 0.75, 1).")
    print(T.round(2).to_string(index=False))
    rc = ["fixed_cv", "envelope"] + (["fixed_oracle"] if "regret_fixed_oracle" in P else [])
    for lab, sub in (("all cases", P), ("initial benchmark (without Blend)", P[P.dataset != "Blend"])):
        print(f"\nRegret over {lab} ({len(sub)} cases): " + "; ".join(
            f"{c} mean {sub[f'regret_{c}'].mean():.2f}, worst {sub[f'regret_{c}'].max():.2f}" for c in rc))
        mp = sub.groupby(["dataset", "protocol"])[[f"regret_{c}" for c in rc]].mean()
        print("  macro over dataset x protocol: " + ", ".join(f"{c} {mp[f'regret_{c}'].mean():.2f}" for c in rc))
        print(f"  envelope <= fixed_cv in {(sub.envelope <= sub.fixed_cv + 1e-9).mean():.0%} of cases")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
