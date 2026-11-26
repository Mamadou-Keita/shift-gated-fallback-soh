#!/usr/bin/env python3
"""
Mean ± standard deviation over seeds for the summary tables of Sections 6.5, 6.6, 6.8 and 6.9.

For each seed: MAE per method = mean per cell within each split, then mean over splits (as in the paper).
Families: expert = mean over expert variants of X; envelope = mean of GATEcell[X]; switch = mean of
SWITCHcell[X]; fallback = base_dQ+T (deterministic ridge: its seed SD should be ~0). Each variant is
averaged over the splits available to it (the +T variants of the blended cells exist in one split only).
Then mean ± SD over seeds of these per-seed values; per-expert ranges use the seed-mean of each variant,
and "worst SD" is the largest seed SD of a single variant.

  python seed_sd.py Tongji=$W/Tongji_s0/cells.csv,$W/Tongji_s1/cells.csv,$W/Tongji_s2/cells.csv \
      Blend=... MICH=... ISU=... ISUrate=... --out $R/review6/seed_sd.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = "base_dQ+T"
NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./formation/ageing",
         "chemtemp": "double shift", "chemcv": "unseen current"}


def per_variant(C):
    """MAE per (protocol, method), split mean then mean over splits."""
    S = C.groupby(["protocol", "split", "method"]).mae.mean()
    return S.groupby(level=["protocol", "method"]).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--out", default="seed_sd.csv")
    args = ap.parse_args()
    rows, var = [], []
    for g in args.groups:
        name, files = g.split("=", 1)
        for s, f in enumerate(files.split(",")):
            if not os.path.exists(f):
                print(f"  missing: {f}")
                continue
            M = per_variant(pd.read_csv(f))
            for proto in M.index.get_level_values(0).unique():
                m = M.loc[proto]
                experts = sorted(k[9:-1] for k in m.index if k.startswith("GATEcell["))
                experts = [e for e in experts if e in m.index]
                rows.append(dict(dataset=name, protocol=proto, seed=s, experts=len(experts),
                                 expert=m[experts].mean(), fallback=m.get(BASE, np.nan),
                                 envelope=m[[f"GATEcell[{e}]" for e in experts]].mean(),
                                 switch=m[[f"SWITCHcell[{e}]" for e in experts if f"SWITCHcell[{e}]" in m.index]].mean()))
                for e in experts:
                    var.append(dict(dataset=name, protocol=proto, seed=s, variant=e, expert=m[e],
                                    envelope=m[f"GATEcell[{e}]"]))
    R, V = pd.DataFrame(rows), pd.DataFrame(var)
    cols = ["expert", "fallback", "envelope", "switch"]
    A = R.groupby(["dataset", "protocol"])[cols].agg(["mean", "std"])
    VM = V.groupby(["dataset", "protocol", "variant"])[["expert", "envelope"]].agg(["mean", "std"])
    rng = VM.groupby(level=["dataset", "protocol"]).agg(
        exp_min=(("expert", "mean"), "min"), exp_max=(("expert", "mean"), "max"),
        env_min=(("envelope", "mean"), "min"), env_max=(("envelope", "mean"), "max"),
        exp_worst_sd=(("expert", "std"), "max"), env_worst_sd=(("envelope", "std"), "max"))
    out = []
    for (d, p), r in A.iterrows():
        q = rng.loc[(d, p)]
        out.append({"dataset": d, "protocol": NAMES.get(p, p),
                    **{c: f"{r[(c, 'mean')]:.2f} ± {r[(c, 'std')]:.2f}" for c in cols},
                    "expert range": f"{q.exp_min:.2f}–{q.exp_max:.2f}",
                    "envelope range": f"{q.env_min:.2f}–{q.env_max:.2f}",
                    "worst SD expert/env": f"{q.exp_worst_sd:.2f} / {q.env_worst_sd:.2f}"})
    O = pd.DataFrame(out)
    # paired over seeds: envelope − switch and envelope − expert, computed seed by seed
    R["env_minus_switch"] = R.envelope - R.switch
    R["env_minus_expert"] = R.envelope - R.expert
    P = R.groupby(["dataset", "protocol"])[["env_minus_switch", "env_minus_expert"]].agg(["mean", "std"])
    O["envelope − switch (paired)"] = [f"{P.loc[(d, p)][('env_minus_switch', 'mean')]:+.2f} ± {P.loc[(d, p)][('env_minus_switch', 'std')]:.2f}"
                                       for d, p in A.index]
    O["envelope − expert (paired)"] = [f"{P.loc[(d, p)][('env_minus_expert', 'mean')]:+.2f} ± {P.loc[(d, p)][('env_minus_expert', 'std')]:.2f}"
                                       for d, p in A.index]
    O.to_csv(args.out, index=False)
    VM.round(3).to_csv(args.out.replace(".csv", "_per_variant.csv"))
    pd.set_option("display.width", 250)
    print("MAE (SOH points). Family columns: mean ± SD over seeds of the mean over expert variants. Ranges: over "
          "variants of their seed-mean. Worst SD: largest seed SD of a single variant.")
    print(O.to_string(index=False))
    print(f"\n{args.out}, {args.out.replace('.csv', '_per_variant.csv')}")


if __name__ == "__main__":
    main()
