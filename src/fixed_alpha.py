#!/usr/bin/env python3
"""
Does the shift detection itself matter? Envelope vs a fixed blend without detection.

Fixed blend: ŷ = α·ŷ_X + (1 − α)·ŷ_b with the same α for every cell, α ∈ {0, 0.25, 0.5, 0.75, 1}
(α = 0: fallback alone, α = 1: expert alone). Two ways to choose α, per expert and seed:
  α_in     : best α on in-domain test cells of the same dataset (protocol "cell"), i.e. what a practitioner
             would tune without knowing the shift; applied unchanged to every protocol;
  α_oracle : one α per expert and dataset, the best on average over ALL its protocols, chosen with the
             test labels (hindsight; not deployable).
Envelope = GATEcell[X]. MAE = mean per cell within each split, then mean over splits; mean over seeds.
Regret = MAE minus that of the better of {X, fallback} (same definition as strategy_table.py).

Inputs: wrap/cells.csv of gated_wrapper.py (with the FIX0.xx[X] methods), one per seed, grouped by dataset:
  python fixed_alpha.py Tongji=a.csv,b.csv,c.csv MICH=... ISU=... ISUrate=... Blend=... --out $R/review6/fixed_alpha
In-domain reference for datasets without an in-domain protocol: ISUrate -> ISU, Blend -> Tongji.
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = "base_dQ+T"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
REF = {"ISUrate": "ISU", "Blend": "Tongji"}
NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./formation/ageing",
         "chemtemp": "double shift", "chemcv": "unseen current"}


def method(a, x):
    return BASE if a == 0 else x if a == 1 else f"FIX{a:.2f}[{x}]"


def per_seed(name, files):
    rows = []
    for s, f in enumerate(files):
        if not os.path.exists(f):
            print(f"  missing: {f}")
            continue
        C = pd.read_csv(f)
        experts = sorted({m[9:-1] for m in C.method.unique() if m.startswith("GATEcell[")})
        for proto, sub in C.groupby("protocol"):
            W = sub.pivot_table(index=["split", "cell"], columns="method", values="mae")
            M = W.groupby(level="split").mean().mean()  # split-mean MAE per method
            for x in experts:
                cols = [method(a, x) for a in ALPHAS] + [f"GATEcell[{x}]"]
                if any(c not in M for c in cols):
                    continue
                r = dict(dataset=name, seed=s, protocol=proto, expert=x, envelope=M[f"GATEcell[{x}]"])
                r.update({f"a{a:.2f}": M[method(a, x)] for a in ALPHAS})
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="NAME=cells1.csv,cells2.csv,...")
    ap.add_argument("--out", default="fixed_alpha")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A = pd.concat([per_seed(g.split("=", 1)[0], g.split("=", 1)[1].split(",")) for g in args.groups])
    acols = [f"a{a:.2f}" for a in ALPHAS]

    # α_in: best α on in-domain cells of the (reference) dataset, per expert and seed
    ind = A[A.protocol == "cell"].set_index(["dataset", "seed", "expert"])[acols]
    a_in = ind.idxmin(axis=1)
    # α_oracle: best α on average over all protocols of the dataset, per expert (and seed)
    a_or = A.groupby(["dataset", "seed", "expert"])[acols].mean().idxmin(axis=1)
    out = []
    for _, r in A.iterrows():
        key = (REF.get(r.dataset, r.dataset), r.seed, r.expert)
        ai = a_in.get(key, np.nan)
        ao = a_or.get((r.dataset, r.seed, r.expert), np.nan)
        best = min(r["a0.00"], r["a1.00"])
        out.append(dict(dataset=r.dataset, seed=r.seed, protocol=r.protocol, expert=r.expert,
                        alpha_in=float(ai[1:]) if isinstance(ai, str) else np.nan,
                        alpha_oracle=float(ao[1:]) if isinstance(ao, str) else np.nan,
                        expert_mae=r["a1.00"], fallback_mae=r["a0.00"], envelope=r.envelope,
                        fixed_in=r[ai] if isinstance(ai, str) else np.nan,
                        fixed_oracle=r[ao] if isinstance(ao, str) else np.nan,
                        best_of_two=best))
    O = pd.DataFrame(out)
    for c in ("envelope", "fixed_in", "fixed_oracle"):
        O[f"regret_{c}"] = O[c] - O.best_of_two
    O.to_csv(os.path.join(args.out, "fixed_alpha_per_seed.csv"), index=False)
    # mean over seeds -> one case per dataset x protocol x expert, as in strategy_table.py
    P = O.groupby(["dataset", "protocol", "expert"]).mean(numeric_only=True).reset_index()
    P.to_csv(os.path.join(args.out, "fixed_alpha_per_expert.csv"), index=False)

    pd.set_option("display.width", 250)
    T = P.groupby(["dataset", "protocol"]).agg(
        experts=("expert", "size"), alpha_in=("alpha_in", "mean"), alpha_oracle=("alpha_oracle", "mean"),
        expert=("expert_mae", "mean"), fallback=("fallback_mae", "mean"), fixed_in=("fixed_in", "mean"),
        fixed_oracle=("fixed_oracle", "mean"), envelope=("envelope", "mean")).reset_index()
    T["protocol"] = T.protocol.map(NAMES).fillna(T.protocol)
    print("MAE (SOH points), mean over experts and seeds. alpha = mean chosen α (1 = expert alone).")
    print("fixed_in: α tuned on in-domain cells, no detection; fixed_oracle: one α per expert chosen in hindsight "
          "over all protocols of the dataset.")
    print(T.round(2).to_string(index=False))
    for lab, sub in (("all cases", P), ("initial benchmark (without Blend)", P[P.dataset != "Blend"])):
        print(f"\nRegret over {lab} ({len(sub)} cases): " + "; ".join(
            f"{c} mean {sub[f'regret_{c}'].mean():.2f}, worst {sub[f'regret_{c}'].max():.2f}"
            for c in ("fixed_in", "fixed_oracle", "envelope")))
        print("  envelope <= fixed_in in "
              f"{(sub.envelope <= sub.fixed_in + 1e-9).mean():.0%} of cases, <= fixed_oracle in "
              f"{(sub.envelope <= sub.fixed_oracle + 1e-9).mean():.0%}")
    print(f"\n{args.out}/fixed_alpha_per_expert.csv")


if __name__ == "__main__":
    main()
