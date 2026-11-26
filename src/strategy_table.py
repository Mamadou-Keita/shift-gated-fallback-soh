#!/usr/bin/env python3
"""
Expert vs fallback vs shift-gated envelope, on every dataset and protocol.

Answers the question "why not always use the transferable fallback?". For each expert X and each
protocol, three strategies are compared on the same test cells:
  expert   = X alone
  fallback = base_dQ+T alone
  envelope = GATEcell[X]
MAE = mean per-cell error within each split, then mean over splits (as in the paper tables).
Regret of a strategy = its MAE minus the MAE of the better of {expert, fallback}, the choice being made once
per dataset x protocol x expert variant x seed (protocol-level oracle); a per-cell oracle is also reported
(0 = as good as an oracle that knew which of the two to use). Cell-level shares say on how many test
cells the envelope beats the fallback and the expert.

Inputs are wrap/cells.csv files (gated_wrapper.py), one per seed, grouped by dataset:

  python strategy_table.py \
     Tongji=$R/wrap/cells.csv,$R/seeds/Tongji_s1/wrap/cells.csv,$R/seeds/Tongji_s2/wrap/cells.csv \
     MICH=$R/MICH/wrap/cells.csv,$R/seeds/MICH_s1/wrap/cells.csv,$R/seeds/MICH_s2/wrap/cells.csv \
     ISU=$R/ISU_c5/wrap/cells.csv,$R/seeds/ISU_c5_s1/wrap/cells.csv,$R/seeds/ISU_c5_s2/wrap/cells.csv \
     ISUrate=$R/ISU_rate/wrap/cells.csv,$R/seeds/ISU_rate_s1/wrap/cells.csv,$R/seeds/ISU_rate_s2/wrap/cells.csv \
     --out $R/strategy

Outputs: strategy_per_expert.csv, strategy_summary.csv (one row per dataset x protocol), printed tables.
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = "base_dQ+T"
NAMES = {"cell": "in-domain", "temp": "unseen T", "chem": "unseen chem./formation/ageing path",
         "chemtemp": "double shift", "chemcv": "unseen current"}


def load(files):
    out = []
    for s, f in enumerate(files):
        if not os.path.exists(f):
            print(f"  missing: {f}")
            continue
        C = pd.read_csv(f)
        C["seed"] = s
        out.append(C)
    if not out:
        raise SystemExit("no cells.csv found")
    return pd.concat(out)


def analyse(name, C):
    rows = []
    experts = sorted({m[9:-1] for m in C.method.unique() if m.startswith("GATEcell[")})
    for (proto, seed), sub in C.groupby(["protocol", "seed"]):
        # one value per (split, cell): a cell is tested once per split
        W = sub.pivot_table(index=["split", "cell"], columns="method", values="mae")
        if BASE not in W:
            continue
        for x in experts:
            env = f"GATEcell[{x}]"
            if x not in W or env not in W:
                continue
            m = W[[x, BASE, env]].dropna()
            # MAE as in the paper tables: mean per cell within each split, then mean over splits
            mae = m.groupby(level="split").mean().mean()
            best = min(mae[x], mae[BASE])
            # per-cell oracle: the better of expert and fallback chosen separately for every test cell
            cell_best = np.minimum(m[x], m[BASE]).groupby(level="split").mean().mean()
            rows.append(dict(dataset=name, protocol=proto, seed=seed, expert=x,
                             regret_envelope_cell_oracle=mae[env] - cell_best,
                             mae_expert=mae[x], mae_fallback=mae[BASE], mae_envelope=mae[env],
                             regret_expert=mae[x] - best, regret_fallback=mae[BASE] - best,
                             regret_envelope=mae[env] - best,
                             cells_env_le_fallback=float((m[env] <= m[BASE]).mean()),
                             cells_env_le_expert=float((m[env] <= m[x]).mean()), n_cells=len(m)))
    R = pd.DataFrame(rows)
    # mean over seeds
    return R.groupby(["dataset", "protocol", "expert"]).mean(numeric_only=True).drop(columns="seed").reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="NAME=cells1.csv,cells2.csv,...")
    ap.add_argument("--out", default="strategy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per = pd.concat([analyse(g.split("=", 1)[0], load(g.split("=", 1)[1].split(","))) for g in args.groups])
    per.to_csv(os.path.join(args.out, "strategy_per_expert.csv"), index=False)

    agg = per.groupby(["dataset", "protocol"]).agg(
        experts=("expert", "size"),
        expert_min=("mae_expert", "min"), expert_max=("mae_expert", "max"),
        fallback=("mae_fallback", "mean"),
        envelope_min=("mae_envelope", "min"), envelope_max=("mae_envelope", "max"),
        regret_expert=("regret_expert", "mean"), regret_fallback=("regret_fallback", "mean"),
        regret_envelope=("regret_envelope", "mean"),
        regret_envelope_cell_oracle=("regret_envelope_cell_oracle", "mean"),
        env_beats_fallback=("mae_envelope", lambda s: np.nan),
        cells_env_le_fallback=("cells_env_le_fallback", "mean"),
        cells_env_le_expert=("cells_env_le_expert", "mean")).reset_index()
    nb = per.assign(b=per.mae_envelope <= per.mae_fallback).groupby(["dataset", "protocol"]).b.sum()
    agg["env_beats_fallback"] = [f"{int(nb.loc[(d, p)])}/{n}" for d, p, n in zip(agg.dataset, agg.protocol, agg.experts)]
    agg.to_csv(os.path.join(args.out, "strategy_summary.csv"), index=False)

    pd.set_option("display.width", 250)
    T = agg.copy()
    T["protocol"] = T.protocol.map(NAMES).fillna(T.protocol)
    T["Expert alone"] = T.apply(lambda r: f"{r.expert_min:.2f}-{r.expert_max:.2f}", axis=1)
    T["Fallback alone"] = T.fallback.map(lambda v: f"{v:.2f}")
    T["Envelope"] = T.apply(lambda r: f"{r.envelope_min:.2f}-{r.envelope_max:.2f}", axis=1)
    T["Regret E / F / Env"] = T.apply(lambda r: f"{r.regret_expert:.2f} / {r.regret_fallback:.2f} / {r.regret_envelope:.2f}", axis=1)
    T["Env<=F (models)"] = T.env_beats_fallback
    T["cells Env<=F / Env<=E"] = T.apply(lambda r: f"{r.cells_env_le_fallback:.0%} / {r.cells_env_le_expert:.0%}", axis=1)
    print("MAE per cell (SOH points), mean over seeds; range over experts. Regret = MAE - MAE of the better of "
          "{expert, fallback} for that expert and protocol (mean over experts).")
    print(T[["dataset", "protocol", "Expert alone", "Fallback alone", "Envelope", "Regret E / F / Env",
             "Env<=F (models)", "cells Env<=F / Env<=E"]].to_string(index=False))

    tot = per[["regret_expert", "regret_fallback", "regret_envelope"]]
    print("\nOver every dataset x protocol x expert "
          f"({len(per)} cases): mean regret expert {tot.regret_expert.mean():.2f}, fallback "
          f"{tot.regret_fallback.mean():.2f}, envelope {tot.regret_envelope.mean():.2f}; "
          f"worst case expert {tot.regret_expert.max():.2f}, fallback {tot.regret_fallback.max():.2f}, "
          f"envelope {tot.regret_envelope.max():.2f}")
    # macro averages: every dataset x protocol weighs the same, whatever its number of expert variants
    mp = per.groupby(["dataset", "protocol"])[["regret_expert", "regret_fallback", "regret_envelope"]].mean()
    md = mp.groupby(level="dataset").mean()
    print("Macro average over dataset x protocol (each weighs the same): expert "
          f"{mp.regret_expert.mean():.2f}, fallback {mp.regret_fallback.mean():.2f}, envelope {mp.regret_envelope.mean():.2f}")
    print("Per dataset (mean over its protocols):\n" + md.round(2).to_string())
    print("Macro average over datasets: " + ", ".join(f"{c.split('_')[1]} {md[c].mean():.2f}" for c in md.columns))
    print(f"Cases = dataset x protocol x expert variant, each averaged over seeds: "
          f"{per.groupby(['dataset', 'protocol']).size().to_dict()}")
    print(f"Against a per-cell oracle (better of expert and fallback chosen for each test cell), "
          f"the envelope's mean regret is {per.regret_envelope_cell_oracle.mean():.2f} "
          f"(worst {per.regret_envelope_cell_oracle.max():.2f}).")
    print(f"\n{args.out}/strategy_summary.csv, strategy_per_expert.csv")


if __name__ == "__main__":
    main()
