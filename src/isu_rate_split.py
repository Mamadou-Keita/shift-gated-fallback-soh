#!/usr/bin/env python3
"""
ISU-ILCC : MAE par cellule selon le courant de TEST (C/5 ou C/2), courant de test jamais vu (rate)
contre les deux courants vus (ratemix), moyenne ± écart-type sur les graines.

Chaque fichier est un wrap/cells.csv ; le chemin dit l'expérience (« ratemix » ou « rate »).

  python isu_rate_split.py $R/ISU_ratemix/wrap/cells.csv $R/seeds/ISU_ratemix_s*/wrap/cells.csv \
                           $R/ISU_rate/wrap/cells.csv    $R/seeds/ISU_rate_s*/wrap/cells.csv
"""
import sys

import pandas as pd

KEEP = ["base_dQ+T", "HGB", "GPR", "MLP", "CNN", "LSTM", "CPMLP", "CPTransformer", "iTransformer",
        "PatchTST", "DLinear", "ConvTimeNet", "PatchMLP"]
rows = []
for f in sys.argv[1:]:
    C = pd.read_csv(f)
    C["test"] = C.cell.str.extract(r"^ISU_(c\d)_")[0]
    exp = "vu" if "ratemix" in f else "jamais_vu"
    m = C.groupby(["method", "test"]).mae.mean().reset_index()
    m["exp"], m["file"] = exp, f
    rows.append(m)
A = pd.concat(rows)
S = A.groupby(["method", "exp", "test"]).mae.agg(["mean", "std", "count"]).reset_index()
S["txt"] = S.apply(lambda r: f"{r['mean']:.2f} ± {0 if pd.isna(r['std']) else r['std']:.2f}", axis=1)
P = S.pivot_table(index="method", columns=["exp", "test"], values="txt", aggfunc="first")
order = [k for k in KEEP + [f"GATEcell[{x}]" for x in KEEP[1:]] if k in P.index]
cols = [c for c in [("jamais_vu", "c2"), ("vu", "c2"), ("jamais_vu", "c5"), ("vu", "c5")] if c in P.columns]
pd.set_option("display.width", 250)
n = S.groupby("exp")["count"].max().to_dict()
print(f"MAE moyenne par cellule (points de SOH), moyenne ± écart-type sur les graines {n}, selon le courant de test")
print(P.loc[order, cols].to_string())
S.to_csv("isu_rate_split.csv", index=False)
print("\nisu_rate_split.csv")
