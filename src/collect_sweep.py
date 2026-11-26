#!/usr/bin/env python3
"""Rassemble les results_summary.csv du balayage de fenêtres en un seul tableau.

python collect_sweep.py $WORK/Inversion/results
"""
import glob
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []
for d in sorted(glob.glob(os.path.join(root, "ident_*"))):
    f = os.path.join(d, "results_folds.csv")
    if not os.path.exists(f):
        print(f"(pas de résultats dans {d})")
        continue
    res = pd.read_csv(f)
    best = res.groupby("features")["MAE"].apply(
        lambda s: s.groupby(res.loc[s.index, "model"]).mean().min())
    best.name = os.path.basename(d).replace("ident_", "")
    rows.append(best)

if not rows:
    sys.exit("Aucun résultat trouvé.")
tab = pd.DataFrame(rows)
tab.index.name = "fenetre_V"
cols = [c for c in ["mean", "dQ", "shape", "ic", "dQ+shape", "dQ+ic"] if c in tab]
tab = tab[cols]
tab["gain_forme_vs_dQ_%"] = 100 * (1 - tab[["dQ+shape", "dQ+ic"]].min(axis=1) / tab["dQ"])
print("MAE SOH (points de %), meilleur modèle par jeu de features — validation par cellule\n")
print(tab.round(3).to_string())
tab.to_csv(os.path.join(root, "sweep_summary.csv"))
