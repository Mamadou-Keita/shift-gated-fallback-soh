#!/usr/bin/env python3
"""
Tableau de synthèse pour l'article : rassemble les results.csv de plusieurs runs de
lit_baselines.py (un dossier par protocole × groupe de modèles) et affiche, pour chaque méthode
et chaque protocole, la MAE moyenne par cellule sur les plis et le pire pli.

Les dossiers sont nommés <préfixe>_<protocole>_<n'importe quoi> (ex. lit_chem_GPR) ; le protocole
est lu dans le nom (cell, temp, chem, chemtemp). Les lignes de référence (FUS, base, tout+T),
recalculées à chaque run, sont dédupliquées.

  python lit_summary.py $R/lit_*  --out $R/lit_summary.csv
"""
import argparse
import os
import re

import pandas as pd

PROTOS = ["cell", "temp", "chem", "chemtemp", "chemcv"]
NAMES = {"cell": "Dans le domaine", "temp": "Température jamais vue", "chem": "Chimie jamais vue",
         "chemtemp": "Chimie + température", "chemcv": "Condition jamais vue (cellules disjointes)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="lit_summary.csv")
    args = ap.parse_args()
    frames = []
    for d in args.dirs:
        f = os.path.join(d, "results.csv")
        if not os.path.exists(f):
            continue
        m = re.search(r"_(cell|temp|chemtemp|chemcv|chem)(_|$)", os.path.basename(d.rstrip("/")))
        if not m:
            print(f"  protocole introuvable dans le nom : {d}"); continue
        if os.path.getsize(f) < 5:
            print(f"  vide (aucun pli pour ce protocole) : {d}"); continue
        R = pd.read_csv(f); R["protocol"] = m.group(1)
        frames.append(R)
    if not frames:
        raise SystemExit("Aucun results.csv exploitable.")
    R = pd.concat(frames).drop_duplicates(["protocol", "split", "method"])
    S = R.groupby(["method", "protocol"]).MAE_cell.agg(["mean", "max", "count"]).reset_index()
    S.to_csv(args.out, index=False)
    S["txt"] = S.apply(lambda x: f"{x['mean']:.2f} [{x['max']:.2f}]", axis=1)
    P = S.pivot(index="method", columns="protocol", values="txt")
    P = P[[p for p in PROTOS if p in P.columns]]
    order = S.pivot(index="method", columns="protocol", values="mean")[P.columns].mean(axis=1).sort_values().index
    P = P.loc[order].rename(columns=NAMES)
    pd.set_option("display.width", 250)
    print("MAE moyenne par cellule sur les plis [pire pli], points de SOH ; tri par moyenne des protocoles")
    print(P.to_string())
    n = S.pivot(index="method", columns="protocol", values="count")
    if n.nunique().max() > 1:
        print("\nAttention : nombre de plis différent selon la méthode (run incomplet ?)")
        print(n.to_string())
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
