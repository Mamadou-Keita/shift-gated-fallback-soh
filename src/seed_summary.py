#!/usr/bin/env python3
"""
Robustesse aux initialisations : agrège les résultats de l'enveloppe (gated_wrapper.py, fichier
wrap/results.csv) de plusieurs runs indépendants (graines 0, 1, 2) d'un même jeu de données.

Pour chaque méthode et chaque protocole : MAE moyenne sur les plis, puis moyenne ± écart-type
sur les graines. Effet de l'enveloppe MAE(GATE[X]) − MAE(X) : moyenne ± écart-type sur les
graines et nombre de graines où l'enveloppe améliore (signe constant = résultat robuste).

  python seed_summary.py --name Tongji $R/wrap $R/seeds/Tongji_s1/wrap $R/seeds/Tongji_s2/wrap
"""
import argparse
import os

import pandas as pd

PROTOS = ["cell", "temp", "chem", "chemtemp", "chemcv"]
NAMES = {"cell": "Domaine", "temp": "Température", "chem": "Chimie/formation", "chemtemp": "Double", "chemcv": "Condition (cellules disjointes)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wrap_dirs", nargs="+", help="Dossiers wrap/ (un par graine)")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate", default="GATEcell", choices=["GATEcell", "GATE"],
                    help="variante de l'enveloppe dont on compte l'effet (GATEcell = variante principale)")
    args = ap.parse_args()

    frames = []
    for i, d in enumerate(args.wrap_dirs):
        f = os.path.join(d, "results.csv")
        if not os.path.exists(f):
            print(f"  absent : {f}"); continue
        R = pd.read_csv(f)
        R = R.groupby(["protocol", "method"]).MAE_cell.mean().reset_index()  # moyenne sur les plis
        R["seed"] = i
        frames.append(R)
    if len(frames) < 2:
        raise SystemExit("Il faut au moins 2 graines.")
    A = pd.concat(frames)
    n_seeds = A.seed.nunique()
    S = A.groupby(["method", "protocol"]).MAE_cell.agg(["mean", "std", "count"]).reset_index()
    out = args.out or f"seed_summary_{args.name or 'run'}.csv"
    S.to_csv(out, index=False)

    S["txt"] = S.apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)
    P = S.pivot(index="method", columns="protocol", values="txt")
    cols = [p for p in PROTOS if p in P.columns]
    order = S.pivot(index="method", columns="protocol", values="mean")[cols].mean(axis=1).sort_values().index
    pd.set_option("display.width", 250)
    print(f"=== {args.name} : MAE (moyenne des plis), moyenne ± écart-type sur {n_seeds} graines ===")
    print(P.loc[order, cols].rename(columns=NAMES).to_string())

    # effet de l'enveloppe, graine par graine
    W = A.pivot_table(index=["seed", "protocol"], columns="method", values="MAE_cell")
    rows = []
    pre = args.gate + "["
    for m in W.columns:
        if not m.startswith(pre) or m[len(pre):-1] not in W.columns:
            continue
        x = m[len(pre):-1]
        d = (W[m] - W[x]).dropna()
        for proto, g in d.groupby(level="protocol"):
            rows.append(dict(model=x, protocol=proto, diff=g.mean(), sd=g.std(), n=len(g),
                             n_better=int((g < 0).sum())))
    E = pd.DataFrame(rows)
    E.to_csv(out.replace(".csv", f"_envelope_{args.gate}.csv"), index=False)
    E["txt"] = E.apply(lambda x: f"{x['diff']:+.2f} ± {x.sd:.2f} ({x.n_better}/{x.n})", axis=1)
    print(f"\n=== Effet de l'enveloppe MAE({args.gate}[X]) − MAE(X) : moyenne ± écart-type sur les graines "
          f"(graines où l'enveloppe améliore / total) ===")
    print(E.pivot(index="model", columns="protocol", values="txt")[cols].rename(columns=NAMES).to_string())
    print(f"\n{out}")


if __name__ == "__main__":
    main()
