#!/usr/bin/env python3
"""
Comparaison appariée par cellule entre méthodes (même cellules de test, mêmes cycles).

Pour chaque paire (méthode A, méthode B) et chaque métrique calculée par cellule
(MAE, couverture 90 %, largeur 90 %, CRPS), donne la différence moyenne A − B,
un intervalle de confiance bootstrap à 95 % (rééchantillonnage des cellules)
et le test de Wilcoxon apparié.

  python paired_test.py --pred prob_hust_filter3/predictions.csv.gz --ref gen_filter \
      --others hgb_filter hgb_cqr_ema gen_Tcal --out paired_hust
  # restreint à une tranche de SOH :
  python paired_test.py --pred ... --soh-max 80 --out paired_hust_eol
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def per_cell(P):
    P = P.assign(err=(P.pred - P.soh).abs(),
                 cov=((P.soh >= P.lo90) & (P.soh <= P.hi90)).astype(float),
                 width=P.hi90 - P.lo90)
    agg = dict(MAE=("err", "mean"), cov90=("cov", "mean"), width90=("width", "mean"))
    if P.crps.notna().any():
        agg["CRPS"] = ("crps", "mean")
    return P.groupby("cell").agg(**agg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", default="gen_filter")
    ap.add_argument("--others", nargs="+", default=["hgb_filter", "hgb_cqr_ema", "hgb_cqr", "gen_Tcal", "ens_hgb"])
    ap.add_argument("--soh-min", type=float, default=-np.inf)
    ap.add_argument("--soh-max", type=float, default=np.inf)
    ap.add_argument("--domain", default=None, help="ID ou OOD (runs de transfert)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="paired_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    P = pd.read_csv(args.pred)
    P = P[(P.soh >= args.soh_min) & (P.soh <= args.soh_max)]
    if args.domain and "domain" in P:
        P = P[P.domain == args.domain]
    ref = per_cell(P[P.method == args.ref])
    rng = np.random.default_rng(0)
    rows = []
    for o in args.others:
        if o not in P.method.unique():
            continue
        oth = per_cell(P[P.method == o])
        cells = ref.index.intersection(oth.index)
        for m in ref.columns:
            if m not in oth:
                continue
            d = (ref.loc[cells, m] - oth.loc[cells, m]).values
            boots = np.array([rng.choice(d, len(d)).mean() for _ in range(args.n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            try:
                pval = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pval = np.nan
            better = {"MAE": d < 0, "CRPS": d < 0, "width90": d < 0, "cov90": d > 0}[m].mean()
            rows.append(dict(ref=args.ref, other=o, metric=m, n_cells=len(cells),
                             ref_mean=ref.loc[cells, m].mean(), other_mean=oth.loc[cells, m].mean(),
                             diff_mean=d.mean(), ci95_lo=lo, ci95_hi=hi, wilcoxon_p=pval,
                             frac_cells_ref_better=better))
    R = pd.DataFrame(rows).round(4)
    R.to_csv(os.path.join(args.out, "paired_results.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"Référence : {args.ref} | SOH ∈ [{args.soh_min}, {args.soh_max}] | différence = réf − autre\n")
    print(R.drop(columns="ref").to_string(index=False))
    print("\nIC 95 % excluant 0 = écart significatif au niveau des cellules. "
          "Pour MAE/CRPS/largeur, négatif = référence meilleure ; pour la couverture, regarder l'écart à 0,90.")


if __name__ == "__main__":
    main()
