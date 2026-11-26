#!/usr/bin/env python3
"""
Évaluation de la détection hors distribution (OOD) et de la prédiction sélective,
à partir du predictions.csv.gz d'un run de transfert de prob_inversion.py
(lancé avec --test-npz et --id-holdout > 0 : domaines ID et OOD prédits par le même modèle).

Scores comparés (plus grand = plus suspect) :
  gen : -log_evidence        -log p(x) du modèle génératif (par cycle)
  gen : -log_evidence_cum    moyenne causale de -log p(x) sur les cycles passés de la cellule
  gen : pca_recon_err        résidu de reconstruction PCA (directions ignorées par le modèle)
  ens_hgb : width90          dispersion de l'ensemble (référence classique)
  hgb_hetero : width90       variance prédite par la régression hétéroscédastique

Sorties :
  ood_auroc.csv          AUROC ID vs OOD, par cycle et par cellule
  selective_risk.csv     MAE en fonction de la fraction de cycles conservés (score le plus bas d'abord)
  fig_ood_scores.png     distributions des scores ID / OOD
  fig_selective.png      courbes risque / rétention

  python ood_eval.py --pred transfer_hust_matr/predictions.csv.gz --out ood_hust_matr
"""
import argparse
import os

import numpy as np
import pandas as pd


def causal_mean(df, col):
    df = df.sort_values(["cell", "cycle"])
    return df.groupby("cell")[col].transform(lambda s: s.expanding().mean())


def auroc(y_ood, score):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(score)
    if len(np.unique(y_ood[ok])) < 2:
        return np.nan
    return roc_auc_score(y_ood[ok], score[ok])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="ood_out")
    ap.add_argument("--methods", nargs="+", default=["gen_filter", "gen_Tcal", "hgb_cqr", "hgb_filter", "ens_hgb"],
                    help="Méthodes dont on mesure la MAE après rejet")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    P = pd.read_csv(args.pred)
    if "domain" not in P or P.domain.nunique() < 2:
        raise SystemExit("Il faut un run de transfert avec --id-holdout > 0 (domaines ID et OOD).")
    P["width90"] = P.hi90 - P.lo90
    key = ["cell", "cycle"]

    # --- scores, un par (cellule, cycle)
    g = P[P.method == "gen_Tcal"][key + ["domain", "log_evidence", "pca_recon_err"]].copy()
    g["neg_log_evidence"] = -g.log_evidence
    g = g.sort_values(key)
    g["neg_log_evidence_cum"] = causal_mean(g, "neg_log_evidence")
    g["pca_recon_err_cum"] = causal_mean(g, "pca_recon_err")
    # test de typicalité (bilatéral) : une log-évidence anormalement HAUTE est aussi suspecte.
    # Référence = médiane / dispersion de -log p(x) sur les cellules ID (diagnostic : légère fuite,
    # à remplacer par les cellules de validation pour un résultat publiable)
    ref = g.loc[g.domain == "ID", "neg_log_evidence"]
    med, mad = ref.median(), 1.4826 * (ref - ref.median()).abs().median() + 1e-9
    g["typicality"] = ((g.neg_log_evidence - med) / mad).abs()
    g["typicality_cum"] = causal_mean(g, "typicality")
    g["nle_minus_med"] = g.neg_log_evidence - med
    print(f"-log p(x) médiane ID = {med:.1f} ; médiane OOD = {g.loc[g.domain == 'OOD', 'neg_log_evidence'].median():.1f}"
          f" (une médiane OOD plus basse = le modèle trouve les courbes OOD *plus* probables)")
    S = g[key + ["domain", "neg_log_evidence", "neg_log_evidence_cum", "typicality", "typicality_cum",
                 "pca_recon_err", "pca_recon_err_cum"]]
    for m in ["ens_hgb", "hgb_hetero"]:
        if m in P.method.unique():
            w = P[P.method == m][key + ["width90"]].rename(columns={"width90": f"{m}_width90"})
            S = S.merge(w, on=key, how="left")
            S[f"{m}_width90_cum"] = causal_mean(S, f"{m}_width90")
    S["is_ood"] = (S.domain == "OOD").astype(int)
    scores = [c for c in S.columns if c not in key + ["domain", "is_ood"]]

    # --- AUROC par cycle et par cellule (score moyen de la cellule)
    rows = []
    C = S.groupby("cell").agg({**{c: "mean" for c in scores}, "is_ood": "first"})
    for c in scores:
        rows.append(dict(score=c, auroc_cycle=auroc(S.is_ood.values, S[c].values),
                         auroc_cell=auroc(C.is_ood.values, C[c].values)))
    A = pd.DataFrame(rows).set_index("score").round(3)
    A.to_csv(os.path.join(args.out, "ood_auroc.csv"))
    print(f"ID : {C.is_ood.eq(0).sum()} cellules / OOD : {C.is_ood.sum()} cellules")
    print("\n=== AUROC ID vs OOD (1 = séparation parfaite, 0,5 = hasard) ===")
    print(A.to_string())

    # --- prédiction sélective : on garde les cycles les moins suspects
    keep = np.linspace(1.0, 0.1, 19)
    rows = []
    for m in args.methods:
        if m not in P.method.unique():
            continue
        Pm = P[P.method == m][key + ["pred", "soh", "lo90", "hi90"]].merge(S, on=key)
        Pm["err"] = (Pm.pred - Pm.soh).abs()
        Pm["cov"] = (Pm.soh >= Pm.lo90) & (Pm.soh <= Pm.hi90)
        own = {"gen_filter": "neg_log_evidence_cum", "gen_Tcal": "neg_log_evidence",
               "ens_hgb": "ens_hgb_width90_cum", "hgb_filter": "hgb_hetero_width90_cum",
               "hgb_cqr": "ens_hgb_width90"}
        for sc in dict.fromkeys([own.get(m, "neg_log_evidence_cum"), "neg_log_evidence_cum",
                                 "ens_hgb_width90_cum"]):
            if sc not in Pm:
                continue
            order = Pm[sc].rank(method="first").values
            n = len(Pm)
            for k in keep:
                sel = order <= int(np.ceil(k * n))
                rows.append(dict(method=m, score=sc, kept=round(k, 2), MAE=Pm.err[sel].mean(),
                                 cov90=Pm["cov"][sel].mean(), frac_ood_kept=Pm.is_ood[sel].mean()))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out, "selective_risk.csv"), index=False)
    print("\n=== MAE après rejet des cycles les plus suspects (kept = fraction conservée) ===")
    piv = R[R.kept.isin([1.0, 0.8, 0.6, 0.5, 0.4])].pivot_table(
        index=["method", "score"], columns="kept", values="MAE").round(2)
    print(piv[sorted(piv.columns, reverse=True)].to_string())

    # --- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [c for c in ["neg_log_evidence", "typicality_cum", "pca_recon_err", "ens_hgb_width90_cum"] if c in S]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.4))
    for ax, c in zip(np.atleast_1d(axes), show):
        v = S[c].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = np.percentile(v, [1, 99])
        bins = np.linspace(lo, hi, 60)
        for d, col in (("ID", "#4C78A8"), ("OOD", "#E45756")):
            ax.hist(S.loc[S.domain == d, c].clip(lo, hi), bins=bins, alpha=0.6, density=True, label=d, color=col)
        ax.set_title(f"{c}\nAUROC cycle {A.loc[c, 'auroc_cycle']:.2f} / cellule {A.loc[c, 'auroc_cell']:.2f}",
                     fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_ood_scores.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (m, sc), d in R.groupby(["method", "score"]):
        ax.plot(d.kept, d.MAE, marker="o", ms=2.5, label=f"{m} | {sc}")
    ax.invert_xaxis()
    ax.set(xlabel="fraction des cycles conservés", ylabel="MAE SOH (points de %)",
           title="Prédiction sélective (rejet des cycles les plus suspects)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_selective.png"), dpi=140); plt.close(fig)
    print(f"\nFichiers et figures dans {args.out}/")


if __name__ == "__main__":
    main()
