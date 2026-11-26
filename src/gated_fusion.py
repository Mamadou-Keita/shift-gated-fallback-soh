#!/usr/bin/env python3
"""
Fusion pondérée par le décalage : une base transférable + une correction de forme activée
seulement quand le cycle test ressemble au domaine d'entraînement.

  base       ŷ_b = ridge([ΔQ/Q_nom, T])                       (se transfère entre chimies)
  correction r̂   = ridge([PCA(forme), PCA(dQ/dV), T]) appris sur y − ŷ_b HORS-PLI
                    (validation groupée par cellule : la correction apprend ce que la base rate
                     sur des cellules qu'elle n'a pas vues, pas son sur-apprentissage)
  pondération w  = 1 si s ≤ q, 0 si s ≥ k·q, log-linéaire entre les deux, où
                    s = d²_Mahalanobis(Ledoit-Wolf) des scores PCA forme/dQ/dV
                        + résidu de reconstruction PCA normalisé (sortie du sous-espace)
                    q = quantile (--gate-q, 0,99) de s sur des cellules d'entraînement
                        TENUES À L'ÉCART (étalonnage hors-pli : « une cellule nouvelle mais du même domaine »)
  prédiction ŷ   = ŷ_b + w · r̂

Comparaisons (mêmes plis, MAE moyenne par cellule + IC bootstrap de cellules) :
  base_dQ+T, forme+T (shape+ic, M1), tout+T (dQ+shape+ic, M1), FUS_toujours (w = 1), FUS_pondérée,
  et FUS_cellule (w agrégé par cellule = médiane des w des cycles déjà vus, causal).

Protocoles : ceux de cond_models.py (cell, temp, chem, chemtemp).

  python gated_fusion.py --npz benchT_Tongji/samples.npz --include-regex '^Tongji[12]' \
      --temp-regex 'CY(\\d+)-' --chem-regex '^(Tongji\\d)' --mode chem --out fus_chem
"""
import argparse
import os

import numpy as np
import pandas as pd

from cond_models import boot_ci, cell_mae, design, features, fit_ridge, load, splits


def pca_blocks(D, tr, te, n_pca, keys=("shape", "ic")):
    """Scores PCA standardisés + résidu de reconstruction relatif, par bloc (ajusté sur tr)."""
    from sklearn.decomposition import PCA
    Ztr, Zte, Rtr, Rte = [], [], [], []
    for k in keys:
        p = PCA(n_components=n_pca).fit(D[k][tr])
        for idx, Z, R in ((tr, Ztr, Rtr), (te, Zte, Rte)):
            s = p.transform(D[k][idx])
            Z.append(s)
            R.append(((D[k][idx] - p.inverse_transform(s)) ** 2).sum(1))
    Ztr, Zte = np.hstack(Ztr), np.hstack(Zte)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-12
    return (Ztr - mu) / sd, (Zte - mu) / sd, np.vstack(Rtr).T, np.vstack(Rte).T


class Gate:
    """Score de décalage s(x) = d²_LW(z) + Σ_blocs r_bloc / médiane_train(r_bloc).
    score_parts (ablation) : "full" (défaut, publié) ; "maha" (Mahalanobis seul) ;
    "recon" (résidus de reconstruction seuls, forme + dQ/dV) ; "maha_shape" (Mahalanobis + résidu forme)."""

    def __init__(self, parts="full"):
        self.parts = parts

    def fit(self, Z, R):
        from sklearn.covariance import LedoitWolf
        self.lw = LedoitWolf().fit(Z)
        self.rmed = np.median(R, 0) + 1e-12
        return self

    def score(self, Z, R):
        m, r = self.lw.mahalanobis(Z), R / self.rmed
        if self.parts == "maha":
            return m
        if self.parts == "recon":
            return r.sum(1)
        if self.parts == "maha_shape":
            return m + r[:, 0]  # bloc 0 = forme (pca_blocks keys=("shape", "ic"))
        return m + r.sum(1)


def weighted_quantile(x, q, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, c, x))


def calibrate_gate(D, tr, n_pca, q, k_folds=5, weighting="cycle", first_k=5, parts="full"):
    """Quantile q du score sur des cellules d'entraînement tenues à l'écart (hors-pli).

    Les k plis internes (GroupKFold par cellule) donnent chacun les scores hors-pli de leurs cellules
    tenues à l'écart ; tous ces scores sont réunis, puis un seul quantile est pris.
    weighting = "cycle"   : chaque cycle compte pareil (défaut fixé a priori, résultats publiés) ;
                "cell"    : chaque cellule compte pareil (poids 1 / nb de cycles de la cellule) ;
                "cellmed" : médiane causale du score sur les cycles de chaque cellule tenue à l'écart
                            (même unité que la décision au niveau cellule) ;
                "firstk"  : une valeur par cellule tenue à l'écart, médiane de ses first_k premiers cycles."""
    from sklearn.model_selection import GroupKFold
    g = D["groups"][tr]
    k = min(k_folds, len(np.unique(g)))
    s_oof, g_oof = [], []
    for i, j in GroupKFold(n_splits=k).split(tr, tr, g):
        Ztr, Zte, Rtr, Rte = pca_blocks(D, tr[i], tr[j], n_pca)
        s_oof.append(Gate(parts).fit(Ztr, Rtr).score(Zte, Rte)); g_oof.append(g[j])
    s, gg = np.concatenate(s_oof), np.concatenate(g_oof)
    if weighting in ("cellmed", "firstk"):
        # same unit as the cell-level decision: running median of the score over each held-out cell's
        # cycles in time order (the cell-level weight is a monotone function of it)
        cyc = D["cycle"][tr] if "cycle" in D else np.arange(len(tr))
        c_oof = np.concatenate([cyc[j] for _, j in GroupKFold(n_splits=k).split(tr, tr, g)])
        med, firsts = np.empty_like(s), []
        for c in np.unique(gg):
            idx = np.flatnonzero(gg == c)
            idx = idx[np.argsort(c_oof[idx], kind="stable")]
            med[idx] = [np.median(s[idx[: i + 1]]) for i in range(len(idx))]
            firsts.append(np.median(s[idx[:first_k]]))
        if weighting == "cellmed":
            return float(np.quantile(med, q))
        return float(np.quantile(np.array(firsts), q))  # one value per held-out cell
    if weighting == "cell":
        _, inv, cnt = np.unique(gg, return_inverse=True, return_counts=True)
        return weighted_quantile(s, q, 1.0 / cnt[inv])
    return float(np.quantile(s, q))


def weights(s, q, k):
    """1 sous q, 0 au-delà de k·q, décroissance linéaire en log(s) entre les deux."""
    w = 1 - np.log(np.maximum(s, q) / q) / np.log(k)
    return np.clip(w, 0, 1)


def causal_cell_median(w, cells, cycles):
    """Pour chaque cycle : médiane des w de sa cellule sur les cycles ≤ celui-ci (causal)."""
    out = np.empty_like(w)
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        idx = idx[np.argsort(cycles[idx], kind="stable")]
        ww = w[idx]
        out[idx] = [np.median(ww[: i + 1]) for i in range(len(ww))]
    return out


def ridge_T(fs, D, tr, te, args, alphas, tmu, tsd):
    Xtr, Xte = features(fs, D, tr, te, args.n_pca)
    T = D["T"]
    m, a = fit_ridge(design(Xtr, T[tr], "M1", tmu, tsd), D["soh"][tr], D["groups"][tr], alphas)
    return m, a, Xtr, Xte


def fusion_predict(D, tr, te, args, alphas):
    """Prédictions de toutes les variantes (dict) + informations sur la pondération."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    y, g, T = D["soh"], D["groups"], D["T"]
    cyc = D["cycle"] if "cycle" in D else np.arange(len(y))
    if len(set(T[tr])) < 2:
        tmu, tsd = T[tr].mean(), 1.0  # T constante : sa colonne est nulle, M1 = M0
    else:
        tmu, tsd = T[tr].mean(), T[tr].std() + 1e-9
    preds, info = {}, {}

    # références M1
    for lab, fs in (("base_dQ+T", "dQ"), ("forme+T", "shape+ic"), ("tout+T", "dQ+shape+ic")):
        m, a, Xtr, Xte = ridge_T(fs, D, tr, te, args, alphas, tmu, tsd)
        preds[lab] = m.predict(design(Xte, T[te], "M1", tmu, tsd))
        if lab == "base_dQ+T":
            a_base = a

    # base hors-pli sur l'entraînement (α fixé à celui choisi sur tout l'entraînement)
    k = min(5, len(np.unique(g[tr])))
    base_oof = np.empty(len(tr))
    for i, j in GroupKFold(n_splits=k).split(tr, tr, g[tr]):
        Xa, Xb = features("dQ", D, tr[i], tr[j], args.n_pca)
        mb = Ridge(alpha=a_base).fit(design(Xa, T[tr[i]], "M1", tmu, tsd), y[tr[i]])
        base_oof[j] = mb.predict(design(Xb, T[tr[j]], "M1", tmu, tsd))

    # correction sur le résidu hors-pli
    Xtr, Xte = features(args.corr_features, D, tr, te, args.n_pca)
    mc, a_corr = fit_ridge(design(Xtr, T[tr], "M1", tmu, tsd), y[tr] - base_oof, g[tr], alphas)
    corr = mc.predict(design(Xte, T[te], "M1", tmu, tsd))

    # pondération
    q = calibrate_gate(D, tr, args.n_pca, args.gate_q)
    Ztr, Zte, Rtr, Rte = pca_blocks(D, tr, te, args.n_pca)
    s = Gate().fit(Ztr, Rtr).score(Zte, Rte)
    w = weights(s, q, args.gate_k)
    wc = causal_cell_median(w, g[te], cyc[te])

    preds["FUS_toujours"] = preds["base_dQ+T"] + corr
    preds["FUS_pondérée"] = preds["base_dQ+T"] + w * corr
    preds["FUS_cellule"] = preds["base_dQ+T"] + wc * corr
    wcell = pd.Series(w).groupby(g[te]).mean()
    gate = dict(w_mean=w.mean(), frac_w1=np.mean(w >= 0.999), frac_w0=np.mean(w <= 1e-3),
                w_cell_min=wcell.min(), w_cell_max=wcell.max(), q_gate=q,
                s_median_test=float(np.median(s)), alpha_base=a_base, alpha_corr=a_corr)
    return preds, gate


def run_split(D, name, tr, te, args, alphas):
    y, g = D["soh"], D["groups"]
    preds, gate = fusion_predict(D, tr, te, args, alphas)
    gate = dict(split=name, **gate)
    rows, percell = [], {}
    for lab, p in preds.items():
        cm = cell_mae(p - y[te], g[te])
        percell[lab] = cm
        lo, hi = boot_ci(cm.values)
        rows.append(dict(split=name, method=lab, MAE_cell=cm.mean(), ci_lo=lo, ci_hi=hi,
                         n_test_cells=len(cm), n_train_cells=len(np.unique(g[tr]))))

    comps = []
    for a_ in ("FUS_pondérée", "FUS_cellule"):
        for b_ in ("base_dQ+T", "forme+T", "tout+T", "FUS_toujours"):
            d = (percell[a_] - percell[b_]).values
            lo, hi = boot_ci(d, seed=1)
            comps.append(dict(split=name, comparison=f"{a_} − {b_}", diff=d.mean(), ci_lo=lo, ci_hi=hi,
                              frac_cells_better=np.mean(d < 0)))
    # « oracle » : meilleur des deux experts choisi a posteriori pour ce pli (borne, pas une méthode)
    best = min(percell["base_dQ+T"].mean(), percell["forme+T"].mean())
    gate["oracle_best_of_2"] = best
    return rows, comps, gate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="fus_out")
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", default=None)
    ap.add_argument("--mode", choices=["cell", "temp", "chem", "chemtemp", "chemcv"], default="cell")
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--corr-features", default="shape+ic",
                    help="Features de la correction (ex. shape+ic, shape, ic, dQ+shape+ic)")
    ap.add_argument("--gate-q", type=float, default=0.99, help="Quantile hors-pli du score : w = 1 en dessous")
    ap.add_argument("--gate-k", type=float, default=10.0, help="w = 0 au-delà de k × seuil")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    alphas = np.logspace(-3, 4, 15)

    D = load(args)
    tab = pd.crosstab(pd.Series(D["chem"], name="chimie"), pd.Series(D["T"], name="T (°C)"),
                      values=D["groups"], aggfunc=lambda s: len(set(s)))
    print(f"{len(D['soh'])} échantillons, {len(set(D['groups']))} cellules. Cellules par chimie × température :")
    print(tab.fillna(0).astype(int).to_string(), "\n")

    rows, comps, gates = [], [], []
    for name, tr, te in splits(D, args.mode):
        r, c, gt = run_split(D, name, tr, te, args, alphas)
        rows += r; comps += c; gates.append(gt)
        print(f"  {name} : w moyen {gt['w_mean']:.2f} (w=1 : {gt['frac_w1']:.0%}, w=0 : {gt['frac_w0']:.0%})")

    R, C, G = pd.DataFrame(rows), pd.DataFrame(comps), pd.DataFrame(gates)
    R.to_csv(os.path.join(args.out, "results.csv"), index=False)
    C.to_csv(os.path.join(args.out, "paired.csv"), index=False)
    G.to_csv(os.path.join(args.out, "gate.csv"), index=False)
    pd.set_option("display.width", 250)
    order = ["base_dQ+T", "forme+T", "tout+T", "FUS_toujours", "FUS_pondérée", "FUS_cellule"]
    R["txt"] = R.apply(lambda x: f"{x.MAE_cell:.2f} [{x.ci_lo:.2f}-{x.ci_hi:.2f}]", axis=1)
    print("\n=== MAE moyenne par cellule (points de SOH) [IC 95 % bootstrap de cellules] ===")
    P = R.pivot(index="split", columns="method", values="txt")[order]
    P["oracle(min 2)"] = G.set_index("split").oracle_best_of_2.map("{:.2f}".format)
    print(P.to_string())
    if args.mode == "cell":
        M = R.pivot(index="split", columns="method", values="MAE_cell")[order]
        print("\nMoyenne des plis :", "  ".join(f"{c} {M[c].mean():.2f}" for c in order))
    print("\n=== Pondération ===")
    print(G[["split", "w_mean", "frac_w1", "frac_w0", "w_cell_min", "w_cell_max", "q_gate", "s_median_test"]]
          .round(3).to_string(index=False))
    C["txt"] = C.apply(lambda x: f"{x['diff']:+.2f} [{x.ci_lo:+.2f};{x.ci_hi:+.2f}] ({x.frac_cells_better:.0%})", axis=1)
    print("\n=== Écarts appariés par cellule (négatif = la fusion est meilleure) [IC 95 %] (part des cellules améliorées) ===")
    print(C.pivot(index="split", columns="comparison", values="txt").to_string())
    print(f"\nFichiers dans {args.out}/")


if __name__ == "__main__":
    main()
