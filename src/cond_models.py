#!/usr/bin/env python3
"""
Effet des conditions d'essai (température) sur l'estimation du SOH à partir de la forme des
courbes : comparaison de trois modèles linéaires, sur les mêmes cellules et les mêmes plis.

  M0 : SOH ~ X                       (aucune information sur la condition)
  M1 : SOH ~ X + T                   (correction additive par la température)
  M2 : SOH ~ X + T + X·T             (la température modifie la relation courbe → SOH)

X = ΔQ / Q_nom et/ou composantes PCA de la forme Q(V)/ΔQ et/ou de dQ/dV, PCA et standardisation
ajustées sur l'entraînement de chaque pli ; régularisation ridge choisie par validation interne
GROUPÉE PAR CELLULE (pas par cycle). Référence non linéaire : gradient boosting avec T.

Protocoles (--mode) :
  cell      dans le domaine : validation croisée groupée par cellule (5 plis), toutes conditions
  temp     laisser une température de côté (toutes chimies sélectionnées confondues)
  chem      entraîner sur une chimie, tester sur l'autre, résultats par température
  chemtemp  entraîner sur la chimie A sans la température t, tester sur la chimie B à t
            (changement simultané de chimie et de température)

Stabilité (--jackknife) : en mode temp, si une température d'entraînement compte ≤ 8 cellules,
le pli est refait en retirant chacune de ces cellules à tour de rôle (l'effet de T repose sur elles).

Exemple (Tongji, NCA + NCM, 0,5C) :
  python cond_models.py --npz benchT_Tongji/samples.npz --include-regex '^Tongji[12]' \
      --temp-regex 'CY(\\d+)-' --chem-regex '^(Tongji\\d)' --mode temp --jackknife --out cond_temp
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def load(args):
    D = {k: v for k, v in np.load(args.npz, allow_pickle=False).items()}
    D["groups"] = D["groups"].astype(str)
    if args.include_regex:
        rx = re.compile(args.include_regex)
        keep = np.array([bool(rx.search(c)) for c in D["groups"]])
        D = {k: (v[keep] if k != "v_grid" and np.ndim(v) and len(v) == len(keep) else v) for k, v in D.items()}
    rx = re.compile(args.temp_regex)
    D["T"] = np.array([float(rx.search(c).group(1)) if rx.search(c) else np.nan for c in D["groups"]])
    if np.isnan(D["T"]).any():
        sys.exit("--temp-regex ne trouve pas la température dans certains noms de cellules.")
    if args.chem_regex:
        rc = re.compile(args.chem_regex)
        D["chem"] = np.array([rc.search(c).group(1) if rc.search(c) else "?" for c in D["groups"]])
    else:
        D["chem"] = np.array(["all"] * len(D["groups"]))
    if "qnom" in D and np.isfinite(D["qnom"]).all():
        D["dqn"] = D["dq"][:, 0] / D["qnom"]
    else:
        print("  (qnom absent : ΔQ non normalisé)")
        D["dqn"] = D["dq"][:, 0]
    D["soh"] = D["soh"] * 100.0
    # unité physique : pour ISU-ILCC, partie après « __ » dans le nom du groupe (une même cellule mesurée
    # à deux courants) ; sinon le groupe lui-même (Tongji, MICH : comportement inchangé)
    D["unit"] = np.array([c.split("__", 1)[1] if c.startswith("ISU_") and "__" in c else c for c in D["groups"]])
    return D


def features(fs, D, tr, te, n_pca):
    from sklearn.decomposition import PCA
    A, B = [], []
    parts = fs.split("+")
    if "dQ" in parts:
        A.append(D["dqn"][tr, None]); B.append(D["dqn"][te, None])
    for key in ("shape", "ic"):
        if key in parts:
            p = PCA(n_components=n_pca).fit(D[key][tr])
            A.append(p.transform(D[key][tr])); B.append(p.transform(D[key][te]))
    Xtr, Xte = np.hstack(A), np.hstack(B)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


def design(X, T, model, tmu, tsd, scale=1.0):
    t = ((T - tmu) / tsd)[:, None]
    if model == "M0":
        return X
    if model == "M1":
        return np.hstack([X, t])
    # M2 / M2p : interactions X·T ; « scale » = 1/sqrt(m) multiplie la pénalité ridge des
    # interactions par m (m → ∞ : M2p retombe sur M1)
    return np.hstack([X, t, scale * X * t])


def fit_ridge_m2p(Xtr, ttr, ytr, gtr, alphas, mults, tmu, tsd):
    """M2 à pénalité séparée : (α, m) choisis ensemble en validation interne groupée par cellule."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    k = min(4, len(np.unique(gtr)))
    best = None
    for m in mults:
        Z = design(Xtr, ttr, "M2", tmu, tsd, 1 / np.sqrt(m))
        for a in alphas:
            err = []
            for i, j in GroupKFold(n_splits=k).split(Z, ytr, gtr):
                err.append(np.abs(Ridge(alpha=a).fit(Z[i], ytr[i]).predict(Z[j]) - ytr[j]).mean())
            e = float(np.mean(err))
            if best is None or e < best[0]:
                best = (e, a, m)
    _, a, m = best
    return Ridge(alpha=a).fit(design(Xtr, ttr, "M2", tmu, tsd, 1 / np.sqrt(m)), ytr), a, m


def fit_ridge(Xtr, ytr, gtr, alphas):
    """α choisi par validation interne groupée par cellule (MAE)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    cells = np.unique(gtr)
    k = min(4, len(cells))
    best = None
    for a in alphas:
        err = []
        for i, j in GroupKFold(n_splits=k).split(Xtr, ytr, gtr):
            m = Ridge(alpha=a).fit(Xtr[i], ytr[i])
            err.append(np.abs(m.predict(Xtr[j]) - ytr[j]).mean())
        e = float(np.mean(err))
        if best is None or e < best[0]:
            best = (e, a)
    return Ridge(alpha=best[1]).fit(Xtr, ytr), best[1]


def cell_mae(err, cells):
    return pd.Series(np.abs(err)).groupby(cells).mean()


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v)
    b = np.random.default_rng(seed).choice(v, size=(n, len(v)), replace=True).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def splits(D, mode):
    T, C = D["T"], D["chem"]
    temps, chems = sorted(set(T)), sorted(set(C))
    out = []
    if mode == "cell":
        from sklearn.model_selection import GroupKFold
        g = D["unit"]
        for k, (tr, te) in enumerate(GroupKFold(n_splits=min(5, len(set(g)))).split(g, g, g)):
            out.append((f"cellules pli {k + 1}", tr, te))
    elif mode == "temp":
        for t in temps:
            out.append((f"test {t:g}°C", np.flatnonzero(T != t), np.flatnonzero(T == t)))
    elif mode == "chem":
        for a in chems:
            for b in chems:
                if a == b:
                    continue
                for t in temps:
                    te = np.flatnonzero((C == b) & (T == t))
                    if len(te):
                        out.append((f"{a}→{b} @ {t:g}°C", np.flatnonzero(C == a), te))
    elif mode == "chemcv":
        # A → B sur des cellules physiques disjointes (GroupKFold sur les unités) : sert quand une
        # même cellule existe dans les deux conditions (ISU-ILCC : fenêtres C/5 et C/2 des mêmes RPT)
        from sklearn.model_selection import GroupKFold
        u = D["unit"]
        units = np.unique(u)
        for k, (_, te_u) in enumerate(GroupKFold(n_splits=min(5, len(units))).split(units, units, units)):
            test_units = set(units[te_u])
            in_test = np.array([x in test_units for x in u])
            for a in chems:
                for b in chems:
                    if a != b:
                        out.append((f"{a}→{b} pli {k + 1}", np.flatnonzero((C == a) & ~in_test),
                                    np.flatnonzero((C == b) & in_test)))
    elif mode == "chemtemp":
        for a in chems:
            for b in chems:
                if a == b:
                    continue
                for t in temps:
                    tr = np.flatnonzero((C == a) & (T != t))
                    te = np.flatnonzero((C == b) & (T == t))
                    # ≥ 2 températures à l'entraînement quand c'est possible (effet de T identifiable) ;
                    # avec seulement 2 températures au total (ex. MICH), on accepte 1 (T non identifiable)
                    if len(te) and len(set(T[tr])) >= min(2, len(temps) - 1):
                        out.append((f"{a}(sans {t:g}°C)→{b} @ {t:g}°C", tr, te))
    return out


def run_split(D, name, tr, te, args, alphas):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y, g, T = D["soh"], D["groups"], D["T"]
    rows, percell = [], {}
    tmu, tsd = T[tr].mean(), T[tr].std() + 1e-9
    for fs in args.feature_sets:
        Xtr, Xte = features(fs, D, tr, te, args.n_pca)
        for model in ("M0", "M1", "M2", "M2p"):
            if model != "M0" and len(set(T[tr])) < 2:
                continue  # T constante à l'entraînement : M1/M2 non identifiables
            if model == "M2p":
                m, a, mult = fit_ridge_m2p(Xtr, T[tr], y[tr], g[tr], alphas, args.inter_mults, tmu, tsd)
                err = m.predict(design(Xte, T[te], "M2", tmu, tsd, 1 / np.sqrt(mult))) - y[te]
            else:
                m, a = fit_ridge(design(Xtr, T[tr], model, tmu, tsd), y[tr], g[tr], alphas)
                mult = np.nan
                err = m.predict(design(Xte, T[te], model, tmu, tsd)) - y[te]
            cm = cell_mae(err, g[te])
            percell[(fs, model)] = cm
            lo, hi = boot_ci(cm.values)
            rows.append(dict(split=name, features=fs, model=model, MAE_cell=cm.mean(), ci_lo=lo, ci_hi=hi,
                             alpha=a, inter_mult=mult, n_test_cells=len(cm),
                             n_train_cells=len(np.unique(g[tr]))))
        if args.hgb:
            h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0)
            h.fit(np.hstack([Xtr, T[tr, None]]), y[tr])
            err = h.predict(np.hstack([Xte, T[te, None]])) - y[te]
            cm = cell_mae(err, g[te])
            lo, hi = boot_ci(cm.values)
            rows.append(dict(split=name, features=fs, model="HGB+T", MAE_cell=cm.mean(), ci_lo=lo, ci_hi=hi,
                             alpha=np.nan, n_test_cells=len(cm), n_train_cells=len(np.unique(g[tr]))))
    # comparaisons appariées par cellule : M1 − M0 et M2 − M1
    comps = []
    for fs in args.feature_sets:
        for a_, b_ in (("M1", "M0"), ("M2", "M1"), ("M2p", "M1")):
            if (fs, a_) in percell and (fs, b_) in percell:
                d = (percell[(fs, a_)] - percell[(fs, b_)]).values
                lo, hi = boot_ci(d, seed=1)
                comps.append(dict(split=name, features=fs, comparison=f"{a_}−{b_}", diff=d.mean(),
                                  ci_lo=lo, ci_hi=hi, frac_cells_better=np.mean(d < 0)))
    return rows, comps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="cond_out")
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", default=None)
    ap.add_argument("--mode", choices=["cell", "temp", "chem", "chemtemp", "chemcv"], default="temp")
    ap.add_argument("--feature-sets", nargs="+",
                    default=["dQ", "shape", "ic", "dQ+shape", "dQ+ic", "dQ+shape+ic"])
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--hgb", action="store_true", help="Ajouter la référence gradient boosting + T")
    ap.add_argument("--inter-mults", type=float, nargs="+", default=[1, 10, 100, 1e3, 1e4, 1e6],
                    help="M2p : multiplicateurs candidats de la pénalité des interactions")
    ap.add_argument("--jackknife", action="store_true")
    ap.add_argument("--jk-max-cells", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    alphas = np.logspace(-3, 4, 15)

    D = load(args)
    tab = pd.crosstab(pd.Series(D["chem"], name="chimie"), pd.Series(D["T"], name="T (°C)"),
                      values=D["groups"], aggfunc=lambda s: len(set(s)))
    print(f"{len(D['soh'])} échantillons, {len(set(D['groups']))} cellules. Cellules par chimie × température :")
    print(tab.fillna(0).astype(int).to_string(), "\n")

    rows, comps, jk = [], [], []
    for name, tr, te in splits(D, args.mode):
        r, c = run_split(D, name, tr, te, args, alphas)
        rows += r; comps += c
        print(f"  {name} : {len(np.unique(D['groups'][tr]))} cellules d'entraînement, "
              f"{len(np.unique(D['groups'][te]))} de test")
        if args.jackknife and args.mode == "temp":
            Ttr = D["T"][tr]
            for t in sorted(set(Ttr)):
                cells_t = np.unique(D["groups"][tr][Ttr == t])
                if len(cells_t) > args.jk_max_cells:
                    continue
                for c_out in cells_t:
                    tr2 = tr[D["groups"][tr] != c_out]
                    r2, _ = run_split(D, name, tr2, te, args, alphas)
                    for x in r2:
                        x.update(jk_removed=c_out, jk_temp=t)
                    jk += r2

    R = pd.DataFrame(rows); C = pd.DataFrame(comps)
    R.to_csv(os.path.join(args.out, "results.csv"), index=False)
    C.to_csv(os.path.join(args.out, "paired.csv"), index=False)
    pd.set_option("display.width", 220)
    R["cell"] = R.apply(lambda x: f"{x.MAE_cell:.2f} [{x.ci_lo:.2f}-{x.ci_hi:.2f}]"
                        + (f" m={x.inter_mult:g}" if x.model == "M2p" else ""), axis=1)
    print("\n=== MAE moyenne par cellule [IC 95 % bootstrap de cellules] ===")
    for fs in args.feature_sets:
        sub = R[R.features == fs].pivot(index="split", columns="model", values="cell")
        print(f"\n-- {fs}"); print(sub.to_string())
    if len(C):
        C["txt"] = C.apply(lambda x: f"{x['diff']:+.2f} [{x.ci_lo:+.2f};{x.ci_hi:+.2f}] ({x.frac_cells_better:.0%})", axis=1)
        print("\n=== Écarts appariés par cellule (négatif = le modèle plus riche est meilleur) "
              "[IC 95 %] (part des cellules améliorées) ===")
        print(C.pivot_table(index=["split", "features"], columns="comparison", values="txt",
                            aggfunc="first").to_string())
    if jk:
        J = pd.DataFrame(jk); J.to_csv(os.path.join(args.out, "jackknife.csv"), index=False)
        s = J.groupby(["split", "features", "model"]).MAE_cell.agg(["min", "max"]).round(2)
        base = R.set_index(["split", "features", "model"]).MAE_cell.round(2).rename("complet")
        print("\n=== Stabilité : MAE en retirant une à une les cellules de la température minoritaire ===")
        print(s.join(base).reset_index().pivot_table(index=["split", "features"], columns="model",
              values=["complet", "min", "max"]).round(2).to_string())
    print(f"\nFichiers dans {args.out}/")


if __name__ == "__main__":
    main()
