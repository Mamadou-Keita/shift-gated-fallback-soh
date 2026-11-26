#!/usr/bin/env python3
"""
La pondération par le décalage comme ENVELOPPE DE SÉCURITÉ autour de n'importe quel modèle :

    ŷ = base + w · (ŷ_expert − base)

  base     ridge(ΔQ/Q_nom, T) : se transfère entre chimies
  expert   n'importe quel modèle précis dans le domaine (GPR+T, CNN+T, LSTM+T, …) — ici les
           prédictions DÉJÀ CALCULÉES par lit_baselines.py (dossiers $R/lit_<protocole>_*/preds),
           donc aucun réentraînement : seuls la base et la pondération sont recalculées
  w        même pondération que gated_fusion.py (Mahalanobis Ledoit-Wolf + résidu PCA sur la
           forme et dQ/dV, seuil = quantile 0,99 sur des cellules d'entraînement tenues à l'écart,
           k = 10 fixé a priori)

Les plis sont reconstruits par cond_models.splits (déterministes) : les prédictions en cache
correspondent pli à pli. Contrôle : la base recalculée doit coïncider avec celle du cache.

  python gated_wrapper.py --npz $NPZ --include-regex '^Tongji[12]' --temp-regex 'CY(\\d+)-' \
      --chem-regex '^(Tongji\\d)' --lit-root $R --out $R/wrap
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from cond_models import boot_ci, cell_mae, load, splits
from gated_fusion import Gate, calibrate_gate, causal_cell_median, pca_blocks, ridge_T, weights

PROTOS = ["cell", "temp", "chem", "chemtemp", "chemcv"]
FIXED_ALPHAS = (0.25, 0.5, 0.75)  # α = 0 is the fallback, α = 1 the expert
NAMES = {"cell": "Dans le domaine", "temp": "Température jamais vue", "chem": "Chimie jamais vue",
         "chemtemp": "Chimie + température", "chemcv": "Condition jamais vue (cellules disjointes)"}


def label_of(fname_part):
    # nom de fichier : « + » remplacé par « p » à la sauvegarde (ex. GPRpT, base_dQpT, toutpT)
    return re.sub(r"p(T)$", r"+\1", fname_part)


def paired_tests(Cc, out_dir):
    """Tests appariés par cellule, toutes les cellules de test d'un protocole regroupées (chaque cellule
    n'est testée qu'une fois par protocole) : écart moyen, IC 95 % bootstrap de cellules, Wilcoxon."""
    from scipy.stats import wilcoxon
    Cc = Cc.drop_duplicates(["protocol", "split", "method", "cell"])
    W = Cc.pivot_table(index=["protocol", "split", "cell"], columns="method", values="mae")
    experts = sorted({m[5:-1] for m in W.columns if m.startswith("GATE[")})
    pairs = [(f"GATE[{e}]", e) for e in experts] + [(f"GATE[{e}]", "FUS_pondérée") for e in experts] \
        + [("FUS_pondérée", e) for e in experts]
    res = []
    for proto, sub in W.groupby(level="protocol"):
        for a, b in pairs:
            if a not in sub or b not in sub:
                continue
            d = (sub[a] - sub[b]).dropna().values
            if len(d) < 3:
                continue
            lo, hi = boot_ci(d, seed=1)
            try:
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = np.nan
            res.append(dict(protocol=proto, comparison=f"{a} − {b}", diff=d.mean(), ci_lo=lo, ci_hi=hi,
                            p_wilcoxon=p, frac_better=np.mean(d < 0), n_cells=len(d)))
    T = pd.DataFrame(res)
    T.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)
    if not len(T):
        return
    T["txt"] = T.apply(lambda x: f"{x['diff']:+.2f} [{x.ci_lo:+.2f};{x.ci_hi:+.2f}] p={x.p_wilcoxon:.1g} ({x.frac_better:.0%})", axis=1)
    P = T.pivot(index="comparison", columns="protocol", values="txt")
    P = P[[p for p in PROTOS if p in P.columns]].rename(columns=NAMES)
    print("\n=== Tests appariés par cellule (toutes les cellules de test du protocole) : écart moyen de MAE "
          "[IC 95 %] p Wilcoxon (part des cellules où le 1er est meilleur) ; négatif = le 1er est meilleur ===")
    print(P.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--lit-root", required=True, help="Dossier contenant les lit_<protocole>_<modèles>/")
    ap.add_argument("--out", default="wrap_out")
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", default=None)
    ap.add_argument("--protocols", nargs="+", default=PROTOS, choices=PROTOS)
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--gate-q", type=float, default=0.99)
    ap.add_argument("--gate-k", type=float, default=10.0)
    ap.add_argument("--calib", choices=["cycle", "cell", "cellmed", "firstk"], default="cycle",
                    help="Seuil q : cycles (défaut), cellules à poids égal, médiane causale par cellule, "
                         "ou médiane des K premiers cycles par cellule")
    ap.add_argument("--calib-k", type=int, default=5, help="K pour --calib firstk")
    ap.add_argument("--fallback", choices=["nominal", "selfref"], default="nominal",
                    help="Repli : ridge sur [ΔQ/Q_nom, T] (nominal, publié) ou sur [ΔQ / médiane des --ref-k "
                         "premières fenêtres de la même cellule, T] (selfref, suppose la cellule vue neuve)")
    ap.add_argument("--ref-k", type=int, default=3)
    ap.add_argument("--ref-mode", choices=["fixed", "causal"], default="fixed",
                    help="fixed : médiane des ref_k premières fenêtres pour toute la cellule (publié) ; "
                         "causal : à la r-ième fenêtre, médiane des min(r, ref_k) premières, rien du futur")
    ap.add_argument("--score", choices=["full", "maha", "recon", "maha_shape"], default="full",
                    help="Composantes du score de décalage (ablation) ; full = score publié")
    ap.add_argument("--dump-cycles", nargs="*", default=None, metavar="EXPERT",
                    help="Écrit cycles.csv (erreur par cycle, rang du cycle dans la cellule, w, w cellule) "
                         "pour ces experts, ex. CNN+T CPMLP+T")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = load(args)
    if args.fallback == "selfref":
        # amplitude référencée à la cellule elle-même : médiane de ses ref_k premières fenêtres (ordre des cycles)
        cyc0 = D["cycle"] if "cycle" in D else np.arange(len(D["soh"]))
        ref = np.empty(len(D["dqn"]))
        for c in np.unique(D["groups"]):
            idx = np.flatnonzero(D["groups"] == c)
            order = idx[np.argsort(cyc0[idx], kind="stable")]
            if args.ref_mode == "fixed":
                ref[idx] = np.median(D["dqn"][order[: args.ref_k]])
            else:
                for j, i in enumerate(order):
                    ref[i] = np.median(D["dqn"][order[: min(j + 1, args.ref_k)]])
        D["dqn"] = D["dqn"] / ref
        print(f"  repli auto-référencé ({args.ref_mode}) : ΔQ/Q_nom divisé par la médiane des {args.ref_k} "
              f"premières fenêtres de chaque cellule (mesures de charge seulement)")
    y, g = D["soh"], D["groups"]
    cyc = D["cycle"] if "cycle" in D else np.arange(len(y))

    rows, gates, cells, dumps = [], [], [], []
    for proto in args.protocols:
        for i, (name, tr, te) in enumerate(splits(D, proto)):
            P = {}
            for d in glob.glob(os.path.join(args.lit_root, f"lit_{proto}_*")):
                if not re.fullmatch(rf"lit_{proto}_[A-Za-z]+", os.path.basename(d)):
                    continue
                for f in glob.glob(os.path.join(d, "preds", f"s{i:02d}__*.npy")):
                    P[label_of(os.path.basename(f)[5:-4])] = np.load(f)
            if "base_dQ+T" not in P:
                print(f"  {proto} / {name} : pas de prédictions en cache, ignoré"); continue
            bad = [k for k, v in P.items() if len(v) != len(te)]
            if bad:
                raise SystemExit(f"Plis incohérents avec le cache ({proto}, {name}, {bad}) : mêmes options que lit_baselines ?")
            if args.fallback == "selfref":
                T_ = D["T"]
                tmu, tsd = (T_[tr].mean(), 1.0) if len(set(T_[tr])) < 2 else (T_[tr].mean(), T_[tr].std() + 1e-9)
                from cond_models import design
                m_, _, _, Xte_ = ridge_T("dQ", D, tr, te, args, np.logspace(-3, 4, 15), tmu, tsd)
                P["base_dQ+T"] = m_.predict(design(Xte_, T_[te], "M1", tmu, tsd))
            base = P["base_dQ+T"]
            q = calibrate_gate(D, tr, args.n_pca, args.gate_q, weighting=args.calib, first_k=args.calib_k,
                               parts=args.score)
            Ztr, Zte, Rtr, Rte = pca_blocks(D, tr, te, args.n_pca)
            s_te = Gate(args.score).fit(Ztr, Rtr).score(Zte, Rte)
            w = weights(s_te, q, args.gate_k)
            wc = causal_cell_median(w, g[te], cyc[te])
            gates.append(dict(protocol=proto, split=name, w_mean=w.mean()))
            experts = [k for k in P if k not in ("base_dQ+T", "tout+T", "FUS_pondérée")]
            allp = dict(P)
            for e in experts:
                allp[f"GATE[{e}]"] = base + w * (P[e] - base)
                allp[f"GATEcell[{e}]"] = base + wc * (P[e] - base)
                # decision-rule ablation: hard switch with the same score and threshold
                #   SWITCH[X]     : expert if s <= q on this cycle, fallback otherwise
                #   SWITCHcell[X] : expert if the cell's running median of s is <= q (i.e. w_cell = 1)
                allp[f"SWITCH[{e}]"] = np.where(w >= 1 - 1e-9, P[e], base)
                allp[f"SWITCHcell[{e}]"] = np.where(wc >= 1 - 1e-9, P[e], base)
                # no-detection baseline: fixed blend α·expert + (1−α)·fallback, same α for every cell
                for a in FIXED_ALPHAS:
                    allp[f"FIX{a:.2f}[{e}]"] = base + a * (P[e] - base)
            for k, p in allp.items():
                cm = cell_mae(p - y[te], g[te])
                rows.append(dict(protocol=proto, split=name, method=k, MAE_cell=cm.mean()))
                for c, v in cm.items():
                    cells.append(dict(protocol=proto, split=name, method=k, cell=c, mae=v))
            if args.dump_cycles is not None:
                gt, ct = g[te], cyc[te]
                rank = pd.Series(ct).groupby(gt).rank(method="first").astype(int).values
                Dd = pd.DataFrame(dict(protocol=proto, split=name, cell=gt, cycle=ct, rank=rank, y=y[te],
                                       base=base, w=w, wc=wc, sq=s_te / q))
                for e in (args.dump_cycles or experts):
                    if e in P:
                        Dd[f"pred[{e}]"] = P[e]
                dumps.append(Dd)
            print(f"  {proto} / {name} : w moyen {w.mean():.2f}", flush=True)

    R = pd.DataFrame(rows).drop_duplicates(["protocol", "split", "method"])
    R.to_csv(os.path.join(args.out, "results.csv"), index=False)
    pd.DataFrame(gates).to_csv(os.path.join(args.out, "gate.csv"), index=False)
    S = R.groupby(["method", "protocol"]).MAE_cell.agg(["mean", "max"]).reset_index()
    S["txt"] = S.apply(lambda x: f"{x['mean']:.2f} [{x['max']:.2f}]", axis=1)
    Pt = S.pivot(index="method", columns="protocol", values="txt")
    cols = [p for p in PROTOS if p in Pt.columns]
    Mn = S.pivot(index="method", columns="protocol", values="mean")[cols]
    Mx = S.pivot(index="method", columns="protocol", values="max")[cols]
    Pt = Pt[cols]
    Pt["moyenne"] = Mn.mean(axis=1).round(2)
    Pt["pire (tous)"] = Mx.max(axis=1).round(2)
    Pt = Pt.loc[Mn.mean(axis=1).sort_values().index].rename(columns=NAMES)
    pd.set_option("display.width", 250)
    print("\nMAE moyenne par cellule sur les plis [pire pli] ; GATE[X] = base + w·(X − base) ; "
          "GATEcell = w médian causal par cellule")
    print(Pt.to_string())
    # gain de l'enveloppe pour chaque expert
    print("\n=== Effet de l'enveloppe : MAE(GATE[X]) − MAE(X), moyenne des plis (négatif = l'enveloppe améliore) ===")
    out = {}
    for e in sorted({m[5:-1] for m in Mn.index if m.startswith("GATE[")}):
        if e in Mn.index:
            out[e] = (Mn.loc[f"GATE[{e}]"] - Mn.loc[e]).round(2)
    print(pd.DataFrame(out).T.rename(columns=NAMES).to_string())
    pd.DataFrame(cells).to_csv(os.path.join(args.out, "cells.csv"), index=False)  # erreur par cellule
    if dumps:
        pd.concat(dumps).to_csv(os.path.join(args.out, "cycles.csv"), index=False)
    paired_tests(pd.DataFrame(cells), args.out)
    print(f"\nFichiers dans {args.out}/")


if __name__ == "__main__":
    main()
