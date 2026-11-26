#!/usr/bin/env python3
"""
Test d'identifiabilité du SOH à partir d'une fenêtre de charge CC (format BatteryLife / BatteryML).

Question posée : sur des cellules jamais vues, la FORME de la courbe de charge dans une
fenêtre de tension fixe contient-elle de l'information sur le SOH, au-delà de la simple
capacité chargée ΔQ dans cette fenêtre ?

Deux sous-commandes :

  1) inspect : résume les cellules (chimie, protocoles, nb de cycles, plage de tension des
     segments CC) pour choisir un sous-ensemble homogène et une fenêtre de tension.

     python identifiability_test.py inspect --data-dir ./MATR --out ./inspect_out

  2) run : extrait les fenêtres, construit les features et compare, en validation croisée
     groupée par cellule :
        - dQ        : ΔQ seul (amplitude)
        - shape     : PCA de Q(V)/ΔQ (forme, amplitude neutralisée)
        - ic        : PCA de dQ/dV normalisé (forme, amplitude neutralisée)
        - dQ+shape  : les deux
        - mean      : baseline constante (moyenne du train)

     python identifiability_test.py run --data-dir ./MATR --v-low 3.35 --v-high 3.55 \
         --cc-crate 1.0 --out ./ident_out

Sorties : results_folds.csv, results_summary.csv, extraction_log.csv, samples.npz et
figures (formes par SOH, PCA, ΔQ vs SOH, comparaison des erreurs).
"""
import argparse
import glob
import os
import pickle
import sys
from collections import Counter

import numpy as np

# fichiers annexes du pipeline (métadonnées, features…) à ignorer
EXCLUDE = (".meta.", ".features.", ".soh_raw.")
import pandas as pd

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def _as_dict(obj):
    """BatteryLife stocke des dicts ; BatteryML peut stocker des objets. On gère les deux."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return vars(obj)


def load_cell(path):
    with open(path, "rb") as f:
        cell = _as_dict(pickle.load(f))
    cycles = [_as_dict(c) for c in cell.get("cycle_data", [])]
    cell["cycle_data"] = cycles
    cell.setdefault("cell_id", os.path.splitext(os.path.basename(path))[0])
    return cell


def list_files(data_dir, pattern):
    files = sorted(f for f in glob.glob(os.path.join(data_dir, pattern), recursive=True)
                   if not any(x in os.path.basename(f) for x in EXCLUDE))
    if not files:
        sys.exit(f"Aucun fichier trouvé : {os.path.join(data_dir, pattern)}")
    return files


def arr(cyc, key):
    v = cyc.get(key)
    if v is None:
        return None
    a = np.asarray(v, dtype=float).ravel()
    return a if a.size else None


# ---------------------------------------------------------------------------
# Extraction du segment de charge CC
# ---------------------------------------------------------------------------

def contiguous_runs(mask):
    """Renvoie la liste (start, stop) des segments True contigus."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def cc_charge_segment(cyc, nominal, args):
    """Isole le segment de charge à courant constant visé.

    Si --cc-crate est donné : points dont |I|/Q_nom ≈ cc_crate (± tol).
    Sinon : segment le plus long où le courant de charge est constant (± tol relative).
    Parmi les segments candidats, on garde celui qui atteint la tension la plus haute
    (pour MATR : l'étape 1C finale, commune à tous les protocoles).
    """
    V = arr(cyc, "voltage_in_V")
    I = arr(cyc, "current_in_A")
    if V is None or I is None or len(V) != len(I) or len(V) < 10:
        return None, "missing_V_or_I"
    I = I * args.charge_sign  # charge > 0
    t = arr(cyc, "time_in_s")
    Qc = arr(cyc, "charge_capacity_in_Ah")

    if Qc is None or len(Qc) != len(V):
        if t is None or len(t) != len(V):
            return None, "no_capacity_no_time"
        Qc = np.concatenate([[0], np.cumsum(np.clip(I[1:], 0, None) * np.diff(t) / 3600.0)])

    charging = I > args.min_charge_current
    if not charging.any():
        return None, "no_charge"

    if args.cc_crate is not None:
        target = args.cc_crate * nominal
        mask = charging & (np.abs(I - target) <= args.cc_tol * target)
    else:
        ref = np.median(I[charging])
        mask = charging & (np.abs(I - ref) <= args.cc_tol * abs(ref))
        # on raffine en prenant le courant du plus long segment
        runs = contiguous_runs(mask)
        if runs:
            s, e = max(runs, key=lambda r: r[1] - r[0])
            ref = np.median(I[s:e])
            mask = charging & (np.abs(I - ref) <= args.cc_tol * abs(ref))

    # Exclure la phase CV : tension collée à la tension de coupure
    if args.exclude_cv:
        vcut = np.nanmax(V[charging])
        mask &= V < vcut - args.cv_margin

    runs = [r for r in contiguous_runs(mask) if r[1] - r[0] >= args.min_run_points]
    if not runs:
        return None, "no_cc_run"
    s, e = select_run(runs, V, Qc, args.run_select)
    return (V[s:e], Qc[s:e]), "ok"


def select_run(runs, V, Qc, how):
    """Choix du segment CC parmi les candidats.
    qspan : celui qui charge le plus de capacité (robuste, défaut)
    last  : le dernier avant la fin de charge
    maxv  : celui qui atteint la tension la plus haute (ancien comportement)
    """
    if how == "last":
        return runs[-1]
    if how == "maxv":
        return max(runs, key=lambda r: np.nanmax(V[r[0]:r[1]]))
    return max(runs, key=lambda r: np.nanmax(Qc[r[0]:r[1]]) - np.nanmin(Qc[r[0]:r[1]]))


def window_curve(Vseg, Qseg, v_grid):
    """Q(V) sur la grille de tension fixe ; renvoie (Q_rel, ΔQ) ou None."""
    ok = np.isfinite(Vseg) & np.isfinite(Qseg)
    Vseg, Qseg = Vseg[ok], Qseg[ok]
    if Vseg.size < 5:
        return None, "too_few_points"
    if Vseg.min() > v_grid[0] or Vseg.max() < v_grid[-1]:
        return None, "window_not_covered"
    # Rendre V strictement croissant (bruit de mesure) : on garde les nouveaux maxima
    keep = np.concatenate([[True], Vseg[1:] > np.maximum.accumulate(Vseg)[:-1]])
    Vm, Qm = Vseg[keep], Qseg[keep]
    if Vm.size < 5:
        return None, "non_monotonic"
    Q = np.interp(v_grid, Vm, Qm)
    Q = Q - Q[0]
    dQ = Q[-1]
    if not np.isfinite(dQ) or dQ <= 0:
        return None, "nonpositive_dQ"
    return Q, "ok"


def smooth_ic(Q, v_grid, window):
    """dQ/dV lissé (Savitzky-Golay) et normalisé (aire = 1 → amplitude neutralisée)."""
    from scipy.signal import savgol_filter
    dV = v_grid[1] - v_grid[0]
    w = max(5, window if window % 2 else window + 1)
    w = min(w, len(Q) if len(Q) % 2 else len(Q) - 1)
    ic = savgol_filter(Q, window_length=w, polyorder=2, deriv=1, delta=dV)
    ic = np.clip(ic, 0, None)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    area = trapz(ic, v_grid)
    return ic / area if area > 0 else None


# ---------------------------------------------------------------------------
# Labels SOH
# ---------------------------------------------------------------------------

def cycle_discharge_capacity(cyc, args):
    Qd = arr(cyc, "discharge_capacity_in_Ah")
    if Qd is not None:
        return np.nanmax(Qd)
    I, t = arr(cyc, "current_in_A"), arr(cyc, "time_in_s")
    if I is None or t is None or len(I) != len(t):
        return np.nan
    I = -I * args.charge_sign
    return np.sum(np.clip(I[1:], 0, None) * np.diff(t)) / 3600.0


def reference_capacity(caps, nominal, mode):
    caps = np.asarray(caps, float)
    if mode == "nominal":
        return nominal
    first = caps[np.isfinite(caps)][:10]
    return np.nanmedian(first) if first.size else np.nan


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in list_files(args.data_dir, args.pattern):
        try:
            cell = load_cell(path)
        except Exception as ex:
            rows.append({"file": os.path.basename(path), "error": repr(ex)})
            continue
        cycles = cell["cycle_data"]
        nominal = float(cell.get("nominal_capacity_in_Ah") or np.nan)
        vmins, vmaxs, qspans, temps = [], [], [], []
        sample = cycles[:: max(1, len(cycles) // 10)] if cycles else []
        for cyc in sample:
            seg, why = cc_charge_segment(cyc, nominal, args)
            if seg is not None:
                vmins.append(np.nanmin(seg[0]))
                vmaxs.append(np.nanmax(seg[0]))
                qspans.append(np.nanmax(seg[1]) - np.nanmin(seg[1]))
            T = arr(cyc, "temperature_in_C")
            if T is not None:
                temps.append(np.nanmedian(T))
        rows.append({
            "cell_id": cell.get("cell_id"),
            "file": os.path.basename(path),
            "cathode": cell.get("cathode_material"),
            "anode": cell.get("anode_material"),
            "form_factor": cell.get("form_factor"),
            "nominal_Ah": nominal,
            "n_cycles": len(cycles),
            "charge_protocol": str(cell.get("charge_protocol"))[:200],
            "discharge_protocol": str(cell.get("discharge_protocol"))[:200],
            "SOC_interval": str(cell.get("SOC_interval")),
            "temp_median_C": np.nanmedian(temps) if temps else np.nan,
            "cc_seg_Vmin_median": np.median(vmins) if vmins else np.nan,
            "cc_seg_Vmax_median": np.median(vmaxs) if vmaxs else np.nan,
            "cc_seg_Qspan_frac_nominal": (np.median(qspans) / nominal) if qspans else np.nan,
            "cc_seg_found_frac": len(vmins) / max(1, len(sample)),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(args.out, "cells_summary.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} cellules -> {out}")
    if "cathode" in df:
        print("\nCathodes :", dict(Counter(df["cathode"].astype(str))))
    if "cc_seg_Vmin_median" in df and df["cc_seg_Vmin_median"].notna().any():
        lo = df["cc_seg_Vmin_median"].quantile(0.9)
        hi = df["cc_seg_Vmax_median"].quantile(0.1)
        qf = df["cc_seg_Qspan_frac_nominal"].median()
        print(f"\nSegment CC retenu : capacité chargée médiane = {qf:.1%} de Q_nom")
        if hi - lo < 0.02:
            print("⚠ Fenêtre quasi nulle : le segment retenu n'est probablement pas l'étape CC "
                  "visée. Lance `diagnose` sur une cellule pour voir les étapes de charge.")
        else:
            print(f"Fenêtre de tension couverte par ~90 % des cellules : "
                  f"[{lo:.3f}, {hi:.3f}] V  (resserre-la un peu pour être sûr)")


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

def cmd_diagnose(args):
    """Décrit les étapes de charge de quelques cycles d'une cellule (texte + figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    files = list_files(args.data_dir, args.pattern)
    match = [f for f in files if args.cell in os.path.basename(f)] if args.cell else files[:1]
    if not match:
        sys.exit(f"Aucune cellule ne correspond à '{args.cell}'")
    cell = load_cell(match[0])
    cycles = cell["cycle_data"]
    nominal = float(cell.get("nominal_capacity_in_Ah") or np.nan)
    print(f"Cellule {cell['cell_id']} — {len(cycles)} cycles — Q_nom = {nominal} Ah")
    print(f"charge_protocol : {str(cell.get('charge_protocol'))[:300]}")
    print(f"Clés d'un cycle : {sorted(cycles[0].keys())}\n")

    idx = [int(i) for i in args.cycles.split(",")] if args.cycles else \
          [1, len(cycles) // 2, len(cycles) - 2]
    idx = [i for i in idx if 0 <= i < len(cycles)]
    fig, ax = plt.subplots(len(idx), 3, figsize=(14, 3.2 * len(idx)), squeeze=False)

    for row, k in enumerate(idx):
        cyc = cycles[k]
        V, I = arr(cyc, "voltage_in_V"), arr(cyc, "current_in_A") * args.charge_sign
        t, Qc = arr(cyc, "time_in_s"), arr(cyc, "charge_capacity_in_Ah")
        if t is None or len(t) != len(V):
            t = np.arange(len(V), dtype=float)
        c = I / nominal
        ch = I > args.min_charge_current
        print(f"--- cycle index {k} (cycle_number={cyc.get('cycle_number')}) : {len(V)} points, "
              f"pas de temps médian {np.median(np.diff(t)):.1f} s")
        if ch.any():
            print(f"    charge : V ∈ [{V[ch].min():.3f}, {V[ch].max():.3f}] V, "
                  f"C-rate percentiles 5/25/50/75/95 = {np.round(np.percentile(c[ch], [5,25,50,75,95]), 2)}")
            # étapes à C-rate ~constant (arrondi au 0.1C)
            lvl = np.round(c / 0.1) * 0.1
            runs = contiguous_runs(ch)
            for s, e in runs:
                seg_l = lvl[s:e]
                bounds = np.flatnonzero(np.diff(seg_l)) + 1
                for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(seg_l)]):
                    if b - a < args.min_run_points:
                        continue
                    sl = slice(s + a, s + b)
                    dq = (np.nanmax(Qc[sl]) - np.nanmin(Qc[sl])) if Qc is not None and len(Qc) == len(V) else np.nan
                    print(f"      étape {seg_l[a]:5.1f}C : {b - a:5d} pts, "
                          f"{t[sl][-1] - t[sl][0]:7.0f} s, V {V[sl].min():.3f}→{V[sl].max():.3f}, "
                          f"ΔQ {dq:.3f} Ah")
        seg, why = cc_charge_segment(cyc, nominal, args)
        if seg is not None:
            print(f"    → segment retenu ({args.run_select}) : {len(seg[0])} pts, "
                  f"V {seg[0].min():.3f}→{seg[0].max():.3f}, ΔQ {seg[1].max() - seg[1].min():.3f} Ah")
        else:
            print(f"    → aucun segment retenu ({why})")

        ax[row, 0].plot(t, V); ax[row, 0].set(ylabel="V", title=f"cycle {k}")
        ax[row, 1].plot(t, c); ax[row, 1].set(ylabel="I / Q_nom (C-rate)")
        if args.cc_crate:
            ax[row, 1].axhline(args.cc_crate, ls="--", color="k", lw=0.8)
        if Qc is not None and len(Qc) == len(V):
            ax[row, 2].plot(Qc[ch], V[ch], lw=0.8)
            if seg is not None:
                ax[row, 2].plot(seg[1], seg[0], color="r", lw=2, label="segment retenu")
                ax[row, 2].legend()
            ax[row, 2].set(xlabel="Q charge (Ah)", ylabel="V")
    for a in ax[-1, :2]:
        a.set_xlabel("temps (s)")
    out = os.path.join(args.out, f"diag_{cell['cell_id']}.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nFigure : {out}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def extract_dataset(args):
    v_grid = np.linspace(args.v_low, args.v_high, args.n_points)
    shapes, ics, dqs, sohs, groups, cycnums, protos, temps, qnoms = [], [], [], [], [], [], [], [], []
    log = []
    for path in list_files(args.data_dir, args.pattern):
        cell = load_cell(path)
        cid = str(cell["cell_id"])
        proto = str(cell.get("charge_protocol"))
        nominal = float(cell.get("nominal_capacity_in_Ah") or np.nan)
        cycles = cell["cycle_data"]
        caps = [cycle_discharge_capacity(c, args) for c in cycles]
        qref = reference_capacity(caps, nominal, args.soh_ref)
        reasons = Counter()
        kept = 0
        for k in range(args.skip_first, len(cycles), args.cycle_stride):
            cyc = cycles[k]
            soh = caps[k] / qref if qref and np.isfinite(qref) else np.nan
            if not np.isfinite(soh) or not (args.soh_min <= soh <= args.soh_max):
                reasons["soh_out_of_range"] += 1
                continue
            seg, why = cc_charge_segment(cyc, nominal, args)
            if seg is None:
                reasons[why] += 1
                continue
            Q, why = window_curve(seg[0], seg[1], v_grid)
            if Q is None:
                reasons[why] += 1
                continue
            ic = smooth_ic(Q, v_grid, args.ic_window)
            if ic is None:
                reasons["ic_failed"] += 1
                continue
            shapes.append(Q / Q[-1])
            # température mesurée pendant la charge (disponible en usage réel, contrairement au SOH)
            T_ = arr(cyc, "temperature_in_C"); I_ = arr(cyc, "current_in_A")
            if T_ is not None and I_ is not None and len(T_) == len(I_):
                chg = I_ * args.charge_sign > args.min_charge_current
                temps.append(float(np.nanmedian(T_[chg])) if chg.any() else np.nan)
            else:
                temps.append(np.nan)
            ics.append(ic)
            dqs.append(Q[-1])
            qnoms.append(nominal)
            sohs.append(soh)
            groups.append(cid)
            protos.append(proto)
            cycnums.append(cyc.get("cycle_number", k))
            kept += 1
        log.append({"cell_id": cid, "n_cycles": len(cycles), "kept": kept,
                    "Q_ref_Ah": qref, **dict(reasons)})
        print(f"  {cid:40s} gardés {kept:5d} / {len(cycles)}")
    if not sohs:
        sys.exit("Aucun échantillon extrait : vérifie --v-low/--v-high, --cc-crate et "
                 "--charge-sign (voir extraction_log.csv).")
    return dict(v_grid=v_grid, shape=np.array(shapes), ic=np.array(ics),
                dq=np.array(dqs)[:, None], soh=np.array(sohs),
                groups=np.array(groups), cycle=np.array(cycnums),
                protocol=np.array(protos), temp=np.array(temps, float),
                qnom=np.array(qnoms, float)), pd.DataFrame(log)


def build_features(name, D, tr, te, n_pca, with_temp=False, dq_norm=False):
    from sklearn.decomposition import PCA
    parts_tr, parts_te = [], []
    dq = D["dq"]
    if dq_norm:  # ΔQ en fraction de la capacité nominale (comparable entre cellules de Q_nom différentes)
        if "qnom" not in D or not np.isfinite(D["qnom"]).all():
            sys.exit("--dq-norm : capacité nominale absente du samples.npz (ré-extraire avec la nouvelle version).")
        dq = dq / D["qnom"][:, None]
    if with_temp:  # température mesurée, valeurs manquantes remplacées par la médiane d'entraînement
        t = D["temp"].astype(float).copy()
        med = np.nanmedian(t[tr]) if np.isfinite(t[tr]).any() else 25.0
        t[~np.isfinite(t)] = med
        parts_tr.append(t[tr, None]); parts_te.append(t[te, None])
    if "dQ" in name:
        parts_tr.append(dq[tr]); parts_te.append(dq[te])
    for key in ("shape", "ic"):
        if key in name.split("+"):
            p = PCA(n_components=min(n_pca, len(tr) - 1)).fit(D[key][tr])
            parts_tr.append(p.transform(D[key][tr])); parts_te.append(p.transform(D[key][te]))
    return np.hstack(parts_tr), np.hstack(parts_te)


def make_model(kind):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         min_samples_leaf=20, random_state=0)


def evaluate(D, args):
    from sklearn.model_selection import GroupKFold
    y = D["soh"] * 100.0
    if args.group_by == "protocol":
        g = D["protocol"]
    elif args.group_by == "regex":
        # condition extraite du nom de cellule (ex. température, formation, chimie) :
        # chaque pli laisse de côté TOUTES les cellules d'une ou plusieurs conditions
        import re
        rx = re.compile(args.group_regex)
        g = np.array([(rx.search(str(c)).group(1) if rx.search(str(c)) else "autre") for c in D["groups"]])
        vals, counts = np.unique(g, return_counts=True)
        print("  conditions :", dict(zip(vals, counts)))
    else:
        g = D["groups"]
    n_cells = len(np.unique(g))
    print(f"  groupes ({args.group_by}) : {n_cells}")
    k = min(args.folds, n_cells)
    if k < 2:
        sys.exit("Il faut au moins 2 cellules pour une validation groupée.")
    with_temp = bool(getattr(args, "with_temp", False))
    dq_norm = bool(getattr(args, "dq_norm", False))
    if getattr(args, "temp_regex", None):
        # température de consigne lue dans le nom de cellule (condition d'essai connue en usage)
        import re
        rx = re.compile(args.temp_regex)
        D = dict(D)
        D["temp"] = np.array([float(rx.search(str(c)).group(1)) if rx.search(str(c)) else np.nan
                              for c in D["groups"]])
        print(f"  température lue dans les noms : {sorted(set(D['temp'][np.isfinite(D['temp'])]))}")
    if with_temp:
        if "temp" not in D or not np.isfinite(D["temp"]).any():
            sys.exit("--with-temp : pas de température dans ce samples.npz (ré-extraire avec la nouvelle version).")
        print(f"  température mesurée : {np.nanmin(D['temp']):.1f} à {np.nanmax(D['temp']):.1f} °C "
              f"({np.mean(~np.isfinite(D['temp'])):.0%} manquantes)")
    feats = (["T"] if with_temp else []) + ["dQ", "shape", "ic", "dQ+shape", "dQ+ic"]
    models = ["ridge"] + (["hgb"] if not args.no_hgb else [])
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(y, y, g)):
        err = y[te] - y[tr].mean()
        cond = ",".join(sorted(map(str, np.unique(g[te])))) if args.group_by == "regex" else str(fold)
        rows.append({"fold": fold, "features": "mean", "model": "-", "test_cond": cond,
                     "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                     "n_test_cells": len(np.unique(D["groups"][te]))})
        for f in feats:
            if f == "T":
                Xtr, Xte = build_features("", D, tr, te, args.n_pca, with_temp=True)
            else:
                Xtr, Xte = build_features(f, D, tr, te, args.n_pca, with_temp=with_temp, dq_norm=dq_norm)
            for m in models:
                pred = make_model(m).fit(Xtr, y[tr]).predict(Xte)
                err = y[te] - pred
                # MAE moyenne PAR CELLULE + IC bootstrap sur les cellules (les cycles d'une cellule
                # ne sont pas indépendants)
                cm = pd.Series(np.abs(err)).groupby(D["groups"][te]).mean().values
                rng = np.random.default_rng(fold)
                bs = rng.choice(cm, size=(1000, len(cm)), replace=True).mean(1)
                rows.append({"fold": fold, "features": f + ("+T" if with_temp and f != "T" else ""),
                             "model": m, "test_cond": cond,
                             "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                             "MAE_cell": cm.mean(), "ci_lo": np.percentile(bs, 2.5),
                             "ci_hi": np.percentile(bs, 97.5),
                             "n_test_cells": len(np.unique(D["groups"][te]))})
        print(f"  pli {fold + 1}/{k} terminé")
    res = pd.DataFrame(rows)
    summ = (res.groupby(["features", "model"])[["MAE", "RMSE"]]
               .agg(["mean", "std"]).round(3).sort_values(("MAE", "mean")))
    if args.group_by == "regex":
        # détail par condition laissée de côté (meilleur des modèles pour chaque jeu de features)
        piv = res.groupby(["test_cond", "features"]).MAE.min().unstack().round(2)
        n = res.groupby("test_cond").n_test_cells.first()
        piv.insert(0, "n_cellules_test", n)
        print("\n=== MAE par condition laissée de côté (meilleur modèle par jeu de features) ===")
        print(piv.to_string())
        # même famille de modèles pour toutes les représentations, avec IC bootstrap par cellules
        for m in models:
            r = res[res.model == m].copy()
            r["cell"] = r.apply(lambda x: f"{x.MAE_cell:.2f} [{x.ci_lo:.2f}-{x.ci_hi:.2f}]", axis=1)
            print(f"\n--- {m} uniquement : MAE moyenne par cellule [IC 95 % bootstrap de cellules] ---")
            print(r.pivot(index="test_cond", columns="features", values="cell").to_string())
    return res, summ


def make_figures(D, res, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    v, soh = D["v_grid"], D["soh"] * 100
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(soh.min(), soh.max())

    # 1. Formes moyennes par tranche de SOH
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(soh.min(), soh.max(), 9)
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (soh >= lo) & (soh <= hi)
        if sel.sum() < 3:
            continue
        c = cmap(norm((lo + hi) / 2))
        ax[0].plot(v, D["shape"][sel].mean(0), color=c)
        ax[1].plot(v, D["ic"][sel].mean(0), color=c)
    ax[0].set(xlabel="Tension (V)", ylabel="Q(V) / ΔQ", title="Forme normalisée")
    ax[1].set(xlabel="Tension (V)", ylabel="dQ/dV normalisé", title="Capacité incrémentale")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="SOH (%)")
    fig.savefig(os.path.join(args.out, "fig_shapes_by_soh.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. PCA (ajustée sur tout le jeu, uniquement pour la visualisation)
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for row, key in enumerate(("shape", "ic")):
        p = PCA(n_components=min(5, D[key].shape[1])).fit(D[key])
        Z = p.transform(D[key])
        sc = ax[row, 0].scatter(Z[:, 0], Z[:, 1], c=soh, s=4, cmap=cmap)
        ax[row, 0].set(xlabel="PC1", ylabel="PC2",
                       title=f"{key} — var. expl. {np.round(p.explained_variance_ratio_[:3], 3)}")
        ax[row, 1].scatter(Z[:, 0], soh, s=4, c=pd.factorize(D["groups"])[0], cmap="tab20")
        ax[row, 1].set(xlabel="PC1", ylabel="SOH (%)", title=f"{key} : PC1 vs SOH (couleur = cellule)")
    fig.colorbar(sc, ax=ax[:, 0], label="SOH (%)")
    fig.savefig(os.path.join(args.out, "fig_pca.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. ΔQ vs SOH
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(D["dq"][:, 0], soh, s=4, c=pd.factorize(D["groups"])[0], cmap="tab20")
    ax.set(xlabel=f"ΔQ dans [{args.v_low}, {args.v_high}] V (Ah)", ylabel="SOH (%)",
           title="Amplitude seule (couleur = cellule)")
    fig.savefig(os.path.join(args.out, "fig_dq_vs_soh.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Comparaison des erreurs
    agg = res.groupby(["features", "model"])["MAE"].agg(["mean", "std"]).reset_index()
    agg["label"] = agg["features"] + (" / " + agg["model"]).where(agg["model"] != "-", "")
    agg = agg.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(agg) + 1.5))
    ax.barh(agg["label"], agg["mean"], xerr=agg["std"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set(xlabel="MAE SOH (points de %) — validation groupée par cellule",
           title="Identifiabilité : amplitude vs forme")
    fig.savefig(os.path.join(args.out, "fig_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def cmd_run(args):
    os.makedirs(args.out, exist_ok=True)
    if args.from_npz:
        print(f"Chargement des échantillons déjà extraits : {args.from_npz}")
        D = {k: v for k, v in np.load(args.from_npz, allow_pickle=False).items()}
        if args.include_regex:
            import re
            rx = re.compile(args.include_regex)
            keep = np.array([bool(rx.search(str(c))) for c in D["groups"]])
            D = {k: (v[keep] if (k != "v_grid" and np.ndim(v) and len(v) == len(keep)) else v) for k, v in D.items()}
            print(f"  filtre --include-regex : {keep.sum()} échantillons, {len(np.unique(D['groups']))} cellules gardés")
        if "protocol" not in D:
            D["protocol"] = D["groups"]
        if args.v_low is None:
            args.v_low, args.v_high = float(D["v_grid"][0]), float(D["v_grid"][-1])
        log = pd.DataFrame()
    else:
        print("Extraction des fenêtres de charge…")
        D, log = extract_dataset(args)
    log.to_csv(os.path.join(args.out, "extraction_log.csv"), index=False)
    smin = pd.Series(D["soh"]).groupby(D["groups"]).min()
    print(f"SOH minimal atteint par cellule : médiane {smin.median()*100:.1f} %, "
          f"{(smin < 0.85).mean():.0%} des cellules descendent sous 85 %")
    if len(log) and "window_not_covered" in log:
        print(f"Cycles rejetés car fenêtre non couverte : {int(log['window_not_covered'].fillna(0).sum())}")
    if not args.from_npz:
        np.savez_compressed(os.path.join(args.out, "samples.npz"), **D)
    n_cells = len(np.unique(D["groups"]))
    print(f"\n{len(D['soh'])} échantillons, {n_cells} cellules, "
          f"SOH ∈ [{D['soh'].min()*100:.1f}, {D['soh'].max()*100:.1f}] %")

    print("\nValidation croisée groupée par cellule…")
    res, summ = evaluate(D, args)
    res.to_csv(os.path.join(args.out, "results_folds.csv"), index=False)
    summ.to_csv(os.path.join(args.out, "results_summary.csv"))
    make_figures(D, res, args)

    print("\n=== MAE / RMSE du SOH (points de %) ===")
    print(summ.to_string())
    r2 = res.assign(features=res.features.str.replace(r"\+T$", "", regex=True))
    best = r2[r2.features != "T"].groupby(["features", "model"])["MAE"].mean()
    dq = best.loc["dQ"].min()
    shape = best.loc[[i for i in best.index if i[0] in ("shape", "ic")]].min()
    comb = best.loc[[i for i in best.index if "+" in i[0]]].min()
    mean = best.loc[("mean", "-")]
    print(f"\nRéférence constante : {mean:.2f} | ΔQ seul : {dq:.2f} | "
          f"forme seule : {shape:.2f} | ΔQ + forme : {comb:.2f}")
    if shape > 0.8 * mean:
        print("→ La forme seule apporte peu d'information sur ce sous-ensemble : "
              "l'inversion de la forme est mal posée ici.")
    if comb > 0.9 * dq:
        print("→ La forme n'améliore pas nettement ΔQ : le gain d'un modèle génératif "
              "devra venir de l'incertitude/OOD, pas de la précision.")
    print(f"\nRésultats et figures dans {args.out}")


# ---------------------------------------------------------------------------

def main():
    if "--n-jobs" in sys.argv:
        n = sys.argv[sys.argv.index("--n-jobs") + 1]
        os.environ["OMP_NUM_THREADS"] = n  # avant l'import de scikit-learn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data-dir", required=True, help="Dossier contenant les .pkl d'un sous-jeu")
        p.add_argument("--pattern", default="**/*.pkl")
        p.add_argument("--out", default="./ident_out")
        p.add_argument("--charge-sign", type=float, default=1.0,
                       help="+1 si le courant de charge est positif (convention BatteryML), -1 sinon")
        p.add_argument("--cc-crate", type=float, default=None,
                       help="C-rate de l'étape CC visée (ex. 1.0 pour MATR) ; sinon détection auto")
        p.add_argument("--cc-tol", type=float, default=0.05, help="Tolérance relative sur le courant CC")
        p.add_argument("--min-charge-current", type=float, default=1e-3)
        p.add_argument("--run-select", choices=["qspan", "last", "maxv"], default="qspan",
                       help="Règle de choix du segment CC (défaut : celui qui charge le plus)")
        p.add_argument("--exclude-cv", action=argparse.BooleanOptionalAction, default=True,
                       help="Exclure les points collés à la tension de coupure (phase CV)")
        p.add_argument("--cv-margin", type=float, default=0.005, help="Marge (V) sous la coupure")
        p.add_argument("--min-run-points", type=int, default=5)

    pi = sub.add_parser("inspect", help="Résumé des cellules pour choisir sous-ensemble et fenêtre")
    common(pi)

    pd_ = sub.add_parser("diagnose", help="Détail des étapes de charge d'une cellule")
    common(pd_)
    pd_.add_argument("--cell", default=None, help="Sous-chaîne du nom de fichier (ex. b1c0)")
    pd_.add_argument("--cycles", default=None, help="Indices de cycles, ex. 1,200,500")

    pr = sub.add_parser("run", help="Extraction + test d'identifiabilité")
    common(pr)
    pr.add_argument("--v-low", type=float, default=None)
    pr.add_argument("--v-high", type=float, default=None)
    pr.add_argument("--from-npz", default=None,
                    help="Sauter l'extraction : réutiliser le samples.npz d'un run précédent")
    pr.add_argument("--n-points", type=int, default=128)
    pr.add_argument("--ic-window", type=int, default=15, help="Fenêtre Savitzky-Golay pour dQ/dV")
    pr.add_argument("--soh-ref", choices=["first", "nominal"], default="first",
                    help="Q_ref = médiane des 10 premiers cycles ('first') ou capacité nominale")
    pr.add_argument("--soh-min", type=float, default=0.5)
    pr.add_argument("--soh-max", type=float, default=1.1)
    pr.add_argument("--skip-first", type=int, default=1, help="Cycles initiaux ignorés (formation)")
    pr.add_argument("--cycle-stride", type=int, default=1, help="Sous-échantillonnage des cycles")
    pr.add_argument("--n-pca", type=int, default=8)
    pr.add_argument("--n-jobs", type=int, default=None,
                    help="Threads pour le gradient boosting (ex. 2 sur un frontal partagé)")
    pr.add_argument("--folds", type=int, default=5)
    pr.add_argument("--no-hgb", action="store_true", help="Ne garder que la régression ridge")
    pr.add_argument("--include-regex", default=None,
                    help="Avec --from-npz : ne garder que les cellules dont le nom correspond (ex. '^Tongji[12]')")
    pr.add_argument("--temp-regex", default=None,
                    help="Température prise dans le nom de cellule (1er groupe capturé), ex. 'CY(\\d+)-' ; "
                         "remplace la température mesurée")
    pr.add_argument("--dq-norm", action="store_true",
                    help="ΔQ divisé par la capacité nominale (indispensable entre chimies de Q_nom différentes)")
    pr.add_argument("--with-temp", action="store_true",
                    help="Ajouter la température mesurée pendant la charge à toutes les features (+ baseline T seule)")
    pr.add_argument("--group-regex", default=None,
                    help="Avec --group-by regex : motif dont le 1er groupe capturé définit la condition, "
                         "ex. '_(\\d+C)_' pour la température")
    pr.add_argument("--group-by", choices=["cell", "protocol", "regex"], default="cell",
                    help="Unité de la validation croisée : cellule ou protocole de charge")

    args = ap.parse_args()
    if args.cmd == "inspect":
        cmd_inspect(args)
    elif args.cmd == "diagnose":
        cmd_diagnose(args)
    else:
        if args.from_npz is None and (args.v_low is None or args.v_high is None):
            sys.exit("--v-low et --v-high sont requis (sauf avec --from-npz)")
        if args.v_low is not None and args.v_high <= args.v_low:
            sys.exit("--v-high doit être > --v-low")
        cmd_run(args)


if __name__ == "__main__":
    main()
