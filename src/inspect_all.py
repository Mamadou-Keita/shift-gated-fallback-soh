#!/usr/bin/env python3
"""
Inventaire des étapes de charge à courant constant (CC) dans tous les sous-jeux BatteryLife.

Pour chaque sous-jeu (un dossier par sous-jeu sous --root) :
  - échantillonne --n-cells cellules et --n-cycles cycles par cellule (début, milieu, fin) ;
  - détecte la convention de signe du courant (charge > 0 ou < 0) ;
  - découpe la charge en étapes à C-rate constant, en écartant la phase CV ;
  - regroupe les étapes par C-rate et résume : fréquence, plage de tension, ΔQ/Q_nom, nb de points ;
  - teste si la fenêtre [--v-low, --v-high] est couverte par une étape CC, et à quel C-rate.

    python inspect_all.py --root $WORK/SODA_Battery/datasets/soh_raw \
        --v-low 3.42 --v-high 3.58 --out inspect_all_out

Sorties :
  steps_all.csv     une ligne par étape CC détectée (cellule, cycle, C-rate, V début/fin, ΔQ…)
  datasets.csv      une ligne par sous-jeu (chimie, Q_nom, signe, couverture de la fenêtre…)
  step_types.csv    étapes CC typiques par sous-jeu, regroupées par C-rate
"""
import argparse
import glob
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

EXCLUDE = ('.meta.', '.features.', '.soh_raw.')
import pandas as pd


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _as_dict(obj):
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
    cell["cycle_data"] = [_as_dict(c) for c in cell.get("cycle_data", []) or []]
    cell.setdefault("cell_id", os.path.splitext(os.path.basename(path))[0])
    return cell


def arr(cyc, key):
    v = cyc.get(key)
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return a if a.size else None


def contiguous_runs(mask):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def sample_indices(n, k):
    """k indices répartis sur [1, n-2] (on évite le 1er et le dernier cycle)."""
    if n <= 2:
        return list(range(n))
    return sorted(set(np.linspace(1, n - 2, k).round().astype(int).tolist()))


def to_scalar(x):
    if isinstance(x, (list, tuple)):
        return "/".join(map(str, x))
    return x


# ---------------------------------------------------------------------------
# Signe du courant
# ---------------------------------------------------------------------------

def detect_charge_sign(cycles):
    """+1 si la charge a un courant positif, -1 sinon.

    1) corrélation entre I et l'accroissement de charge_capacity_in_Ah ;
    2) à défaut, signe du courant quand la tension monte.
    """
    votes = []
    for cyc in cycles:
        I, V, Qc = arr(cyc, "current_in_A"), arr(cyc, "voltage_in_V"), arr(cyc, "charge_capacity_in_Ah")
        if I is None or V is None or len(I) != len(V) or len(I) < 10:
            continue
        if Qc is not None and len(Qc) == len(I):
            dq = np.diff(Qc)
            inc = dq > 1e-9
            if inc.sum() > 5:
                votes.append(np.sign(np.median(I[1:][inc])))
                continue
        dv = np.diff(V)
        up = dv > 0
        if up.sum() > 5:
            votes.append(np.sign(np.median(I[1:][up])))
    votes = [v for v in votes if v != 0]
    if not votes:
        return 1.0, "unknown"
    s = np.sign(np.sum(votes))
    return (s if s != 0 else 1.0), f"{int(np.mean(np.array(votes) == s) * 100)}%"


# ---------------------------------------------------------------------------
# Découpage de la charge en étapes CC
# ---------------------------------------------------------------------------

def cc_steps(cyc, nominal, sign, args):
    V, I = arr(cyc, "voltage_in_V"), arr(cyc, "current_in_A")
    if V is None or I is None or len(V) != len(I) or len(V) < 10:
        return [], "missing_V_or_I"
    I = I * sign
    t = arr(cyc, "time_in_s")
    if t is None or len(t) != len(V):
        t = np.full(len(V), np.nan)
    Qc = arr(cyc, "charge_capacity_in_Ah")
    if Qc is None or len(Qc) != len(V):
        if np.isfinite(t).all():
            Qc = np.concatenate([[0], np.cumsum(np.clip(I[1:], 0, None) * np.diff(t) / 3600.0)])
        else:
            Qc = np.full(len(V), np.nan)

    c = I / nominal
    ch = c > args.min_crate
    if not ch.any():
        return [], "no_charge"
    vcut = np.nanmax(V[ch])

    # niveaux de C-rate sur échelle logarithmique (tolérance relative)
    lvl = np.full(len(c), -10**9)
    lvl[ch] = np.round(np.log(c[ch]) / np.log(1 + args.crate_tol)).astype(int)

    steps = []
    for s0, e0 in contiguous_runs(ch):
        seg = lvl[s0:e0]
        bounds = np.flatnonzero(np.diff(seg)) + 1
        for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(seg)]):
            s, e = s0 + a, s0 + b
            if e - s < args.min_points:
                continue
            v = V[s:e]
            vrange = np.nanmax(v) - np.nanmin(v)
            # phase CV : tension quasi constante au voisinage de la coupure
            if vrange < args.min_vrange:
                continue
            dq = np.nanmax(Qc[s:e]) - np.nanmin(Qc[s:e])
            steps.append(dict(
                crate=float(np.median(c[s:e])),
                v_start=float(v[0]), v_end=float(v[-1]),
                v_min=float(np.nanmin(v)), v_max=float(np.nanmax(v)),
                dq_frac=float(dq / nominal) if np.isfinite(dq) else np.nan,
                n_points=int(e - s),
                duration_s=float(t[e - 1] - t[s]) if np.isfinite(t[[s, e - 1]]).all() else np.nan,
                dt_median_s=float(np.nanmedian(np.diff(t[s:e]))) if np.isfinite(t[s:e]).sum() > 2 else np.nan,
                ends_at_cutoff=bool(np.nanmax(v) >= vcut - args.cv_margin),
            ))
    # fusion des étapes adjacentes de même niveau séparées par un trou court : pas nécessaire ici
    return steps, "ok" if steps else "no_cc_step"


# ---------------------------------------------------------------------------
# Inventaire d'un sous-jeu
# ---------------------------------------------------------------------------

def inspect_dataset(name, files, args, rng):
    picked = sorted(rng.choice(files, size=min(args.n_cells, len(files)), replace=False))
    meta, rows, reasons = [], [], Counter()
    sign, sign_conf = None, None
    for path in picked:
        try:
            cell = load_cell(path)
        except Exception as ex:
            reasons[f"load_error:{type(ex).__name__}:{str(ex)[:60]}"] += 1
            continue
        cycles = cell["cycle_data"]
        nominal = cell.get("nominal_capacity_in_Ah")
        try:
            nominal = float(nominal)
        except (TypeError, ValueError):
            nominal = np.nan
        if not np.isfinite(nominal) or nominal <= 0:
            # repli : capacité de décharge médiane des premiers cycles
            caps = [np.nanmax(arr(c, "discharge_capacity_in_Ah"))
                    for c in cycles[:10] if arr(c, "discharge_capacity_in_Ah") is not None]
            nominal = float(np.nanmedian(caps)) if caps else np.nan
            reasons["nominal_from_data"] += 1
        if not np.isfinite(nominal) or not cycles:
            reasons["no_nominal_or_cycles"] += 1
            continue
        if sign is None:
            sign, sign_conf = detect_charge_sign(cycles[:: max(1, len(cycles) // 20)][:20])
        meta.append(dict(
            cell_id=str(cell["cell_id"]), n_cycles=len(cycles), nominal_Ah=nominal,
            cathode=to_scalar(cell.get("cathode_material")),
            anode=to_scalar(cell.get("anode_material")),
            form_factor=to_scalar(cell.get("form_factor")),
            vmin_limit=cell.get("min_voltage_limit_in_V"),
            vmax_limit=cell.get("max_voltage_limit_in_V"),
        ))
        for k in sample_indices(len(cycles), args.n_cycles):
            steps, why = cc_steps(cycles[k], nominal, sign, args)
            reasons[why] += 1
            pos = "début" if k < len(cycles) / 3 else ("milieu" if k < 2 * len(cycles) / 3 else "fin")
            for j, st in enumerate(steps):
                rows.append(dict(dataset=name, cell_id=str(cell["cell_id"]), cycle_index=k,
                                 cycle_pos=pos, step_order=j, **st))
    return pd.DataFrame(meta), pd.DataFrame(rows), sign, sign_conf, reasons


def summarize_dataset(name, meta, steps, sign, sign_conf, reasons, n_files, args):
    out = dict(dataset=name, n_files=n_files, n_cells_sampled=len(meta),
               charge_sign=sign, sign_agreement=sign_conf)
    if len(meta):
        out.update(
            cathode=", ".join(sorted(set(map(str, meta["cathode"])))),
            nominal_Ah=", ".join(f"{x:g}" for x in sorted(set(meta["nominal_Ah"].round(3)))),
            n_cycles_median=int(meta["n_cycles"].median()),
            vmax_limit=", ".join(sorted(set(map(str, meta["vmax_limit"])))),
        )
    n_cyc = sum(v for k, v in reasons.items() if k in ("ok", "no_cc_step", "no_charge", "missing_V_or_I"))
    out["cycles_sampled"] = n_cyc
    out["cycles_with_cc"] = reasons.get("ok", 0)
    out["issues"] = "; ".join(f"{k}={v}" for k, v in reasons.items() if k != "ok") or ""

    if len(steps):
        cov = steps[(steps.v_min <= args.v_low) & (steps.v_max >= args.v_high)]
        cyc_ids = steps[["cell_id", "cycle_index"]].drop_duplicates().shape[0]
        cov_ids = cov[["cell_id", "cycle_index"]].drop_duplicates().shape[0]
        out["window_coverage"] = round(cov_ids / max(1, cyc_ids), 2)
        out["window_cells_covered"] = f"{cov['cell_id'].nunique()}/{len(meta)}"
        if len(cov):
            out["window_crates"] = ", ".join(f"{x:.2f}" for x in sorted(set(cov.crate.round(2))))[:80]
            out["window_points_median"] = int(
                np.median(cov.n_points * (args.v_high - args.v_low) /
                          np.maximum(cov.v_max - cov.v_min, 1e-6)))
            out["window_cov_end_of_life"] = round(
                cov[cov.cycle_pos == "fin"][["cell_id", "cycle_index"]].drop_duplicates().shape[0] /
                max(1, steps[steps.cycle_pos == "fin"][["cell_id", "cycle_index"]]
                    .drop_duplicates().shape[0]), 2)
    else:
        out["window_coverage"] = 0.0
    return out


def step_types(steps, n_cycles_with_steps):
    """Regroupe les étapes par C-rate arrondi : quelles étapes reviennent dans la plupart des cycles ?"""
    if steps.empty:
        return pd.DataFrame()
    s = steps.copy()
    s["crate_bin"] = (np.round(s.crate / 0.05) * 0.05).round(2)
    g = s.groupby(["dataset", "crate_bin"])
    t = g.agg(
        n_steps=("crate", "size"),
        n_cells=("cell_id", "nunique"),
        v_start_med=("v_start", "median"), v_end_med=("v_end", "median"),
        v_min_p10=("v_min", lambda x: np.percentile(x, 10)),
        v_max_p90=("v_max", lambda x: np.percentile(x, 90)),
        dq_frac_med=("dq_frac", "median"),
        n_points_med=("n_points", "median"),
        dt_med_s=("dt_median_s", "median"),
        ends_at_cutoff=("ends_at_cutoff", "mean"),
    ).reset_index()
    t["freq_in_cycles"] = [
        round(s[(s.dataset == d) & (s.crate_bin == b)][["cell_id", "cycle_index"]]
              .drop_duplicates().shape[0] / max(1, n_cycles_with_steps.get(d, 1)), 2)
        for d, b in zip(t.dataset, t.crate_bin)]
    return t.sort_values(["dataset", "freq_in_cycles", "n_steps"], ascending=[True, False, False])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Dossier contenant un sous-dossier par sous-jeu")
    ap.add_argument("--datasets", default=None, help="Liste séparée par des virgules (défaut : tous)")
    ap.add_argument("--pattern", default="**/*.pkl")
    ap.add_argument("--out", default="inspect_all_out")
    ap.add_argument("--n-cells", type=int, default=8)
    ap.add_argument("--n-cycles", type=int, default=6)
    ap.add_argument("--v-low", type=float, default=3.42)
    ap.add_argument("--v-high", type=float, default=3.58)
    ap.add_argument("--crate-tol", type=float, default=0.06, help="Tolérance relative sur le courant CC")
    ap.add_argument("--min-crate", type=float, default=0.02)
    ap.add_argument("--min-points", type=int, default=8)
    ap.add_argument("--min-vrange", type=float, default=0.01, help="V ; en dessous = phase CV, ignorée")
    ap.add_argument("--cv-margin", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names = sorted(d for d in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, d)) and not d.startswith(("_", ".")))
    if args.datasets:
        keep = set(args.datasets.split(","))
        names = [n for n in names if n in keep]
    if not names:
        sys.exit(f"Aucun sous-jeu trouvé dans {args.root}")

    all_steps, summaries = [], []
    for name in names:
        t0 = time.time()
        files = sorted(f for f in glob.glob(os.path.join(args.root, name, args.pattern), recursive=True)
                       if not any(x in os.path.basename(f) for x in EXCLUDE))
        if not files:
            print(f"[{name}] aucun fichier {args.pattern}")
            summaries.append(dict(dataset=name, n_files=0, issues="no_files"))
            continue
        meta, steps, sign, conf, reasons = inspect_dataset(name, files, args, rng)
        summaries.append(summarize_dataset(name, meta, steps, sign, conf, reasons, len(files), args))
        all_steps.append(steps)
        s = summaries[-1]
        if sign is None:
            print(f"[{name:11s}] {len(files):4d} fichiers | AUCUNE cellule exploitable : {s.get('issues')}")
            continue
        print(f"[{name:11s}] {len(files):4d} fichiers | signe {sign:+.0f} ({conf}) | "
              f"cycles avec CC {s.get('cycles_with_cc', 0)}/{s.get('cycles_sampled', 0)} | "
              f"fenêtre couverte {s.get('window_coverage', 0):.0%} "
              f"(C-rate {s.get('window_crates', '-')}) | {time.time() - t0:.0f} s")

    steps = pd.concat([s for s in all_steps if len(s)], ignore_index=True) if any(len(s) for s in all_steps) \
        else pd.DataFrame()
    summ = pd.DataFrame(summaries)
    steps.to_csv(os.path.join(args.out, "steps_all.csv"), index=False)
    summ.to_csv(os.path.join(args.out, "datasets.csv"), index=False)

    n_cyc = steps.groupby("dataset").apply(
        lambda d: d[["cell_id", "cycle_index"]].drop_duplicates().shape[0]).to_dict() if len(steps) else {}
    types = step_types(steps, n_cyc)
    types.to_csv(os.path.join(args.out, "step_types.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 40)
    cols = [c for c in ["dataset", "n_files", "cathode", "nominal_Ah", "charge_sign", "cycles_with_cc",
                        "cycles_sampled", "window_coverage", "window_cells_covered", "window_crates",
                        "window_points_median", "window_cov_end_of_life"] if c in summ]
    print(f"\n=== Couverture de la fenêtre [{args.v_low}, {args.v_high}] V par une étape CC de charge ===")
    print(summ[cols].to_string(index=False))

    if len(types):
        print("\n=== Étapes CC présentes dans ≥ 50 % des cycles échantillonnés ===")
        tc = ["dataset", "crate_bin", "freq_in_cycles", "n_cells", "v_start_med", "v_end_med",
              "dq_frac_med", "n_points_med", "dt_med_s", "ends_at_cutoff"]
        print(types[types.freq_in_cycles >= 0.5][tc].round(3).to_string(index=False))
    print(f"\nFichiers écrits dans {args.out}/ (datasets.csv, step_types.csv, steps_all.csv)")


if __name__ == "__main__":
    main()
