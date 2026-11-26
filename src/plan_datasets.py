#!/usr/bin/env python3
"""
Sélection reproductible des sous-jeux BatteryLife et de leur fenêtre de tension,
à partir de l'inventaire produit par inspect_all.py (steps_all.csv, datasets.csv).

Pour chaque sous-jeu, on cherche l'étape de charge CC « dominante » (même C-rate dans la plupart
des cycles et des cellules), puis on applique des critères d'inclusion explicites :

  C1  au moins --min-cells cellules dans le sous-jeu
  C2  une étape CC présente dans au moins --min-freq des cycles échantillonnés
  C3  cette étape existe dans toutes les phases de vie (début, milieu, fin)
  C4  la fenêtre commune (couverte par ~90 % des étapes, fin de vie comprise) fait au moins --min-width V
  C5  résolution suffisante : au moins --min-points mesures réelles dans la fenêtre

Fenêtres proposées :
  full     = [p90 des tensions de début, p10 des tensions max] resserrée de --margin V
  partial  = la première moitié (en tension) de full : cas de charge partielle réaliste,
             proposée quand full couvre plus de --partial-above de la capacité (ΔQ ≈ capacité sinon)

Sorties : eligibility.csv (critères, fenêtres, raisons d'exclusion) et run_extraction.sh
(commandes identifiability_test.py pour tous les sous-jeux retenus).

  python plan_datasets.py --inv $R/inspect_all_big --data-root $P --results $R
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inv", required=True, help="Dossier de sortie de inspect_all.py")
    ap.add_argument("--data-root", default="$WORK/SODA_Battery/datasets/processed")
    ap.add_argument("--results", default="$WORK/Inversion/results")
    ap.add_argument("--min-cells", type=int, default=15)
    ap.add_argument("--min-freq", type=float, default=0.8)
    ap.add_argument("--min-width", type=float, default=0.08)
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--partial-above", type=float, default=0.6)
    ap.add_argument("--cycle-stride", type=int, default=10)
    args = ap.parse_args()

    S = pd.read_csv(os.path.join(args.inv, "steps_all.csv"))
    M = pd.read_csv(os.path.join(args.inv, "datasets.csv")).set_index("dataset")
    S["crate_bin"] = (np.round(S.crate / 0.05) * 0.05).round(2)

    rows, cmds = [], []
    for ds in M.index:
        info = M.loc[ds]
        r = dict(dataset=ds, n_cells=int(info.get("n_files", 0) or 0),
                 cathode=info.get("cathode", ""), nominal_Ah=info.get("nominal_Ah", ""))
        d = S[S.dataset == ds]
        reasons = []
        if r["n_cells"] < args.min_cells:
            reasons.append(f"C1 : {r['n_cells']} cellules < {args.min_cells}")
        if d.empty:
            reasons.append("aucune étape CC détectée")
            rows.append({**r, "eligible": False, "reasons": " ; ".join(reasons)}); continue

        n_cyc = d[["cell_id", "cycle_index"]].drop_duplicates().shape[0]
        best = None
        for b, g in d.groupby("crate_bin"):
            freq = g[["cell_id", "cycle_index"]].drop_duplicates().shape[0] / n_cyc
            # étape la plus longue en tension, la plus fréquente
            score = (freq, (g.v_max - g.v_min).median())
            if best is None or score > best[0]:
                best = (score, b, g, freq)
        _, crate, g, freq = best
        lo = g.v_min.quantile(0.9) + args.margin          # début de l'étape, fin de vie comprise
        hi = g.v_max.quantile(0.1) - args.margin - 0.005   # sous la coupure (phase CV exclue)
        width = hi - lo
        phases = set(g.cycle_pos)
        pts = float(np.median(g.n_points * np.clip(width, 0, None) / np.maximum(g.v_max - g.v_min, 1e-6)))
        cap_frac = float(np.median(g.dq_frac * np.clip(width, 0, None) / np.maximum(g.v_max - g.v_min, 1e-6)))
        r.update(crate=float(g.crate.median()), step_freq=round(freq, 2), phases="/".join(sorted(phases)),
                 v_low=round(lo, 3), v_high=round(hi, 3), width_V=round(width, 3),
                 points_in_window=round(pts), capacity_frac_in_window=round(cap_frac, 2),
                 dt_median_s=round(float(g.dt_median_s.median()), 1))
        if freq < args.min_freq:
            reasons.append(f"C2 : étape CC dans {freq:.0%} des cycles < {args.min_freq:.0%}")
        if not {"début", "milieu", "fin"} <= phases:
            reasons.append(f"C3 : étape absente en phase(s) {sorted({'début','milieu','fin'} - phases)}")
        if width < args.min_width:
            reasons.append(f"C4 : fenêtre commune {width:.3f} V < {args.min_width} V")
        if pts < args.min_points:
            reasons.append(f"C5 : ~{pts:.0f} points dans la fenêtre < {args.min_points} (dt ≈ {r['dt_median_s']} s)")
        r["eligible"] = not reasons
        r["reasons"] = " ; ".join(reasons)
        if r["eligible"]:
            wins = [("full", lo, hi)]
            if cap_frac > args.partial_above:
                wins.append(("part", lo, lo + 0.5 * width))
            r["windows"] = " ; ".join(f"{n}=[{a:.3f}, {b:.3f}]" for n, a, b in wins)
            for n, a, b in wins:
                cmds.append(f"python identifiability_test.py run --data-dir {args.data_root}/{ds} "
                            f"--v-low {a:.3f} --v-high {b:.3f} --cc-crate {r['crate']:.2f} --cc-tol 0.08 "
                            f"--cycle-stride {args.cycle_stride} --n-jobs 2 --out {args.results}/bench_{n}_{ds}")
        rows.append(r)

    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(args.inv, "eligibility.csv"), index=False)
    with open(os.path.join(args.inv, "run_extraction.sh"), "w") as f:
        f.write("#!/bin/bash\n# Généré par plan_datasets.py — un run par sous-jeu retenu et par fenêtre\nset -e\n")
        f.write("\n".join(cmds) + "\n")
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
    cols = ["dataset", "n_cells", "cathode", "crate", "step_freq", "v_low", "v_high", "width_V",
            "points_in_window", "capacity_frac_in_window", "eligible"]
    print(E[[c for c in cols if c in E]].to_string(index=False))
    print("\n=== Raisons d'exclusion ===")
    for _, x in E[~E.eligible].iterrows():
        print(f"  {x.dataset:11s} {x.reasons}")
    print(f"\n{E.eligible.sum()} sous-jeux retenus ; {len(cmds)} extractions dans {args.inv}/run_extraction.sh")


if __name__ == "__main__":
    main()
