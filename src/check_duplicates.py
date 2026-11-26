#!/usr/bin/env python3
"""
Détecte les cellules présentes en double dans les sous-jeux BatteryLife (même nom de cellule,
ou même contenu sous un autre nom), pour éviter qu'une même cellule soit à la fois en
entraînement et en test.

Empreinte de contenu = hachage des tensions arrondies (mV) des 3 premiers cycles non vides.

  python check_duplicates.py --root $WORK/SODA_Battery/datasets/processed --out dup_report.csv
"""
import argparse
import glob
import hashlib
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

EXCLUDE = (".meta.", ".features.", ".soh_raw.")


def fingerprint(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    d = d if isinstance(d, dict) else vars(d)
    cyc = d.get("cycle_data", []) or []
    h = hashlib.sha1()
    n = 0
    for c in cyc:
        c = c if isinstance(c, dict) else vars(c)
        v = c.get("voltage_in_V")
        if v is None or len(v) < 10:
            continue
        h.update(np.round(np.asarray(v, float) * 1000).astype(np.int32).tobytes())
        n += 1
        if n == 3:
            break
    return str(d.get("cell_id", os.path.basename(path))), h.hexdigest() if n else None, len(cyc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="dup_report.csv")
    args = ap.parse_args()

    rows = []
    for ds in sorted(os.listdir(args.root)):
        files = [f for f in glob.glob(os.path.join(args.root, ds, "**", "*.pkl"), recursive=True)
                 if not any(x in os.path.basename(f) for x in EXCLUDE)]
        for f in files:
            try:
                cid, fp, n = fingerprint(f)
            except Exception as ex:
                cid, fp, n = os.path.basename(f), None, -1
            rows.append(dict(dataset=ds, file=os.path.basename(f), cell_id=cid, fingerprint=fp, n_cycles=n))
        print(f"{ds:12s} {len(files)} fichiers")
    D = pd.DataFrame(rows)

    by_fp = D[D.fingerprint.notna()].groupby("fingerprint")
    dup = by_fp.filter(lambda g: len(g) > 1).sort_values(["fingerprint", "dataset"])
    D.to_csv(args.out, index=False)
    dup.to_csv(args.out.replace(".csv", "_duplicates.csv"), index=False)

    print(f"\n{len(D)} fichiers, {D.fingerprint.nunique()} contenus distincts, "
          f"{dup.fingerprint.nunique()} contenus présents plusieurs fois")
    if len(dup):
        pairs = defaultdict(int)
        for _, g in dup.groupby("fingerprint"):
            pairs[" + ".join(sorted(g.dataset))] += 1
        print("\nDoublons par combinaison de sous-jeux (nombre de cellules) :")
        for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]):
            print(f"  {k:40s} {v}")
    print(f"\nDétail : {args.out.replace('.csv', '_duplicates.csv')}")


if __name__ == "__main__":
    main()
