#!/usr/bin/env python3
"""
Fixed blend with α chosen by internal validation on the TRAINING cells of each split (no test label).

For each split of each protocol and each expert X: α* = the α of the grid {0, 0.25, 0.5, 0.75, 1} that
minimises the per-cell MAE of base_oof + α (X_oof − base_oof) on the training cells, using the
out-of-fold predictions written by inner_oof.py. α* is then applied unchanged to the split's test cells
(cached test predictions of lit_baselines.py). One output row per protocol x split x expert.

  python alpha_cv.py --lit-root $R --npz ... --include-regex ... --temp-regex ... --chem-regex ... \
      --protocols cell temp chem chemtemp --out $R/review7/alpha_cv/Tongji_s0.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from cond_models import cell_mae, load, splits
from gated_wrapper import label_of

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lit-root", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", default=None)
    ap.add_argument("--protocols", nargs="+", default=["cell", "temp", "chem", "chemtemp"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = load(args)
    y, g = D["soh"], D["groups"]
    rows, missing = [], set()
    for proto in args.protocols:
        dirs = [d for d in glob.glob(os.path.join(args.lit_root, f"lit_{proto}_*"))
                if re.fullmatch(rf"lit_{proto}_[A-Za-z]+", os.path.basename(d))]
        for i, (name, tr, te) in enumerate(splits(D, proto)):
            test, inner = {}, {}
            for d in dirs:
                for f in glob.glob(os.path.join(d, "preds", f"s{i:02d}__*.npy")):
                    test[label_of(os.path.basename(f)[5:-4])] = np.load(f)
                for f in glob.glob(os.path.join(d, "inner", f"s{i:02d}__*.npy")):
                    inner[label_of(os.path.basename(f)[5:-4])] = np.load(f)
            if "base_dQ+T" not in test or "base_dQ+T" not in inner:
                missing.add((proto, name, "base"))
                continue
            if "train_idx" in inner and not np.array_equal(inner["train_idx"], tr):
                raise SystemExit(f"inner predictions do not match the split {proto}/{name}")
            b_oof, b_te = inner["base_dQ+T"], test["base_dQ+T"]
            experts = [k for k in test if k not in ("base_dQ+T", "tout+T", "FUS_pondérée")]
            for x in experts:
                if x not in inner or np.isnan(inner[x]).any():
                    missing.add((proto, name, x))
                    continue
                val = {a: cell_mae(b_oof + a * (inner[x] - b_oof) - y[tr], g[tr]).mean() for a in GRID}
                a_star = min(val, key=val.get)
                te_mae = cell_mae(b_te + a_star * (test[x] - b_te) - y[te], g[te]).mean()
                rows.append(dict(protocol=proto, split=name, expert=x, alpha_cv=a_star, mae_fixcv=te_mae,
                                 **{f"val_{a:.2f}": v for a, v in val.items()}))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(args.out, index=False)
    if missing:
        print(f"  {len(missing)} split x model combinations without inner predictions, e.g. {sorted(missing)[:5]}")
    if len(R):
        print(R.groupby("protocol")[["alpha_cv", "mae_fixcv"]].mean().round(2).to_string())
    print(args.out)


if __name__ == "__main__":
    main()
