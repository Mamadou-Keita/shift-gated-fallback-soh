#!/usr/bin/env python3
"""
MATR (BatteryLife): does the final ~1C CC charge step (80 % SoC -> 3.6 V) survive until end of life,
and which voltage window does it cover as a function of SOH?

In MATR the fast-charge step stops at 80 % of NOMINAL capacity. Once the cell has faded, the
fast-charge step can hit 3.6 V before that point, and the 1C step then disappears. If it
disappears at high SOH, a charge window on this step cannot label aged cells.

  python matr_1c_check.py --data-dir $P/MATR --out $R/matr_1c_check.csv
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identifiability_test import arr, cc_charge_segment, list_files, load_cell  # noqa: E402

WINDOWS = [(3.40, 3.58), (3.42, 3.58), (3.45, 3.58), (3.45, 3.55), (3.48, 3.58), (3.50, 3.58)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pattern", default="**/*.pkl")
    ap.add_argument("--out", default="matr_1c_check.csv")
    ap.add_argument("--cc-crate", type=float, default=0.91, help="C-rate of the final step relative to Q_nom")
    ap.add_argument("--cc-tol", type=float, default=0.10)
    ap.add_argument("--stride", type=int, default=10, help="inspect one cycle out of N")
    args = ap.parse_args()
    seg_args = SimpleNamespace(charge_sign=1.0, cc_crate=args.cc_crate, cc_tol=args.cc_tol,
                               min_charge_current=1e-3, run_select="maxv", exclude_cv=True,
                               cv_margin=0.005, min_run_points=5)
    rows = []
    files = list_files(args.data_dir, args.pattern)
    for k, path in enumerate(files):
        cell = load_cell(path)
        qn = float(cell.get("nominal_capacity_in_Ah") or 1.1)
        name = os.path.basename(path)
        batch = name.split("_")[1][:2] if "_" in name else "?"
        cycles = cell["cycle_data"]
        for i in range(1, len(cycles), args.stride):
            cyc = cycles[i]
            Qd = arr(cyc, "discharge_capacity_in_Ah")
            soh = np.nanmax(Qd) / qn if Qd is not None and len(Qd) else np.nan
            seg, why = cc_charge_segment(cyc, qn, seg_args)
            r = dict(cell=name, batch=batch, cycle=i, soh=soh, found=seg is not None)
            if seg is not None:
                V, Q = seg
                r.update(vmin=np.nanmin(V), vmax=np.nanmax(V), npts=len(V), qspan=(np.nanmax(Q) - np.nanmin(Q)) / qn)
            rows.append(r)
        print(f"\r{k + 1}/{len(files)} {name}", end="", flush=True)
    print()
    D = pd.DataFrame(rows)
    D.to_csv(args.out, index=False)
    D = D[(D.soh > 0.5) & (D.soh < 1.2)]

    pd.set_option("display.width", 200)
    D["soh_bin"] = pd.cut(D.soh, [0.5, 0.8, 0.85, 0.9, 0.95, 1.0, 1.2])
    print("\n=== Share of cycles where the ~1C step exists, by batch and SOH ===")
    print(D.pivot_table(index="batch", columns="soh_bin", values="found", aggfunc="mean", observed=True)
          .round(2).to_string())
    F = D[D.found]
    if len(F):
        print("\n=== Step voltage range when it exists (median [p10, p90]) and points ===")
        g = F.groupby("batch")
        print(pd.DataFrame({
            "vmin": g.vmin.median(), "vmin_p90": g.vmin.quantile(.9),
            "vmax": g.vmax.median(), "vmax_p10": g.vmax.quantile(.1),
            "npts": g.npts.median(), "qspan": g.qspan.median()}).round(3).to_string())
        print("\n=== Share of ALL cycles (not only those with the step) whose step covers the window ===")
        for a, b in WINDOWS:
            cov = (D.found & (D.vmin <= a) & (D.vmax >= b))
            by = cov.groupby(D.soh_bin, observed=True).mean()
            print(f"  {a:.2f}-{b:.2f} V: all {cov.mean():.0%} | by SOH " +
                  "  ".join(f"{str(k)}: {v:.0%}" for k, v in by.items()))
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
