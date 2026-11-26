#!/usr/bin/env python3
"""Compare gate calibrations: MAE of every GATEcell[X] minus that of the per-cycle calibration (default).
  python calib_compare.py $R/wrap $R/wrap_cellcal $R/wrap_cellmed $R/wrap_firstk
The first directory is the reference."""
import os
import sys

import pandas as pd

f = lambda d: (pd.read_csv(os.path.join(d, "results.csv")).query("method.str.startswith('GATEcell[')", engine="python")
               .groupby(["method", "protocol"]).MAE_cell.mean().unstack())
ref = f(sys.argv[1])
pd.set_option("display.width", 200)
print(f"Reference: {sys.argv[1]} (mean MAE over GATEcell variants: {ref.mean().round(3).to_dict()})")
for d in sys.argv[2:]:
    diff = (f(d) - ref)
    print(f"\n{os.path.basename(d.rstrip('/'))} - reference (MAE, SOH points; mean / min / max over the variants):")
    print(diff.describe().loc[["mean", "min", "max"]].round(3).to_string())
