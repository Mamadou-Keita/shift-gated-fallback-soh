#!/usr/bin/env python3
"""
How close to the start of life are each cell's first windows? For every cell, the SOH label of its 1st, 3rd,
5th and 10th observed windows (chronological order). The labels are used ONLY for this check, never by the
self-referenced fallback, which reads the charge amplitude alone.

  python soh_first_windows.py Tongji=$R/benchT_Tongji/samples.npz:^Tongji[12] Blend=$R/benchT_Tongji/samples.npz:^Tongji3 \
      MICH=$R/benchT_MICH/samples.npz:^MICH ISU=$R/ISU/samples.npz:^ISU_c5_
"""
import re
import sys

import numpy as np

for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1)
    path, rx = rest.rsplit(":", 1)
    Z = np.load(path, allow_pickle=True)
    g, y = Z["groups"].astype(str), Z["soh"].astype(float)
    cyc = Z["cycle"] if "cycle" in Z.files else np.arange(len(y))
    y = y * (100 if np.nanmax(y) <= 1.5 else 1)
    out = {r: [] for r in (1, 3, 5, 10)}
    cells = [c for c in np.unique(g) if re.search(rx, c)]
    for c in cells:
        idx = np.flatnonzero(g == c)
        idx = idx[np.argsort(cyc[idx], kind="stable")]
        for r in out:
            if len(idx) >= r:
                out[r].append(y[idx[r - 1]])
    print(f"{name:7s} {len(cells):4d} cells | " + " | ".join(
        f"window {r}: SOH median {np.median(v):.1f}, min {np.min(v):.1f}" for r, v in out.items() if v))
