#!/usr/bin/env python3
"""
Why does the amplitude fallback fail on Tongji's blended cells? Ridge on [dQ/Q_nom, T] trained on Tongji1+2
(all temperatures), tested on Tongji3 (25 C). Prints the signed error (bias) and the slope of SOH vs amplitude
per population, then a variant normalised by each cell's own first-window amplitude (needs the first
windows of a new cell, an extra assumption at deployment).

  python blend_diag.py $R/benchT_Tongji/samples.npz
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

D = np.load(sys.argv[1], allow_pickle=True)
g = pd.Series(D["groups"].astype(str))
pop = g.str.extract(r"^(Tongji\d)")[0].values
T = g.str.extract(r"CY(\d+)-")[0].astype(float).values
x = D["dq"][:, 0] / D["qnom"]
y = D["soh"] * 100
cyc = D["cycle"]
first = pd.Series(x).groupby([g.values]).transform(lambda s: s.iloc[np.argsort(cyc[s.index])[:3]].median()).values
xr = x / first

print("SOH vs amplitude per population (25 C): slope, intercept, amplitude range")
for p in ("Tongji1", "Tongji2", "Tongji3"):
    m = (pop == p) & (T == 25)
    b = np.polyfit(x[m], y[m], 1)
    print(f"  {p}: SOH = {b[0]:.1f} * dQ/Qnom + {b[1]:.1f}   dQ/Qnom in [{x[m].min():.3f}, {x[m].max():.3f}]  "
          f"first-window dQ/Qnom median {np.median(first[m]):.3f}")

tr, te = np.isin(pop, ["Tongji1", "Tongji2"]), pop == "Tongji3"
for name, feat in (("dQ/Qnom + T (paper's fallback)", x), ("dQ / own first windows + T", xr)):
    X = np.c_[feat, (T - T[tr].mean()) / T[tr].std()]
    r = RidgeCV(alphas=np.logspace(-4, 3, 30)).fit(X[tr], y[tr])
    e = r.predict(X[te]) - y[te]
    cell = pd.Series(np.abs(e)).groupby(g[te].values).mean()
    print(f"\n{name}: test on Tongji3  MAE per cell {cell.mean():.2f}  signed bias {e.mean():+.2f}  "
          f"(train MAE {np.abs(r.predict(X[tr]) - y[tr]).mean():.2f})")

print("\nSame two fallbacks on the main chemistry transfers (train one of Tongji1/2 at all T, test the other at t):")
rows = []
for a, b in (("Tongji1", "Tongji2"), ("Tongji2", "Tongji1")):
    for t in (25, 35, 45):
        tr, te = pop == a, (pop == b) & (T == t)
        for name, feat in (("absolute", x), ("self-referenced", xr)):
            X = np.c_[feat, (T - T[tr].mean()) / T[tr].std()]
            r = RidgeCV(alphas=np.logspace(-4, 3, 30)).fit(X[tr], y[tr])
            e = np.abs(r.predict(X[te]) - y[te])
            rows.append(dict(transfer=f"{a}->{b} @ {t}C", fallback=name,
                             mae=pd.Series(e).groupby(g[te].values).mean().mean()))
    # in-domain check: 5-fold by cell on all Tongji1+2
from sklearn.model_selection import GroupKFold
m12 = np.isin(pop, ["Tongji1", "Tongji2"])
idx = np.flatnonzero(m12)
for name, feat in (("absolute", x), ("self-referenced", xr)):
    errs = []
    for i, j in GroupKFold(5).split(idx, idx, g.values[idx]):
        tr, te = idx[i], idx[j]
        X = np.c_[feat, (T - T[tr].mean()) / T[tr].std()]
        r = RidgeCV(alphas=np.logspace(-4, 3, 30)).fit(X[tr], y[tr])
        errs.append(pd.Series(np.abs(r.predict(X[te]) - y[te])).groupby(g.values[te]).mean().mean())
    rows.append(dict(transfer="in-domain (5 folds by cell)", fallback=name, mae=np.mean(errs)))
print(pd.DataFrame(rows).pivot(index="transfer", columns="fallback", values="mae").round(2).to_string())
