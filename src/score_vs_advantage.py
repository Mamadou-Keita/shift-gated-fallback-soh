#!/usr/bin/env python3
"""
Is the shift score informative about the DECISION, not only about the population?

For every test cell: x = median over its windows of s/q (1 = threshold), y = advantage of the fallback,
MAE(expert) − MAE(fallback) on that cell, averaged over the expert variants (y > 0: the fallback was the
better choice for this cell). One point per cell and seed. Reads cycles.csv written by
gated_wrapper.py --dump-cycles (all experts).

  python score_vs_advantage.py Tongji=a.csv,b.csv,c.csv Blend=... MICH=... ISU=... ISUrate=... \
      --out $R/review6/score_vs_adv
Outputs: cells_score_adv.csv, a summary by s/q bin, Spearman correlations, and fig_score_vs_advantage.png/.pdf.
"""
import argparse
import os

import numpy as np
import pandas as pd

SHIFT = {"cell": "in-domain", "chem": "chemistry / formation / ageing path", "temp": "temperature",
         "chemtemp": "double shift", "chemcv": "charging current"}
BINS = [0, 0.5, 1, 2, 10, np.inf]
BIN_LAB = ["< 0.5", "0.5–1", "1–2", "2–10", "> 10"]


def kind(ds, proto):
    if ds == "Blend":
        return "blended cells (other format)"
    if proto == "chem" and ds == "Tongji":
        return "chemistry (NCA ↔ NCM)"
    if proto == "chem":
        return "formation (MICH) / ageing path (ISU)"
    return SHIFT.get(proto, proto)


def load(name, files):
    out = []
    for s, f in enumerate(files):
        if not os.path.exists(f):
            print(f"  missing: {f}")
            continue
        C = pd.read_csv(f)
        experts = [c for c in C.columns if c.startswith("pred[")]
        e_b = (C.base - C.y).abs()
        adv = pd.DataFrame({c: (C[c] - C.y).abs() for c in experts}).sub(e_b, axis=0)
        C["adv"] = adv.mean(axis=1)  # per window, mean over experts
        g = C.groupby(["protocol", "split", "cell"])
        # per cell: MAE difference = mean over windows of (|e_X| - |e_b|), averaged over experts
        R = pd.DataFrame(dict(sq=g.sq.median(), adv=g.adv.mean(), n=g.size())).reset_index()
        R["dataset"], R["seed"] = name, s
        out.append(R)
    return pd.concat(out) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="NAME=cycles1.csv,cycles2.csv,...")
    ap.add_argument("--out", default="score_vs_adv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A = pd.concat([load(g.split("=", 1)[0], g.split("=", 1)[1].split(",")) for g in args.groups])
    A["shift"] = [kind(d, p) for d, p in zip(A.dataset, A.protocol)]
    A["bin"] = pd.cut(A.sq, BINS, labels=BIN_LAB, right=False)
    A.to_csv(os.path.join(args.out, "cells_score_adv.csv"), index=False)
    pd.set_option("display.width", 250)

    from scipy.stats import spearmanr
    print("Spearman correlation between s/q and the fallback's advantage, per dataset (all protocols pooled):")
    for d, sub in A.groupby("dataset"):
        r, p = spearmanr(sub.sq, sub.adv)
        print(f"  {d:8s} rho = {r:+.2f} (p = {p:.1g}, {len(sub)} cell-seed pairs)")
    r, p = spearmanr(A.sq, A.adv)
    print(f"  {'all':8s} rho = {r:+.2f} (p = {p:.1g}, {len(A)} cell-seed pairs)")

    print("\nFallback advantage by score bin (SOH points; > 0 = fallback better), and share of cells where the "
          "fallback is better:")
    B = A.groupby("bin", observed=False).agg(cells=("adv", "size"), mean_adv=("adv", "mean"),
                                             median_adv=("adv", "median"),
                                             fallback_better=("adv", lambda v: (v > 0).mean()))
    print(B.round(2).to_string())

    print("\nDecision table (cell level): gate closing (median s/q > 1) vs which strategy was better")
    A["closing"] = A.sq > 1
    A["fb_better"] = A.adv > 0
    print(pd.crosstab(A.closing, A.fb_better, rownames=["s/q > 1"], colnames=["fallback better"]).to_string())

    print("\nPer shift type: median s/q, share of cells with s/q > 1, mean fallback advantage")
    S = A.groupby(["dataset", "shift"]).agg(cells=("sq", "size"), median_sq=("sq", "median"),
                                            closing=("closing", "mean"), mean_adv=("adv", "mean"),
                                            fallback_better=("fb_better", "mean"))
    print(S.round(2).to_string())

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = ["in-domain", "formation (MICH) / ageing path (ISU)", "temperature", "charging current",
             "double shift", "chemistry (NCA ↔ NCM)", "blended cells (other format)"]
    cols = ["#8c8c8c", "#9CC9BC", "#E39B2D", "#C2553B", "#5B7B86", "#1F8A70", "#0B3C49"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    for k, c in zip(kinds, cols):
        sub = A[A["shift"] == k]
        if len(sub):
            ax.scatter(sub.sq.clip(lower=0.05), sub.adv, s=9, alpha=0.55, color=c, label=f"{k} ({len(sub)})",
                       edgecolors="none")
    ax.set_xscale("log")
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.axvline(10, color="k", lw=0.6, ls=":")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Shift score s/q, median over the cell's windows (1 = threshold, 10 = gate fully closed)")
    ax.set_ylabel("MAE(expert) − MAE(fallback)\n(SOH points, > 0: fallback better)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_score_vs_advantage.{ext}"))
    print(f"\n{args.out}/fig_score_vs_advantage.png")


if __name__ == "__main__":
    main()
