#!/usr/bin/env python3
"""
Audit of the constant-current charging windows (Tongji, MICH): which cycles were kept, which were
excluded and why, and whether excluded cycles concentrate at some SOH (the window can stop being covered
as the cell ages).

Three sub-commands:

  log      summarise the extraction_log.csv written by identifiability_test.py run (per-cell reason counts)
           python audit_windows.py log Tongji=$R/benchT_Tongji/extraction_log.csv MICH=$R/benchT_MICH/extraction_log.csv

  rebuild  re-run the extraction on the raw BatteryLife files with the same rules, recording the reason of
           EVERY cycle and its SOH, then check that the kept cycles match samples.npz one to one
           python audit_windows.py rebuild --data-dir $P/Tongji --npz $R/benchT_Tongji/samples.npz \
               --cycle-stride 10 --out $R/audit/Tongji
           (window bounds are read from samples.npz; pass the other extraction options if they differed)

  npz      quality checks on samples.npz alone: windows per cell, SOH range, monotonic Q(V), amplitude
           outliers with respect to the cell's own trend, repeated cycles
           python audit_windows.py npz Tongji=$R/benchT_Tongji/samples.npz MICH=$R/benchT_MICH/samples.npz
"""
import argparse
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

REASONS = ["kept", "soh_out_of_range", "missing_V_or_I", "no_capacity_no_time", "no_charge", "no_cc_run",
           "too_few_points", "window_not_covered", "non_monotonic", "nonpositive_dQ", "ic_failed"]
SOH_BANDS = [0, 80, 85, 90, 95, 100, 200]
pd.set_option("display.width", 250)


# ---------------------------------------------------------------- log
def cmd_log(args):
    for spec in args.logs:
        name, f = spec.split("=", 1)
        L = pd.read_csv(f)
        if L.empty or "kept" not in L:
            print(f"\n{name}: {f} is empty (the npz was probably reused with --from-npz); use 'rebuild'.")
            continue
        cols = [c for c in L.columns if c not in ("cell_id", "n_cycles", "Q_ref_Ah")]
        tot = L[cols].sum()
        considered = tot.sum()
        print(f"\n=== {name}: {len(L)} cells, {int(L.n_cycles.sum())} recorded cycles, "
              f"{int(considered)} examined after stride/skip, {int(tot['kept'])} kept "
              f"({tot['kept'] / considered:.1%}) ===")
        print((tot.sort_values(ascending=False) / considered).map(lambda v: f"{v:.2%}").to_string())
        L["kept_share"] = L.kept / L[cols].sum(axis=1)
        print(f"kept share per cell: median {L.kept_share.median():.1%}, min {L.kept_share.min():.1%} "
              f"({L.loc[L.kept_share.idxmin(), 'cell_id']})")
        low = L[L.kept_share < 0.8]
        if len(low):
            print(f"{len(low)} cells keep less than 80 % of their examined cycles:")
            print(low.sort_values("kept_share")[["cell_id", "kept_share"] + [c for c in cols if c != "kept"]]
                  .head(15).to_string(index=False))


# ---------------------------------------------------------------- rebuild
def cmd_rebuild(args):
    import identifiability_test as IT
    os.makedirs(args.out, exist_ok=True)
    Z = np.load(args.npz, allow_pickle=False)
    v_grid = Z["v_grid"]
    opt = SimpleNamespace(charge_sign=args.charge_sign, cc_crate=args.cc_crate, cc_tol=args.cc_tol,
                          min_charge_current=args.min_charge_current, run_select=args.run_select,
                          exclude_cv=args.exclude_cv, cv_margin=args.cv_margin, min_run_points=args.min_run_points)
    rows = []
    files = IT.list_files(args.data_dir, args.pattern)
    for path in files:
        cell = IT.load_cell(path)
        cid = str(cell["cell_id"])
        nominal = float(cell.get("nominal_capacity_in_Ah") or np.nan)
        cycles = cell["cycle_data"]
        caps = [IT.cycle_discharge_capacity(c, opt) for c in cycles]
        qref = IT.reference_capacity(caps, nominal, args.soh_ref)
        for k in range(args.skip_first, len(cycles), args.cycle_stride):
            cyc = cycles[k]
            soh = caps[k] / qref if qref and np.isfinite(qref) else np.nan
            r = dict(cell=cid, k=k, cycle=cyc.get("cycle_number", k), soh=100 * soh if np.isfinite(soh) else np.nan)
            if not np.isfinite(soh) or not (args.soh_min <= soh <= args.soh_max):
                rows.append({**r, "reason": "soh_out_of_range"}); continue
            seg, why = IT.cc_charge_segment(cyc, nominal, opt)
            if seg is not None:
                r["v_seg_min"], r["v_seg_max"] = float(np.nanmin(seg[0])), float(np.nanmax(seg[0]))
                Q, why = IT.window_curve(seg[0], seg[1], v_grid)
                if Q is not None and IT.smooth_ic(Q, v_grid, args.ic_window) is None:
                    why = "ic_failed"
            rows.append({**r, "reason": why if why != "ok" else "kept"})
        print(f"  {cid:40s} kept {sum(1 for x in rows if x['cell'] == cid and x['reason'] == 'kept'):5d}", flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(args.out, "cycles_audit.csv"), index=False)

    print(f"\n=== {len(files)} cells, {len(C)} cycles examined, {int((C.reason == 'kept').sum())} kept ===")
    print(C.reason.value_counts(normalize=True).map(lambda v: f"{v:.2%}").to_string())
    C["band"] = pd.cut(C.soh, SOH_BANDS, right=False,
                       labels=["<80", "80–85", "85–90", "90–95", "95–100", "≥100"])
    T = pd.crosstab(C.band, C.reason, normalize="index").reindex(columns=[r for r in REASONS if r in set(C.reason)])
    print("\nShare of examined cycles per reason, by SOH band (does exclusion depend on ageing?):")
    print(T.map(lambda v: f"{v:.1%}").to_string())
    nc = C[C.reason == "window_not_covered"]
    if len(nc):
        print(f"\nwindow_not_covered: {len(nc)} cycles in {nc.cell.nunique()} cells; CC segment started above the "
              f"window low edge in {(nc.v_seg_min > v_grid[0]).mean():.0%} and ended below its high edge in "
              f"{(nc.v_seg_max < v_grid[-1]).mean():.0%} of them")

    # one-to-one check with samples.npz
    g = Z["groups"].astype(str)
    cyc = Z["cycle"] if "cycle" in Z.files else None
    K = C[C.reason == "kept"]
    if cyc is not None:
        a = set(zip(K.cell, K.cycle.astype(int)))
        b = set(zip(g, cyc.astype(int)))
        print(f"\nCheck vs samples.npz: rebuilt kept {len(a)}, npz {len(b)}, common {len(a & b)}, "
              f"only rebuilt {len(a - b)}, only npz {len(b - a)}")
        if a != b:
            print("  mismatch: the extraction options differ from those used for the npz (stride, skip-first, "
                  "soh range, cc-tol ...) or the npz was filtered afterwards")
    else:
        print(f"\nCheck vs samples.npz: rebuilt kept {len(K)}, npz {len(g)}")
    print(f"\n{args.out}/cycles_audit.csv")


# ---------------------------------------------------------------- npz
def cmd_npz(args):
    for spec in args.npzs:
        name, f = spec.split("=", 1)
        Z = np.load(f, allow_pickle=False)
        g, soh, dq = Z["groups"].astype(str), Z["soh"] * 100, Z["dq"][:, 0]
        cyc = Z["cycle"] if "cycle" in Z.files else np.arange(len(g))
        qn = Z["qnom"] if "qnom" in Z.files else np.ones(len(g))
        a = dq / qn
        Q = Z["shape"]
        print(f"\n=== {name}: {len(g)} windows, {len(set(g))} cells, window {Z['v_grid'][0]:.3f}–{Z['v_grid'][-1]:.3f} V ===")
        n = pd.Series(g).value_counts()
        print(f"windows per cell: median {n.median():.0f}, min {n.min()} ({n.idxmin()}), max {n.max()}")
        print(f"SOH: {soh.min():.1f} to {soh.max():.1f} %; above 100 %: {(soh > 100).mean():.1%}; "
              f"cells reaching 80 %: {(pd.Series(soh).groupby(g).min() <= 80).mean():.0%}")
        nonmono = (np.diff(Q, axis=1) < -1e-9).any(1)
        print(f"non-monotonic Q(V)/ΔQ: {nonmono.sum()} windows; shape end point != 1: "
              f"{(np.abs(Q[:, -1] - 1) > 1e-6).sum()}")
        dup = pd.DataFrame(dict(g=g, c=cyc)).duplicated().sum()
        print(f"repeated (cell, cycle): {dup}")
        # amplitude outliers: residual from a robust linear trend of a against SOH within each cell
        out = np.zeros(len(g), bool)
        for c in np.unique(g):
            i = np.flatnonzero(g == c)
            if len(i) < 8:
                continue
            p = np.polyfit(soh[i], a[i], 1)
            r = a[i] - np.polyval(p, soh[i])
            mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-12
            out[i] = np.abs(r) > args.z * mad
        print(f"amplitude outliers (> {args.z:g} robust SD from the cell's own ΔQ/Q_nom–SOH line): "
              f"{out.sum()} windows ({out.mean():.2%}) in {len(set(g[out]))} cells")
        if out.any():
            top = pd.Series(g[out]).value_counts().head(5)
            print("  most affected cells: " + ", ".join(f"{k} ({v})" for k, v in top.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("log"); p.add_argument("logs", nargs="+")
    p = sub.add_parser("npz"); p.add_argument("npzs", nargs="+"); p.add_argument("--z", type=float, default=5.0)
    p = sub.add_parser("rebuild")
    p.add_argument("--data-dir", required=True); p.add_argument("--pattern", default="**/*.pkl")
    p.add_argument("--npz", required=True); p.add_argument("--out", required=True)
    p.add_argument("--charge-sign", type=float, default=1.0); p.add_argument("--cc-crate", type=float, default=None)
    p.add_argument("--cc-tol", type=float, default=0.05); p.add_argument("--min-charge-current", type=float, default=1e-3)
    p.add_argument("--run-select", choices=["qspan", "last", "maxv"], default="qspan")
    p.add_argument("--exclude-cv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cv-margin", type=float, default=0.005); p.add_argument("--min-run-points", type=int, default=5)
    p.add_argument("--ic-window", type=int, default=15); p.add_argument("--soh-ref", choices=["first", "nominal"], default="first")
    p.add_argument("--soh-min", type=float, default=0.5); p.add_argument("--soh-max", type=float, default=1.1)
    p.add_argument("--skip-first", type=int, default=1); p.add_argument("--cycle-stride", type=int, default=1)
    args = ap.parse_args()
    {"log": cmd_log, "rebuild": cmd_rebuild, "npz": cmd_npz}[args.cmd](args)


if __name__ == "__main__":
    main()
