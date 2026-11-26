#!/usr/bin/env python3
"""
ISU-ILCC (raw JSON release) -> samples.npz in the format of identifiability_test.py, so that
cond_models.py, gated_fusion.py, lit_baselines.py and gated_wrapper.py run unchanged.

One sample = one RPT x one charge rate:
  window : CC part of the RPT charge (C/5 or C/2, CC-CV to 4.2 V from 3.0 V), voltage window [v_low, v_high]
  label  : SOH = C/5 RPT discharge capacity / capacity at the first RPT (week 0)
  cycle  : RPT index (chronological), used by the cell-level causal gate

Group names encode every factor used to build the shift protocols (regex-friendly):

    ISU_<rate>_T30_ch<A|B|C>_dod<A|B|C>_dis<A|B|C>__<cell>
      rate : c5 (C/5) or c2 (C/2)  -> measurement-current shift
      ch   : charge C-rate in cycling   A <= 1.2C < B < 2.0C <= C
      dod  : mean depth of discharge    A < 45 % <= B < 85 % <= C
      dis  : discharge C-rate           A <= 0.6C < B <= 1.5C < C
      after "__": physical cell id (the same cell appears once per rate)

  python -I isu_to_npz.py --root ./isu --out $R/ISU/samples.npz --v-low 3.70 --v-high 4.00
"""
import argparse
import glob
import json
import os
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np

Q_NOM = 0.250  # Ah, rated capacity

COND = {
    1: (0.5, 0.5, 25.90), 2: (0.5, 0.5, 46.11), 3: (0.5, 0.5, 67.77), 4: (1.0, 0.5, 47.27),
    5: (1.0, 0.5, 97.54), 6: (2.0, 0.5, 27.23), 7: (2.0, 0.5, 46.86), 8: (2.0, 0.5, 67.24),
    9: (2.0, 0.5, 96.93), 10: (2.5, 0.5, 47.21), 11: (2.5, 0.5, 95.98), 12: (3.0, 0.5, 28.60),
    13: (3.0, 0.5, 47.26), 14: (3.0, 0.5, 66.32), 16: (0.5, 0.5, 97.62), 17: (1.0, 0.5, 67.53),
    18: (2.5, 0.5, 27.53), 19: (2.5, 0.5, 65.70), 20: (0.8, 0.5, 98.16), 21: (1.2, 0.5, 97.98),
    22: (1.4, 0.5, 97.88), 23: (1.6, 0.5, 97.61), 24: (1.8, 0.5, 97.51), 25: (1.8, 0.6, 51.43),
    26: (1.4, 2.2, 4.16), 27: (0.6, 2.4, 57.85), 28: (2.4, 1.6, 80.99), 29: (1.6, 1.8, 52.48),
    30: (0.8, 0.8, 77.59), 31: (1.2, 1.0, 61.43), 32: (1.0, 1.4, 38.86), 33: (2.0, 1.2, 40.65),
    34: (2.2, 2.0, 27.60), 35: (1.825, 0.5, 97.02), 36: (2.075, 0.5, 97.06), 37: (0.725, 0.5, 98.08),
    38: (1.875, 0.5, 95.83), 39: (1.475, 0.5, 97.14), 40: (1.825, 1.025, 31.94),
    41: (2.075, 1.775, 67.02), 42: (0.725, 2.375, 36.32), 43: (1.875, 2.325, 38.00),
    44: (0.775, 1.275, 37.12), 45: (1.125, 1.725, 80.26), 46: (1.225, 2.025, 75.36),
    47: (2.325, 1.925, 48.27), 48: (2.375, 2.225, 58.57), 49: (0.975, 0.675, 18.82),
    50: (2.425, 1.625, 20.10), 51: (2.275, 1.875, 53.64), 52: (1.425, 0.875, 83.55),
    53: (2.025, 0.825, 91.54), 54: (0.925, 1.125, 27.05), 55: (1.025, 2.475, 61.44),
    56: (2.175, 0.975, 53.81), 57: (1.775, 1.175, 17.42), 58: (2.475, 0.575, 42.03),
    59: (1.325, 1.825, 31.64), 60: (0.675, 1.325, 83.63), 61: (2.125, 1.975, 54.07),
    62: (1.575, 2.425, 64.19), 63: (1.975, 1.675, 80.57), 64: (1.175, 1.425, 91.44),
}


def cls_ch(c):
    return "A" if c <= 1.2 else ("B" if c < 2.0 else "C")


def cls_dod(d):
    return "A" if d < 45 else ("B" if d < 85 else "C")


def cls_dis(c):
    return "A" if c <= 0.6 else ("B" if c <= 1.5 else "C")


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    return json.loads(d) if isinstance(d, str) else d


def as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def cc_part(V, Q, I):
    """Longest run where the charge current is constant (the CC phase, CV excluded)."""
    V, Q, I = (np.asarray(a, float).ravel() for a in (V, Q, I))
    n = min(len(V), len(Q), len(I))
    V, Q, I = V[:n], Q[:n], np.abs(I[:n])
    ok = np.isfinite(V) & np.isfinite(Q) & np.isfinite(I)
    if ok.sum() < 10 or I[ok].max() <= 0:
        return None
    i_cc = np.median(I[ok & (I > 0.5 * I[ok].max())])
    m = ok & (np.abs(I - i_cc) < 0.05 * i_cc)
    if m.sum() < 10:
        return None
    # longest contiguous run
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    k = np.argmax(ends - starts)
    s, e = starts[k], ends[k]
    return V[s:e], Q[s:e], i_cc


def window_curve(V, Q, v_grid):
    if V.min() > v_grid[0] or V.max() < v_grid[-1]:
        return None, "window_not_covered"
    keep = np.concatenate([[True], V[1:] > np.maximum.accumulate(V)[:-1]])
    Vm, Qm = V[keep], Q[keep]
    if ((Vm >= v_grid[0]) & (Vm <= v_grid[-1])).sum() < 20:
        return None, "too_few_points"
    q = np.interp(v_grid, Vm, Qm)
    q = q - q[0]
    if not np.isfinite(q[-1]) or q[-1] <= 0:
        return None, "nonpositive_dQ"
    return q, "ok"


def smooth_ic(q, v_grid, window):
    from scipy.signal import savgol_filter
    dV = v_grid[1] - v_grid[0]
    w = max(5, window if window % 2 else window + 1)
    ic = np.clip(savgol_filter(q, window_length=w, polyorder=2, deriv=1, delta=dV), 0, None)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    area = trapz(ic, v_grid)
    return ic / area if area > 0 else None


def process_cell(job):
    path, v_low, v_high, n_points, ic_window, soh_min, soh_max = job
    cell = os.path.basename(path)[:-5]
    g = int(re.match(r"G(\d+)C", cell).group(1))
    c_ch, c_dis, dod = COND[g]
    tag = f"T30_ch{cls_ch(c_ch)}_dod{cls_dod(dod)}_dis{cls_dis(c_dis)}__{cell}"
    v_grid = np.linspace(v_low, v_high, n_points)
    out, why = [], Counter()
    try:
        d = load_json(path)
    except Exception as e:
        why[f"load_error:{type(e).__name__}"] += 1
        return cell, out, why
    cap = []
    for c in as_list(d.get("capacity_discharge_C_5")):
        try:
            cap.append(float(c) if c not in (None, "", []) else np.nan)
        except (TypeError, ValueError):
            cap.append(np.nan)
    cap = np.array(cap, float)
    if not np.isfinite(cap).any() or not np.isfinite(cap[0]):
        why["no_reference_capacity"] += 1
        return cell, out, why
    q_ref = cap[0]
    for rate, key in (("c5", "QV_charge_C_5"), ("c2", "QV_charge_C_2")):
        qv = d.get(key) or {}
        Vs, Qs, Is = as_list(qv.get("V")), as_list(qv.get("Q")), as_list(qv.get("I"))
        for k in range(min(len(cap), len(Vs), len(Qs), len(Is))):
            soh = cap[k] / q_ref
            if not np.isfinite(soh) or not (soh_min <= soh <= soh_max):
                why[f"{rate}:soh_out_of_range_or_missing"] += 1; continue
            if Vs[k] is None or len(Vs[k]) < 10:
                why[f"{rate}:empty_rpt"] += 1; continue
            seg = cc_part(Vs[k], Qs[k], Is[k])
            if seg is None:
                why[f"{rate}:no_cc"] += 1; continue
            V, Q, i_cc = seg
            q, r = window_curve(V, Q, v_grid)
            if q is None:
                why[f"{rate}:{r}"] += 1; continue
            ic = smooth_ic(q, v_grid, ic_window)
            if ic is None:
                why[f"{rate}:ic_failed"] += 1; continue
            out.append(dict(group=f"ISU_{rate}_{tag}", shape=q / q[-1], ic=ic, dq=q[-1], soh=soh,
                            cycle=k, i_cc=i_cc, c_ch=c_ch, c_dis=c_dis, dod=dod))
            why[f"{rate}:ok"] += 1
    return cell, out, why


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with RPT_json/ (sub-folders 'Release x.y' allowed)")
    ap.add_argument("--out", default="ISU/samples.npz")
    ap.add_argument("--v-low", type=float, default=3.70)
    ap.add_argument("--v-high", type=float, default=4.00)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--ic-window", type=int, default=15)
    ap.add_argument("--soh-min", type=float, default=0.5)
    ap.add_argument("--soh-max", type=float, default=1.1)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "RPT_json", "**", "G*C*.json"), recursive=True))
    if not files:
        raise SystemExit(f"No RPT_json/**/G*C*.json under {args.root}")
    jobs = [(f, args.v_low, args.v_high, args.n_points, args.ic_window, args.soh_min, args.soh_max)
            for f in files]
    rows, reasons = [], Counter()
    with Pool(args.n_jobs) as pool:
        for i, (cell, out, why) in enumerate(pool.imap(process_cell, jobs)):
            rows += out; reasons.update(why)
            print(f"\r{i + 1}/{len(files)} {cell:8s}", end="", flush=True)
    print()
    if not rows:
        raise SystemExit(f"No sample extracted. Reasons: {dict(reasons)}")

    D = dict(
        v_grid=np.linspace(args.v_low, args.v_high, args.n_points),
        shape=np.array([r["shape"] for r in rows]), ic=np.array([r["ic"] for r in rows]),
        dq=np.array([r["dq"] for r in rows])[:, None], soh=np.array([r["soh"] for r in rows]),
        groups=np.array([r["group"] for r in rows]), cycle=np.array([r["cycle"] for r in rows]),
        protocol=np.array([r["group"].split("__")[0] for r in rows]),
        temp=np.full(len(rows), 30.0), qnom=np.full(len(rows), Q_NOM),
        i_cc=np.array([r["i_cc"] for r in rows]), c_ch=np.array([r["c_ch"] for r in rows]),
        c_dis=np.array([r["c_dis"] for r in rows]), dod=np.array([r["dod"] for r in rows]),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **D)

    import pandas as pd
    S = pd.DataFrame({"group": D["groups"], "soh": D["soh"], "dqn": D["dq"][:, 0] / Q_NOM})
    S["rate"] = S.group.str.extract(r"^ISU_(c\d)_")
    S["cell"] = S.group.str.split("__").str[-1]
    for fac in ("ch", "dod", "dis"):
        S[fac] = S.group.str.extract(rf"_{fac}([ABC])_")
    print(f"\n{len(S)} samples, {S.cell.nunique()} cells, window {args.v_low:.2f}-{args.v_high:.2f} V")
    print("Extraction outcome per rate:", {k: v for k, v in sorted(reasons.items())})
    print("\nSamples / cells per rate:")
    print(S.groupby("rate").agg(samples=("soh", "size"), cells=("cell", "nunique"),
                                soh_min=("soh", "min"), dqn_med=("dqn", "median")).round(3).to_string())
    for fac in ("ch", "dod", "dis"):
        print(f"\nCells per {fac} class (C/5):")
        sub = S[S.rate == "c5"]
        print(sub.groupby(fac).agg(cells=("cell", "nunique"), samples=("soh", "size"),
                                   soh_min=("soh", "min")).round(3).to_string())
    r = np.corrcoef(S.dqn[S.rate == "c5"], S.soh[S.rate == "c5"])[0, 1]
    print(f"\ncorr(dQ/Q_nom, SOH) at C/5: {r:.3f}")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
