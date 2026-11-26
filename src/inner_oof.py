#!/usr/bin/env python3
"""
Out-of-fold predictions on the TRAINING cells of every split, for choosing a fixed blend weight α without
any test label (review: "select α by internal validation grouped by cell, on the training data of each
transfer").

For one lit_<protocol>_<models> directory (written by lit_baselines.py), and for every split of that
protocol: the training cells are cut into K folds grouped by cell (GroupKFold, deterministic); every model
of the directory, and the fallback ridge(ΔQ/Q_nom, T), is refitted on K−1 folds and predicts the held-out
fold, with exactly the same inputs, hyper-parameters and seed as in lit_baselines.py. Results go to
<lit dir>/inner/s<split>__<label>.npy, aligned with the split's training indices (also saved). Every
(split, label) is saved as soon as it is computed, so a killed job resumes where it stopped.

  python inner_oof.py --lit-dir $R/lit_chem_HGB --npz ... --include-regex ... --temp-regex ... \
      --chem-regex ... --seed-offset 0 --ext-repo $WORK/Inversion/v12/BatteryMFormer
"""
import argparse
import os
import re
import time

import numpy as np
from sklearn.model_selection import GroupKFold

import lit_baselines as L
from cond_models import design, load, splits
from gated_fusion import ridge_T

MODELS = L.ALL_MODELS + L.EXT_MODELS


def models_of(dirname):
    """lit_<mode>_<tag>: tag = model names concatenated (e.g. MLPCNNLSTM, CPMLPCPTransformerDLinear)."""
    m = re.fullmatch(r"lit_([a-z]+)_([A-Za-z]+)", os.path.basename(dirname.rstrip("/")))
    if not m:
        raise SystemExit(f"not a lit_<mode>_<models> directory: {dirname}")
    mode, tag, out = m.group(1), m.group(2), []
    names = sorted(MODELS, key=len, reverse=True)
    while tag:
        hit = next((n for n in names if tag.startswith(n)), None)
        if hit is None:
            raise SystemExit(f"cannot parse model names in {dirname}")
        out.append(hit)
        tag = tag[len(hit):]
    return mode, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lit-dir", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", default=None)
    ap.add_argument("--k-inner", type=int, default=5)
    ap.add_argument("--ext-repo", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--gpr-max-n", type=int, default=600)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-threads", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    mode, models = models_of(args.lit_dir)
    args.mode = mode
    if any(m in L.EXT_MODELS for m in models):
        if not args.ext_repo:
            raise SystemExit("--ext-repo required for " + ", ".join(m for m in models if m in L.EXT_MODELS))
        L.EXT_REPO = os.path.abspath(args.ext_repo)
    if any(m not in ("GPR", "HGB") for m in models):
        import torch
        torch.set_num_threads(args.n_threads)
    out = os.path.join(args.lit_dir, "inner")
    os.makedirs(out, exist_ok=True)
    D = load(args)
    y, g, T = D["soh"], D["groups"], D["T"]
    alphas = np.logspace(-3, 4, 15)
    print(f"{args.lit_dir}: protocol {mode}, models {models}, seed {args.seed_offset}", flush=True)
    for i, (name, tr, te) in enumerate(splits(D, mode)):
        path = lambda lab: os.path.join(out, f"s{i:02d}__{lab.replace('+', 'p')}.npy")
        np.save(path("train_idx"), tr)
        k = min(args.k_inner, len(np.unique(g[tr])))
        folds = list(GroupKFold(n_splits=k).split(tr, tr, g[tr]))
        t_ok = len(set(T[tr])) >= 2
        labels = ["base_dQ+T"] + [f"{m}{'+T' if t else ''}" for m in models for t in (False, True) if t_ok or not t]
        for lab in labels:
            if os.path.exists(path(lab)):
                continue
            t0 = time.time()
            oof = np.full(len(tr), np.nan)
            for a, b in folds:
                itr, iva = tr[a], tr[b]
                if lab == "base_dQ+T":
                    tmu, tsd = (T[itr].mean(), 1.0) if len(set(T[itr])) < 2 else (T[itr].mean(), T[itr].std() + 1e-9)
                    m_, _, _, Xva = ridge_T("dQ", D, itr, iva, args, alphas, tmu, tsd)
                    oof[b] = m_.predict(design(Xva, T[iva], "M1", tmu, tsd))
                    continue
                use_t = lab.endswith("+T")
                m = lab[:-2] if use_t else lab
                ttr, tva = L.t_col(D, itr, iva)
                ps = []
                for s in range(args.seed_offset, args.seed_offset + args.seeds):
                    if m in ("GPR", "HGB"):
                        Ptr, Pva = L.pca_inputs(D, itr, iva, args.n_pca)
                        A = np.hstack([Ptr, ttr]) if use_t else Ptr
                        B = np.hstack([Pva, tva]) if use_t else Pva
                        ps.append(L.fit_gpr(A, y[itr], B, s, args.gpr_max_n) if m == "GPR" else L.fit_hgb(A, y[itr], B, s))
                    else:
                        Ctr, Cva = L.curves(D, itr, iva)
                        ps.append(L.fit_net(m, Ctr, ttr if use_t else None, y[itr], g[itr], Cva,
                                            tva if use_t else None, s, args))
                oof[b] = np.mean(ps, 0)
            np.save(path(lab), oof)
            print(f"  {name} / {lab}: {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
