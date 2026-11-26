#!/usr/bin/env python3
"""
Figures of the paper (PDF vector + PNG 300 dpi), from the results already on disk.

  Fig. 2  Charging-window curves: Q(V)/Q_nom and dQ/dV at three SOH levels, per chemistry (Tongji)
          and per formation protocol (MICH) - the shape is chemistry-specific.
  Fig. 3  What transfers: SOH vs window amplitude dQ/Q_nom (shared across chemistries) and vs the
          first shape component (separated by chemistry).
  Fig. 4  Benchmark heat maps: MAE of every model variant x protocol, alone vs enveloped
          (3-seed means), Tongji and MICH.
  Fig. 5  Dumbbells under unseen chemistry and double shift (Tongji): each model alone vs enveloped,
          mean +/- s.d. over seeds, fallback as reference line.
  Fig. 6  Gate behaviour: distribution of the shift score s/q per protocol, both datasets.
  Fig. 7  Gate trajectories: cycle-level weight and cell-level running median for example cells.
  Fig. 8  Parity plots under unseen chemistry: one expert alone vs enveloped.
  Fig. 9  Sensitivity to the gate width k (FUS, Tongji, seed 0).
  Fig. 11 Expert alone vs fallback alone vs envelope: mean regret per dataset x protocol.
  Fig. 10 ISU-ILCC: error per model with the test current unseen vs seen in training (C/2 and C/5).

Each figure is independent: a missing input skips that figure with a message.

  python make_figures.py --R $WORK/Inversion/results --out $WORK/Inversion/figures
"""
import argparse
import glob
import os
import re
import traceback
from types import SimpleNamespace
import matplotlib.figure as _mf
_mf.Figure.suptitle = lambda self, *a, **k: None  # no titles in paper figures

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None  # pas de titre dans les figures de la revue  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# validated categorical slots (dataviz reference palette, light mode) + neutrals
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
SEQ = ["#9ec5f0", "#4f94e0", "#1d5aa8"]  # one hue, light -> dark (SOH high -> low)
MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM  # Elsevier single / double column

FAMILIES = ["GPR", "HGB", "MLP", "CNN", "LSTM", "CPMLP", "CPTransformer", "PatchTST", "iTransformer",
            "DLinear", "ConvTimeNet", "PatchMLP"]
DISPLAY = {"CNN": "1D-CNN"}
PROTOS = ["cell", "temp", "chem", "chemtemp"]
PNAME = {"Tongji": {"cell": "In-domain", "temp": "Unseen T", "chem": "Unseen chemistry", "chemtemp": "Double shift"},
         "MICH": {"cell": "In-domain", "temp": "Unseen T", "chem": "Unseen formation", "chemtemp": "Double shift"}}
CHEM_LABEL = {"Tongji1": "NCA", "Tongji2": "NCM", "BL": "BL formation", "MC": "MC formation"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def save(fig, out, name):
    if fig._suptitle is not None:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  ok  {name}")


def log_axis(axis, lo, hi):
    """Readable ticks (0.5, 1, 2, 5, 10 ...) on a log axis instead of 2x10^0."""
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo * 0.8 <= t <= hi * 1.25]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def load_ds(npz, include, temp_rx, chem_rx):
    from cond_models import load
    return load(SimpleNamespace(npz=npz, include_regex=include, temp_regex=temp_rx, chem_regex=chem_rx))


def chem_name(c):
    return CHEM_LABEL.get(c, c)


def split_label(n):
    """English, short labels for the split names produced by cond_models.splits."""
    n = n.replace("cellules pli", "fold").replace("test ", "").replace("sans", "without").replace(" @ ", " at ")
    for k, v in CHEM_LABEL.items():
        n = n.replace(k, v.split()[0])
    return n


# ------------------------------------------------------------------ Fig. 2
def fig_curves(D, name, out, temp=None):
    v = D["v_grid"]
    chems = sorted(set(D["chem"]))
    fig, axes = plt.subplots(2, len(chems), figsize=(W2, 105 * MM), sharex=True, sharey="row")
    for j, c in enumerate(chems):
        m = D["chem"] == c
        if temp is not None and (m & (D["T"] == temp)).any():
            m &= D["T"] == temp
        cells = pd.Series(D["soh"][m]).groupby(D["groups"][m]).agg(["min", "max"])
        cell = (cells["max"] - cells["min"]).idxmax()  # the cell with the widest SOH span
        idx = np.flatnonzero(D["groups"] == cell)
        soh = D["soh"][idx]
        targets = np.linspace(soh.max(), soh.min(), 3)
        for k, tgt in enumerate(targets):
            i = idx[np.argmin(np.abs(soh - tgt))]
            q = D["shape"][i] * D["dqn"][i]
            lab = f"SOH {D['soh'][i]:.0f} %"
            axes[0, j].plot(v, q * 100, color=SEQ[k], label=lab)
            axes[1, j].plot(v, D["ic"][i], color=SEQ[k], label=lab)
        axes[0, j].set_title(f"{chem_name(c)} · {D['T'][idx[0]]:.0f} °C · one cell", loc="left")
        axes[1, j].set_xlabel("Voltage (V)")
        axes[0, j].legend(loc="upper left")
    axes[0, 0].set_ylabel(r"$Q(V)/Q_{\mathrm{nom}}$ (%)")
    axes[1, 0].set_ylabel("dQ/dV (normalised)")
    head = {"Tongji": "Tongji: the window amplitude shrinks with ageing; the dQ/dV peaks differ between NCA and NCM",
            "MICH": "MICH: the window amplitude shrinks with ageing; both formation protocols share the same shape"}
    fig.suptitle(head.get(name, name), x=0.01, ha="left", fontsize=9, fontweight="bold")
    save(fig, out, f"fig2_curves_{name}")


# ------------------------------------------------------------------ Fig. 3
def fig_transfer(D, name, out, max_points=4000, seed=0):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(D["soh"]))[:max_points]
    pc1 = PCA(n_components=1).fit(D["shape"]).transform(D["shape"])[:, 0]
    chems = sorted(set(D["chem"]))
    cols = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 75 * MM), sharey=True)
    for c, col in zip(chems, cols):
        m = idx[D["chem"][idx] == c]
        axes[0].scatter(D["dqn"][m] * 100, D["soh"][m], s=4, color=col, alpha=0.45, linewidths=0, label=chem_name(c))
        axes[1].scatter(pc1[m], D["soh"][m], s=4, color=col, alpha=0.45, linewidths=0, label=chem_name(c))
    axes[0].set_xlabel("Window amplitude ΔQ / Q_nom (%)")
    axes[1].set_xlabel("First principal component of the shape Q(V)/ΔQ")
    axes[0].set_ylabel("SOH (%)")
    ta = {"Tongji": "(a) Amplitude: NCA and NCM share each band (bands = temperature)",
          "MICH": "(a) Amplitude: one relation for both formation protocols"}
    tb = {"Tongji": "(b) Shape: the SOH relation shifts between chemistries",
          "MICH": "(b) Shape: both formation protocols overlap"}
    axes[0].set_title(ta.get(name, "(a) Amplitude"), loc="left")
    axes[1].set_title(tb.get(name, "(b) Shape"), loc="left")
    leg = axes[0].legend(markerscale=3, loc="lower right")
    for h in leg.legend_handles:
        h.set_alpha(1)
    save(fig, out, f"fig3_transfer_{name}")


# ------------------------------------------------------------------ Fig. 4
def fig_heatmap(summary_csv, name, out, gate="GATEcell"):
    S = pd.read_csv(summary_csv)
    M = S.pivot(index="method", columns="protocol", values="mean")
    rows = [f"{f}{t}" for f in FAMILIES for t in ("", "+T") if f"{f}{t}" in M.index]
    protos = [p for p in PROTOS if p in M.columns]
    cols, labels = [], []
    for p in protos:
        for kind in ("alone", "env"):
            cols.append((p, kind))
            labels.append("alone" if kind == "alone" else "enveloped")
    A = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, (p, kind) in enumerate(cols):
            key = r if kind == "alone" else f"{gate}[{r}]"
            if key in M.index and p in M.columns:
                A[i, j] = M.loc[key, p]
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(W2, (len(rows) * 5.2 + 30) * MM))
    finite = A[np.isfinite(A)]
    im = ax.imshow(A, aspect="auto", cmap="Blues", norm=LogNorm(vmin=max(finite.min(), 0.2), vmax=finite.max()))
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(A[i, j]):
                dark = A[i, j] > np.exp(np.log(finite.min()) + 0.62 * np.log(finite.max() / finite.min()))
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), labels, fontsize=7)
    ax.set_yticks(range(len(rows)), [DISPLAY.get(r.replace("+T", ""), r.replace("+T", "")) + (" +T" if r.endswith("+T") else "")
                                     for r in rows], fontsize=7)
    ax.xaxis.tick_top()
    ax.grid(False)
    for j in range(2, len(cols), 2):
        ax.axvline(j - 0.5, color="white", lw=2.5)
    for k, p in enumerate(protos):
        ax.text(2 * k + 0.5, -1.9, PNAME[name][p], ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("MAE (SOH points, log scale)")
    log_axis(cb.ax.yaxis, finite.min(), finite.max())
    head = {"Tongji": "Tongji, mean MAE over 3 seeds: the envelope removes the high errors under chemistry shift",
            "MICH": "MICH, mean MAE over 3 seeds: the envelope changes little (benign formation shift, weak fallback)"}
    fig.suptitle(head.get(name, name) + "; blank = +T undefined (one training temperature)" * (name == "MICH"),
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    save(fig, out, f"fig4_heatmap_{name}")


# ------------------------------------------------------------------ Fig. 5
def fig_dumbbell(summary_csv, name, out, protos=("chem", "chemtemp"), suffix="+T", gate="GATEcell"):
    S = pd.read_csv(summary_csv).set_index(["method", "protocol"])
    models = [f"{f}{suffix}" for f in FAMILIES if (f"{f}{suffix}", protos[0]) in S.index]
    fig, axes = plt.subplots(1, len(protos), figsize=(W2, (len(models) * 6 + 28) * MM), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(models))[::-1]
    for ax, p in zip(axes, protos):
        for yi, mdl in zip(y, models):
            if (mdl, p) not in S.index or (f"{gate}[{mdl}]", p) not in S.index:
                continue
            a, e = S.loc[(mdl, p)], S.loc[(f"{gate}[{mdl}]", p)]
            ax.plot([a["mean"], e["mean"]], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.errorbar(a["mean"], yi, xerr=np.nan_to_num(a["std"]), fmt="o", color=ORANGE, ms=5, capsize=2, zorder=2,
                        label="Model alone" if yi == y[0] else None)
            ax.errorbar(e["mean"], yi, xerr=np.nan_to_num(e["std"]), fmt="o", color=BLUE, ms=5, capsize=2, zorder=3,
                        label="Enveloped (cell-level gate)" if yi == y[0] else None)
        if ("base_dQ+T", p) in S.index:
            b = S.loc[("base_dQ+T", p), "mean"]
            ax.axvline(b, color=INK2, ls="--", lw=1, label="Fallback \u0394Q/Q_nom + T" if p == protos[0] else None)
        ax.set_xscale("log")
        vals = [v for v in S.xs(p, level="protocol")["mean"].values if np.isfinite(v)]
        log_axis(ax.xaxis, min(vals), max(vals))
        ax.set_xlabel("MAE (SOH points, log scale)")
        ax.set_title(PNAME[name][p], loc="left")
    axes[0].set_yticks(y, [DISPLAY.get(m.replace(suffix, ""), m.replace(suffix, "")) + suffix.replace("+T", " +T")
                           for m in models])
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{name}: every model collapses under unseen chemistry; enveloped, all sit near the fallback",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, frameon=False)
    fig.savefig(os.path.join(out, f"fig5_dumbbell_{name}.pdf"), dpi=300)
    fig.savefig(os.path.join(out, f"fig5_dumbbell_{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  ok  fig5_dumbbell_{name}")


# ------------------------------------------------------------------ gate helpers
def gate_split(D, tr, te, n_pca=8, q=0.99, k=10.0):
    from gated_fusion import Gate, calibrate_gate, causal_cell_median, pca_blocks, weights
    qq = calibrate_gate(D, tr, n_pca, q)
    Ztr, Zte, Rtr, Rte = pca_blocks(D, tr, te, n_pca)
    s = Gate().fit(Ztr, Rtr).score(Zte, Rte)
    w = weights(s, qq, k)
    cyc = D["cycle"] if "cycle" in D else np.arange(len(D["soh"]))
    return s / qq, w, causal_cell_median(w, D["groups"][te], cyc[te])


# ------------------------------------------------------------------ Fig. 6
def fig_gate_scores(datasets, out, k=10.0):
    from cond_models import splits
    fig, axes = plt.subplots(1, len(datasets), figsize=(W2, 72 * MM), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, D) in zip(axes, datasets):
        data, labels = [], []
        for p in PROTOS:
            sp = splits(D, p)
            if not sp:
                continue
            r = np.concatenate([gate_split(D, tr, te)[0] for _, tr, te in sp])
            data.append(np.maximum(r, 1e-3))
            labels.append(PNAME[name][p].replace(" ", "\n", 1))
        parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor(BLUE); b.set_edgecolor(BLUE); b.set_alpha(0.35)
        parts["cmedians"].set_color(INK)
        ax.set_yscale("log")
        ax.axhline(1, color=AQUA, lw=1.2, ls="--", label="q: weight 1 below")
        ax.axhline(k, color=ORANGE, lw=1.2, ls="--", label="k\u00b7q: weight 0 above")
        allv = np.concatenate(data)
        log_axis(ax.yaxis, np.percentile(allv, 0.5), max(allv.max(), k))
        ax.set_xticks(range(1, len(data) + 1), labels)
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("Shift score s / q (log scale)")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.suptitle("The shape score stays below q in-domain and under a formation change, and exceeds 10q under chemistry shift",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    save(fig, out, "fig6_gate_scores")


# ------------------------------------------------------------------ Fig. 7
def fig_gate_traj(D, name, out, n_cells=3, seed=0):
    from cond_models import splits
    cyc = D["cycle"] if "cycle" in D else np.arange(len(D["soh"]))
    sel = []
    for p in ("cell", "temp", "chem"):
        sp = splits(D, p)
        if sp:
            sel.append((p, sp[0]))
    fig, axes = plt.subplots(1, len(sel), figsize=(W2, 62 * MM), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(seed)
    for ax, (p, (split_name, tr, te)) in zip(axes, sel):
        _, w, wc = gate_split(D, tr, te)
        cells = rng.permutation(np.unique(D["groups"][te]))[:n_cells]
        for c, col in zip(cells, [BLUE, ORANGE, AQUA]):
            m = D["groups"][te] == c
            o = np.argsort(cyc[te][m])
            ax.plot(cyc[te][m][o], w[m][o], color=col, lw=0.8, alpha=0.35)
            ax.plot(cyc[te][m][o], wc[m][o], color=col, lw=1.8, label="_".join(str(c).split("_")[:3])[:26])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Cycle number")
        ax.set_title(f"{PNAME[name][p]}: {split_label(split_name)}", loc="left", fontsize=7.5)
        ax.legend(loc="best", fontsize=6)
    axes[0].set_ylabel("Gate weight w")
    fig.suptitle(f"{name}: cell-level gate weight (thick; thin = per cycle) over each test cell's life",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    save(fig, out, f"fig7_gate_traj_{name}")


# ------------------------------------------------------------------ Fig. 8
def cached(lit_root, proto, i, lab):
    fn = f"s{i:02d}__{lab.replace('+', 'p')}.npy"
    for d in glob.glob(os.path.join(lit_root, f"lit_{proto}_*")):
        if re.fullmatch(rf"lit_{proto}_[A-Za-z]+", os.path.basename(d)):
            f = os.path.join(d, "preds", fn)
            if os.path.exists(f):
                return np.load(f)
    return None


def fig_parity(D, name, lit_root, out, expert="CNN+T", proto="chem"):
    from cond_models import splits
    ys, pa, pe, ch = [], [], [], []
    for i, (_, tr, te) in enumerate(splits(D, proto)):
        px, pb = cached(lit_root, proto, i, expert), cached(lit_root, proto, i, "base_dQ+T")
        if px is None or pb is None:
            continue
        _, _, wc = gate_split(D, tr, te)
        ys.append(D["soh"][te]); pa.append(px); pe.append(pb + wc * (px - pb)); ch.append(D["chem"][te])
    if not ys:
        raise FileNotFoundError(f"no cached predictions for {expert} / {proto} in {lit_root}")
    y, pa, pe, ch = map(np.concatenate, (ys, pa, pe, ch))
    lo, hi = np.nanmin([y.min(), pa.min(), pe.min()]), np.nanmax([y.max(), pa.max(), pe.max()])
    fig, axes = plt.subplots(1, 2, figsize=(W2, 100 * MM), sharex=True, sharey=True)
    lab = DISPLAY.get(expert.replace("+T", ""), expert.replace("+T", "")) + ("+T" if expert.endswith("+T") else "")
    for ax, p, title in ((axes[0], pa, f"(a) {lab} alone"), (axes[1], pe, f"(b) {lab} enveloped")):
        for c, col in zip(sorted(set(ch)), [BLUE, ORANGE]):
            m = ch == c
            ax.scatter(y[m], p[m], s=4, color=col, alpha=0.45, linewidths=0, label=f"test cells: {chem_name(c)}")
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
        mae = np.mean(np.abs(p - y))
        ax.set_title(f"{title} · MAE {mae:.2f} (per cycle)", loc="left")
        ax.set_xlabel("True SOH (%)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted SOH (%)")
    leg = axes[0].legend(markerscale=3, loc="upper left")
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.suptitle(f"{name}, {PNAME[name][proto].lower()}: the envelope pulls predictions back onto the diagonal",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    save(fig, out, f"fig8_parity_{name}_{expert.replace('+', 'p')}")


# ------------------------------------------------------------------ Fig. 9
def fig_parity_grid(D, name, lit_root, out, experts=("CNN+T", "CPMLP+T", "PatchTST+T"), proto="chem"):
    from cond_models import splits
    data = []
    for expert in experts:
        ys, pa, pe, ch = [], [], [], []
        for i, (_, tr, te) in enumerate(splits(D, proto)):
            px, pb = cached(lit_root, proto, i, expert), cached(lit_root, proto, i, "base_dQ+T")
            if px is None or pb is None:
                continue
            _, _, wc = gate_split(D, tr, te)
            ys.append(D["soh"][te]); pa.append(px); pe.append(pb + wc * (px - pb)); ch.append(D["chem"][te])
        if not ys:
            raise FileNotFoundError(f"no cached predictions for {expert} / {proto} in {lit_root}")
        data.append((expert, *map(np.concatenate, (ys, pa, pe, ch))))
    lo = min(min(y.min(), a.min(), e.min()) for _, y, a, e, _ in data)
    hi = max(max(y.max(), a.max(), e.max()) for _, y, a, e, _ in data)
    fig, axes = plt.subplots(2, len(experts), figsize=(W2, 125 * MM), sharex=True, sharey=True)
    for j, (expert, y, pa, pe, ch) in enumerate(data):
        lab = DISPLAY.get(expert.replace("+T", ""), expert.replace("+T", "")) + ("+T" if expert.endswith("+T") else "")
        for r, (p, kind) in enumerate(((pa, "alone"), (pe, "enveloped"))):
            ax = axes[r, j]
            for c, col in zip(sorted(set(ch)), [BLUE, ORANGE]):
                m = ch == c
                ax.scatter(y[m], p[m], s=3, color=col, alpha=0.45, linewidths=0,
                           label=f"test cells: {chem_name(c)}")
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
            ax.set_aspect("equal", adjustable="box")
            ax.text(0.04, 0.96, f"MAE {np.mean(np.abs(p - y)):.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)
            if r == 0:
                ax.set_title(lab, loc="center")
            if r == 1:
                ax.set_xlabel("True SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)\nalone")
    axes[1, 0].set_ylabel("Predicted SOH (%)\nenveloped")
    axes[0, 0].legend(loc="lower right", fontsize=7, markerscale=3)
    save(fig, out, f"fig8_parity_grid_{name}")


def fig_k(R, out, ks=(3, 10, 30, 100), protos=("temp", "chem", "chemtemp")):
    mean, worst = {}, {}
    for p in protos:
        for k in ks:
            f = os.path.join(R, f"fus_{p}" if k == 10 else f"fus_k{k}_{p}", "results.csv")
            if not os.path.exists(f):
                continue
            r = pd.read_csv(f)
            r = r[r.method == "FUS_pondérée"]
            mean[(p, k)], worst[(p, k)] = r.MAE_cell.mean(), r.MAE_cell.max()
    if not mean:
        raise FileNotFoundError("no fus_* results")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 65 * MM), sharex=True)
    for p, col in zip(protos, [BLUE, ORANGE, AQUA]):
        kk = [k for k in ks if (p, k) in mean]
        axes[0].plot(kk, [mean[(p, k)] for k in kk], "o-", color=col, ms=4, label=PNAME["Tongji"][p])
        axes[1].plot(kk, [worst[(p, k)] for k in kk], "o-", color=col, ms=4, label=PNAME["Tongji"][p])
    for ax, t in zip(axes, ("(a) Mean over folds", "(b) Worst fold")):
        ax.axvline(10, color=INK2, ls="--", lw=1)
        ax.set_xscale("log"); ax.set_xticks(ks, [str(k) for k in ks]); ax.minorticks_off()
        ax.set_xlabel("Gate width k (log scale)"); ax.set_title(t, loc="left")
    axes[0].set_ylabel("MAE (SOH points)")
    axes[0].legend()
    axes[1].text(10, 1.0, " k = 10, fixed a priori", color=INK2, fontsize=7, va="bottom",
                 transform=blended_transform_factory(axes[1].transData, axes[1].transAxes))
    fig.suptitle("k between 10 and 30 is a plateau; a narrow gate hurts unseen T, a wide one lets a 6-point case through",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    save(fig, out, "fig9_k_ablation")


def isu_current_table(rate_cells, mix_cells):
    """Mean per-cell MAE by model and test current, test current unseen (rate) vs seen (ratemix).
    Each argument is a list of cells.csv files (one per seed): per-seed means, then mean over seeds."""
    out = {}
    for tag, files in (("unseen", rate_cells), ("seen", mix_cells)):
        files = [f for f in files if os.path.exists(f)]
        if not files:
            raise FileNotFoundError(f"no cells.csv for {tag}")
        per_seed = []
        for f in files:
            C = pd.read_csv(f)
            C["rate"] = C.cell.str.extract(r"^ISU_(c\d)_")[0]
            per_seed.append(C.groupby(["method", "rate"]).mae.mean())
        m = pd.concat(per_seed, axis=1).mean(axis=1).unstack()
        for r in m.columns:
            out[f"{tag}_{r}"] = m[r]
        print(f"  fig10: {tag} averaged over {len(files)} seed(s)")
    return pd.DataFrame(out)


def fig_isu_current(T, out):
    """Fig. 10 - ISU-ILCC: error per model when the test current was unseen vs seen in training."""
    models = [f for f in FAMILIES if f in T.index] + (["base_dQ+T"] if "base_dQ+T" in T.index else [])
    y = np.arange(len(models))[::-1].astype(float)
    if models[-1] == "base_dQ+T":
        y[-1] -= 0.6  # small gap before the reference row
    fig, axes = plt.subplots(1, 2, figsize=(W2, (len(models) * 6 + 30) * MM), sharey=True, sharex=True)
    for ax, r, title in zip(axes, ("c2", "c5"), ("Test cells measured at C/2", "Test cells measured at C/5")):
        for yi, m in zip(y, models):
            a, b = T.loc[m, f"unseen_{r}"], T.loc[m, f"seen_{r}"]
            base = m == "base_dQ+T"
            ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.plot(a, yi, "o", ms=5, color=INK2 if base else ORANGE, mfc="white" if base else ORANGE,
                    zorder=2, label="Test current unseen in training" if yi == y[0] else None)
            ax.plot(b, yi, "o", ms=5, color=INK2 if base else BLUE, zorder=3,
                    label="Both currents in training" if yi == y[0] else None)
        ax.set_title(title, loc="left")
        ax.set_xlabel("MAE per cell (SOH points)")
        ax.set_xlim(0, max(6.5, float(np.nanmax(T.values)) * 1.05))
    axes[0].set_yticks(y, [("Amplitude base" if m == "base_dQ+T" else DISPLAY.get(m, m)) for m in models])
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("ISU-ILCC: an unseen measurement current costs about 4×; seen in training, the collapse disappears",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig10_isu_current.{ext}"), dpi=300)
    plt.close(fig)
    print("  ok  fig10_isu_current")


SIT_ORDER = [("Tongji", "cell", "Tongji · in-domain"), ("Tongji", "chem", "Tongji · unseen chemistry"),
             ("Tongji", "chemtemp", "Tongji · double shift"), ("Tongji", "temp", "Tongji · unseen T"),
             ("MICH", "cell", "MICH · in-domain"), ("MICH", "chem", "MICH · unseen formation"),
             ("MICH", "temp", "MICH · unseen T"), ("MICH", "chemtemp", "MICH · double shift"),
             ("ISU", "cell", "ISU-ILCC · in-domain"), ("ISU", "chem", "ISU-ILCC · unseen ageing path"),
             ("ISUrate", "chemcv", "ISU-ILCC · unseen current")]


def fig_strategy(summary_csv, out):
    """Fig. 11 - mean regret of expert alone, fallback alone and envelope, per dataset x protocol."""
    S = pd.read_csv(summary_csv).set_index(["dataset", "protocol"])
    rows = [(lab, S.loc[(d, p)]) for d, p, lab in SIT_ORDER if (d, p) in S.index]
    y = np.arange(len(rows))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(W2, (len(rows) * 9 + 26) * MM))
    h = 0.26
    for dy, key, col, lab in ((h, "regret_expert", ORANGE, "Expert alone"),
                              (0, "regret_fallback", INK2, "Fallback alone (ΔQ/Q_nom + T)"),
                              (-h, "regret_envelope", BLUE, "Shift-gated envelope (cell level)")):
        v = np.array([max(r[key], 0) for _, r in rows])
        ax.barh(y + dy, v, height=h * 0.9, color=col, label=lab)
        for yi, vi in zip(y + dy, v):
            if vi >= 0.05:  # values near 0 are read from the bar length; labels would collide
                ax.text(vi + 0.04, yi, f"{vi:.2f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.set_xlabel("Mean regret: MAE minus that of the better of {expert, fallback} (SOH points, mean over experts and seeds)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r["regret_expert"] for _, r in rows) * 1.12)
    h_, l_ = ax.get_legend_handles_labels()
    fig.suptitle("No single model is safe everywhere; the envelope stays near the better of the two",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.legend(h_, l_, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=3, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig11_strategy.{ext}"), dpi=300)
    plt.close(fig)
    print("  ok  fig11_strategy")


SHIFT_STYLE = [  # (label, rows selector, colour)
    ("In-domain", lambda d, s: s == "in-domain", "#9a9893"),
    ("Formation (MICH), ageing path (ISU-ILCC)", lambda d, s: s.startswith("formation"), AQUA),
    ("Unseen temperature", lambda d, s: s == "temperature", ORANGE),
    ("Unseen current (ISU-ILCC)", lambda d, s: s == "charging current", YELLOW),
    ("Chemistry, double shift (Tongji, MICH)", lambda d, s: s.startswith("chemistry") or s == "double shift", BLUE),
    ("Blended cells, nominal fallback", lambda d, s: s.startswith("blended"), MAGENTA),
]
POINT_LABEL = {("Tongji", "chemistry (NCA ↔ NCM)"): "Tongji chemistry and double shift",
               ("Tongji", "temperature"): "Tongji T", ("MICH", "temperature"): "MICH T",
               ("MICH", "double shift"): "MICH double", ("ISUrate", "charging current"): "ISU current",
               ("Blend", "blended cells (other format)"): "Blended cells",
               ("ISU", "formation (MICH) / ageing path (ISU)"): "ISU ageing path",
               ("MICH", "formation (MICH) / ageing path (ISU)"): "MICH formation"}


def fig_score_advantage(csv, out):
    """Fig. 8 - shift score vs the fallback's advantage: (a) per cell, (b) per shift type."""
    A = pd.read_csv(csv)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 78 * MM), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for lab, sel, col in SHIFT_STYLE:
        m = np.array([sel(d, s) for d, s in zip(A.dataset, A["shift"])])
        a.scatter(A.sq[m].clip(lower=0.06), A.adv[m], s=4, alpha=0.35, color=col, edgecolors="none", rasterized=True,
                  label=lab)
    # median advantage per score bin
    edges = np.array([0.06, 0.25, 0.5, 1, 2, 5, 10, 50])
    mid = np.sqrt(edges[:-1] * edges[1:])
    med = [A.adv[(A.sq >= lo) & (A.sq < hi)].median() for lo, hi in zip(edges[:-1], edges[1:])]
    a.plot(mid, med, color=INK, lw=1.4, marker="o", ms=3, label="Median per score bin")
    a.set_xscale("log")
    log_axis(a.xaxis, 0.06, 45)
    a.axvline(1, color=INK2, lw=0.8, ls="--")
    a.axvline(10, color=INK2, lw=0.6, ls=":")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_ylim(-8, 12.5)
    a.set_xlabel("Shift score s/q, median over the cell's windows")
    a.set_ylabel("MAE(expert) − MAE(fallback), SOH points")
    tr = blended_transform_factory(a.transData, a.transAxes)
    a.text(1.08, 0.98, "gate starts\nto close", transform=tr, fontsize=6.5, color=INK2, va="top")
    a.text(10.8, 0.98, "fully\nclosed", transform=tr, fontsize=6.5, color=INK2, va="top")
    a.text(0.07, 11.8, "fallback better", fontsize=6.5, color=INK2, va="top")
    a.text(0.07, -7.6, "expert better", fontsize=6.5, color=INK2, va="bottom")
    a.set_title("a  One point per test cell and seed", loc="left", fontweight="bold")

    # (b) per shift type: how often the gate closes vs how often it should have
    A["closing"], A["fb"] = A.sq > 1, A.adv > 0
    G = A.groupby(["dataset", "shift"]).agg(closing=("closing", "mean"), fb=("fb", "mean"), n=("sq", "size"))
    b.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    offs = {"Tongji chemistry and double shift": (-6, 7, "right"), "Tongji T": (6, 0, "left"),
            "MICH T": (6, 0, "left"), "MICH double": (6, -2, "left"), "ISU current": (6, 0, "left"),
            "Blended cells": (-6, 0, "right"), "ISU ageing path": (6, 4, "left"), "MICH formation": (8, 4, "left")}
    for (d, s), r in G.iterrows():
        col = next(c for _, sel, c in SHIFT_STYLE if sel(d, s))
        b.scatter(r.closing, r.fb, s=26, color=col, edgecolors="white", linewidths=0.6, zorder=3)
        lab = POINT_LABEL.get((d, s))
        if lab:
            dx, dy, ha = offs.get(lab, (6, 0, "left"))
            b.annotate(lab, (r.closing, r.fb), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                       color=INK2, ha=ha, va="center")
    b.set_xlim(-0.05, 1.08)
    b.set_ylim(-0.05, 1.08)
    b.set_xlabel("Share of cells where the gate closes (s/q > 1)")
    b.set_ylabel("Share of cells where the fallback is better")
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.text(0.02, 1.04, "gate too open", fontsize=6.5, color=INK2, va="top")
    b.text(1.04, 0.02, "gate closes,\nfallback wrong", fontsize=6.5, color=INK2, ha="right", va="bottom")
    b.set_title("b  Per dataset and shift", loc="left", fontweight="bold")
    h_, l_ = a.get_legend_handles_labels()
    for hh in h_[:-1]:
        hh.set_alpha(1)
        hh.set_sizes([14])
    fig.legend(h_, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=6.8, handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_score_advantage.{ext}"), dpi=300)
    plt.close(fig)
    print("  ok  fig_score_advantage")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", required=True, help="results root ($WORK/Inversion/results)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tongji-npz", default=None)
    ap.add_argument("--mich-npz", default=None)
    ap.add_argument("--tongji-opts", nargs=3, default=[r"^Tongji[12]", r"CY(\d+)-", r"^(Tongji\d)"],
                    metavar=("INCLUDE", "TEMP", "CHEM"))
    ap.add_argument("--mich-opts", nargs=3, default=[r"^MICH", r"_(\d+)C_0", r"MICH_(BL|MC)Form"],
                    metavar=("INCLUDE", "TEMP", "CHEM"))
    ap.add_argument("--parity-experts", nargs="+", default=["CNN+T", "CPMLP+T", "PatchTST+T"])
    ap.add_argument("--only", nargs="+", default=None, help="e.g. fig2 fig5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = args.R
    tnpz = args.tongji_npz or os.path.join(R, "benchT_Tongji", "samples.npz")
    mnpz = args.mich_npz or os.path.join(R, "benchT_MICH", "samples.npz")
    DS = {}
    for name, npz, opts in (("Tongji", tnpz, args.tongji_opts), ("MICH", mnpz, args.mich_opts)):
        if os.path.exists(npz):
            DS[name] = load_ds(npz, *opts)
        else:
            print(f"  --  {name}: {npz} not found")

    jobs = []
    for name, D in DS.items():
        jobs.append(("fig2", f"curves {name}", lambda D=D, n=name: fig_curves(D, n, args.out, temp=25.0)))
        jobs.append(("fig3", f"transfer {name}", lambda D=D, n=name: fig_transfer(D, n, args.out)))
        jobs.append(("fig7", f"gate trajectories {name}", lambda D=D, n=name: fig_gate_traj(D, n, args.out)))
    for name in ("Tongji", "MICH"):
        csv = os.path.join(R, "seeds", f"summary_{name}.csv")
        jobs.append(("fig4", f"heatmap {name}", lambda c=csv, n=name: fig_heatmap(c, n, args.out)))
    jobs.append(("fig5", "dumbbell Tongji +T", lambda: fig_dumbbell(os.path.join(R, "seeds", "summary_Tongji.csv"),
                                                                      "Tongji", args.out)))
    if DS:
        jobs.append(("fig6", "gate scores", lambda: fig_gate_scores(list(DS.items()), args.out)))
    if "Tongji" in DS:
        for e in args.parity_experts:
            jobs.append(("fig8", f"parity {e}", lambda e=e: fig_parity(DS["Tongji"], "Tongji", R, args.out, expert=e)))
    jobs.append(("fig9", "k ablation", lambda: fig_k(R, args.out)))
    jobs.append(("fig10", "ISU current seen / unseen", lambda: fig_isu_current(
        isu_current_table([os.path.join(R, "ISU_rate", "wrap", "cells.csv")]
                          + sorted(glob.glob(os.path.join(R, "seeds", "ISU_rate_s*", "wrap", "cells.csv"))),
                          [os.path.join(R, "ISU_ratemix", "wrap", "cells.csv")]
                          + sorted(glob.glob(os.path.join(R, "seeds", "ISU_ratemix_s*", "wrap", "cells.csv")))),
        args.out)))
    jobs.append(("fig11", "expert / fallback / envelope", lambda: fig_strategy(
        os.path.join(R, "strategy", "strategy_summary.csv"), args.out)))

    for tag, label, fn in jobs:
        if args.only and tag not in args.only:
            continue
        try:
            fn()
        except Exception as ex:  # one missing input must not stop the other figures
            print(f"  --  {tag} {label}: skipped ({type(ex).__name__}: {ex})")
            if os.environ.get("FIG_DEBUG"):
                traceback.print_exc()
    print(f"\nFigures in {args.out}/ (PDF + PNG 300 dpi)")


if __name__ == "__main__":
    main()
