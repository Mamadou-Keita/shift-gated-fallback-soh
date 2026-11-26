#!/usr/bin/env python3
"""
Modèles d'estimation du SOH de la littérature, évalués avec LES MÊMES protocoles et LES MÊMES
plis que la fusion pondérée (cond_models.splits), dans un tableau commun.

Entrée des modèles de la littérature = la courbe de charge partielle elle-même, comme dans les
articles : Q(V)/Q_nom sur la fenêtre (= forme × ΔQ/Q_nom, amplitude ET forme) et dQ/dV,
128 points chacun, standardisés point par point sur l'entraînement. Variante « +T » : la
température de la condition est ajoutée en entrée (tête dense) ; variante « sans T » : non.

  GPR    processus gaussien (RBF à pertinence automatique + noyau linéaire + bruit blanc) sur PCA(courbe) + PCA(dQ/dV)
  HGB    gradient boosting sur les mêmes composantes
  MLP    réseau dense sur les 2 × 128 points
  CNN    CNN 1D (3 couches de convolution) sur les 2 canaux × 128 points
  LSTM   LSTM sur la courbe parcourue en tension (2 variables par pas, 64 pas)

Réseaux : Adam, arrêt précoce sur 20 % des cellules d'entraînement (séparation PAR CELLULE),
cible standardisée ; --seeds n > 1 moyenne n initialisations.

Référence dans le même tableau : FUS_pondérée (gated_fusion.py), base_dQ+T et tout+T.

  python lit_baselines.py --npz benchT_Tongji/samples.npz --include-regex '^Tongji[12]' \
      --temp-regex 'CY(\\d+)-' --chem-regex '^(Tongji\\d)' --mode chem --out lit_chem
"""
import argparse
import os
import time
import warnings

# limiter les threads BLAS : sur le nœud de connexion, le temps CPU de TOUS les threads est décompté
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

from cond_models import boot_ci, cell_mae, load, splits
from gated_fusion import fusion_predict

ALL_MODELS = ["GPR", "HGB", "MLP", "CNN", "LSTM"]
# architectures du dépôt BatteryLife / BatteryMFormer (Tan et al.), importées telles quelles (--ext-repo)
EXT_MODELS = ["CPMLP", "CPTransformer", "PatchTST", "iTransformer", "DLinear", "ConvTimeNet", "PatchMLP"]
EXT_REPO = None


# ---------------------------------------------------------------- entrées
def curves(D, tr, te):
    """Canaux (n, 2, L) : Q(V)/Q_nom et dQ/dV, standardisés point par point sur tr."""
    q = D["shape"] * D["dqn"][:, None]
    X = np.stack([q, D["ic"]], 1).astype(np.float32)
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    return (X[tr] - mu) / sd, (X[te] - mu) / sd


def pca_inputs(D, tr, te, n_pca):
    from sklearn.decomposition import PCA
    q = D["shape"] * D["dqn"][:, None]
    A, B = [], []
    for M in (q, D["ic"]):
        p = PCA(n_components=n_pca).fit(M[tr])
        A.append(p.transform(M[tr])); B.append(p.transform(M[te]))
    A, B = np.hstack(A), np.hstack(B)
    mu, sd = A.mean(0), A.std(0) + 1e-12
    return (A - mu) / sd, (B - mu) / sd


def t_col(D, tr, te):
    T = D["T"]
    mu, sd = T[tr].mean(), T[tr].std()
    sd = sd if sd > 0 else 1.0
    return ((T[tr] - mu) / sd)[:, None].astype(np.float32), ((T[te] - mu) / sd)[:, None].astype(np.float32)


# ---------------------------------------------------------------- modèles classiques
def fit_gpr(Xtr, ytr, Xte, seed, max_n):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(max_n, len(Xtr)), replace=False)
    # RBF à pertinence automatique + composante linéaire : sans la partie linéaire, le RBF ignore
    # les tendances globales (T notamment) et revient à la moyenne hors des données d'entraînement
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(Xtr.shape[1]), length_scale_bounds=(1e-2, 1e3)) \
        + ConstantKernel(0.1) * DotProduct(1.0) + WhiteKernel(1e-2, (1e-6, 1e1))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xtr[idx], ytr[idx])
    return gp.predict(Xte)


def fit_hgb(Xtr, ytr, Xte, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=seed)
    return h.fit(Xtr, ytr).predict(Xte)


# ---------------------------------------------------------------- réseaux
def load_ext(name):
    """Importe models/<name>.py du dépôt externe sans passer par models/__init__.py (qui importe
    tous les modèles) ; reformer_pytorch (attention LSH, inutilisée ici) est remplacé par un bouchon."""
    import importlib
    import sys
    import types
    if EXT_REPO not in sys.path:
        sys.path.insert(0, EXT_REPO)
    if "reformer_pytorch" not in sys.modules:
        try:
            import reformer_pytorch  # noqa: F401
        except ImportError:
            stub = types.ModuleType("reformer_pytorch"); stub.LSHSelfAttention = object
            sys.modules["reformer_pytorch"] = stub
    if not hasattr(sys.modules.get("models"), "__path__") or EXT_REPO not in str(sys.modules["models"].__path__):
        pkg = types.ModuleType("models"); pkg.__path__ = [os.path.join(EXT_REPO, "models")]
        sys.modules["models"] = pkg
    return importlib.import_module(f"models.{name}").Model


def build_ext(kind, L, use_t):
    """Modèle externe sur UN cycle : entrée [B, 1 cycle, 3 variables, L] = Q(V)/Q_nom, dQ/dV et la
    grille de tension (3 variables comme V, I, Q dans BatteryLife), sortie = 1 valeur (SOH).
    Variante +T : terme additif appris a·T + b en sortie. Pour les modèles qui normalisent chaque
    variable par instance (PatchTST, iTransformer, PatchMLP), une colonne T constante serait
    effacée par cette normalisation : c'est pourquoi T entre en sortie et non comme 4e variable."""
    import torch
    import torch.nn as nn
    cfg = types_ns(task_name="long_term_forecast", input_mode="current_voltage", seq_len=1,
                   early_cycle_threshold=1, charge_discharge_length=L, pred_len=1, d_model=64, d_ff=128,
                   dropout=0.1, e_layers=2, d_layers=1, n_heads=4, factor=1, activation="gelu",
                   patch_len=16, stride=8, enc_in=3, embed="timeF", freq="h", moving_avg=25,
                   output_attention=False, individual=False, label_len=0)
    Model = load_ext(kind)

    class Ext(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = Model(cfg)
            self.register_buffer("vg", torch.linspace(-1, 1, L).view(1, 1, L))
            self.tlin = nn.Linear(1, 1) if use_t else None

        def forward(self, x, t):
            B = x.shape[0]
            z = torch.cat([x, self.vg.expand(B, 1, L)], 1).unsqueeze(1)  # [B, 1, 3, L]
            out = self.m(cycle_curve_data=z, curve_attn_mask=torch.ones(B, 1, device=x.device))
            out = (out[0] if isinstance(out, tuple) else out).reshape(B)
            return out + self.tlin(t).squeeze(1) if t is not None else out

    return Ext()


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def build_net(kind, L, use_t):
    import torch.nn as nn
    import torch
    if kind in EXT_MODELS:
        return build_ext(kind, L, use_t)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            nt = 1 if use_t else 0
            if kind == "MLP":
                self.enc = nn.Sequential(nn.Flatten(), nn.Linear(2 * L, 128), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(128, 64), nn.ReLU())
                d = 64
            elif kind == "CNN":
                self.enc = nn.Sequential(
                    nn.Conv1d(2, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten())
                d = 32 * (L // 8)
            elif kind == "LSTM":
                self.pool = nn.AvgPool1d(2)
                self.lstm = nn.LSTM(2, 32, batch_first=True)
                d = 32
            self.kind = kind
            self.head = nn.Sequential(nn.Linear(d + nt, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, t):
            if self.kind == "LSTM":
                _, (h, _) = self.lstm(self.pool(x).transpose(1, 2))
                z = h[-1]
            else:
                z = self.enc(x)
            if t is not None:
                z = torch.cat([z, t], 1)
            return self.head(z).squeeze(1)

    return Net()


def fit_net(kind, Xtr, ttr, ytr, gtr, Xte, tte, seed, args):
    """LSTM : cuDNN désactivé (sur Jean Zay, l'environnement conda + module pytorch-gpu donne
    CUDNN_STATUS_BAD_PARAM à l'initialisation des poids RNN) ; noyaux CUDA natifs, mêmes résultats."""
    import contextlib
    import torch
    ctx = torch.backends.cudnn.flags(enabled=False) if kind == "LSTM" else contextlib.nullcontext()
    with ctx:
        return _fit_net(kind, Xtr, ttr, ytr, gtr, Xte, tte, seed, args)


def _fit_net(kind, Xtr, ttr, ytr, gtr, Xte, tte, seed, args):
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    torch.manual_seed(seed); np.random.seed(seed)
    ym, ys = ytr.mean(), ytr.std() + 1e-9
    yz = ((ytr - ym) / ys).astype(np.float32)
    i_fit, i_val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(Xtr, ytr, gtr))
    use_t = ttr is not None
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                       (args.device if args.device != "auto" else "cpu"))
    net = build_net(kind, Xtr.shape[2], use_t).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tt = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    Xf, Xv, yf, yv = tt(Xtr[i_fit]), tt(Xtr[i_val]), tt(yz[i_fit]), tt(yz[i_val])
    tf = tt(ttr[i_fit]) if use_t else None
    tv = tt(ttr[i_val]) if use_t else None
    best, best_state, wait = np.inf, None, 0
    for ep in range(args.max_epochs):
        net.train()
        perm = torch.randperm(len(Xf), device=dev)
        for b in range(0, len(Xf), args.batch):
            j = perm[b:b + args.batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xf[j], tf[j] if use_t else None), yf[j])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.nn.functional.l1_loss(net(Xv, tv), yv).item()
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: x.clone() for k, x in net.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        p = net(tt(Xte), tt(tte) if use_t else None).cpu().numpy()
    return p * ys + ym


# ---------------------------------------------------------------- un pli
def run_split(D, i_split, name, tr, te, args, alphas):
    """Chaque prédiction (pli × méthode) est sauvegardée dès qu'elle est calculée : un run tué
    (limite de temps CPU du nœud de connexion) reprend là où il s'est arrêté."""
    y, g = D["soh"], D["groups"]
    cache = os.path.join(args.out, "preds")
    os.makedirs(cache, exist_ok=True)
    path = lambda lab: os.path.join(cache, f"s{i_split:02d}__{lab.replace('+', 'p')}.npy")
    preds = {}
    t0 = time.time()
    refs = ("base_dQ+T", "tout+T", "FUS_pondérée")
    if all(os.path.exists(path(k)) for k in refs):
        for k in refs:
            preds[k] = np.load(path(k))
    else:
        fp, _ = fusion_predict(D, tr, te, args, alphas)
        for k in refs:
            preds[k] = fp[k]; np.save(path(k), fp[k])
    t_ok = len(set(D["T"][tr])) >= 2
    inputs = {}

    def get(kind):
        if kind not in inputs:
            inputs[kind] = pca_inputs(D, tr, te, args.n_pca) if kind == "pca" else curves(D, tr, te)
        return inputs[kind]

    ttr, tte = t_col(D, tr, te)
    for m in args.models:
        for use_t in (False, True):
            if use_t and not t_ok:
                continue  # T constante à l'entraînement : variante +T sans objet
            lab = f"{m}{'+T' if use_t else ''}"
            if os.path.exists(path(lab)):
                preds[lab] = np.load(path(lab)); continue
            t1 = time.time()
            ps = []
            for s in range(args.seed_offset, args.seed_offset + args.seeds):
                if m in ("GPR", "HGB"):
                    Ptr, Pte = get("pca")
                    A = np.hstack([Ptr, ttr]) if use_t else Ptr
                    B = np.hstack([Pte, tte]) if use_t else Pte
                    ps.append(fit_gpr(A, y[tr], B, s, args.gpr_max_n) if m == "GPR" else fit_hgb(A, y[tr], B, s))
                else:
                    Ctr, Cte = get("curves")
                    ps.append(fit_net(m, Ctr, ttr if use_t else None, y[tr], g[tr], Cte,
                                      tte if use_t else None, s, args))
            preds[lab] = np.mean(ps, 0)
            np.save(path(lab), preds[lab])
            print(f"    {name} / {lab} : {time.time() - t1:.0f} s", flush=True)
    rows, percell = [], {}
    for lab, p in preds.items():
        cm = cell_mae(p - y[te], g[te])
        percell[lab] = cm
        lo, hi = boot_ci(cm.values)
        rows.append(dict(split=name, method=lab, MAE_cell=cm.mean(), ci_lo=lo, ci_hi=hi, n_test_cells=len(cm)))
    comps = []
    for lab in preds:
        if lab == "FUS_pondérée":
            continue
        d = (percell["FUS_pondérée"] - percell[lab]).values
        lo, hi = boot_ci(d, seed=1)
        comps.append(dict(split=name, comparison=f"FUS − {lab}", diff=d.mean(), ci_lo=lo, ci_hi=hi,
                          frac_cells_fus_better=np.mean(d < 0)))
    print(f"  {name} : {len(np.unique(g[tr]))} cellules d'entraînement, {len(np.unique(g[te]))} de test "
          f"({time.time() - t0:.0f} s)", flush=True)
    return rows, comps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="lit_out")
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--temp-regex", required=True)
    ap.add_argument("--chem-regex", default=None)
    ap.add_argument("--mode", choices=["cell", "temp", "chem", "chemtemp", "chemcv"], default="cell")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS + EXT_MODELS)
    ap.add_argument("--ext-repo", default=None, help="Clone de github.com/Ruifeng-Tan/BatteryMFormer (pour CPMLP, CPTransformer, PatchTST, …)")
    ap.add_argument("--seeds", type=int, default=1, help="Nombre d'initialisations MOYENNÉES (ensemble)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Graine de départ : runs indépendants = --seed-offset 0, 1, 2 dans des dossiers séparés")
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--gpr-max-n", type=int, default=600, help="Sous-échantillon d'entraînement du GPR (coût n³)")
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-threads", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto (GPU si disponible), cpu ou cuda")
    # paramètres de la fusion (identiques à gated_fusion.py ; ne pas les régler sur les résultats de test)
    ap.add_argument("--corr-features", default="shape+ic")
    ap.add_argument("--gate-q", type=float, default=0.99)
    ap.add_argument("--gate-k", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global EXT_REPO
    if any(m in EXT_MODELS for m in args.models):
        if not args.ext_repo:
            raise SystemExit("--ext-repo requis pour " + ", ".join(m for m in args.models if m in EXT_MODELS))
        EXT_REPO = os.path.abspath(args.ext_repo)
    if any(m not in ("GPR", "HGB") for m in args.models):
        import torch
        torch.set_num_threads(args.n_threads)
    alphas = np.logspace(-3, 4, 15)

    D = load(args)
    print(f"{len(D['soh'])} échantillons, {len(set(D['groups']))} cellules, protocole {args.mode}")
    rows, comps = [], []
    for i_split, (name, tr, te) in enumerate(splits(D, args.mode)):
        r, c = run_split(D, i_split, name, tr, te, args, alphas)
        rows += r; comps += c
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "results.csv"), index=False)  # sauvegarde au fil de l'eau
    R, C = pd.DataFrame(rows), pd.DataFrame(comps)
    R.to_csv(os.path.join(args.out, "results.csv"), index=False)
    C.to_csv(os.path.join(args.out, "paired.csv"), index=False)

    pd.set_option("display.width", 250)
    M = R.pivot(index="split", columns="method", values="MAE_cell")
    order = ["FUS_pondérée", "base_dQ+T", "tout+T"] + [c for c in M.columns if c not in
                                                       ("FUS_pondérée", "base_dQ+T", "tout+T")]
    M = M[order]
    print("\n=== MAE moyenne par cellule (points de SOH), par pli ===")
    print(M.round(2).to_string())
    S = pd.DataFrame({"moyenne": M.mean(), "pire pli": M.max(),
                      "rang moyen": M.rank(axis=1).mean()}).sort_values("moyenne")
    print("\n=== Résumé du protocole ===")
    print(S.round(2).to_string())
    C["txt"] = C.apply(lambda x: f"{x['diff']:+.2f} [{x.ci_lo:+.2f};{x.ci_hi:+.2f}] ({x.frac_cells_fus_better:.0%})", axis=1)
    print("\n=== FUS_pondérée − modèle, apparié par cellule (négatif = la fusion est meilleure) "
          "[IC 95 %] (part des cellules où la fusion est meilleure) ===")
    print(C.pivot(index="comparison", columns="split", values="txt").to_string())
    print(f"\nFichiers dans {args.out}/")


if __name__ == "__main__":
    main()
