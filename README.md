# Shift-gated fallback for battery SOH estimation under distribution shift

Code for the paper:

> M. Keita et al., *An architecture-agnostic shift-gated fallback for battery
> state-of-health estimation under distribution shift*, submitted to *Energy and AI* (2026).

The method wraps any SOH estimator (the "expert") with three components:

1. a **transferable fallback**: ridge regression on the window charge amplitude
   ΔQ/Q_nom and the temperature T;
2. a **shift score** on the curve shape: Mahalanobis distance (Ledoit–Wolf covariance) of
   PCA scores of Q(V)/ΔQ and dQ/dV, plus normalized PCA reconstruction residuals; its
   threshold q is the 0.99 quantile of scores on held-out training cells (5 folds grouped by cell);
3. a **cell-level weight** w̄ (running median over the cell's windows, causal) that moves the
   estimate from the expert to the fallback: ŷ = ŷ_b + w̄ (ŷ_X − ŷ_b), with w = 1 below q,
   0 above k·q (k = 10) and log-linear in between.

No target label and no knowledge of which condition changed are used.

## Repository layout

```
src/            Python scripts (extraction, experts, envelope, analyses, figures)
scripts/slurm/  SLURM jobs used on the Jean Zay cluster (GENCI/IDRIS)
requirements.txt, PYTHON_VERSION   exact versions used for the paper
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.12
pip install -r requirements.txt
```

Seven of the twelve experts (CPMLP, CPTransformer, iTransformer, PatchTST, DLinear,
ConvTimeNet, PatchMLP) come from the BatteryMFormer repository, used at commit `febe174`:

```bash
git clone https://github.com/Ruifeng-Tan/BatteryMFormer ext/BatteryMFormer
git -C ext/BatteryMFormer checkout febe174
```

Pass `--ext-repo ext/BatteryMFormer` to the scripts that train these models.

## Data

All datasets are public and are not redistributed here.

| Dataset | Cells | Source |
|---|---|---|
| Tongji | 105 (50 NCA, 55 NCM) + 9 blended NCM+NCA | BatteryLife processed release |
| MICH | 40 NMC (BL / MC formation) | BatteryLife processed release |
| ISU-ILCC | 251 NMC | original release |

Set `DATA` to the folder holding the BatteryLife processed sub-datasets (`$DATA/Tongji`,
`$DATA/MICH`) and `R` to a results folder.

### Window extraction

One partial constant-current charging window every 10th cycle, 128 points on a fixed voltage
grid (Supplementary Section S1):

```bash
cd src
python identifiability_test.py run --data-dir $DATA/Tongji --v-low 3.326 --v-high 4.185 \
    --cc-crate 0.50 --cycle-stride 10 --out $R/benchT_Tongji        # 114 cells, 5,499 windows
python identifiability_test.py run --data-dir $DATA/MICH --v-low 3.763 --v-high 4.185 \
    --cc-crate 1.00 --cycle-stride 10 --out $R/benchT_MICH          # 40 cells, 1,816 windows
python isu_to_npz.py --help                                          # ISU-ILCC raw files -> samples.npz
```

Check that the extraction matches the paper (cycle by cycle):

```bash
python audit_windows.py rebuild --data-dir $DATA/Tongji --npz $R/benchT_Tongji/samples.npz \
    --cycle-stride 10 --out $R/audit/Tongji
```

The voltage windows were chosen by `inspect_all.py` (inventory of CC steps) and
`plan_datasets.py` (explicit inclusion criteria).

## Reproducing the paper

| Step | Main scripts |
|---|---|
| Train the 12 experts on every protocol (3 seeds) | `lit_baselines.py`; jobs `run_lit_gpu.slurm`, `run_mich_gpu.slurm`, `run_isu_gpu.slurm`, `run_blend_gpu.slurm`, `run_seed_gpu.slurm` |
| Fallback, shift score, calibration, cell-level envelope | `gated_wrapper.py`, `gated_fusion.py` |
| Expert vs fallback vs envelope, regret | `strategy_table.py`, `seed_summary.py`, `seed_sd.py` |
| False alarms | `false_alarm.py`, `false_alarm_boot.py` |
| Score ablation | `score_ablation.py` (job `run_ablation.slurm`) |
| Progressive weight vs hard switch | `decision_rule_compare.py` |
| Fixed blend without detection | `fixed_alpha.py`, `inner_oof.py`, `alpha_cv.py` |
| Blended cells and self-referenced fallback | `blend_check.py`, `late_init.py`, `selfref_k_summary.py` (jobs `run_selfref*.slurm`) |
| ISU-ILCC current split | `isu_rate_split.py` |
| Supplementary S2 to S5 | `calib_compare.py`, `startup_analysis.py`, `soh_first_windows.py`, `model_tables_s5.py` |
| Figures and graphical abstract | `make_figures.py`, `score_vs_advantage.py`, `graphical_abstract.py` |

Example (Tongji, unseen-chemistry protocol, envelope around all experts):

```bash
python gated_wrapper.py --npz $R/benchT_Tongji/samples.npz --include-regex '^Tongji[12]' \
    --temp-regex 'CY(\d+)-' --chem-regex '^(Tongji\d)' --help
```

Each script documents its inputs and outputs in its header (`python <script>.py --help`).
The SLURM scripts expect `$WORK`-based paths and an account name: edit
`--account=<your_account>` and the paths before submitting.

## Citation

```bibtex
@article{keita2026shiftgated,
  author  = {Keita, Mamadou},
  title   = {An architecture-agnostic shift-gated fallback for battery state-of-health estimation under distribution shift},
  journal = {Energy and AI},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT (see `LICENSE`). Third-party models keep their own licenses.
