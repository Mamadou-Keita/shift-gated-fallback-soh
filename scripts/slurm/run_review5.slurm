#!/bin/bash
# Reviewer-2 checks, no training: decision-rule ablation (progressive vs hard switch), cell-level false alarms
# with CIs, aggregation of the chemistry transfers, and a look at Tongji's blended cells.
#   sbatch --export=ALL run_review5.slurm
#SBATCH --job-name=review5
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/review5_%j.out
#SBATCH --error=logs/review5_%j.out
set -u
cd $WORK/Inversion/v12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R=$WORK/Inversion/results
T="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)"
M="--npz $R/benchT_MICH/samples.npz --include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form"
I5="--npz $R/ISU/samples.npz --include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_ --protocols cell chem"
IR="--npz $R/ISU/samples.npz --temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_ --protocols chemcv"
root () { if [ "$2" = 0 ]; then echo "$1"; else echo "$R/seeds/${3}_s$2"; fi; }

for s in 0 1 2; do
  echo "=== seed $s"
  python gated_wrapper.py $T  --lit-root $(root $R $s Tongji)          --dump-cycles CNN+T --out $R/dr/Tongji_s$s   > /dev/null
  python gated_wrapper.py $M  --lit-root $(root $R/MICH $s MICH)       --dump-cycles CNN   --out $R/dr/MICH_s$s     > /dev/null
  python gated_wrapper.py $I5 --lit-root $(root $R/ISU_c5 $s ISU_c5)   --dump-cycles CNN   --out $R/dr/ISU_c5_s$s   > /dev/null
  python gated_wrapper.py $IR --lit-root $(root $R/ISU_rate $s ISU_rate) --dump-cycles CNN --out $R/dr/ISU_rate_s$s > /dev/null
done
d () { echo "$R/dr/${1}_s0,$R/dr/${1}_s1,$R/dr/${1}_s2"; }
c () { echo "$R/dr/${1}_s0/cycles.csv,$R/dr/${1}_s1/cycles.csv,$R/dr/${1}_s2/cycles.csv"; }
echo; echo "######## 1. Decision rule: progressive weight vs hard switch"
python decision_rule_compare.py Tongji=$(d Tongji) MICH=$(d MICH) ISU=$(d ISU_c5) ISUrate=$(d ISU_rate) --out $R/dr/decision_rule.csv
echo; echo "######## 2. False alarms on in-domain cells"
python false_alarm.py Tongji=$(c Tongji) MICH=$(c MICH) ISU=$(c ISU_c5)
echo; echo "######## 3. Splits of the chemistry protocol (Tongji) and cells per split"
python - <<'PY'
import os, pandas as pd
R = os.path.expandvars("$WORK/Inversion/results")
C = pd.read_csv(f"{R}/dr/Tongji_s0/cells.csv")
print(C[C.protocol == "chem"].groupby("split").cell.nunique().to_string())
PY
echo; echo "######## 4. Tongji blended cells in the samples file?"
python - <<'PY'
import os, re, numpy as np, pandas as pd
R = os.path.expandvars("$WORK/Inversion/results")
D = np.load(f"{R}/benchT_Tongji/samples.npz", allow_pickle=True)
g = pd.Series(D["groups"].astype(str))
pref = g.str.extract(r"^(Tongji\d)")[0]
T = g.str.extract(r"CY(\d+)-")[0]
print(pd.crosstab(pref, T, values=g, aggfunc="nunique").fillna(0).astype(int).to_string())
print("window:", D["v_grid"][0], "-", D["v_grid"][-1])
for k in ("qnom", "soh"):
    if k in D: print(k, pd.Series(D[k]).groupby(pref).describe()[["min", "50%", "max"]].round(3).to_string())
PY
echo done
