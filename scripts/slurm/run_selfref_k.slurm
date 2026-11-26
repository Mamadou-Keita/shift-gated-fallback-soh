#!/bin/bash
# Sensitivity of the self-referenced fallback: ref_k = 1, 3, 5, 10 (fixed) and causal reference (k = 3, 5, 10).
# One array task per configuration; 5 settings x 3 seeds each. No training.
#   sbatch --export=ALL run_selfref_k.slurm
#SBATCH --job-name=selfrefk
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/selfrefk_%A_%a.out
#SBATCH --error=logs/selfrefk_%A_%a.out
set -u
cd $WORK/Inversion/v12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R=$WORK/Inversion/results
CFGS=("fixed 1" "fixed 5" "fixed 10" "causal 3" "causal 5" "causal 10")
read MODE K <<< "${CFGS[$SLURM_ARRAY_TASK_ID]}"
O=$R/selfref_k/${MODE}${K}
echo "config $MODE k=$K -> $O"
T="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)"
B="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji --temp-regex CY(\d+)- --chem-regex ^(Tongji3|Tongji) --protocols chem"
M="--npz $R/benchT_MICH/samples.npz --include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form"
I5="--npz $R/ISU/samples.npz --include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_ --protocols cell chem"
IR="--npz $R/ISU/samples.npz --temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_ --protocols chemcv"
F="--fallback selfref --ref-mode $MODE --ref-k $K"
root () { if [ "$2" = 0 ]; then echo "$1"; else echo "$R/seeds/${3}_s$2"; fi; }
for s in 0 1 2; do
  echo "=== seed $s"
  python gated_wrapper.py $T  $F --lit-root $(root $R $s Tongji)                       --out $O/Tongji_s$s       > /dev/null
  python gated_wrapper.py $B  $F --lit-root $(root $R/Tongji_blend $s Tongji_blend)    --out $O/Tongji_blend_s$s > /dev/null
  python gated_wrapper.py $M  $F --lit-root $(root $R/MICH $s MICH)                    --out $O/MICH_s$s         > /dev/null
  python gated_wrapper.py $I5 $F --lit-root $(root $R/ISU_c5 $s ISU_c5)                --out $O/ISU_c5_s$s       > /dev/null
  python gated_wrapper.py $IR $F --lit-root $(root $R/ISU_rate $s ISU_rate)            --out $O/ISU_rate_s$s     > /dev/null
done
sr () { echo "$O/${1}_s0/cells.csv,$O/${1}_s1/cells.csv,$O/${1}_s2/cells.csv"; }
python strategy_table.py Tongji=$(sr Tongji) Blend=$(sr Tongji_blend) MICH=$(sr MICH) ISU=$(sr ISU_c5) ISUrate=$(sr ISU_rate) \
    --out $O/strategy > $O/strategy.log
echo done
