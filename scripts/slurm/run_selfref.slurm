#!/bin/bash
# Self-referenced fallback inside the full envelope, 4 settings + Tongji blend, 3 seeds. No training.
#   sbatch --export=ALL run_selfref.slurm
#SBATCH --job-name=selfref
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/selfref_%j.out
#SBATCH --error=logs/selfref_%j.out
set -u
cd $WORK/Inversion/v12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R=$WORK/Inversion/results
T="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)"
B="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji --temp-regex CY(\d+)- --chem-regex ^(Tongji3|Tongji) --protocols chem"
M="--npz $R/benchT_MICH/samples.npz --include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form"
I5="--npz $R/ISU/samples.npz --include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_ --protocols cell chem"
IR="--npz $R/ISU/samples.npz --temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_ --protocols chemcv"
root () { if [ "$2" = 0 ]; then echo "$1"; else echo "$R/seeds/${3}_s$2"; fi; }
for s in 0 1 2; do
  echo "=== seed $s"
  python gated_wrapper.py $T  --fallback selfref --lit-root $(root $R $s Tongji)               --out $R/selfref/Tongji_s$s       > /dev/null
  python gated_wrapper.py $B  --fallback selfref --lit-root $(root $R/Tongji_blend $s Tongji_blend) --out $R/selfref/Tongji_blend_s$s > /dev/null
  python gated_wrapper.py $M  --fallback selfref --lit-root $(root $R/MICH $s MICH)            --out $R/selfref/MICH_s$s         > /dev/null
  python gated_wrapper.py $I5 --fallback selfref --lit-root $(root $R/ISU_c5 $s ISU_c5)        --out $R/selfref/ISU_c5_s$s       > /dev/null
  python gated_wrapper.py $IR --fallback selfref --lit-root $(root $R/ISU_rate $s ISU_rate)    --out $R/selfref/ISU_rate_s$s     > /dev/null
done
sr () { echo "$R/selfref/${1}_s0/cells.csv,$R/selfref/${1}_s1/cells.csv,$R/selfref/${1}_s2/cells.csv"; }
nm () { echo "$R/dr/${1}_s0/cells.csv,$R/dr/${1}_s1/cells.csv,$R/dr/${1}_s2/cells.csv"; }
bl () { echo "$R/Tongji_blend/wrap/cells.csv,$R/seeds/Tongji_blend_s1/wrap/cells.csv,$R/seeds/Tongji_blend_s2/wrap/cells.csv"; }
echo; echo "######## Nominal fallback (paper)"
python strategy_table.py Tongji=$(nm Tongji) Blend=$(bl) MICH=$(nm MICH) ISU=$(nm ISU_c5) ISUrate=$(nm ISU_rate) --out $R/selfref/strategy_nominal
echo; echo "######## Self-referenced fallback"
python strategy_table.py Tongji=$(sr Tongji) Blend=$(sr Tongji_blend) MICH=$(sr MICH) ISU=$(sr ISU_c5) ISUrate=$(sr ISU_rate) --out $R/selfref/strategy_selfref
echo done
