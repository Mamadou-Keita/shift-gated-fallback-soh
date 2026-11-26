#!/bin/bash
# Shift-score ablation (4 variants x 4 datasets) + calibration sensitivity on MICH and ISU-ILCC.
# Reuses cached predictions: no training. sbatch --export=ALL run_ablation.slurm
#SBATCH --job-name=score_abl
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.out
set -u
cd $WORK/Inversion/v12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R=$WORK/Inversion/results
T="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)"
M="--npz $R/benchT_MICH/samples.npz --include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form"
I5="--npz $R/ISU/samples.npz --include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_ --protocols cell chem"
IR="--npz $R/ISU/samples.npz --temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_ --protocols chemcv"

for v in full maha recon maha_shape; do
  A=$R/ablation/$v
  echo "=== score $v"
  python gated_wrapper.py $T  --lit-root $R          --score $v --dump-cycles CNN+T --out $A/Tongji   > /dev/null
  python gated_wrapper.py $M  --lit-root $R/MICH     --score $v --dump-cycles CNN   --out $A/MICH     > /dev/null
  python gated_wrapper.py $I5 --lit-root $R/ISU_c5   --score $v --dump-cycles CNN   --out $A/ISU_c5   > /dev/null
  python gated_wrapper.py $IR --lit-root $R/ISU_rate --score $v --dump-cycles CNN   --out $A/ISU_rate > /dev/null
done
python score_ablation.py --variant full $R/ablation/full --variant maha $R/ablation/maha \
    --variant recon $R/ablation/recon --variant maha_shape $R/ablation/maha_shape --out $R/ablation/score_ablation.csv

echo "=== calibration sensitivity, MICH and ISU-ILCC (seed 0)"
for cal in cell cellmed firstk; do
  python gated_wrapper.py $M  --lit-root $R/MICH   --calib $cal --out $R/calib/MICH_$cal   > /dev/null
  python gated_wrapper.py $I5 --lit-root $R/ISU_c5 --calib $cal --out $R/calib/ISU_c5_$cal > /dev/null
done
python calib_compare.py $R/MICH/wrap   $R/calib/MICH_cell   $R/calib/MICH_cellmed   $R/calib/MICH_firstk
python calib_compare.py $R/ISU_c5/wrap $R/calib/ISU_c5_cell $R/calib/ISU_c5_cellmed $R/calib/ISU_c5_firstk
echo done
