#!/bin/bash
# One array task = one lit_<protocol>_<models> directory: out-of-fold predictions of its models (and of the
# fallback) on the training cells of every split (inner_oof.py). Resumable: rerunning the same task skips
# what is already saved. Submit through make_inner_tasks.sh.
#SBATCH --job-name=inneroof
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/inner_%A_%a.out
#SBATCH --error=logs/inner_%A_%a.out
set -u
cd $WORK/Inversion/v12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R=$WORK/Inversion/results
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inner_tasks.txt)
IFS='|' read -r DIR SET SEED <<< "$LINE"
case $SET in
  Tongji)   O="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)" ;;
  Blend)    O="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji --temp-regex CY(\d+)- --chem-regex ^(Tongji3|Tongji)" ;;
  MICH)     O="--npz $R/benchT_MICH/samples.npz --include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form" ;;
  ISU_c5)   O="--npz $R/ISU/samples.npz --include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_" ;;
  ISU_rate) O="--npz $R/ISU/samples.npz --temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_" ;;
  *) echo "unknown setting $SET"; exit 1 ;;
esac
echo "task $SLURM_ARRAY_TASK_ID: $DIR ($SET, seed $SEED)"
python -X faulthandler inner_oof.py --lit-dir $DIR $O --seed-offset $SEED --n-threads 8 \
  --ext-repo $WORK/Inversion/v12/BatteryMFormer
