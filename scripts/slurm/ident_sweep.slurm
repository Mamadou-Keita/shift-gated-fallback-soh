#!/bin/bash
# Test d'identifiabilité SOH sur MATR : balayage de fenêtres de tension en job array (CPU).
# Le test n'utilise que numpy/scikit-learn : pas besoin de GPU, donc pas d'heures V100 consommées.
#
# Soumission :   sbatch ident_sweep.slurm
# Suivi :        squeue -u $USER ; tail -f logs/ident_*.out

#SBATCH --job-name=ident_soh
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-dev          # 2 h max, démarrage rapide ; qos_cpu-t3 si plus long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/ident_%A_%a.out
#SBATCH --error=logs/ident_%A_%a.out

# Fenêtres testées (v_low v_high) : une par tâche du job array
WINDOWS=(
  "3.42 3.58"
  "3.42 3.50"
  "3.50 3.58"
  "3.45 3.55"
  "3.40 3.59"
  "3.55 3.59"
)
read VLOW VHIGH <<< "${WINDOWS[$SLURM_ARRAY_TASK_ID]}"

DATA=$WORK/Inversion/data/MATR
CODE=$WORK/Inversion/v12
OUT=$WORK/Inversion/results/ident_${VLOW}_${VHIGH}

module purge
module load pytorch-gpu/py3/2.5.0     # fournit numpy, scipy, pandas, scikit-learn, matplotlib
# Si tu préfères ton env : conda activate $WORK/envs/mamba3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $CODE
mkdir -p logs
echo "Fenêtre [$VLOW, $VHIGH] V -> $OUT"
srun python identifiability_test.py run \
    --data-dir $DATA \
    --v-low $VLOW --v-high $VHIGH \
    --cc-crate 0.91 \
    --cycle-stride 5 \
    --out $OUT
