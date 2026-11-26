#!/bin/bash
# Inventaire des étapes CC de charge sur les 18 sous-jeux (CPU, ~quelques minutes à 1 h selon les E/S).
#   sbatch inspect_all.slurm

#SBATCH --job-name=inspect_all
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/inspect_all_%j.out
#SBATCH --error=logs/inspect_all_%j.out

ROOT=$WORK/SODA_Battery/datasets/soh_raw
CODE=$WORK/Inversion/v12
OUT=$WORK/Inversion/results/inspect_all

module purge
module load pytorch-gpu/py3/2.5.0
# ou : conda activate $WORK/envs/mamba3

cd $CODE
srun python inspect_all.py --root $ROOT --out $OUT \
    --v-low 3.42 --v-high 3.58 --n-cells 8 --n-cycles 6
