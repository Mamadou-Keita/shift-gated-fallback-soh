#!/bin/bash
# Run INDÉPENDANT d'une graine pour un jeu de données : tous les modèles + enveloppe.
# La graine change l'initialisation des réseaux, la séparation de validation (arrêt précoce),
# le sous-échantillon du GPR et le HGB. La graine 0 = les runs déjà faits ($R pour Tongji, $R/MICH).
#
#   sbatch --export=ALL,DATASET=Tongji,SEED=1 run_seed_gpu.slurm
#   sbatch --export=ALL,DATASET=MICH,SEED=2   run_seed_gpu.slurm
#SBATCH --job-name=seed_soh
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=logs/seed_%x_%j.out
#SBATCH --error=logs/seed_%x_%j.out

set -u
cd $WORK/Inversion/v12
echo "python : $(which python) | DATASET=$DATASET SEED=$SEED"
python -c "import sklearn, torch; print('GPU :', torch.cuda.is_available())" || exit 1

export R=$WORK/Inversion/results
case $DATASET in
  Tongji) export NPZ=$R/benchT_Tongji/samples.npz
          export OPTS_STR="--include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)" ;;
  MICH)   export NPZ=$R/benchT_MICH/samples.npz
          export OPTS_STR="--include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form" ;;
  *) echo "DATASET inconnu : $DATASET"; exit 1 ;;
esac
export OUTROOT=$R/seeds/${DATASET}_s${SEED}
mkdir -p $OUTROOT

NT=8 \
MODEL_GROUPS="HGB,MLP CNN LSTM,CPMLP CPTransformer DLinear,PatchTST iTransformer PatchMLP,ConvTimeNet,GPR" \
EXTRA="--ext-repo $WORK/Inversion/v12/BatteryMFormer --seed-offset $SEED" \
bash run_lit.sh || { echo "run_lit.sh a échoué"; exit 1; }

python gated_wrapper.py --npz $NPZ $OPTS_STR --lit-root $OUTROOT --out $OUTROOT/wrap \
  > logs/seeds_${DATASET}_s${SEED}_wrap.log 2>&1
echo "terminé : $OUTROOT/wrap/results.csv"
