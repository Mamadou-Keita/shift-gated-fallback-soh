#!/bin/bash
# Baselines réseaux (MLP/CNN/LSTM + 7 architectures BatteryLife/BatteryMFormer) sur un nœud GPU.
# Reprend les prédictions déjà sauvegardées (dossiers $R/lit_*/preds) : rien n'est recalculé deux fois.
#
# À ADAPTER : la ligne --account (voir « idracct » : utiliser une allocation GPU qui a des heures).
#   V100 : --account=XXX@v100                       (pas de ligne -C)
#   A100 : --account=XXX@a100  et décommenter  #SBATCH -C a100
#   H100 : --account=XXX@h100  et remplacer par  #SBATCH -C h100
#SBATCH --job-name=lit_soh
#SBATCH --account=<your_account>@v100
##SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/lit_gpu_%j.out
#SBATCH --error=logs/lit_gpu_%j.out
#SBATCH --export=ALL

set -u
cd $WORK/Inversion/v12

# Environnement : HÉRITÉ du terminal qui lance sbatch (--export=ALL). Lancer sbatch depuis un
# terminal où « mamba3 » est activé. Pas de « module purge » : il retirerait l'environnement.
echo "python : $(which python)"
python -c "import sklearn, torch" || { echo "ERREUR : environnement incorrect. Lancer sbatch depuis un terminal où mamba3 est activé."; exit 1; }

python -c "import torch; print('torch', torch.__version__, 'GPU :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

export R=$WORK/Inversion/results
export NPZ=$R/benchT_Tongji/samples.npz
NT=8 \
MODEL_GROUPS="MLP CNN LSTM,CPMLP CPTransformer DLinear,PatchTST iTransformer PatchMLP,ConvTimeNet" \
EXTRA="--ext-repo $WORK/Inversion/v12/BatteryMFormer" \
bash run_lit.sh || { echo "run_lit.sh a échoué : enveloppe non calculée"; exit 1; }

# enveloppe + tests appariés sur tous les modèles
python gated_wrapper.py --npz $NPZ --include-regex '^Tongji[12]' --temp-regex 'CY(\d+)-' \
  --chem-regex '^(Tongji\d)' --lit-root $R --out $R/wrap 2>&1 | tee logs/wrap.log
