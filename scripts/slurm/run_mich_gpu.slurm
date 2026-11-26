#!/bin/bash
# MICH (NMC pouch, 2 températures × 2 protocoles de formation) : TOUT le pipeline sur un nœud GPU.
# Le 2e facteur (« chem » dans les scripts) est ici le protocole de formation BL / MC.
# Reprend les prédictions déjà sauvegardées (dossiers $R/lit_*/preds) : rien n'est recalculé deux fois.
#
# À ADAPTER : la ligne --account (voir « idracct » : utiliser une allocation GPU qui a des heures).
#   V100 : --account=XXX@v100                       (pas de ligne -C)
#   A100 : --account=XXX@a100  et décommenter  #SBATCH -C a100
#   H100 : --account=XXX@h100  et remplacer par  #SBATCH -C h100
#SBATCH --job-name=lit_mich
#SBATCH --account=<your_account>@v100
##SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/mich_gpu_%j.out
#SBATCH --error=logs/mich_gpu_%j.out
#SBATCH --export=ALL

set -u
cd $WORK/Inversion/v12

# Environnement : HÉRITÉ du terminal qui lance sbatch (--export=ALL). Lancer sbatch depuis un
# terminal où « mamba3 » est activé. Pas de « module purge » : il retirerait l'environnement.
echo "python : $(which python)"
python -c "import sklearn, torch" || { echo "ERREUR : environnement incorrect. Lancer sbatch depuis un terminal où mamba3 est activé."; exit 1; }

python -c "import torch; print('torch', torch.__version__, 'GPU :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

export R=$WORK/Inversion/results
export NPZ=$R/benchT_MICH/samples.npz
export OUTROOT=$R/MICH
export OPTS_STR="--include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form"
mkdir -p $OUTROOT

NT=8 \
MODEL_GROUPS="HGB,MLP CNN LSTM,CPMLP CPTransformer DLinear,PatchTST iTransformer PatchMLP,ConvTimeNet,GPR" \
EXTRA="--ext-repo $WORK/Inversion/v12/BatteryMFormer" \
bash run_lit.sh || { echo "run_lit.sh a échoué : enveloppe non calculée"; exit 1; }

# fusion seule (tableaux détaillés + pondération) pour les 4 protocoles
for m in cell temp chem chemtemp; do
  python gated_fusion.py --npz $NPZ $OPTS_STR --mode $m --out $OUTROOT/fus_$m > logs/mich_fus_$m.log 2>&1
done

# enveloppe + tests appariés
python gated_wrapper.py --npz $NPZ $OPTS_STR --lit-root $OUTROOT --out $OUTROOT/wrap 2>&1 | tee logs/mich_wrap.log
python lit_summary.py $OUTROOT/lit_*_* --out $OUTROOT/lit_summary.csv 2>&1 | tee logs/mich_lit_summary.log
