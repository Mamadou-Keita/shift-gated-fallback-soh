#!/bin/bash
# Second unseen chemistry: Tongji NCA+NCM cells (Tongji1+2) -> blended NCM+NCA cells (Tongji3, 2.5 Ah, 9 cells),
# at the shared temperature 25 °C (mode chem: train on one chemistry label, test on the other, same T).
#   sbatch --export=ALL,SEED=0 run_blend_gpu.slurm   (then SEED=1, SEED=2)
#SBATCH --job-name=blend
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/blend_%x_%j.out
#SBATCH --error=logs/blend_%x_%j.out
set -u
SEED=${SEED:-0}
cd $WORK/Inversion/v12
python -c "import sklearn, torch; print('GPU :', torch.cuda.is_available())" || exit 1
export R=$WORK/Inversion/results
export NPZ=$R/benchT_Tongji/samples.npz
export OPTS_STR="--include-regex ^Tongji --temp-regex CY(\d+)- --chem-regex ^(Tongji3|Tongji)"
export MODES="chem"
if [ "$SEED" = "0" ]; then OUTROOT=$R/Tongji_blend; else OUTROOT=$R/seeds/Tongji_blend_s$SEED; fi
export OUTROOT
mkdir -p $OUTROOT logs
NT=8 \
MODEL_GROUPS="HGB,MLP CNN LSTM,CPMLP CPTransformer DLinear,PatchTST iTransformer PatchMLP,ConvTimeNet,GPR" \
EXTRA="--ext-repo $WORK/Inversion/v12/BatteryMFormer --seed-offset $SEED" \
bash run_lit.sh || { echo "run_lit.sh failed"; exit 1; }
python gated_wrapper.py --npz $NPZ $OPTS_STR --protocols chem --lit-root $OUTROOT --dump-cycles CNN \
  --out $OUTROOT/wrap > logs/Tongji_blend_s${SEED}_wrap.log 2>&1
grep -A 40 "MAE moyenne par cellule" logs/Tongji_blend_s${SEED}_wrap.log | head -45
echo "done : $OUTROOT/wrap/results.csv"
