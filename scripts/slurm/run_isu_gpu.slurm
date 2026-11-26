#!/bin/bash
# ISU-ILCC (NMC, 30 °C, 251 cells, 63 cycling conditions): all models + envelope, one experiment per job.
#
#   EXP=c5   : in-domain (C/5 windows, held-out cells, mode cell)
#              + ageing-path shift (train on one charge-C-rate class, test on another, mode chem)
#   EXP=rate : measurement-current shift C/5 <-> C/2 on held-out cells (mode chemcv)
#   EXP=dod  : ageing-path shift by depth of discharge (C/5 windows, mode chem)
#   EXP=ratemix : control, both currents in training, current as covariate (mode cell)
#
#   sbatch --export=ALL,EXP=c5   run_isu_gpu.slurm
#   sbatch --export=ALL,EXP=rate run_isu_gpu.slurm
#   sbatch --export=ALL,EXP=dod,SEED=1 run_isu_gpu.slurm
#SBATCH --job-name=isu_soh
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=16:00:00
#SBATCH --output=logs/isu_%x_%j.out
#SBATCH --error=logs/isu_%x_%j.out

set -u
SEED=${SEED:-0}
cd $WORK/Inversion/v12
echo "python : $(which python) | EXP=$EXP SEED=$SEED"
python -c "import sklearn, torch; print('GPU :', torch.cuda.is_available())" || exit 1

export R=$WORK/Inversion/results
export NPZ=$R/ISU/samples.npz
[ -f "$NPZ" ] || { echo "missing $NPZ (run isu_to_npz.py first)"; exit 1; }
case $EXP in
  c5)   export OPTS_STR="--include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_"
        export MODES="cell chem" ;;
  rate) export OPTS_STR="--temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_"
        export MODES="chemcv" ;;
  # control: both currents seen in training (held-out cells), current used as the known covariate
  # in the "T" slot (5 = C/5, 2 = C/2) -> tells whether the C/5 <-> C/2 collapse is only an extrapolation problem
  ratemix) export OPTS_STR="--temp-regex ^ISU_c(\d)_ --chem-regex ^ISU_(c\d)_"
        export MODES="cell" ;;
  dod)  export OPTS_STR="--include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _dod(A|B|C)_"
        export MODES="chem" ;;
  *) echo "unknown EXP : $EXP"; exit 1 ;;
esac
if [ "$SEED" = "0" ]; then OUTROOT=$R/ISU_$EXP; else OUTROOT=$R/seeds/ISU_${EXP}_s${SEED}; fi
export OUTROOT
mkdir -p $OUTROOT logs

# single temperature: the "+T" variants are skipped automatically
NT=8 \
MODEL_GROUPS="HGB,MLP CNN LSTM,CPMLP CPTransformer DLinear,PatchTST iTransformer PatchMLP,ConvTimeNet,GPR" \
EXTRA="--ext-repo $WORK/Inversion/v12/BatteryMFormer --seed-offset $SEED" \
bash run_lit.sh || { echo "run_lit.sh failed"; exit 1; }

python gated_wrapper.py --npz $NPZ $OPTS_STR --protocols $MODES --lit-root $OUTROOT --out $OUTROOT/wrap \
  > logs/ISU_${EXP}_s${SEED}_wrap.log 2>&1
echo "done : $OUTROOT/wrap/results.csv"
