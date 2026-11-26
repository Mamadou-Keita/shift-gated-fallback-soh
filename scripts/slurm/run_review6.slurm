#!/bin/bash
# Review 6: (1) fixed-weight blend without detection, (2) shift score vs the fallback's real advantage,
# (3) late initialisation of the self-referenced fallback, (4) NCA/NCM shape separation at matched SOH,
# (5) cluster-bootstrap false alarms, plus macro-averaged regrets. No training: cached predictions only.
#   sbatch --export=ALL run_review6.slurm
#SBATCH --job-name=review6
#SBATCH --account=<your_account>@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/review6_%j.out
#SBATCH --error=logs/review6_%j.out
set -u
cd $WORK/Inversion/v12
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R=$WORK/Inversion/results
O=$R/review6
T="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)"
B="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji --temp-regex CY(\d+)- --chem-regex ^(Tongji3|Tongji) --protocols chem"
M="--npz $R/benchT_MICH/samples.npz --include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form"
I5="--npz $R/ISU/samples.npz --include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_ --protocols cell chem"
IR="--npz $R/ISU/samples.npz --temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_ --protocols chemcv"
root () { if [ "$2" = 0 ]; then echo "$1"; else echo "$R/seeds/${3}_s$2"; fi; }
for s in 0 1 2; do
  echo "=== seed $s : envelope + fixed blends + per-window dump (all experts)"
  python gated_wrapper.py $T  --dump-cycles --lit-root $(root $R $s Tongji)                    --out $O/wrap/Tongji_s$s       > /dev/null
  python gated_wrapper.py $B  --dump-cycles --lit-root $(root $R/Tongji_blend $s Tongji_blend) --out $O/wrap/Tongji_blend_s$s > /dev/null
  python gated_wrapper.py $M  --dump-cycles --lit-root $(root $R/MICH $s MICH)                 --out $O/wrap/MICH_s$s         > /dev/null
  python gated_wrapper.py $I5 --dump-cycles --lit-root $(root $R/ISU_c5 $s ISU_c5)             --out $O/wrap/ISU_c5_s$s       > /dev/null
  python gated_wrapper.py $IR --dump-cycles --lit-root $(root $R/ISU_rate $s ISU_rate)         --out $O/wrap/ISU_rate_s$s     > /dev/null
  echo "=== seed $s : late initialisation"
  python late_init.py $T  --lit-root $(root $R $s Tongji)                    --out $O/late/Tongji_s$s       > /dev/null
  python late_init.py $B  --lit-root $(root $R/Tongji_blend $s Tongji_blend) --out $O/late/Tongji_blend_s$s > /dev/null
  python late_init.py $M  --lit-root $(root $R/MICH $s MICH)                 --out $O/late/MICH_s$s         > /dev/null
  python late_init.py $I5 --lit-root $(root $R/ISU_c5 $s ISU_c5)             --out $O/late/ISU_c5_s$s       > /dev/null
done
f () { echo "$O/$1/${2}_s0/$3,$O/$1/${2}_s1/$3,$O/$1/${2}_s2/$3"; }
echo; echo "######## (1) Fixed blend without detection"
python fixed_alpha.py Tongji=$(f wrap Tongji cells.csv) Blend=$(f wrap Tongji_blend cells.csv) MICH=$(f wrap MICH cells.csv) \
  ISU=$(f wrap ISU_c5 cells.csv) ISUrate=$(f wrap ISU_rate cells.csv) --out $O/fixed_alpha
echo; echo "######## (2) Shift score vs the fallback's advantage"
python score_vs_advantage.py Tongji=$(f wrap Tongji cycles.csv) Blend=$(f wrap Tongji_blend cycles.csv) MICH=$(f wrap MICH cycles.csv) \
  ISU=$(f wrap ISU_c5 cycles.csv) ISUrate=$(f wrap ISU_rate cycles.csv) --out $O/score_vs_adv
echo; echo "######## (3) Late initialisation of the self-referenced fallback"
python late_summary.py Tongji=$(f late Tongji cells.csv) Blend=$(f late Tongji_blend cells.csv) MICH=$(f late MICH cells.csv) \
  ISU=$(f late ISU_c5 cells.csv) --out $O/late_summary.csv
echo; echo "######## (4) NCA / NCM shape separation at matched SOH"
python shape_matched_soh.py --npz $R/benchT_Tongji/samples.npz --include-regex '^Tongji[12]' --temp-regex 'CY(\d+)-' \
  --chem-regex '^(Tongji\d)' --temp 25 --out $O/shape_matched_25C.csv
python shape_matched_soh.py --npz $R/benchT_Tongji/samples.npz --include-regex '^Tongji[12]' --temp-regex 'CY(\d+)-' \
  --chem-regex '^(Tongji\d)' --out $O/shape_matched_allT.csv
echo; echo "######## (5) False alarms, cluster bootstrap over cells"
python false_alarm_boot.py Tongji=$(f wrap Tongji cycles.csv) MICH=$(f wrap MICH cycles.csv) ISU=$(f wrap ISU_c5 cycles.csv)
echo; echo "######## Macro-averaged regrets (nominal envelope)"
python strategy_table.py Tongji=$(f wrap Tongji cells.csv) MICH=$(f wrap MICH cells.csv) ISU=$(f wrap ISU_c5 cells.csv) \
  ISUrate=$(f wrap ISU_rate cells.csv) --out $O/strategy | tail -14
echo done
