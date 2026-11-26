#!/bin/bash
# After all inner_oof tasks are done: α by inner validation for every setting and seed, then the summary.
#   bash run_alpha_cv.sh > logs/alpha_cv.log 2>&1 ; tail -40 logs/alpha_cv.log
set -u
R=$WORK/Inversion/results
A=$R/review7/alpha_cv
W=$R/review6/wrap
T="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d) --protocols cell temp chem chemtemp"
B="--npz $R/benchT_Tongji/samples.npz --include-regex ^Tongji --temp-regex CY(\d+)- --chem-regex ^(Tongji3|Tongji) --protocols chem"
M="--npz $R/benchT_MICH/samples.npz --include-regex ^MICH --temp-regex _(\d+)C_0 --chem-regex MICH_(BL|MC)Form --protocols cell temp chem chemtemp"
I5="--npz $R/ISU/samples.npz --include-regex ^ISU_c5_ --temp-regex _T(\d+)_ --chem-regex _ch(A|B|C)_ --protocols cell chem"
IR="--npz $R/ISU/samples.npz --temp-regex _T(\d+)_ --chem-regex ^ISU_(c\d)_ --protocols chemcv"
root () { if [ "$2" = 0 ]; then echo "$1"; else echo "$R/seeds/${3}_s$2"; fi; }
for s in 0 1 2; do
  python alpha_cv.py $T  --lit-root $(root $R $s Tongji)                    --out $A/Tongji_s$s.csv
  python alpha_cv.py $B  --lit-root $(root $R/Tongji_blend $s Tongji_blend) --out $A/Tongji_blend_s$s.csv
  python alpha_cv.py $M  --lit-root $(root $R/MICH $s MICH)                 --out $A/MICH_s$s.csv
  python alpha_cv.py $I5 --lit-root $(root $R/ISU_c5 $s ISU_c5)             --out $A/ISU_c5_s$s.csv
  python alpha_cv.py $IR --lit-root $(root $R/ISU_rate $s ISU_rate)         --out $A/ISU_rate_s$s.csv
done
g () { echo "$A/${1}_s0.csv:$W/${1}_s0/cells.csv,$A/${1}_s1.csv:$W/${1}_s1/cells.csv,$A/${1}_s2.csv:$W/${1}_s2/cells.csv"; }
python alpha_cv_summary.py Tongji=$(g Tongji) Blend=$(g Tongji_blend) MICH=$(g MICH) ISU=$(g ISU_c5) ISUrate=$(g ISU_rate) \
  --oracle $R/review6/fixed_alpha/fixed_alpha_per_seed.csv --out $A/summary.csv
