#!/bin/bash
# Lists every lit_<protocol>_<models> directory used in the paper (5 settings x 3 seeds) into inner_tasks.txt,
# one line per array task: <lit dir>|<setting>|<seed>. Then submits the array.
#   bash make_inner_tasks.sh            (dry run: writes the list and prints the sbatch command)
#   bash make_inner_tasks.sh --submit
set -u
R=$WORK/Inversion/results
: > inner_tasks.txt
add () {  # setting seed root protocols...
  local set=$1 seed=$2 root=$3; shift 3
  for p in "$@"; do
    for d in $root/lit_${p}_*; do
      [ -d "$d" ] || continue
      [[ $(basename $d) =~ ^lit_${p}_[A-Za-z]+$ ]] || continue
      echo "$d|$set|$seed" >> inner_tasks.txt
    done
  done
}
for s in 0 1 2; do
  if [ $s = 0 ]; then rt=$R; rb=$R/Tongji_blend; rm=$R/MICH; r5=$R/ISU_c5; rr=$R/ISU_rate
  else rt=$R/seeds/Tongji_s$s; rb=$R/seeds/Tongji_blend_s$s; rm=$R/seeds/MICH_s$s; r5=$R/seeds/ISU_c5_s$s; rr=$R/seeds/ISU_rate_s$s; fi
  add Tongji $s $rt cell temp chem chemtemp
  add Blend  $s $rb chem
  add MICH   $s $rm cell temp chem chemtemp
  add ISU_c5 $s $r5 cell chem
  add ISU_rate $s $rr chemcv
done
N=$(wc -l < inner_tasks.txt)
echo "$N tasks in inner_tasks.txt"; cut -d'|' -f2,3 inner_tasks.txt | sort | uniq -c
CMD="sbatch --array=1-${N}%${MAXPAR:-24} --export=ALL run_inner_oof.slurm"
if [ "${1:-}" = "--submit" ]; then $CMD; else echo "to submit: $CMD"; fi
