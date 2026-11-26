#!/bin/bash
# Baselines de la littérature + fusion, 4 protocoles, Tongji NCA + NCM.
# Usage (dans le dossier des scripts) :  R=$WORK/Inversion/results nohup bash run_lit.sh > logs/run_lit.log 2>&1 &
# Modèles BatteryLife/BatteryMFormer :
#   R=... MODEL_GROUPS="CPMLP CPTransformer DLinear,PatchTST iTransformer PatchMLP,ConvTimeNet" \
#   EXTRA="--ext-repo $WORK/BatteryMFormer" nohup bash run_lit.sh > logs/run_ext.log 2>&1 &
# Les prédictions sont sauvegardées pli par pli : si le nœud de connexion tue un run (limite de temps CPU),
# il est relancé automatiquement et reprend où il s'était arrêté (jusqu'à MAX_TRY fois).
set -u
NPZ=${NPZ:-$R/benchT_Tongji/samples.npz}
MAX_TRY=${MAX_TRY:-12}
# groupes de modèles séparés par des virgules ; EXTRA : options supplémentaires (ex. --ext-repo ...)
MODEL_GROUPS=${MODEL_GROUPS:-"HGB,MLP CNN LSTM,GPR"}
EXTRA=${EXTRA:-}
IFS=, read -ra GROUPS_ARR <<< "$MODEL_GROUPS"
# jeu de données : options de sélection (sans espaces dans les regex) et dossier de résultats
OPTS_STR=${OPTS_STR:-"--include-regex ^Tongji[12] --temp-regex CY(\d+)- --chem-regex ^(Tongji\d)"}
read -ra OPTS <<< "$OPTS_STR"
OUTROOT=${OUTROOT:-$R}
mkdir -p "$OUTROOT"
NT=${NT:-2}   # threads CPU par processus (2 sur le nœud de connexion ; plus sur un nœud de calcul)
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
mkdir -p logs
MODES=${MODES:-"cell temp chem chemtemp"}   # protocoles à lancer (ISU-ILCC : "cell chemcv" ou "cell chem")
for m in $MODES; do
  for g in "${GROUPS_ARR[@]}"; do
    tag=${g// /}
    out=$OUTROOT/lit_${m}_${tag}
    fast=0
    for try in $(seq 1 $MAX_TRY); do
      if [ -f "$out/paired.csv" ]; then break; fi
      echo "=== $(date +%H:%M) $m / $g (essai $try)"
      t0=$(date +%s)
      python -X faulthandler lit_baselines.py --npz "$NPZ" "${OPTS[@]}" --mode $m --models $g --n-threads $NT $EXTRA --out "$out" \
        >> logs/$(basename $OUTROOT)_lit_${m}_${tag}.log 2>&1 && continue
      # échec en moins d'une minute : erreur de code/environnement (on s'arrête après 3 fois) ou
      # plantage ponctuel, ex. « Segmentation fault » des bibliothèques natives (une relance suffit)
      if [ $(( $(date +%s) - t0 )) -lt 60 ]; then
        fast=$((fast + 1))
        if [ $fast -ge 3 ]; then echo "  ERREUR répétée, arrêt :"; tail -n 15 logs/$(basename $OUTROOT)_lit_${m}_${tag}.log; exit 1; fi
        echo "  échec immédiat ($fast/3), nouvel essai"; sleep 5; continue
      fi
      echo "  interrompu, reprise"
    done
    [ -f "$out/paired.csv" ] || echo "  ÉCHEC DÉFINITIF : $out (voir logs/$(basename $OUTROOT)_lit_${m}_${tag}.log)"
  done
done
python lit_summary.py $OUTROOT/lit_*_* --out $OUTROOT/lit_summary.csv
