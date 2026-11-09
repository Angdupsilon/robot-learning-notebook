#!/usr/bin/env bash
# Stage 2: real benchmarks. Run this only once Stage 1 shows D beating A and A_prime.
#
#   bash run_stage2.sh                    # all tasks below, 3 seeds each
#   TASKS="movi_a" bash run_stage2.sh     # one task
#   SEEDS="0 1 2 3 4" bash run_stage2.sh  # more seeds
#
#   clevrtex  A A_prime D   grouping when appearance misleads (+ OOD materials, CAMO)
#   movi_a    A D E         does motion (common fate) help?  E vs D is the comparison
#   boxes     A A_prime D   binding in text: tracking after more moves than seen in training
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2"}
TASKS=${TASKS:-"clevrtex movi_a boxes"}
declare -A VARIANTS=( [clevrtex]="A A_prime D" [movi_a]="A D E" [boxes]="A A_prime D" )
mkdir -p logs
for task in $TASKS; do
  for seed in $SEEDS; do
    for v in ${VARIANTS[$task]}; do
      if [ -f "runs/$task/$v/seed$seed/results.json" ]; then
        echo "== $task $v seed $seed: already done"
        continue
      fi
      echo "== $task $v seed $seed"
      $PY train.py --task "$task" --variant "$v" --seed "$seed" ${EXTRA:-} 2>&1 | tee "logs/${task}_${v}_seed${seed}.log"
    done
  done
  $PY summarize.py --task "$task"
done
