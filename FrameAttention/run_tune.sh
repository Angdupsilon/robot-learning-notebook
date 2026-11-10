#!/usr/bin/env bash
# The tuning sweep from PREDICTIONS.md on Shapes: learning rates x variants, seed 0 only.
# Runs go to runs_tune/steps<STEPS>/lr<lr>/, kept apart from runs/. Finished runs are skipped.
#
#   PY="uv run python" bash run_tune.sh                     # 30,000 steps, all variants
#   STEPS=60000 PY="uv run python" bash run_tune.sh         # the next step budget
#   VARIANTS="A A_prime K" PY="uv run python" bash run_tune.sh   # only some variants
#
# Baselines run first, so you can stop early and still have their sweep.
# Then read the choice, which uses iid validation only:
#   uv run python pick_tuning.py --steps 30000
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PY=${PY:-python}
STEPS=${STEPS:-30000}
LRS=${LRS:-"3e-4 1e-3 3e-3"}
VARIANTS=${VARIANTS:-"A A_prime K B C D"}
mkdir -p logs
for v in $VARIANTS; do
  for lr in $LRS; do
    out="runs_tune/steps$STEPS/lr$lr"
    if [ -f "$out/shapes/$v/seed0/results.json" ]; then
      echo "== $v lr $lr steps $STEPS: already done"
      continue
    fi
    echo "== $v lr $lr steps $STEPS"
    # 20 evaluations, so that 90% of training lands exactly on one for the plateau check
    $PY train.py --task shapes --variant "$v" --seed 0 --lr "$lr" --steps "$STEPS" \
      --eval-every $((STEPS / 20)) --out "$out" ${EXTRA:-} 2>&1 | tee "logs/tune_${v}_lr${lr}_steps${STEPS}.log"
  done
done
$PY pick_tuning.py --steps "$STEPS" --lrs $LRS
