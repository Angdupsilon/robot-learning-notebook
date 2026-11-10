#!/usr/bin/env bash
# Stage 1: every variant x 5 seeds on the synthetic shapes task, then the summary table.
#
#   bash run_stage1.sh                          # all 30 runs (finished runs are skipped)
#   VARIANTS="K" bash run_stage1.sh             # only some variants
#   VARIANTS="D_nosync D_nomoves D_frozen D_random" bash run_stage1.sh   # the ablations of D
#   SEEDS="0 1" bash run_stage1.sh              # fewer seeds
#   EXTRA="--steps 10000" bash run_stage1.sh    # extra flags for every train.py call
#   PY="uv run python" bash run_stage1.sh       # on the Mac, inside the uv environment
#
# Seeds run in the outer loop, so after the first pass you already have one seed of every variant.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2 3 4"}
VARIANTS=${VARIANTS:-"A A_prime B C D K"}
mkdir -p logs
for seed in $SEEDS; do
  for v in $VARIANTS; do
    if [ -f "runs/shapes/$v/seed$seed/results.json" ]; then
      echo "== shapes $v seed $seed: already done"
      continue
    fi
    echo "== shapes $v seed $seed"
    $PY train.py --task shapes --variant "$v" --seed "$seed" ${EXTRA:-} 2>&1 | tee "logs/shapes_${v}_seed${seed}.log"
  done
done
$PY summarize.py --task shapes
