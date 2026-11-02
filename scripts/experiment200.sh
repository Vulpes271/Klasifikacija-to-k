#!/usr/bin/env bash
# Resume every stage; failed calls remain recorded and are never repeated.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/experiment200
python3 scripts/prepare_loco_sample.py --extend100 --output data/loco_test100 \
  > results/experiment200/download.log 2>&1 || echo 'LOCO download incomplete; continuing with saved photographs.'
bash scripts/stage1_florence.sh --experiment200 > results/experiment200/florence-run.log 2>&1
bash scripts/florence_grounding_llm.sh run --experiment200 > results/experiment200/llm-run.log 2>&1
