#!/usr/bin/env bash
# Existing local images/weights only. Each native process owns the GPU in turn.
set -euo pipefail
main() {
cd "$(dirname "$0")/.."
phase=${1:-prepare}
mkdir -p results/loco/logs
export THUD_PROGRESS_FILE="$PWD/results/loco/progress.md"
native() {
  local model=$1 repeat=$2 pilot=${3:-false} image module
  local seed=$((41+${repeat#r}))
  local extra=()
  if [[ "$pilot" == true ]]; then extra+=(--max-new-calls 1); fi
  image="thud-$model:local"; module="llm_object_classifier.thud_${model//-/_}"
  if [[ "$model" == qwen3-vl ]]; then
    extra+=(--output-dir results/loco/vlm)
  else
    extra+=(--limit 30 --output-dir "results/loco/vlm/$model/$repeat" --repeat "$repeat" --seed "$seed")
  fi
  if [[ "$model" == nvila ]]; then
    extra+=(--system-file configs/ponovitve/nvila-description-v2.txt --schema legacy --temperature 0.3 --max-new-tokens 4096)
  fi
  docker run --rm --gpus all --ipc=host --name "loco-$model" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONFAULTHANDLER=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" -e THUD_PROGRESS_FILE \
    -e PYTHONPATH="$PWD/results/thud-extended/vendor/VILA:$PWD" \
    -v "$PWD:$PWD" -w "$PWD" "$image" python -m "$module" \
    --selection results/loco/inputs.json --no-warmup "${extra[@]}"
}
case "$phase" in
  prepare) python3 -m llm_object_classifier.loco_experiment prepare ;;
  vlm-pilots) for model in florence nvila qwen3-vl; do native "$model" r1 true; done ;;
  vlm)
    for model in florence nvila; do for repeat in r1 r2 r3; do native "$model" "$repeat"; done; done
    native qwen3-vl r1
    ;;
  qwen) native qwen3-vl r1 ;;
  pilot|run)
    # Reuse the installed read-only model server; never call its THUD run phase.
    mkdir -p results/loco/llm/setup
    if [[ ! -f results/loco/llm/setup/server-start.json ]]; then
      python3 - <<'PY'
import json
from datetime import datetime,timezone
from pathlib import Path
Path('results/loco/llm/setup/server-start.json').write_text(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat()}))
PY
    fi
    export THUD_SERVER_NAME=loco-ollama THUD_SERVER_PARALLEL=1
    export THUD_PARALLEL_SLOTS=1 THUD_CLIENT_WORKERS=1
    bash scripts/thud_stage2_parallel.sh prepare
    original_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' klasifikacija-ollama-1)
    execution_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' loco-ollama)
    export THUD_OLLAMA_URL="http://$original_ip:11434"
    export THUD_EXECUTION_OLLAMA_URL="http://$execution_ip:11434"
    exec python3 -m llm_object_classifier.loco_experiment "$phase"
    ;;
  vlm-export|freeze|export|runtime|report|verify)
    python3 -m llm_object_classifier.loco_experiment "$phase" ;;
  pdf)
    report_dir='analiza rezultatov vmesna predstavitev/industrija LOCO'
    bash scripts/compile_thud_report.sh "$report_dir" Rezultati_LOCO.tex
    bash scripts/compile_thud_report.sh "$report_dir" Priloga_LOCO.tex
    ;;
  *) echo 'Use prepare|vlm-pilots|vlm|qwen|vlm-export|freeze|pilot|run|export|runtime|report|pdf|verify' >&2; exit 2 ;;
esac
}
main "$@"
