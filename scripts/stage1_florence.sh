#!/usr/bin/env bash
# Native first step: a complete detailed caption followed by grounding it.
set -euo pipefail
cd "$(dirname "$0")/.."
work=results/florence-caption-grounding-test10
prepare_phase=prepare
export_phase=export
if [[ "${1:-}" == '--loco30' ]]; then
  work=results/florence-caption-grounding-loco20
  prepare_phase=prepare-remaining
  export_phase=export-remaining
fi
python3 -m llm_object_classifier.florence_grounding "$prepare_phase"
mkdir -p "$work/logs"
export THUD_PROGRESS_FILE="$PWD/$work/progress.md"
docker run --rm --gpus all --ipc=host --name florence-caption-grounding \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONFAULTHANDLER=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" -e THUD_PROGRESS_FILE \
  -v "$PWD:$PWD" -w "$PWD" thud-florence:local \
  python -m llm_object_classifier.thud_florence \
  --pipeline-config configs/stage1/florence-caption-grounding.json \
  --selection "$work/inputs.json" \
  --output-dir "$work/native" \
  --limit 30 --seed 42 --repeat caption-grounding --no-warmup \
  > "$work/logs/inference.log" 2>&1
python3 -m llm_object_classifier.florence_grounding "$export_phase"
