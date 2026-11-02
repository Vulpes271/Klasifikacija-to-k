#!/usr/bin/env bash
# Native first step: a complete detailed caption followed by grounding it.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m llm_object_classifier.florence_grounding prepare
mkdir -p results/florence-caption-grounding-test10/logs
export THUD_PROGRESS_FILE="$PWD/results/florence-caption-grounding-test10/progress.md"
docker run --rm --gpus all --ipc=host --name florence-caption-grounding \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONFAULTHANDLER=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" -e THUD_PROGRESS_FILE \
  -v "$PWD:$PWD" -w "$PWD" thud-florence:local \
  python -m llm_object_classifier.thud_florence \
  --pipeline-config configs/stage1/florence-caption-grounding.json \
  --selection results/florence-caption-grounding-test10/inputs.json \
  --output-dir results/florence-caption-grounding-test10/native \
  --limit 10 --seed 42 --repeat caption-grounding --no-warmup \
  > results/florence-caption-grounding-test10/logs/inference.log 2>&1
python3 -m llm_object_classifier.florence_grounding export
