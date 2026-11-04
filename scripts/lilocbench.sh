#!/usr/bin/env bash
# Only LILocBench; all previous experiment folders remain untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
phase="${1:-all}"
if [[ "$phase" == prepare || "$phase" == all ]]; then
  python3 -B scripts/prepare_lilocbench.py inspect
  python3 -B scripts/prepare_lilocbench.py select
  python3 -B scripts/prepare_lilocbench.py download
fi
if [[ "$phase" == vision || "$phase" == all ]]; then
  mkdir -p results/lilocbench
  export THUD_PROGRESS_FILE="$PWD/results/lilocbench/progress.md"
  docker run --rm --gpus all --ipc=host --name lilocbench-florence \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONFAULTHANDLER=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" -e THUD_PROGRESS_FILE \
    -v "$PWD:$PWD" -w "$PWD" thud-florence:local \
    python -m llm_object_classifier.thud_florence \
    --pipeline-config configs/stage1/florence-caption-grounding.json \
    --selection 'results models/lilocbench/native-selection.json' \
    --output-dir 'results models/lilocbench/native' \
    --limit 100 --seed 42 --repeat lilocbench-v1 --no-warmup \
    > results/lilocbench/florence.log 2>&1
fi
if [[ "$phase" == text || "$phase" == all ]]; then
  if [[ -z "${THUD_OLLAMA_URL:-}" ]]; then
    lilocbench_ip="$(docker inspect klasifikacija-ollama-1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"
    export THUD_OLLAMA_URL="http://$lilocbench_ip:11434"
  fi
  python3 -B -m llm_object_classifier.lilocbench run
fi
if [[ "$phase" == export ]]; then
  python3 -B -m llm_object_classifier.lilocbench export
fi
