#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml:compose.gpu.yaml}"
run_name="${1:-vision-two-stage-$(date +%Y%m%d-%H%M%S)}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9._-]+$ ]] || [[ "$run_name" == . || "$run_name" == .. ]]; then
    echo "Invalid run name" >&2
    exit 1
fi
if [[ -e "results/$run_name" ]]; then
    echo "Run directory already exists: results/$run_name" >&2
    exit 1
fi
if [[ ! -f data/nyu20/objects.jsonl ]]; then
    echo "Missing data/nyu20/objects.jsonl. Download the 20-image sample first." >&2
    exit 1
fi
mkdir -p results
printf 'Run: %s\nLog: results/%s.log\n' "$run_name" "$run_name"
exec > >(tee -a "results/$run_name.log") 2>&1
nvidia-smi
docker compose pull ollama
docker compose build app
docker compose up -d --wait --wait-timeout 180 ollama
docker compose exec -T ollama ollama pull qwen3.5:4b
docker compose run --rm --no-deps --entrypoint python app -m llm_object_classifier.gpu_check --model qwen3.5:4b
docker compose run --rm --no-deps --entrypoint python app -m unittest discover -s tests -v
docker compose run --rm --no-deps --entrypoint python app \
    -u -m llm_object_classifier.vision_pipeline \
    --input data/nyu20/objects.jsonl \
    --output-dir "results/$run_name" \
    --vision-model qwen3.5:4b --reason-model qwen3.5:4b \
    --temperatures 0 0.3 0.7 --limit 20 --require-gpu
printf '\nSummary:\n'
cat "results/$run_name/summary.json"
printf '\nModel processors:\n'
docker compose exec -T ollama ollama ps
