#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml:compose.gpu.yaml}"
run_name="${1:-vision-two-stage-$(date +%Y%m%d-%H%M%S)}"
# Remaining flags go to the Python CLI, including --config and model overrides.
if (( $# > 0 )); then shift; fi
pipeline_args=("$@" --output-dir "results/$run_name" --require-gpu)
if [[ ! "$run_name" =~ ^[a-zA-Z0-9._-]+$ ]] || [[ "$run_name" == . || "$run_name" == .. ]]; then
    echo "Invalid run name" >&2
    exit 1
fi
if [[ -e "results/$run_name" ]]; then
    echo "Run directory already exists: results/$run_name" >&2
    exit 1
fi
mkdir -p results
printf 'Run: %s\nLog: results/%s.log\n' "$run_name" "$run_name"
exec > >(tee -a "results/$run_name.log") 2>&1
nvidia-smi
docker compose pull ollama
docker compose build app
docker compose up -d --wait --wait-timeout 180 ollama
docker compose run --rm --no-deps --entrypoint python app -m unittest discover -s tests -v
# Use the same parser for preparation and inference: configuration/CLI changes
# must pull and check the selected models, rather than a hardcoded default.
model_output="$(docker compose run --rm --no-deps --entrypoint python app \
    -m llm_object_classifier.vision_pipeline "${pipeline_args[@]}" --print-models)"
mapfile -t selected_models <<< "$model_output"
if [[ ${#selected_models[@]} -ne 2 ]]; then
    echo "Expected two model names from pipeline configuration" >&2
    exit 1
fi
previous_model=""
for model in "${selected_models[@]}"; do
    if [[ "$model" == "$previous_model" ]]; then continue; fi
    docker compose exec -T ollama ollama pull "$model"
    docker compose run --rm --no-deps --entrypoint python app \
        -m llm_object_classifier.gpu_check --model "$model"
    previous_model="$model"
done
docker compose run --rm --no-deps --entrypoint python app \
    -u -m llm_object_classifier.vision_pipeline \
    "${pipeline_args[@]}"
printf '\nSummary:\n'
cat "results/$run_name/summary.json"
printf '\nModel processors:\n'
docker compose exec -T ollama ollama ps
