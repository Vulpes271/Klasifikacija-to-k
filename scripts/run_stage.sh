#!/usr/bin/env bash
# Shared Docker setup for the two separately selectable stages.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:?Use stage1_describe or stage2_reason}"
shift
case "$stage" in stage1_describe) label=stage1;; stage2_reason) label=stage2;; *) exit 1;; esac
run_name="${1:-$(date +%Y%m%d-%H%M%S)}"
if (( $# > 0 )); then shift; fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then echo 'Invalid run name' >&2; exit 1; fi
run_dir="results/$label/$run_name"
if [[ -e "$run_dir" || -e "$run_dir.log" ]]; then echo "Run exists: $run_dir" >&2; exit 1; fi
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml:compose.gpu.yaml}"
mkdir -p "results/$label" data
exec > >(tee "$run_dir.log") 2>&1
pipeline_args=("$@" --output-dir "$run_dir")
docker compose build app
# Resolve/validate settings using exactly the same CLI used for inference.
model_output="$(docker compose run --rm --no-deps --entrypoint python app \
  -m "llm_object_classifier.$stage" "${pipeline_args[@]}" --list-local-models)"
if [[ -n "$model_output" ]]; then
    nvidia-smi
    docker compose up -d --wait --wait-timeout 180 ollama
    while IFS= read -r model; do
        docker compose exec -T ollama ollama pull "$model"
        docker compose run --rm --no-deps --entrypoint python app \
          -m llm_object_classifier.gpu_check --model "$model"
        # A preflight for both models should not leave them both resident.
        docker compose exec -T ollama ollama stop "$model"
    done <<< "$model_output"
fi
docker compose run --rm -T --no-deps --entrypoint python app \
  -u -m "llm_object_classifier.$stage" "${pipeline_args[@]}"
