#!/usr/bin/env bash
# Legacy experiment retained for reproducibility; new stages: stage1_describe.sh / stage2_reason.sh.
# Same BF16 captions -> quantized/BF16 text reasoning, on NVIDIA GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
run_name="${1:-vision-gpu-$(date +%Y%m%d-%H%M%S)}"
mode="${2:-both}" # both | bf16 | quantized
limit="${LIMIT:-20}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9._-]+$ || "$run_name" == . || "$run_name" == .. ]]; then
    echo 'Invalid run name' >&2; exit 1
fi
if [[ "$mode" != both && "$mode" != bf16 && "$mode" != quantized ]]; then
    echo 'Usage: bash scripts/test_vision_benchmark.sh [run-name] [both|bf16|quantized]' >&2; exit 1
fi
if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]] || (( limit > 20 )); then
    echo 'LIMIT must be 1..20' >&2; exit 1
fi
if [[ -e "results/$run_name" || -e "results/$run_name.log" ]]; then
    echo "Run already exists: results/$run_name" >&2; exit 1
fi
mkdir -p "results/$run_name" data
printf 'Run: %s; mode: %s; frames: %s\nLog: results/%s.log\n' "$run_name" "$mode" "$limit" "$run_name"
exec > >(tee -a "results/$run_name.log") 2>&1
nvidia-smi
docker compose pull ollama
docker compose build app
docker compose up -d --wait --wait-timeout 180 ollama
# These are explicit tags; record actual precision and digest in each manifest.
docker compose exec -T ollama ollama pull qwen3.5:4b-bf16
docker compose run --rm --no-deps --entrypoint python app \
    -m llm_object_classifier.gpu_check --model qwen3.5:4b-bf16 --require-unquantized
docker compose run --rm --no-deps --entrypoint python app -m unittest discover -s tests -v
if [[ ! -f data/nyu20/objects.jsonl ]]; then
    docker compose run --rm --no-deps --entrypoint python app \
        scripts/download_nyu_sample.py --count 20 --output-dir data/nyu20
fi
if [[ "$mode" != bf16 ]]; then
    docker compose exec -T ollama ollama pull qwen3.5:4b-q4_K_M
    docker compose run --rm --no-deps --entrypoint python app \
        -m llm_object_classifier.gpu_check --model qwen3.5:4b-q4_K_M
fi
status=0
run_model() {
    local label="$1" model="$2"
    shift 2
    local rc=0
    docker compose run --rm --no-deps --entrypoint python app \
        -u -m llm_object_classifier.vision_pipeline \
        --input data/nyu20/objects.jsonl --output-dir "results/$run_name/$label" \
        --vision-model qwen3.5:4b-bf16 --reason-model "$model" \
        --temperatures 0 0.3 0.7 --seeds 42 43 --seed 42 \
        --profiles configs/vision-sweep.json --require-gpu --limit "$limit" "$@" || rc=$?
    # Exit 2 means completed with response errors; still run the other precision.
    if (( rc != 0 && rc != 2 )); then exit "$rc"; fi
    if (( rc != 0 )); then status=2; fi
}
if [[ "$mode" != quantized ]]; then
    run_model bf16 qwen3.5:4b-bf16 --require-unquantized
fi
if [[ "$mode" != bf16 ]]; then
    run_model q4 qwen3.5:4b-q4_K_M
fi
docker compose exec -T ollama ollama ps
printf '\nResults: results/%s/{bf16,q4}/summary.json and results.csv\n' "$run_name"
if (( status != 0 )); then echo 'Some responses failed: inspect summary.json and raw JSON logs.'; fi
exit "$status"
