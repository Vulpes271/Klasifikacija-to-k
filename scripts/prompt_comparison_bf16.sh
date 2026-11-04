#!/usr/bin/env bash
# 160 main positions, sequential BF16 models, one short GPU preflight per model.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${THUD_OLLAMA_URL:-}" ]]; then
  docker exec klasifikacija-ollama-1 nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
  bf16_ollama_ip="$(docker inspect klasifikacija-ollama-1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"
  export THUD_OLLAMA_URL="http://$bf16_ollama_ip:11434"
fi
python3 -m llm_object_classifier.prompt_comparison_bf16 "${1:-run}"
