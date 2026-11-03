#!/usr/bin/env bash
# Exactly 120 main + 60 repeat calls; resume skips every saved attempt.
set -euo pipefail
cd "$(dirname "$0")/.."
docker exec klasifikacija-ollama-1 nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
if [[ -z "${THUD_OLLAMA_URL:-}" ]]; then
  prompt40_ollama_ip="$(docker inspect klasifikacija-ollama-1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"
  export THUD_OLLAMA_URL="http://$prompt40_ollama_ip:11434"
fi
python3 -m llm_object_classifier.prompt_comparison40 "${1:-run}"
