#!/usr/bin/env bash
# Uses existing Ollama and stage-2 transport, without loading Florence or images.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${THUD_OLLAMA_URL:-}" ]]; then
  # Fail before inference if a long-running container has lost GPU/NVML access.
  docker exec klasifikacija-ollama-1 nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
  ollama_ip="$(docker inspect klasifikacija-ollama-1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"
  export THUD_OLLAMA_URL="http://$ollama_ip:11434"
fi
phase="${1:-run}"
if (( $# > 0 )); then shift; fi
python3 -m llm_object_classifier.florence_grounding_llm "$phase" "$@"
