#!/usr/bin/env bash
# Existing local Ollama models; never invokes stage one or a PDF compiler.
set -euo pipefail
cd "$(dirname "$0")/.."
phase=${1:-export}
if [[ "$phase" == pilot || "$phase" == run ]]; then
  if [[ -z "${THUD_OLLAMA_URL:-}" ]]; then
    service_id=$(docker compose -f compose.yaml -f compose.gpu.yaml ps -q ollama)
    service_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$service_id")
    export THUD_OLLAMA_URL="http://$service_ip:11434"
  fi
fi
python3 -m llm_object_classifier.thud_stage2_presentation "$phase"
