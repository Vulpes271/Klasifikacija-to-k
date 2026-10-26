#!/usr/bin/env bash
# Reuse installed images/models. Never rebuild, download, or erase past results.
set -euo pipefail
main() {
  cd "$(dirname "$0")/.."
  mkdir -p results/ponovitve/logs
  export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
  if [[ "${1:-all}" != prepare && "${1:-all}" != export && "${1:-all}" != verify ]]; then
    docker compose up -d --wait ollama
    local service_id
    service_id=$(docker compose ps -q ollama)
    THUD_OLLAMA_URL="http://$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$service_id"):11434"
    export THUD_OLLAMA_URL
  fi
  python3 -m scripts.thud_ponovitve --phase "${1:-all}" "${@:2}"
}
main "$@"
