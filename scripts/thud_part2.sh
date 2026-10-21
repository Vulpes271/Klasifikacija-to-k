#!/usr/bin/env bash
# Part two only. Completed part-one requests are read, never rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
phase=all
previous=
for argument in "$@"; do
  if [[ "$previous" == --phase ]]; then phase="$argument"; fi
  previous="$argument"
done
if [[ "$phase" != prepare ]]; then
  docker compose up -d --wait ollama
  docker compose exec -T ollama nvidia-smi --query-gpu=name,memory.total --format=csv
  service_id=$(docker compose ps -q ollama)
  service_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$service_id")
  export THUD_OLLAMA_URL="http://$service_ip:11434"
fi
python3 -m llm_object_classifier.thud_part2 "$@"
if [[ "$phase" == all ]]; then
  python3 -m llm_object_classifier.thud_part2_report --require-complete
  python3 -m llm_object_classifier.thud_report
  bash scripts/compile_thud_report.sh
fi
