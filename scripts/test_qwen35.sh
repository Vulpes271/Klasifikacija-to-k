#!/usr/bin/env bash
# Legacy experiment retained for reproducibility; new stages: stage1_describe.sh / stage2_reason.sh.
set -euo pipefail

# Always run Compose from the project root, even from another directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"
export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml}"

input="data/nyu-small/objects.jsonl"
run_name="${1:-qwen3.5-small-$(date +%Y%m%d-%H%M%S)}"
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Ime testa sme vsebovati samo črke, številke, piko, podčrtaj in vezaj." >&2
  exit 1
fi
if [[ ! -f "$input" ]]; then
  echo "Manjka $input. Najprej prenesi vzorec po navodilih README_DOCKER.md." >&2
  exit 1
fi
if [[ -e "results/$run_name" ]]; then
  echo "Mapa results/$run_name že obstaja. Izberi novo ime testa." >&2
  exit 1
fi
mkdir -p results
docker compose pull ollama
docker compose up -d --wait --wait-timeout 120 ollama
docker compose build app
docker compose exec -T ollama ollama pull qwen3.5:0.8b
docker compose run --rm app run \
  --input "$input" \
  --config configs/docker-qwen3.5-small.json \
  --output-dir results \
  --run-name "$run_name"
printf '\nPovzetek testa: results/%s/summary.json\n' "$run_name"
cat "results/$run_name/summary.json"
printf '\nKlasifikacije: results/%s/results.csv\n' "$run_name"
printf 'Preveri tudi napake v povzetku. Ollama ostane zagnana; ustavi jo z: docker compose down\n'
