#!/usr/bin/env bash
# Separate native-model environments; one GPU model at a time, resumable outputs.
set -euo pipefail
# Parse the complete launcher before a long-running child starts. This also
# avoids resuming halfway through a file edited while a GPU process was running.
main() {
cd "$(dirname "$0")/.."
phase=${1:-all}
limit=${2:-100}
out=results/thud-extended
mkdir -p "$out/logs" "$out/vendor"
case "$phase" in
  prepare|all)
    python3 scripts/download_thud_models.py
    if [[ ! -d "$out/vendor/VILA/.git" ]]; then
      git clone https://github.com/NVlabs/VILA.git "$out/vendor/VILA"
      git -C "$out/vendor/VILA" checkout --detach 57ecbfb904b99b563db69ef7916f6390eecea516
    fi
    if [[ "$(git -C "$out/vendor/VILA" rev-parse HEAD)" != 57ecbfb904b99b563db69ef7916f6390eecea516 ]]; then
      echo 'Unexpected VILA revision; preserve it and use a separate pinned checkout.' >&2
      exit 1
    fi
    python3 scripts/patch_nvila_inference.py "$out/vendor/VILA"
    git -C "$out/vendor/VILA" diff > "$out/setup/vila-inference.patch"
    docker build -f docker/thud-nvila.Dockerfile -t thud-nvila:local .
    docker build -f docker/thud-florence.Dockerfile -t thud-florence:local .
    ;;
esac
if [[ "$phase" == nvila || "$phase" == all ]]; then
  nvila_exit=0
  docker run --rm --gpus all --name thud-nvila --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" \
    -e PYTHONPATH="$PWD/$out/vendor/VILA:$PWD" \
    -v "$PWD:$PWD" -w "$PWD" thud-nvila:local \
    python -m llm_object_classifier.thud_nvila --limit "$limit" || nvila_exit=$?
  if [[ "$nvila_exit" != 0 ]]; then
    echo "NVILA process exit: $nvila_exit; outputs retained" | tee -a "$out/logs/process-errors.log"
    if [[ "$phase" == nvila ]]; then exit "$nvila_exit"; fi
  fi
fi
if [[ "$phase" == florence || "$phase" == all ]]; then
  docker run --rm --gpus all --name thud-florence --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" \
    -v "$PWD:$PWD" -w "$PWD" thud-florence:local \
    python -m llm_object_classifier.thud_florence --limit "$limit"
fi
if [[ "$phase" == reason || "$phase" == all ]]; then
  export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
  docker compose up -d --wait ollama
  service_id=$(docker compose ps -q ollama)
  service_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$service_id")
  THUD_OLLAMA_URL="http://$service_ip:11434" python3 -m llm_object_classifier.thud_extended_reason
fi
if [[ "$phase" == report || "$phase" == all ]]; then
  python3 -m llm_object_classifier.thud_extended_report
  python3 -m llm_object_classifier.thud_report
  bash scripts/compile_thud_report.sh
fi
case "$phase" in prepare|nvila|florence|reason|report|all) ;; *) echo 'Use prepare|nvila|florence|reason|report|all [limit]' >&2; exit 1;; esac
}
main "$@"
