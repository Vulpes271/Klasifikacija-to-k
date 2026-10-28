#!/usr/bin/env bash
# Optional isolated serving of the same read-only local models. No downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
original=klasifikacija-ollama-1
server=${THUD_SERVER_NAME:-thud-stage2-ollama}
parallel=${THUD_SERVER_PARALLEL:-4}
phase=${1:-run}
case "$phase" in
  prepare)
    if docker inspect "$server" >/dev/null 2>&1; then
      docker start "$server" >/dev/null
    else
      image=$(docker inspect -f '{{.Image}}' "$original")
      network=$(docker inspect -f '{{range $name, $v := .NetworkSettings.Networks}}{{$name}}{{end}}' "$original")
      volume=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/root/.ollama"}}{{.Name}}{{end}}{{end}}' "$original")
      docker run -d --name "$server" --gpus all --network "$network" \
        --mount "type=volume,source=$volume,target=/shared-models,readonly" \
        -e OLLAMA_MODELS=/shared-models/models -e OLLAMA_HOST=0.0.0.0:11434 \
        -e "OLLAMA_NUM_PARALLEL=$parallel" -e OLLAMA_MAX_LOADED_MODELS=1 \
        -e OLLAMA_KEEP_ALIVE=5m -e OLLAMA_CONTEXT_LENGTH=4096 "$image" serve
    fi
    server_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$server")
    python3 - "$server_ip" <<'PY'
import sys,time,urllib.request,json
deadline=time.monotonic()+60
while True:
    try:
        data=json.load(urllib.request.urlopen('http://'+sys.argv[1]+':11434/api/tags',timeout=2))
        if not data.get('models'):raise RuntimeError('Shared model volume is empty')
        break
    except Exception:
        if time.monotonic()>deadline:raise
        time.sleep(.5)
PY
    ;;
  run)
    original_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$original")
    execution_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$server")
    export THUD_OLLAMA_URL="http://$original_ip:11434"
    export THUD_EXECUTION_OLLAMA_URL="http://$execution_ip:11434"
    python3 -m llm_object_classifier.thud_stage2_presentation run
    ;;
  stop) docker stop "$server" ;;
  *) echo 'Use prepare | run | stop' >&2; exit 2 ;;
esac
