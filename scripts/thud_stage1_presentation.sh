#!/usr/bin/env bash
# Stage one only: native Qwen3-VL, then export audited existing Florence/NVILA.
set -euo pipefail
cd "$(dirname "$0")/.."
main() {
  local phase=${1:-export}
  local output=results/stage1-presentation
  mkdir -p "$output/logs"
  case "$phase" in
    prepare)
      docker build -f docker/thud-qwen3-vl.Dockerfile -t thud-qwen3-vl:local .
      docker run --rm -v "$PWD:$PWD" -w "$PWD" \
        -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 thud-qwen3-vl:local \
        python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-VL-8B-Instruct",revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",local_dir="results/stage1-presentation/models/qwen3-vl",max_workers=4)'
      ;;
    qwen)
      docker run --rm --gpus all --ipc=host --name thud-qwen3-vl \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" \
        -e THUD_PROGRESS_FILE="$PWD/$output/progress.md" \
        -v "$PWD:$PWD" -w "$PWD" thud-qwen3-vl:local \
        python -m llm_object_classifier.thud_qwen3_vl
      ;;
    export) python3 -m llm_object_classifier.thud_stage1_presentation ;;
    *) echo 'Use prepare | qwen | export' >&2; return 2 ;;
  esac
}
main "$@"
