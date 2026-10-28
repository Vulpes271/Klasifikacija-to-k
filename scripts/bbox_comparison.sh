#!/usr/bin/env bash
# Exactly four images per installed model; no warmup, retries, LLM or PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m llm_object_classifier.bbox_comparison prepare
mkdir -p results/bbox-comparison/logs
export THUD_PROGRESS_FILE="$PWD/results/bbox-comparison/progress.md"
for model in qwen3-vl nvila; do
  extra=()
  if [[ "$model" == qwen3-vl ]]; then
    module=thud_qwen3_vl
    extra+=(--output-dir results/bbox-comparison --repeat r1 --task bbox --instruction-file configs/prompts/loco-bbox.txt)
  else
    module=thud_nvila
    extra+=(--limit 4 --output-dir results/bbox-comparison/nvila/r1 --repeat r1 --seed 42 --schema bbox --system-file configs/prompts/loco-bbox.txt --temperature 0.3 --max-new-tokens 4096)
  fi
  docker run --rm --gpus all --ipc=host --name "bbox-$model" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONFAULTHANDLER=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e THUD_PROGRESS_FILE -e THUD_SOURCE_COMMIT="$(git rev-parse HEAD)" \
    -e PYTHONPATH="$PWD/results/thud-extended/vendor/VILA:$PWD" \
    -v "$PWD:$PWD" -w "$PWD" "thud-$model:local" python -m "llm_object_classifier.$module" \
    --selection results/bbox-comparison/inputs.json --no-warmup --max-new-calls 4 "${extra[@]}" \
    > "results/bbox-comparison/logs/$model.log" 2>&1
done
python3 -m llm_object_classifier.bbox_comparison export
