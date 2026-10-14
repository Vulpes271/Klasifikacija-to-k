# Image observations -> mobility reasoning (Qwen3.5 4B)

Stage 1 receives the photograph and exact semantic-region list, and describes
visible observations. Stage 2 receives only that description and the list, then
assigns fixed/movable/uncertain. The same model weights serve both separate calls.
No paid API, public tunnel or host port is required.

The 20 NYUv2 frames are each described once (temperature 0). Reasoning uses 0,
0.3 and 0.7 on the same cached description. Settings: thinking off, seed 42,
top_p 0.8, top_k 20, presence_penalty 1.5, context 8192, output limit 2048.
One run per temperature is exploratory; use manual references and several seeds
for a reliable quality comparison.

## Run on Linux + NVIDIA GPU

From the repository root, with data/nyu20/objects.jsonl already present:

```bash
bash scripts/test_vision_pipeline.sh vision-first
```

To keep running after closing the terminal/RustDesk:

```bash
nohup bash scripts/test_vision_pipeline.sh vision-first > vision-launch.log 2>&1 < /dev/null &
tail -f results/vision-first.log
```

Ctrl+C stops tail, not the background test. The shell prints its PID. The computer
must stay awake. Startup errors may appear in vision-launch.log. Every new run
needs a new name; existing run directories are refused. The helper rebuilds the
application, starts only this project's Ollama, checks tests and downloads the
model if needed. Other containers and GPU jobs are not stopped.

## Results

```bash
cat results/vision-first/summary.json
head -n 15 results/vision-first/results.csv
```

- frame-001/caption.json: parsed observations, raw response and generation time.
- frame-001/reasoning-t0.7.json: raw response, classification, prompt and time.
- manifest.json: prompts, input hash, model metadata/quantization, Ollama and settings.
- summary.json: updated after each frame; inspect errors and skipped counts.
- results.csv: observations, attachment evidence, classes, reasons, times and errors.
- results/vision-caption-cache: descriptions reused only when image, labels,
  model metadata, prompts and settings match. Cached time is the ORIGINAL caption
  generation time, not the cache lookup time.

Invalid captions skip reasoning for that image and preserve the raw response.
A reasoning failure at one temperature does not stop the others. Errors are not
silently corrected. Process completion alone does not mean all answers succeeded.

## Scope and limitations

This sample has SEMANTIC CLASSES, not separate instances. Union bounding boxes
from the masks guide observation; two chairs remain one class even if different.
This is not instance detection/grounding, SAM, 3D mapping or student-model training.

Size, proximity to a wall, and standing still do not prove permanent attachment.
Prompts require uncertain where evidence is insufficient or same-class instances
conflict. A single photograph cannot expose hidden fixtures. Captions can hallucinate:
review them as well as the final labels. No fixed/movable ground truth is supplied,
so the program reports format success/coverage/time, NOT accuracy.

## Other models

After building and starting Ollama, for example keep 4B image observations and
try 0.8B text reasoning:

```bash
docker compose exec ollama ollama pull qwen3.5:0.8b
docker compose run --rm --no-deps --entrypoint python app \
  -u -m llm_object_classifier.vision_pipeline \
  --input data/nyu20/objects.jsonl --output-dir results/vision-small-reasoner \
  --vision-model qwen3.5:4b --reason-model qwen3.5:0.8b \
  --temperatures 0 0.3 0.7 --limit 20
```

The selected vision model must advertise vision support in Ollama.
Stop this project's service after all tests with docker compose down.
No daemon/driver changes or global container cleanup are performed.

References: https://docs.ollama.com/capabilities/vision,
https://docs.ollama.com/api/chat, https://huggingface.co/Qwen/Qwen3.5-4B.
Automated tests use mocked model responses. The actual vision inference must be
verified on your GPU; it was not executed in the preparation environment.
