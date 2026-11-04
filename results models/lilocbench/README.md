# LILocBench — independent active-pipeline test on pose-selected RGB pairs

Source and citation: **N. Trekel, T. Guadagnino, T. Läbe, L. Wiesmann, P. Aguiar,
J. Behley and C. Stachniss**, *Benchmark for Evaluating Long-Term Localization in
Indoor Environments Under Substantial Static and Dynamic Scene Changes*, IROS 2025.
[Official dataset](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/).
Selected data and adaptations: **CC BY-NC-SA 4.0**; see `LICENSE-DATA.md` and `CITATION.bib`.
The repository's code license is unchanged.

## Selection and transfer

`pairs.json` freezes 50 pairs / 100 unique original PNGs from `mapping` and
`lt_changes_0`, before any classification. All pairs use **camera_front/color**
and satisfy the initial **0.25 m / 10 degree** calibrated-camera thresholds;
the permitted fallback 0.5 m / 20 degrees was not needed. Camera optical forward
direction is verified from the official transformation tree. Poses are the public
AprilTag-derived global `base_link` ground truth, not wheel odometry.

Translations are linearly interpolated and rotations use quaternion SLERP at RGB
timestamps, followed by the calibrated base-to-color-optical transform. Per-view
records include robot/camera poses, nearest GT timestamp, time offset, bracketing
timestamps, interpolation fraction, archive path, CRC and SHA256. GT brackets
greater than 0.1 s or nearest-pose offsets greater than 0.05 s are excluded.

RGB metadata is thinned to 0.5 s. A deterministic spatial maximin selection
prefers new 3 m cells and rejects a candidate when either trajectory has an
already selected center within 0.75 m **and** rotation within 25 degrees.
These cells are coverage aids, not ground-truth room labels. All five contact
sheets were visually checked for view overlap; kitchen, hallway and multiple
office interiors/doorways are represented. This check does not annotate object
identity, motion or stability. See `visual-overlap-review.json`.

`download-ledger.json` and `transfer-summary.json` record actual HTTP response-body
bytes, including retries. Documentation browsing and HTTP/TLS headers are outside
the counter. ZIP/ZIP64 indexes and only selected members are read through HTTP
Range. Every archive response must be 206 with the exact Content-Range; a 200
response is closed before reading its body. CRC and size checks precede SHA256.
No full ZIP, ROS bag, LiDAR payload or depth image is downloaded. Complete archive
indexes and cached range blocks stay under ignored `results/lilocbench/`; only
index summaries and selected-member provenance are published.

## Pipeline and neutrality

Each image is independent: Florence `MORE_DETAILED_CAPTION` → the **complete**
caption in `CAPTION_TO_PHRASE_GROUNDING` → the existing text adapter → five LLMs.
`native-selection.json` and model-facing documents have neutral `view-NNNNNN`
IDs. Sequence names, poses, timestamps, pair IDs and the second image never enter
model messages. They remain only in pair provenance and the human HTML/CSV.

All five installed models use the exact saved evidence-based prompt from
`prompt_comparison40`, T=0.3, seed 42, context 8192, output budget 2048 and the same
sampler options. One main call per valid image/model, maximum 500. No old prompt,
repeats, warmups or old experiment reruns. Empty-prompt model loading and native
tokenization are setup operations, not extra generated responses. Each model is
loaded sequentially and checked for full GPU allocation; CPU fallback is refused.

The existing runner tokenizer counts the actual system/user text. A conservative
256-token margin covers the unchanged two-message text-only chat wrapper; this
is explicitly an estimate of the rendered chat, not an exact chat token count.
Unknown/insufficient budgets are recorded, without shortening inputs or enlarging
context. Actual token counts and context reservation are checked after each call.
Signatures cover frozen input/image SHA, exact messages, weights digest and options.
Completed positions, including errors/skips, are preserved on continuation.

## Review

`index.html` shows each pair side by side, the predicted regions, every model's
classification, full reason and complete raw response. `region-classes.csv`
contains per-region output with whole-response validity flags. `summary.json`
counts calls, schema/API errors, skips and times; it is not content accuracy.

`pair-review-empty.csv` is intentionally blank except for known pair IDs. Fill
region IDs yourself and confirm the same identity before marking movement.
Local region IDs across images do **not** establish object identity. Unchanged
does not mean fixed; disappearance can reflect occlusion or changed viewpoint.
No localization performance or accuracy is reported without manual references.

From the repository root:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory "results models/lilocbench"
```

Open `http://127.0.0.1:8766/`. For single-region blind reference review:

```bash
python3 -m llm_object_classifier.review_viewer --results-dir "results models/lilocbench" --db reviews/lilocbench.sqlite3
```

This is a separate review namespace and database. Existing 40-image ratings are
untouched. One reference applies to all five models; reason review is optional.
Private ratings and backups stay excluded from Git.

To continue only this experiment using existing environments/models:

```bash
bash scripts/lilocbench.sh all
```

Individual phases are `prepare`, `vision`, `text`, `export`. The first setup uses
existing host NumPy/SciPy/PyYAML/Pillow, the existing `thud-florence:local` image,
and the existing Ollama container. No new models or packages are installed.
