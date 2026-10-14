"""Two-stage Ollama vision -> text reasoning, with cached observations and full logs."""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image
from .schema import validate_frame, validate_classifications
from .vision_prompts import PROMPT_VERSION, CAPTION_SYSTEM, REASON_SYSTEM, caption_prompt, reason_prompt

ATTACHMENTS = {"visible_installation", "appears_freestanding", "not_visible", "mixed"}


def typed_id(value):
    if type(value) not in (int, str) or value == "":
        raise ValueError("Object IDs must be numbers or nonempty strings, never booleans.")
    return type(value).__name__, value


def exact_items(objects, items):
    if not isinstance(items, list):
        raise ValueError("Response must contain an objects list.")
    expected = {typed_id(o["id"]) for o in objects}
    found = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each response item must be an object.")
        key = typed_id(item.get("id"))
        if key not in expected or key in found:
            raise ValueError(f"Unexpected, wrong-type or duplicate ID: {item.get('id')!r}")
        found.add(key)
    if found != expected:
        raise ValueError(f"Missing IDs: {expected - found}")


def validate_caption(objects, result):
    if not isinstance(result, dict) or not isinstance(result.get("scene_description"), str):
        raise ValueError("Missing scene_description.")
    if not result["scene_description"].strip():
        raise ValueError("Empty scene_description.")
    exact_items(objects, result.get("objects"))
    for item in result["objects"]:
        if not isinstance(item.get("observation"), str) or not item["observation"].strip():
            raise ValueError("Missing object observation.")
        if item.get("attachment_evidence") not in ATTACHMENTS:
            raise ValueError("Invalid attachment_evidence.")
        if type(item.get("identifiable")) is not bool:
            raise ValueError("identifiable must be a boolean.")
    return result


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def request_json(base_url, path, body=None, timeout=600):
    request = urllib.request.Request(base_url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {message}") from None


def chat(base_url, model, system, prompt, temperature, seed, image=None, timeout=600):
    message = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [base64.b64encode(image).decode()]
    started = time.perf_counter()
    response = request_json(base_url, "/api/chat", {
        "model": model, "messages": [{"role": "system", "content": system}, message],
        "stream": False, "think": False, "format": "json", "keep_alive": "5m",
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192,
                    "num_predict": 2048, "top_p": 0.8, "top_k": 20,
                    "presence_penalty": 1.5}}, timeout)
    return {"content": response.get("message", {}).get("content", ""),
        "elapsed_seconds": time.perf_counter() - started,
        "done_reason": response.get("done_reason"),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count")}


def region_objects(record, data_dir):
    """NYU semantic-class union boxes; explicitly NOT instance detections."""
    objects = [{"id": o["id"], "label": o["label"],
                "region_kind": "semantic_class_region", "bbox_xyxy": None}
               for o in record["objects"]]
    mask_name = record.get("mask_path")
    if mask_name:
        with Image.open(data_dir / mask_name) as mask:
            mask.load()
            if mask.mode not in {"P", "L", "I", "I;16", "I;16B", "I;16L"}:
                raise ValueError("Semantic mask must contain integer IDs.")
            w, h = mask.size
            bounds = {}
            pixels = mask.get_flattened_data() if hasattr(mask, "get_flattened_data") else mask.getdata()
            for index, raw_id in enumerate(pixels):
                if raw_id == 0:
                    continue
                x, y = index % w, index // w
                box = bounds.setdefault(raw_id, [x, y, x + 1, y + 1])
                box[0], box[1] = min(box[0], x), min(box[1], y)
                box[2], box[3] = max(box[2], x + 1), max(box[3], y + 1)
            for obj in objects:
                obj["bbox_xyxy"] = bounds.get(obj["id"])
                obj["image_size"] = [w, h]
    return objects


def run(args):
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "manifest.json").exists():
        raise ValueError("Output run already exists. Choose a new directory; old results are preserved.")
    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()][:args.limit]
    if not records:
        raise ValueError("Input has no frames.")
    for record in records:
        validate_frame(record)
    models = {name: request_json(args.base_url, "/api/show", {"model": name})
              for name in {args.vision_model, args.reason_model}}
    if "vision" not in models[args.vision_model].get("capabilities", []):
        raise ValueError("Selected vision model does not report vision capability.")
    write_json(run_dir / "manifest.json", {
        "prompt_version": PROMPT_VERSION, "caption_system": CAPTION_SYSTEM,
        "reason_system": REASON_SYSTEM, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "models": models, "ollama": request_json(args.base_url, "/api/version"),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "note": "Semantic class regions, not object instances. No fixed/movable ground truth; no accuracy."})
    summary = {"frames": len(records), "caption_ok": 0, "caption_error": 0, "caption_cached": 0,
        "reasoning": {str(t): {"ok": 0, "error": 0, "skipped": 0, "objects": 0} for t in args.temperatures}}
    rows_path = run_dir / "results.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as out:
        fields = ["frame_id", "object_id", "label", "temperature", "status", "classification",
                  "reason", "visual_observation", "attachment_evidence", "caption_seconds",
                  "caption_cached", "reason_seconds", "error"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, 1):
            frame_id = record["frame_id"]
            frame_dir = run_dir / f"frame-{index:03d}"
            frame_dir.mkdir()
            caption = None
            caption_info = {"elapsed_seconds": 0, "cached": False}
            print(f"[{index}/{len(records)}] {frame_id}: opis slike", flush=True)
            try:
                image = (args.input.parent / record["image_path"]).read_bytes()
                objects = region_objects(record, args.input.parent)
                prompt = caption_prompt(frame_id, objects)
                key = hashlib.sha256(image + json.dumps({"prompt": prompt, "system": CAPTION_SYSTEM,
                    "model": args.vision_model, "model_info": models[args.vision_model],
                    "temperature": args.caption_temperature, "seed": args.seed,
                    "version": PROMPT_VERSION, "context": 8192, "output_tokens": 2048},
                    sort_keys=True).encode()).hexdigest()
                cache = args.cache_dir / f"{key}.json"
                if cache.exists():
                    caption_info = json.loads(cache.read_text())
                    caption = validate_caption(objects, caption_info["parsed"])
                    caption_info["cached"] = True
                    summary["caption_cached"] += 1
                else:
                    caption_info = chat(args.base_url, args.vision_model, CAPTION_SYSTEM, prompt,
                        args.caption_temperature, args.seed, image, args.timeout)
                    caption = validate_caption(objects, json.loads(caption_info["content"]))
                    caption_info.update({"parsed": caption, "cached": False})
                    write_json(cache, caption_info)
                write_json(frame_dir / "caption.json", caption_info)
                write_json(frame_dir / "input.json", {"record": record, "regions": objects, "caption_prompt": prompt})
                summary["caption_ok"] += 1
                print(f"  opis OK ({caption_info['elapsed_seconds']:.1f} s, cache={caption_info['cached']})", flush=True)
            except Exception as exc:
                summary["caption_error"] += 1
                write_json(frame_dir / "caption_error.json", {"error": str(exc), "response": caption_info})
                print(f"  NAPAKA OPISA: {exc}", flush=True)
            for temperature in args.temperatures:
                counters = summary["reasoning"][str(temperature)]
                if caption is None:
                    counters["skipped"] += 1
                    writer.writerow({"frame_id": frame_id, "temperature": temperature,
                                     "status": "skipped_caption_error", "error": "No valid visual observations."})
                    continue
                result = {}
                started = time.perf_counter()
                try:
                    prompt = reason_prompt(frame_id, record["objects"], caption)
                    result = chat(args.base_url, args.reason_model, REASON_SYSTEM, prompt,
                                  temperature, args.seed, timeout=args.timeout)
                    parsed = json.loads(result["content"])
                    exact_items(record["objects"], parsed.get("objects"))
                    predictions = validate_classifications(validate_frame(record), parsed)
                    result.update({"status": "ok", "parsed": parsed, "prompt": prompt})
                    counters["ok"] += 1
                    counters["objects"] += len(predictions)
                    labels = {typed_id(o["id"]): o["label"] for o in record["objects"]}
                    observations = {typed_id(o["id"]): o for o in caption["objects"]}
                    for pred in predictions:
                        obs = observations[typed_id(pred.id)]
                        writer.writerow({"frame_id": frame_id, "object_id": pred.id,
                            "label": labels[typed_id(pred.id)], "temperature": temperature, "status": "ok",
                            "classification": pred.classification, "reason": pred.reason,
                            "visual_observation": obs["observation"], "attachment_evidence": obs["attachment_evidence"],
                            "caption_seconds": caption_info["elapsed_seconds"], "caption_cached": caption_info["cached"],
                            "reason_seconds": result["elapsed_seconds"]})
                    print(f"  reasoning T={temperature}: OK {len(predictions)} objektov ({result['elapsed_seconds']:.1f} s)", flush=True)
                except Exception as exc:
                    counters["error"] += 1
                    result.update({"status": "error", "error": str(exc),
                                   "elapsed_seconds": time.perf_counter() - started})
                    writer.writerow({"frame_id": frame_id, "temperature": temperature,
                                     "status": "error", "error": str(exc)})
                    print(f"  reasoning T={temperature}: NAPAKA {exc}", flush=True)
                write_json(frame_dir / f"reasoning-t{temperature}.json", result)
            out.flush()
            write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Rezultati: {run_dir}", flush=True)
    return 0 if any(v["ok"] for v in summary["reasoning"].values()) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/nyu20/objects.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/vision-caption-cache"))
    parser.add_argument("--base-url", default="http://ollama:11434")
    parser.add_argument("--vision-model", default="qwen3.5:4b")
    parser.add_argument("--reason-model", default="qwen3.5:4b")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    parser.add_argument("--caption-temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.limit < 1 or len(set(args.temperatures)) != len(args.temperatures):
        parser.error("limit must be positive and temperatures must be unique.")
    try:
        return run(args)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
