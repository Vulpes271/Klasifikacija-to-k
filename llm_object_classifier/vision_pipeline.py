"""Two-stage Ollama vision -> text reasoning, with cached observations and full logs."""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import json
import itertools
import math
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


DEFAULT_OPTIONS = {"num_ctx": 8192, "num_predict": 2048, "top_p": 0.8,
                   "top_k": 20, "presence_penalty": 1.5, "num_gpu": 999}


class GPURequiredError(RuntimeError):
    """Abort the run when GPU execution cannot be confirmed."""


def loaded_gpu(base_url, model, timeout=600):
    state = request_json(base_url, "/api/ps", timeout=timeout)
    for entry in state.get("models", []):
        if model in (entry.get("name"), entry.get("model")):
            if entry.get("size_vram", 0) > 0:
                return entry
            break
    raise GPURequiredError(f"GPU use not confirmed for {model}. Check NVIDIA Container Toolkit, "
                           "docker compose GPU override, VRAM and Ollama logs. CPU fallback refused.")


def chat(base_url, model, system, prompt, temperature, seed, image=None, timeout=600,
         options=None, require_gpu=False):
    message = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [base64.b64encode(image).decode()]
    effective = {**DEFAULT_OPTIONS, **(options or {}), "temperature": temperature, "seed": seed}
    started = time.perf_counter()
    response = request_json(base_url, "/api/chat", {
        "model": model, "messages": [{"role": "system", "content": system}, message],
        "stream": False, "think": False, "format": "json", "keep_alive": "5m",
        "options": effective}, timeout)
    elapsed = time.perf_counter() - started
    result = {"content": response.get("message", {}).get("content", ""),
        "elapsed_seconds": elapsed, "options": effective,
        "done_reason": response.get("done_reason"),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count")}
    if require_gpu:
        result["gpu"] = loaded_gpu(base_url, model, timeout)
    return result


def variants(args):
    profiles = [{"name": "default", "options": {}}]
    if getattr(args, "profiles", None):
        profiles = json.loads(args.profiles.read_text())
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a nonempty JSON list.")
    result, names = [], set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Each profile must be an object.")
        name, options = profile.get("name"), profile.get("options", {})
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name in names:
            raise ValueError("Profile names must be unique and use letters, digits, _ or -.")
        names.add(name)
        if not isinstance(options, dict) or set(options) - set(DEFAULT_OPTIONS):
            raise ValueError("Unsupported profile options.")
        effective = {**DEFAULT_OPTIONS, "num_ctx": getattr(args, "num_ctx", 8192),
                     "num_predict": getattr(args, "num_predict", 2048), **options}
        for key, value in effective.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid option {key}.")
        if not 0 < effective["top_p"] <= 1 or not 0 <= effective["presence_penalty"] <= 2:
            raise ValueError("top_p must be in (0,1], presence_penalty in [0,2].")
        for key in ("num_ctx", "num_predict", "top_k", "num_gpu"):
            if type(effective[key]) is not int or effective[key] < (0 if key in ("top_k", "num_gpu") else 1):
                raise ValueError(f"Invalid integer option {key}.")
        for temperature, seed in itertools.product(args.temperatures, getattr(args, "seeds", None) or [args.seed]):
            result.append({"id": f"{name}-t{temperature}-s{seed}", "profile": name,
                           "temperature": temperature, "seed": seed, "options": effective})
    if len({v["id"] for v in result}) != len(result):
        raise ValueError("Duplicate experiment settings.")
    return result


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
    experiments = variants(args)
    require_gpu = getattr(args, "require_gpu", False)
    caption_options = {**DEFAULT_OPTIONS, "num_ctx": getattr(args, "num_ctx", 8192),
                       "num_predict": getattr(args, "num_predict", 2048)}
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
    if getattr(args, "require_unquantized", False):
        for name, info in models.items():
            precision = info.get("details", {}).get("quantization_level", "").upper()
            if precision not in {"BF16", "F16", "F32"}:
                raise ValueError(f"{name}: expected unquantized BF16/F16/F32, got {precision!r}.")
    write_json(run_dir / "manifest.json", {
        "prompt_version": PROMPT_VERSION, "caption_system": CAPTION_SYSTEM,
        "reason_system": REASON_SYSTEM, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "experiments": experiments, "caption_options": caption_options,
        "models": models, "ollama": request_json(args.base_url, "/api/version"),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "note": "Semantic class regions, not object instances. No fixed/movable ground truth; no accuracy."})
    summary = {"frames": len(records), "caption_ok": 0, "caption_error": 0, "caption_cached": 0,
        "reasoning": {str(t): {"ok": 0, "error": 0, "skipped": 0, "objects": 0} for t in args.temperatures},
        "experiments": {v["id"]: {"ok": 0, "error": 0, "skipped": 0, "objects": 0, **v}
                        for v in experiments}}
    rows_path = run_dir / "results.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as out:
        fields = ["frame_id", "object_id", "label", "experiment", "profile", "seed", "top_p", "top_k",
                  "presence_penalty", "num_ctx", "num_predict", "temperature", "status", "classification",
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
                    "version": PROMPT_VERSION, "options": caption_options},
                    sort_keys=True).encode()).hexdigest()
                cache = args.cache_dir / f"{key}.json"
                if cache.exists():
                    caption_info = json.loads(cache.read_text())
                    caption = validate_caption(objects, caption_info["parsed"])
                    caption_info["cached"] = True
                    summary["caption_cached"] += 1
                else:
                    caption_info = chat(args.base_url, args.vision_model, CAPTION_SYSTEM, prompt,
                        args.caption_temperature, args.seed, image, args.timeout,
                        options=caption_options, require_gpu=require_gpu)
                    caption = validate_caption(objects, json.loads(caption_info["content"]))
                    caption_info.update({"parsed": caption, "cached": False})
                    write_json(cache, caption_info)
                write_json(frame_dir / "caption.json", caption_info)
                write_json(frame_dir / "input.json", {"record": record, "regions": objects, "caption_prompt": prompt})
                summary["caption_ok"] += 1
                print(f"  opis OK ({caption_info['elapsed_seconds']:.1f} s, cache={caption_info['cached']})", flush=True)
            except GPURequiredError:
                raise
            except Exception as exc:
                summary["caption_error"] += 1
                write_json(frame_dir / "caption_error.json", {"error": str(exc), "response": caption_info})
                print(f"  NAPAKA OPISA: {exc}", flush=True)
            for variant in experiments:
                temperature, seed = variant["temperature"], variant["seed"]
                counters = summary["experiments"][variant["id"]]
                aggregate = summary["reasoning"][str(temperature)]
                row_settings = {"experiment": variant["id"], "profile": variant["profile"],
                                "seed": seed, **{k: v for k, v in variant["options"].items() if k != "num_gpu"}}
                def count(key, value=1):
                    counters[key] += value
                    aggregate[key] += value
                if caption is None:
                    count("skipped")
                    writer.writerow({**row_settings, "frame_id": frame_id, "temperature": temperature,
                                     "status": "skipped_caption_error", "error": "No valid visual observations."})
                    continue
                prompt = reason_prompt(frame_id, record["objects"], caption)
                result = {}
                started = time.perf_counter()
                try:
                    result = chat(args.base_url, args.reason_model, REASON_SYSTEM, prompt,
                                  temperature, seed, timeout=args.timeout,
                                  options=variant["options"], require_gpu=require_gpu)
                    parsed = json.loads(result["content"])
                    exact_items(record["objects"], parsed.get("objects"))
                    predictions = validate_classifications(validate_frame(record), parsed)
                    result.update({"status": "ok", "parsed": parsed, "prompt": prompt, "experiment": variant})
                    count("ok")
                    count("objects", len(predictions))
                    labels = {typed_id(o["id"]): o["label"] for o in record["objects"]}
                    observations = {typed_id(o["id"]): o for o in caption["objects"]}
                    for pred in predictions:
                        obs = observations[typed_id(pred.id)]
                        writer.writerow({**row_settings, "frame_id": frame_id, "object_id": pred.id,
                            "label": labels[typed_id(pred.id)], "temperature": temperature, "status": "ok",
                            "classification": pred.classification, "reason": pred.reason,
                            "visual_observation": obs["observation"], "attachment_evidence": obs["attachment_evidence"],
                            "caption_seconds": caption_info["elapsed_seconds"], "caption_cached": caption_info["cached"],
                            "reason_seconds": result["elapsed_seconds"]})
                    print(f"  reasoning {variant['id']}: OK {len(predictions)} objektov ({result['elapsed_seconds']:.1f} s)", flush=True)
                except GPURequiredError:
                    raise
                except Exception as exc:
                    count("error")
                    result.update({"status": "error", "error": str(exc), "prompt": prompt, "experiment": variant,
                                   "elapsed_seconds": time.perf_counter() - started})
                    writer.writerow({**row_settings, "frame_id": frame_id, "temperature": temperature,
                                     "status": "error", "error": str(exc)})
                    print(f"  reasoning {variant['id']}: NAPAKA {exc}", flush=True)
                write_json(frame_dir / f"reasoning-{variant['id']}.json", result)
            out.flush()
            write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Rezultati: {run_dir}", flush=True)
    return 0 if summary["caption_error"] == 0 and all(
        v["error"] == 0 and v["skipped"] == 0 for v in summary["experiments"].values()) else 2


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
    parser.add_argument("--seeds", type=int, nargs="+", help="Reasoning seeds; caption always uses --seed.")
    parser.add_argument("--profiles", type=Path, help="JSON sampling profiles; crossed with temperatures and seeds.")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--require-gpu", action="store_true", help="Abort on CPU fallback; records /api/ps per request.")
    parser.add_argument("--require-unquantized", action="store_true", help="Check both models' reported weight precision.")
    args = parser.parse_args()
    if args.limit < 1 or len(set(args.temperatures)) != len(args.temperatures):
        parser.error("limit must be positive and temperatures must be unique.")
    if any(not math.isfinite(t) or t < 0 for t in [*args.temperatures, args.caption_temperature]):
        parser.error("Temperatures must be finite and nonnegative.")
    try:
        return run(args)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
