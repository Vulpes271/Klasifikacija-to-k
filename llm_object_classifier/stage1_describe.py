"""Stage 1 only: compare image descriptions, preserving all raw outputs."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from .description_prompts import VERSION, system_prompt, user_prompt, validate_description
from .description_report import write_report
from .experiment import combinations, load_config, new_run, options, sha256, strict_json, write_csv, write_json
from .image_dataset import load_images
from .vision_backends import Backend
from .vision_pipeline import GPURequiredError

CSV_FIELDS = ["frame_id", "model", "variant", "status", "object_id", "label", "observation",
              "attachment_evidence", "identifiable", "elapsed_seconds", "gpu_verified", "error"]


def summarize(results):
    by_model = {}
    for row in results:
        name = row["model_name"]
        counts = by_model.setdefault(name, {"ok": 0, "error": 0, "objects_returned": 0,
                                          "successful_request_seconds": []})
        counts[row["status"]] += 1
        if row["status"] == "ok":
            counts["objects_returned"] += len(row["parsed"]["objects"])
            counts["successful_request_seconds"].append(row["elapsed_seconds"])
    for counts in by_model.values():
        times = counts.pop("successful_request_seconds")
        counts["mean_successful_request_seconds"] = statistics.mean(times) if times else None
        counts["mean_excluding_first_success_seconds"] = statistics.mean(times[1:]) if len(times) > 1 else None
    return {"stage": 1, "models": by_model,
            "note": "Format success and latency, not accuracy. Count of discovered objects is not detection recall."}


def save_tables(run_dir, frames, results):
    rows = []
    for row in results:
        common = {"frame_id": row["frame_id"], "model": row["model_name"],
                  "variant": row["variant"]["id"], "status": row["status"],
                  "elapsed_seconds": row.get("elapsed_seconds"), "gpu_verified": row.get("gpu_verified", False),
                  "error": row.get("error", "")}
        if row["status"] == "ok":
            for obj in row["parsed"]["objects"]:
                rows.append({**common, "object_id": obj["id"], **{k: obj[k] for k in
                             ("label", "observation", "attachment_evidence", "identifiable")}})
        else:
            rows.append(common)
    write_csv(run_dir / "descriptions.csv", rows, CSV_FIELDS)
    write_json(run_dir / "summary.json", summarize(results))
    write_report(run_dir / "comparison.html", frames, results)


def run(args, config):
    if args.limit < 1:
        raise ValueError("limit must be positive.")
    frames = load_images(args.input, args.limit, args.frame_id)
    system = config.get("system_file_text", system_prompt(config["detail"]))
    instruction = config.get("instruction_file_text", "")
    resume = getattr(args, "resume", False)
    warmup = getattr(args, "warmup", False)
    run_dir = Path(args.output_dir) if resume and Path(args.output_dir).exists() else new_run(args.output_dir)
    manifest = {"stage": 1, "prompt_version": VERSION, "system_prompt": system,
                "instruction_prefix": instruction, "config": config,
                "input_sha256": sha256(args.input.read_bytes()),
                "frames": [{"record": f["record"], "image_sha256": f["image_sha256"],
                            "image_path_resolved": f["path"], "supplied_regions": f["regions"]} for f in frames],
                "model_metadata": {}, "fresh_inference": True,
                "note": "No caption cache. Each model sees identical image bytes and prompts. Discovery IDs are local to a model/image."}
    if (run_dir / "manifest.json").exists() and resume:
        from .experiment import read_json
        previous = read_json(run_dir / "manifest.json")
        for key in ("config", "input_sha256", "system_prompt", "frames"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume input changed: {key}")
        manifest = previous
    write_json(run_dir / "manifest.json", manifest)
    results, fatal = [], False
    for spec in config["models"]:
        backend = Backend(spec, config["timeout"])
        metadata_error = None
        try:
            manifest["model_metadata"][spec["name"]] = backend.metadata(vision=True)
        except Exception as exc:
            metadata_error = str(exc)
            manifest["model_metadata"][spec["name"]] = {"error": metadata_error}
        write_json(run_dir / "manifest.json", manifest)
        if warmup and not metadata_error:
            warm_path = run_dir / spec["name"] / "warmup.json"
            if not warm_path.exists():
                first = next(combinations(config))
                try:
                    response = backend.generate(system, user_prompt(frames[0]), options(config, first), frames[0]["image"], frames[0]["mime"])
                    write_json(warm_path, {"excluded_from_quality": True, **response})
                except GPURequiredError as exc:
                    write_json(warm_path, {"excluded_from_quality": True, "error": str(exc), **getattr(exc, "result", {})})
                    raise
                except Exception as exc:
                    write_json(warm_path, {"error": str(exc), "excluded_from_quality": True, **getattr(exc, "result", {})})
        for variant in combinations(config):
            destination = run_dir / spec["name"] / variant["id"]
            for index, frame in enumerate(frames, 1):
                fid = frame["record"]["frame_id"]
                saved_path = destination / f"frame-{index:06d}.json"
                if resume and saved_path.exists():
                    from .experiment import read_json
                    results.append(read_json(saved_path))
                    continue
                prompt = (instruction + "\n" if instruction else "") + user_prompt(frame)
                row = {"stage": 1, "frame_id": fid, "source_frame": frame["record"],
                       "model_name": spec["name"], "model": spec["model"], "variant": variant,
                       "image_sha256": frame["image_sha256"], "system_prompt": system, "user_prompt": prompt,
                       "annotation_mode": "supplied" if frame["regions"] is not None else "model_discovery"}
                print(f"[{spec['name']} / {variant['id']}] {index}/{len(frames)} {fid}: describing...", flush=True)
                try:
                    if metadata_error:
                        raise ValueError("Model preparation failed: " + metadata_error)
                    response = backend.generate(system, prompt, options(config, variant), frame["image"], frame["mime"])
                    row.update(response)
                    if response["finish_reason"] in {"length", "max_tokens"}:
                        raise ValueError("Output truncated; increase max_tokens or reduce description detail.")
                    parsed = validate_description(strict_json(response["content"]), frame["regions"])
                    row.update({"status": "ok", "parsed": parsed})
                    print(f"  OK: {len(parsed['objects'])} objects, {response['elapsed_seconds']:.2f} s", flush=True)
                except GPURequiredError as exc:
                    row.update(getattr(exc, "result", {}))
                    row.update({"status": "error", "error": str(exc)})
                    fatal = True
                except Exception as exc:
                    row.update(getattr(exc, "result", {}))
                    row.update({"status": "error", "error": str(exc)})
                    print(f"  ERROR: {exc}", flush=True)
                results.append(row)
                write_json(destination / f"frame-{index:06d}.json", row)
                # JSON/CSV progress survives interruptions; HTML is built once at completion.
                write_json(run_dir / "summary.json", summarize(results))
                if fatal:
                    break
            if fatal:
                break
        if spec["backend"] == "ollama" and spec["base_url"].rstrip("/") == "http://ollama:11434" and not fatal:
            try:
                backend.unload()
            except Exception as exc:
                manifest.setdefault("unload_warnings", []).append(str(exc))
                write_json(run_dir / "manifest.json", manifest)
        if fatal:
            break
    save_tables(run_dir, frames, results)
    print(f"Results: {run_dir}\nOpen {run_dir / 'comparison.html'}", flush=True)
    return 1 if fatal else (2 if any(r["status"] != "ok" for r in results) else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1/qwen-compare.json"))
    parser.add_argument("--input", type=Path, default=Path("data/selected/frames.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--frame-id", nargs="+", help="Optional exact frame IDs, e.g. one previous photograph.")
    parser.add_argument("--resume", action="store_true", help="Reuse only identical immutable inputs; retain saved failures.")
    parser.add_argument("--warmup", action="store_true", help="Save a separate excluded warm-up call per model.")
    parser.add_argument("--list-local-models", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, 1)
        if args.list_local_models:
            for model in dict.fromkeys(s["model"] for s in config["models"]
                                       if s["backend"] == "ollama" and s["base_url"].rstrip("/") == "http://ollama:11434"):
                print(model)
            return 0
        if args.output_dir is None:
            parser.error("--output-dir is required.")
        return run(args, config)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
