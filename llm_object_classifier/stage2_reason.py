"""Stage 2 only: classify one selected set of saved descriptions; never load images."""
from __future__ import annotations
import argparse
from pathlib import Path
from .description_prompts import validate_description
from .experiment import combinations, load_config, new_run, options, read_json, sha256, strict_json, write_csv, write_json
from .schema import validate_frame, validate_classifications
from .vision_pipeline import exact_items, GPURequiredError
from .vision_prompts import REASON_SYSTEM, REASON_INSTRUCTION, reason_prompt
from .vision_backends import Backend

FIELDS = ["frame_id", "caption_model", "caption_variant", "model", "variant", "status", "object_id", "label",
          "classification", "reason", "observation", "attachment_evidence", "elapsed_seconds", "error"]


def run(args, config):
    paths = sorted(args.captions.glob("frame-*.json"))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive.")
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No stage 1 frame-*.json files. Select a model/variant folder, not the whole run.")
    frozen = [(p, p.read_bytes()) for p in paths]
    inputs = [(p, strict_json(data.decode("utf-8-sig"))) for p, data in frozen]
    if any(d.get("stage") != 1 for _, d in inputs):
        raise ValueError("Stage 2 requires outputs from stage1_describe.")
    # Freeze file bytes before any model calls. Images referenced in source_frame are never opened.
    system = config.get("system_file_text", REASON_SYSTEM)
    instructions = config.get("instruction_file_text", REASON_INSTRUCTION)
    run_dir = new_run(args.output_dir)
    manifest = {"stage": 2, "system_prompt": system, "instructions": instructions, "config": config,
                "source_captions": [{"path": str(p), "sha256": sha256(data)} for p, data in frozen],
                "model_metadata": {}, "note": "All reasoners receive the same saved descriptions. No images sent."}
    write_json(run_dir / "manifest.json", manifest)
    rows, summary = [], {"stage": 2, "models": {}}
    fatal = False
    for spec in config["models"]:
        backend = Backend(spec, config["timeout"])
        counts = summary["models"][spec["name"]] = {"ok": 0, "error": 0, "skipped": 0}
        try:
            manifest["model_metadata"][spec["name"]] = backend.metadata()
        except Exception as exc:
            manifest["model_metadata"][spec["name"]] = {"error": str(exc)}
        write_json(run_dir / "manifest.json", manifest)
        for variant in combinations(config):
            for index, (path, saved) in enumerate(inputs, 1):
                result = {"stage": 2, "frame_id": saved["frame_id"], "caption_source": str(path),
                          "caption_sha256": manifest["source_captions"][index - 1]["sha256"],
                          "model": spec["model"], "model_name": spec["name"], "variant": variant,
                          "system_prompt": system}
                common = {"frame_id": saved["frame_id"], "caption_model": saved["model_name"],
                          "caption_variant": saved["variant"]["id"], "model": spec["name"], "variant": variant["id"]}
                if saved.get("status") != "ok":
                    result.update({"status": "skipped", "error": "Invalid stage 1 description."})
                    rows.append({**common, "status": "skipped", "error": result["error"]})
                    counts["skipped"] += 1
                else:
                    try:
                        caption = validate_description(saved["parsed"])
                        objects = [{"id": o["id"], "label": o["label"]} for o in caption["objects"]]
                        frame = {"frame_id": saved["frame_id"], "objects": objects}
                        prompt = reason_prompt(frame["frame_id"], objects, caption, instructions)
                        result["user_prompt"] = prompt
                        print(f"[{spec['name']} / {variant['id']}] {index}/{len(inputs)} {frame['frame_id']}: reasoning...", flush=True)
                        if "error" in manifest["model_metadata"][spec["name"]]:
                            raise ValueError("Model preparation failed: " + manifest["model_metadata"][spec["name"]]["error"])
                        response = backend.generate(system, prompt, options(config, variant))
                        result.update(response)
                        if response["finish_reason"] in {"length", "max_tokens"}:
                            raise ValueError("Truncated reasoning response.")
                        parsed = strict_json(response["content"])
                        if not isinstance(parsed, dict):
                            raise ValueError("Reasoning response must be an object.")
                        exact_items(objects, parsed.get("objects"))
                        predictions = validate_classifications(validate_frame(frame), parsed)
                        result.update({"status": "ok", "parsed": parsed})
                        counts["ok"] += 1
                        # typed IDs are kept distinct throughout matching.
                        by_id = {(type(o["id"]), o["id"]): o for o in caption["objects"]}
                        for pred in predictions:
                            obs = by_id[(type(pred.id), pred.id)]
                            rows.append({**common, "status": "ok", "object_id": pred.id, "label": obs["label"],
                                         "classification": pred.classification, "reason": pred.reason,
                                         "observation": obs["observation"], "attachment_evidence": obs["attachment_evidence"],
                                         "elapsed_seconds": response["elapsed_seconds"]})
                    except Exception as exc:
                        if isinstance(exc, GPURequiredError):
                            result.update(getattr(exc, "result", {}))
                            fatal = True
                        result.update({"status": "error", "error": str(exc)})
                        rows.append({**common, "status": "error", "error": str(exc)})
                        counts["error"] += 1
                        print(f"  ERROR: {exc}", flush=True)
                write_json(run_dir / spec["name"] / variant["id"] / f"frame-{index:06d}.json", result)
                write_json(run_dir / "summary.json", summary)
                write_csv(run_dir / "classifications.csv", rows, FIELDS)
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
    print(f"Results: {run_dir}", flush=True)
    return 1 if fatal else (2 if any(c["error"] or c["skipped"] for c in summary["models"].values()) else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2/qwen-reason.json"))
    parser.add_argument("--captions", type=Path, help="Stage 1 model/variant directory.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--list-local-models", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, 2)
        if args.list_local_models:
            for model in dict.fromkeys(s["model"] for s in config["models"]
                                       if s["backend"] == "ollama" and s["base_url"].rstrip("/") == "http://ollama:11434"):
                print(model)
            return 0
        if args.captions is None or args.output_dir is None:
            parser.error("--captions and --output-dir are required.")
        return run(args, config)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
